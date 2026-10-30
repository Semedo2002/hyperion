"""Live viewer: runs a simulation and streams it to a browser, with the automatic run monitor (aegis/agent_monitor.py).

A background thread advances the solver; every few steps it publishes a frame: density and temperature on a slice
(z = mid-plane in 3D), temperature contours and in-plane magnetic field lines (traced on the server), the run
monitor's mode spectrum and alerts. The page draws the slice with WebGPU (a colour-mapped texture in a WGSL
shader) when the browser has it, otherwise with a 2D canvas, and overlays contours, field lines and alerts.
Standard library HTTP only (no web framework); frames arrive by Server-Sent Events (/events) or polling (/frame).

    python scripts/server.py                       # 2D liner cross-section implosion, http://127.0.0.1:8765
    python scripts/server.py --preset cad3d        # the 3D CAD corrugated liner (examples/cad_liner_implosion.py)
    python scripts/server.py --port 9000 --every 5

Endpoints: /  (viewer)  /frame  (latest frame, JSON)  /events  (SSE stream)  /alerts  (monitor log)  /status
"""
import os
import sys
import json
import time
import base64
import argparse
import threading
import contextlib
import io
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
with contextlib.redirect_stdout(io.StringIO()):
    from aegis.config import Config, NVAR, iRHO, iVX, iVY, iPR, iBX, iBY, iBZ, iCLR
    from aegis.physics import xp, prim_to_cons, cons_to_prim
    from aegis.solver import MHDSolver
from aegis.agent_monitor import AgentMonitor


# ============================================================
# presets
# ============================================================
def preset_liner2d(n=128, seed_m=4, drive_m3=0.03):
    """2D x-y cross-section of an imploding liner: a dense ring coasting inward (v ~ 1/r) onto an axial seed field,
    with an m = 4 surface corrugation (seeded) and a weak m = 3 asymmetry in the drive velocity (not declared to the
    monitor, as an unknown drive asymmetry would be)"""
    cfg = Config(nx=n, ny=n, x_min=-1.6, x_max=1.6, y_min=-1.6, y_max=1.6, cfl=0.3, gamma=5 / 3,
                 bc_x_type="extrapolation", bc_y_type="extrapolation", diag_interval=10**9, snapshot_times=[])
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver(cfg)
    R = xp.sqrt(s.X**2 + s.Y**2) + 1e-12
    th = xp.arctan2(s.Y, s.X)
    r_in = 1.0 * (1 + 0.01 * xp.cos(seed_m * th))
    w = 0.6 * cfg.dx
    ring = 0.5 * (xp.tanh((R - r_in) / w) - xp.tanh((R - r_in - 0.2) / w))
    W = xp.zeros_like(s.U)
    W[iRHO] = 0.05 + 0.95 * ring
    vr = -1.1 / R * ring * (1 + drive_m3 * xp.cos(3 * th))
    W[iVX], W[iVY] = vr * s.X / R, vr * s.Y / R
    W[iPR] = 0.01
    W[iBZ] = 0.1
    W[iCLR] = ring
    s.U[...] = prim_to_cons(W, cfg.gamma)
    mon = AgentMonitor(seeded_modes=(seed_m,), center=(0.0, 0.0), growth_alert=3.0)
    return s, mon, 0.8


def preset_cad3d(n=48, nz=24):
    from aegis.solver3d import MHDSolver3D
    from aegis import cad_importer as ci
    stl = os.path.join(ROOT, "examples", "data", "corrugated_liner.stl")
    tris = ci.read_cad(stl)
    cfg = Config(nx=n, ny=n, nz=nz, x_min=-1.6, x_max=1.6, y_min=-1.6, y_max=1.6, z_min=0.0, z_max=2.0, cfl=0.3,
                 gamma=5 / 3, bc_x_type="extrapolation", bc_y_type="extrapolation", powell_source=False,
                 diag_interval=10**9, snapshot_times=[])
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver3D(cfg)
    alpha = ci.Geometry(tris, ci.solver_edges(s), sub=2).volume_fraction()
    ng = s.ng
    A = np.zeros((s.nx_tot, s.ny_tot, s.nz_tot))
    A[ng:-ng, ng:-ng, ng:-ng] = alpha
    A = xp.asarray(A)
    X = s.x[:, None, None] * xp.ones((1, s.ny_tot, s.nz_tot))
    Y = s.y[None, :, None] * xp.ones((s.nx_tot, 1, s.nz_tot))
    R = xp.sqrt(X**2 + Y**2) + 1e-12
    W = xp.zeros((NVAR, s.nx_tot, s.ny_tot, s.nz_tot))
    W[iRHO] = 0.05 + 0.95 * A
    vr = -1.1 / R * A
    W[iVX], W[iVY] = vr * X / R, vr * Y / R
    W[iPR], W[iBZ], W[iCLR] = 0.01, 0.1, A
    s.U = prim_to_cons(W, cfg.gamma)
    mon = AgentMonitor(seeded_modes=(4,), slices=[nz // 8], center=(0.0, 0.0))
    return s, mon, 0.75


PRESETS = {"liner2d": preset_liner2d, "cad3d": preset_cad3d}


# ============================================================
# frames
# ============================================================
def _q8(a):
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    q = np.clip((a - lo) / (hi - lo + 1e-300) * 255, 0, 255).astype(np.uint8)
    return dict(w=int(a.shape[0]), h=int(a.shape[1]), lo=lo, hi=hi,
                data=base64.b64encode(np.ascontiguousarray(q.T).tobytes()).decode())   # row-major, y rows


def contours(f, levels):
    """polylines (normalised 0..1 coordinates) of f at the given levels"""
    import contourpy
    gen = contourpy.contour_generator(z=f.T)
    ny, nx = f.T.shape
    out = []
    for lv in levels:
        for line in gen.lines(lv):
            out.append(dict(level=float(lv), pts=np.round(np.column_stack([(line[:, 0] + 0.5) / nx,
                                                                         (line[:, 1] + 0.5) / ny]), 4).ravel().tolist()))
    return out


def field_lines(bx, by, n_seed=12, n_step=120):
    """in-plane field lines (RK2 on the cell-centred field, both directions), normalised coordinates"""
    nx, ny = bx.shape
    bm = np.sqrt(bx**2 + by**2)
    if bm.max() < 1e-12:
        return []
    h = 0.5

    def sample(f, x, y):
        i = np.clip(x, 0, nx - 1.001)
        j = np.clip(y, 0, ny - 1.001)
        i0, j0 = i.astype(int), j.astype(int)
        a, b = i - i0, j - j0
        return (1 - a) * (1 - b) * f[i0, j0] + a * (1 - b) * f[i0 + 1, j0] + (1 - a) * b * f[i0, j0 + 1] + a * b * f[i0 + 1, j0 + 1]
    sx, sy = np.meshgrid(np.linspace(0.1, 0.9, n_seed) * (nx - 1), np.linspace(0.1, 0.9, n_seed) * (ny - 1), indexing="ij")
    sx, sy = sx.ravel(), sy.ravel()
    keep = sample(bm, sx, sy) > 0.05 * bm.max()
    lines = []
    for sgn in (1.0, -1.0):
        x, y = sx[keep].copy(), sy[keep].copy()
        path = [np.column_stack([x, y])]
        for _ in range(n_step):
            ux, uy = sample(bx, x, y), sample(by, x, y)
            nrm = np.sqrt(ux**2 + uy**2) + 1e-300
            xm, ym = x + 0.5 * h * sgn * ux / nrm, y + 0.5 * h * sgn * uy / nrm
            ux, uy = sample(bx, xm, ym), sample(by, xm, ym)
            nrm = np.sqrt(ux**2 + uy**2) + 1e-300
            x, y = x + h * sgn * ux / nrm, y + h * sgn * uy / nrm
            path.append(np.column_stack([x, y]))
        P = np.stack(path, axis=1)                        # (lines, steps, 2)
        for p in P:
            inside = (p[:, 0] >= 0) & (p[:, 0] <= nx - 1) & (p[:, 1] >= 0) & (p[:, 1] <= ny - 1)
            k = int(np.argmin(inside)) if not inside.all() else len(p)
            if k > 2:
                q = p[:k] / [nx - 1, ny - 1]
                lines.append(np.round(q, 4).ravel().tolist())
    return lines


def make_frame(s, mon, step):
    W = cons_to_prim(s.U, s.eos)
    get = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)
    ng = s.ng
    if hasattr(s, "nz_tot"):
        k = ng + s.cfg.nz // 8
        sl = (slice(ng, -ng), slice(ng, -ng), k)
    else:
        sl = (slice(ng, -ng), slice(ng, -ng))
    rho = get(W[iRHO][sl])
    T = get((W[iPR] / W[iRHO])[sl])
    bx, by = get(W[iBX][sl]), get(W[iBY][sl])
    lv = np.quantile(T, [0.5, 0.8, 0.95, 0.99])
    summ = mon.summary() if mon.t else {}
    return dict(step=int(step), t=float(s.t), rho=_q8(np.log10(np.maximum(rho, 1e-12))), T=_q8(T),
                contours=contours(T, lv), fieldlines=field_lines(bx, by), monitor=summ,
                alerts=mon.alerts[-10:], extent=[s.cfg.x_min, s.cfg.x_max, s.cfg.y_min, s.cfg.y_max])


# ============================================================
# simulation thread
# ============================================================
class Session:
    def __init__(self, preset="liner2d", every=10, log_path=None, **kw):
        self.s, self.mon, self.t_end = PRESETS[preset](**kw)
        self.mon.log_path = log_path
        self.every = every
        self.frame = None
        self.version = 0
        self.cond = threading.Condition()
        self.running = False
        self.error = None
        self.step = 0

    def publish(self):
        f = make_frame(self.s, self.mon, self.step)
        with self.cond:
            self.frame = f
            self.version += 1
            self.cond.notify_all()

    def run(self, max_steps=None):
        self.running = True
        s = self.s
        try:
            self.mon.observe(s, 0.0)
            self.publish()
            while s.t < self.t_end - 1e-12 and (max_steps is None or self.step < max_steps):
                s.dt = min(s.compute_dt(cons_to_prim(s.U, s.eos)), self.t_end - s.t)
                s.step_ssprk3()
                s.t += s.dt
                self.step += 1
                if self.step % self.every == 0:
                    self.mon.observe(s)
                    self.publish()
            self.mon.observe(s)
            self.publish()
        except Exception as e:                       # shown on /status instead of killing the server silently
            self.error = repr(e)
        finally:
            self.running = False

    def start(self, max_steps=None):
        th = threading.Thread(target=self.run, kwargs=dict(max_steps=max_steps), daemon=True)
        th.start()
        return th


# ============================================================
# HTTP
# ============================================================
PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>AEGIS-MHD live</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#0f1115;--fg:#e6e6e6;--mut:#9aa0a6;--acc:#f0a35e;--warn:#ff6b6b}
body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}
header{padding:10px 16px;display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
h1{font-size:16px;margin:0}#info{color:var(--mut)}
main{display:flex;flex-wrap:wrap;gap:16px;padding:0 16px 16px}
.view{position:relative;width:min(560px,100%);aspect-ratio:1}
.view canvas{position:absolute;inset:0;width:100%;height:100%}
aside{flex:1;min-width:260px;max-width:520px}
select{background:#1b1e24;color:var(--fg);border:1px solid #333;padding:2px 6px}
.alert{border-left:3px solid var(--warn);padding:4px 8px;margin:6px 0;background:#1b1e24}
table{border-collapse:collapse}td{padding:1px 8px;color:var(--mut)}
</style></head><body>
<header><h1>AEGIS-MHD live</h1><span id="info">connecting...</span>
<label>field <select id="fld"><option value="rho">log10 density</option><option value="T">temperature</option></select></label>
<span id="gpu"></span></header>
<main><div class="view"><canvas id="gl" width="512" height="512"></canvas><canvas id="ov" width="512" height="512"></canvas></div>
<aside><h3>run monitor</h3><table id="modes"></table><div id="alerts"></div></aside></main>
<script>
const gl=document.getElementById('gl'), ov=document.getElementById('ov'), octx=ov.getContext('2d');
let gpu=null, last=null;
const WGSL=`
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
struct VO {@builtin(position) p: vec4f, @location(0) uv: vec2f};
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  var q = array<vec2f,6>(vec2f(-1,-1),vec2f(1,-1),vec2f(-1,1),vec2f(-1,1),vec2f(1,-1),vec2f(1,1));
  var o: VO; o.p = vec4f(q[i],0,1); o.uv = vec2f((q[i].x+1)/2, (1-q[i].y)/2); return o; }
fn cmap(t: f32) -> vec3f {   // magma-like polynomial
  let c0=vec3f(-0.002,-0.0007,-0.0053); let c1=vec3f(0.25,0.68,2.49); let c2=vec3f(8.35,-3.58,0.31);
  let c3=vec3f(-27.67,14.26,-13.65); let c4=vec3f(52.18,-27.94,12.94); let c5=vec3f(-50.77,29.05,4.23);
  let c6=vec3f(18.66,-11.49,-5.6);
  return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), vec3f(0), vec3f(1)); }
@fragment fn fs(v: VO) -> @location(0) vec4f { let t = textureSample(tex, smp, v.uv).r; return vec4f(cmap(t),1); }`;
async function initGPU(){
  if(!navigator.gpu) return null;
  const ad=await navigator.gpu.requestAdapter(); if(!ad) return null;
  const dev=await ad.requestDevice(); const ctx=gl.getContext('webgpu'); const fmt=navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({device:dev,format:fmt,alphaMode:'opaque'});
  const mod=dev.createShaderModule({code:WGSL});
  const pipe=dev.createRenderPipeline({layout:'auto',vertex:{module:mod,entryPoint:'vs'},fragment:{module:mod,entryPoint:'fs',targets:[{format:fmt}]},primitive:{topology:'triangle-list'}});
  return {dev,ctx,pipe,smp:dev.createSampler({magFilter:'linear',minFilter:'linear'})};
}
function cmapJS(t){const c=[[-0.002,-0.0007,-0.0053],[0.25,0.68,2.49],[8.35,-3.58,0.31],[-27.67,14.26,-13.65],[52.18,-27.94,12.94],[-50.77,29.05,4.23],[18.66,-11.49,-5.6]];
  return [0,1,2].map(k=>{let v=0;for(let i=6;i>=0;i--)v=v*t+c[i][k];return Math.max(0,Math.min(255,v*255));});}
function bytes(f){const b=atob(f.data),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return a;}
function draw(fr){
  const f=fr[document.getElementById('fld').value], px=bytes(f), w=f.w, h=f.h;
  if(gpu){
    // rows of px run along x for increasing y; texture v grows downward, so flip rows for y up
    const pad=Math.ceil(w/256)*256, up=new Uint8Array(pad*h);
    for(let j=0;j<h;j++) up.set(px.subarray((h-1-j)*w,(h-j)*w), j*pad);
    const tex=gpu.dev.createTexture({size:[w,h],format:'r8unorm',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
    gpu.dev.queue.writeTexture({texture:tex},up,{bytesPerRow:pad},[w,h]);
    const bg=gpu.dev.createBindGroup({layout:gpu.pipe.getBindGroupLayout(0),entries:[{binding:0,resource:tex.createView()},{binding:1,resource:gpu.smp}]});
    const enc=gpu.dev.createCommandEncoder(); const pass=enc.beginRenderPass({colorAttachments:[{view:gpu.ctx.getCurrentTexture().createView(),loadOp:'clear',storeOp:'store',clearValue:{r:0,g:0,b:0,a:1}}]});
    pass.setPipeline(gpu.pipe); pass.setBindGroup(0,bg); pass.draw(6); pass.end(); gpu.dev.queue.submit([enc.finish()]);
  } else {
    const c2=gl.getContext('2d'), img=c2.createImageData(w,h);
    for(let j=0;j<h;j++)for(let i=0;i<w;i++){const v=px[(h-1-j)*w+i]/255,[r,g,b]=cmapJS(v),o=4*(j*w+i);img.data[o]=r;img.data[o+1]=g;img.data[o+2]=b;img.data[o+3]=255;}
    const tmp=document.createElement('canvas');tmp.width=w;tmp.height=h;tmp.getContext('2d').putImageData(img,0,0);
    c2.imageSmoothingEnabled=true;c2.drawImage(tmp,0,0,gl.width,gl.height);
  }
  const W=ov.width,H=ov.height; octx.clearRect(0,0,W,H);
  octx.lineWidth=1; octx.strokeStyle='rgba(120,200,255,0.8)';
  for(const l of fr.fieldlines){octx.beginPath();for(let k=0;k<l.length;k+=2){const x=l[k]*W,y=(1-l[k+1])*H;k?octx.lineTo(x,y):octx.moveTo(x,y);}octx.stroke();}
  octx.strokeStyle='rgba(255,255,255,0.7)';
  for(const c of fr.contours){const p=c.pts;octx.beginPath();for(let k=0;k<p.length;k+=2){const x=p[k]*W,y=(1-p[k+1])*H;k?octx.lineTo(x,y):octx.moveTo(x,y);}octx.stroke();}
  document.getElementById('info').textContent=`step ${fr.step}  t = ${fr.t.toFixed(4)}  range ${f.lo.toPrecision(3)} .. ${f.hi.toPrecision(3)}`;
  const m=fr.monitor.modes||{}; let rows='<tr><td>m</td><td>amplitude / R</td><td>growth / t</td></tr>';
  for(const k in m) rows+=`<tr><td>${k}</td><td>${m[k].amp.toExponential(2)}</td><td>${isFinite(m[k].growth)?m[k].growth.toFixed(2):'-'}</td></tr>`;
  document.getElementById('modes').innerHTML=rows;
  document.getElementById('alerts').innerHTML=(fr.alerts||[]).slice().reverse().map(a=>`<div class="alert"><b>t = ${a.t.toFixed(3)} ${a.kind}${a.mode?' m = '+a.mode:''}</b><br>${a.recommendation}</div>`).join('');
}
document.getElementById('fld').onchange=()=>last&&draw(last);
(async()=>{ try{gpu=await initGPU();}catch(e){gpu=null;}
  document.getElementById('gpu').textContent=gpu?'WebGPU':'canvas 2D (no WebGPU)';
  const es=new EventSource('/events'); es.onmessage=e=>{last=JSON.parse(e.data);draw(last);};
  es.onerror=()=>{document.getElementById('info').textContent='stream closed';};})();
</script></body></html>"""


def make_handler(session):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype="application/json", code=200):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(PAGE, "text/html; charset=utf-8")
            elif self.path == "/frame":
                with session.cond:
                    f = session.frame
                self._send(json.dumps(f) if f else "null")
            elif self.path == "/alerts":
                self._send(json.dumps(session.mon.alerts))
            elif self.path == "/status":
                self._send(json.dumps(dict(running=session.running, step=session.step, t=float(session.s.t),
                                           frames=session.version, error=session.error)))
            elif self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                seen = 0
                try:
                    while True:
                        with session.cond:
                            session.cond.wait_for(lambda: session.version > seen or not session.running, timeout=15)
                            f, v = session.frame, session.version
                        if v > seen and f is not None:
                            self.wfile.write(b"data: " + json.dumps(f).encode() + b"\n\n")
                            self.wfile.flush()
                            seen = v
                        elif not session.running:
                            break
                        else:
                            self.wfile.write(b": keep-alive\n\n")
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
            else:
                self._send("not found", "text/plain", 404)
    return H


def serve(session, host="127.0.0.1", port=8765):
    httpd = ThreadingHTTPServer((host, port), make_handler(session))
    httpd.daemon_threads = True
    return httpd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="liner2d")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--every", type=int, default=10, help="steps between frames")
    ap.add_argument("--log", default=os.path.join(ROOT, "results", "agent_monitor.jsonl"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    sess = Session(a.preset, every=a.every, log_path=a.log)
    httpd = serve(sess, a.host, a.port)
    sess.start()
    print(f"serving http://{a.host}:{a.port}/  (preset {a.preset}; monitor log {a.log}); Ctrl+C to stop", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
