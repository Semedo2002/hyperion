# Contributing

## Setup

You need an NVIDIA GPU with CUDA 12.

```
python -m venv .venv
.venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install -e .              # or: pip install -r requirements.txt
```

## Before you commit

Run the verification suite and make sure everything passes:

```
python main.py --test all
```

It covers shock tubes, wave convergence in x and z, boundary reflection, energy conservation,
upstream equilibrium, 2D/3D consistency, resistive diffusion, the Galilean frame and the RMI benchmark.
Changes to the solver should keep the energy conservation error at round-off (around 1e-13).

## Results and data

- Saved sweep results live in `experiments/data/`; the paper figures are generated from them
  (`python paper/make_figures.py`).
- Large regenerated output (VTK frames, `frames.npz`, `rmi_output_gpu/`) is not tracked; see `.gitignore`.
- If a change alters results, rerun the affected sweep and commit the new data with the code that produced it.
