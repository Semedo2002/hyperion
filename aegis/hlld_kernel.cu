extern "C" __global__
void hlld_flux_x_kernel(
    const double* __restrict__ WL,
    const double* __restrict__ WR,
    double* __restrict__ F,
    double* __restrict__ smax_arr,
    const int n_faces,
    const double gamma,
    const double ch,
    const double floor_rho,
    const double floor_pr,
    const double* __restrict__ eint_L,    // per-face internal energy per volume and squared sound
    const double* __restrict__ eint_R,    // speed from the EOS, read only when use_eos != 0
    const double* __restrict__ csq_L,     // (otherwise ideal gas with the scalar gamma); for a
    const double* __restrict__ csq_R,     // stiffened gas: (p + g p_inf)/(g - 1) and g (p + p_inf)/rho
    const int use_eos,                    // 0 ideal gas, 1 the arrays above, 2 the device EOS (eos_fused.cu)
    const int axn,                        // face normal axis 0, 1, 2: components are rotated in place (x, y, z),
                                          // (y, x, z), (z, y, x) instead of permuting the arrays on the host
    const double* __restrict__ eos_P,     // use_eos == 2: EOS parameters and tables, evaluated inline per face
    const double* __restrict__ eos_T
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_faces) return;

    const int NVAR = 11;
    const int c1 = (axn == 0) ? 1 : ((axn == 1) ? 2 : 3);   // global velocity index of local x, y, z
    const int c2 = (axn == 1) ? 1 : 2;
    const int c3 = (axn == 2) ? 1 : 3;

    double rhoL = WL[0*n_faces + idx]; if (rhoL < floor_rho) rhoL = floor_rho;
    double rhoR = WR[0*n_faces + idx]; if (rhoR < floor_rho) rhoR = floor_rho;
    double vxL = WL[c1*n_faces + idx], vyL = WL[c2*n_faces + idx], vzL = WL[c3*n_faces + idx];
    double vxR = WR[c1*n_faces + idx], vyR = WR[c2*n_faces + idx], vzR = WR[c3*n_faces + idx];
    double BxL = WL[(c1+3)*n_faces + idx], ByL = WL[(c2+3)*n_faces + idx], BzL = WL[(c3+3)*n_faces + idx];
    double BxR = WR[(c1+3)*n_faces + idx], ByR = WR[(c2+3)*n_faces + idx], BzR = WR[(c3+3)*n_faces + idx];
    // with an EOS the face pressures were already floored above cavitation on the host
    double pL = WL[7*n_faces + idx]; if (!use_eos && pL < floor_pr) pL = floor_pr;
    double pR = WR[7*n_faces + idx]; if (!use_eos && pR < floor_pr) pR = floor_pr;
    double psiL = WL[8*n_faces + idx], psiR = WR[8*n_faces + idx];
    double CL_v = WL[9*n_faces + idx], CR_v = WR[9*n_faces + idx];
    double SL_v = WL[10*n_faces + idx], SR_v = WR[10*n_faces + idx];   // electron entropy s_e, passive like C

    double Bx_f, psi_f;
    if (ch > 0.0) {
        Bx_f = 0.5*(BxL + BxR) - 0.5/ch*(psiR - psiL);
        psi_f = 0.5*(psiL + psiR) - 0.5*ch*(BxR - BxL);
    } else {
        Bx_f = 0.5*(BxL + BxR);
        psi_f = 0.0;
    }
    double Bx2 = Bx_f * Bx_f;

    double BmL2 = Bx2 + ByL*ByL + BzL*BzL;
    double BmR2 = Bx2 + ByR*ByR + BzR*BzR;
    double ptL = pL + 0.5*BmL2;
    double ptR = pR + 0.5*BmR2;

    double aL2, aR2, eiL, eiR;            // squared sound speeds, internal energies per volume
    if (use_eos == 2) {
        aL2 = fmax(ef_c2(eos_P, eos_T, rhoL, pL, CL_v), 0.0); eiL = ef_eps(eos_P, eos_T, rhoL, pL, CL_v);
        aR2 = fmax(ef_c2(eos_P, eos_T, rhoR, pR, CR_v), 0.0); eiR = ef_eps(eos_P, eos_T, rhoR, pR, CR_v);
    } else if (use_eos) {
        aL2 = csq_L[idx]; aR2 = csq_R[idx]; eiL = eint_L[idx]; eiR = eint_R[idx];
    } else {
        aL2 = gamma*pL/rhoL; aR2 = gamma*pR/rhoR; eiL = pL/(gamma - 1.0); eiR = pR/(gamma - 1.0);
    }

    double tmpL = aL2 + BmL2/rhoL;
    double discL = tmpL*tmpL - 4.0*aL2*Bx2/rhoL;
    if (discL < 0.0) discL = 0.0;
    double cfL = sqrt(fmax(0.5*(tmpL + sqrt(discL)), 0.0));

    double tmpR = aR2 + BmR2/rhoR;
    double discR = tmpR*tmpR - 4.0*aR2*Bx2/rhoR;
    if (discR < 0.0) discR = 0.0;
    double cfR = sqrt(fmax(0.5*(tmpR + sqrt(discR)), 0.0));

    double SL = fmin(vxL - cfL, vxR - cfR);
    double SR = fmax(vxL + cfL, vxR + cfR);
    if (ch > 0.0) { SL = fmin(SL, -ch); SR = fmax(SR, ch); }

    double local_smax = fmax(fabs(SL), fabs(SR));
    smax_arr[idx] = local_smax;

    double dSL = SL - vxL;
    double dSR = SR - vxR;
    double den = dSR*rhoR - dSL*rhoL;
    if (fabs(den) < 1e-14) den = (den >= 0.0) ? 1e-14 : -1e-14;

    double SM = (dSR*rhoR*vxR - dSL*rhoL*vxL - ptR + ptL) / den;
    if (SM < SL + 1e-14) SM = SL + 1e-14;
    if (SM > SR - 1e-14) SM = SR - 1e-14;

    double ptS = ptL + rhoL*dSL*(SM - vxL);

    double dSL_SM = SL - SM; if (fabs(dSL_SM) < 1e-14) dSL_SM = 1e-14;
    double dSR_SM = SR - SM; if (fabs(dSR_SM) < 1e-14) dSR_SM = 1e-14;

    double rhoLS = fmax(rhoL*dSL/dSL_SM, floor_rho);
    double rhoRS = fmax(rhoR*dSR/dSR_SM, floor_rho);
    double sqLS = sqrt(rhoLS);
    double sqRS = sqrt(rhoRS);

    // star-state transverse v and B; when the fast and Alfven waves coincide (rho dS (S - SM) = Bx^2) the
    // formulas are 0/0 and the outer state is the correct limit (Miyoshi & Kusano 2005)
    double dnL = rhoL*dSL*dSL_SM - Bx2;
    double dnR = rhoR*dSR*dSR_SM - Bx2;
    double vyLS = vyL, vzLS = vzL, ByLS = ByL, BzLS = BzL;
    double vyRS = vyR, vzRS = vzR, ByRS = ByR, BzRS = BzR;
    if (fabs(dnL) > 1e-8 * fmax(fabs(ptS), 1e-300)) {
        vyLS = vyL - Bx_f*ByL*(SM - vxL)/dnL;
        vzLS = vzL - Bx_f*BzL*(SM - vxL)/dnL;
        ByLS = ByL*(rhoL*dSL*dSL - Bx2)/dnL;
        BzLS = BzL*(rhoL*dSL*dSL - Bx2)/dnL;
    }
    if (fabs(dnR) > 1e-8 * fmax(fabs(ptS), 1e-300)) {
        vyRS = vyR - Bx_f*ByR*(SM - vxR)/dnR;
        vzRS = vzR - Bx_f*BzR*(SM - vxR)/dnR;
        ByRS = ByR*(rhoR*dSR*dSR - Bx2)/dnR;
        BzRS = BzR*(rhoR*dSR*dSR - Bx2)/dnR;
    }

    double vBL = vxL*Bx_f + vyL*ByL + vzL*BzL;
    double vBLS = SM*Bx_f + vyLS*ByLS + vzLS*BzLS;
    double EL = eiL + 0.5*rhoL*(vxL*vxL + vyL*vyL + vzL*vzL) + 0.5*BmL2;
    double ELS = ((SL - vxL)*EL - ptL*vxL + ptS*SM + Bx_f*(vBL - vBLS))/dSL_SM;

    double vBR = vxR*Bx_f + vyR*ByR + vzR*BzR;
    double vBRS = SM*Bx_f + vyRS*ByRS + vzRS*BzRS;
    double ER = eiR + 0.5*rhoR*(vxR*vxR + vyR*vyR + vzR*vzR) + 0.5*BmR2;
    double ERS = ((SR - vxR)*ER - ptR*vxR + ptS*SM + Bx_f*(vBR - vBRS))/dSR_SM;

    double sBx = (Bx_f >= 0.0) ? 1.0 : -1.0;
    double sqLS_safe = fmax(sqLS, 1e-14);
    double sqRS_safe = fmax(sqRS, 1e-14);
    double SAL = SM - fabs(Bx_f)/sqLS_safe;
    double SAR = SM + fabs(Bx_f)/sqRS_safe;
    if (SAL < SL) SAL = SL; if (SAL > SM) SAL = SM;
    if (SAR < SM) SAR = SM; if (SAR > SR) SAR = SR;

    double dds = fmax(sqLS + sqRS, 1e-14);
    double vyDS = (sqLS*vyLS + sqRS*vyRS + (ByRS - ByLS)*sBx)/dds;
    double vzDS = (sqLS*vzLS + sqRS*vzRS + (BzRS - BzLS)*sBx)/dds;
    double ByDS = (sqLS*ByRS + sqRS*ByLS + sqLS*sqRS*(vyRS - vyLS)*sBx)/dds;
    double BzDS = (sqLS*BzRS + sqRS*BzLS + sqLS*sqRS*(vzRS - vzLS)*sBx)/dds;
    double vBDS = SM*Bx_f + vyDS*ByDS + vzDS*BzDS;
    double ELDS = ELS - sqLS*(vBLS - vBDS)*sBx;
    double ERDS = ERS + sqRS*(vBRS - vBDS)*sBx;

    double f[11];

    if (SL >= 0.0) {
        f[0] = rhoL*vxL;
        f[1] = rhoL*vxL*vxL + ptL - Bx2;
        f[2] = rhoL*vyL*vxL - Bx_f*ByL;
        f[3] = rhoL*vzL*vxL - Bx_f*BzL;
        f[4] = (ch > 0.0) ? psi_f : 0.0;
        f[5] = ByL*vxL - Bx_f*vyL;
        f[6] = BzL*vxL - Bx_f*vzL;
        f[7] = (EL + ptL)*vxL - Bx_f*vBL;
        f[8] = (ch > 0.0) ? Bx_f*ch*ch : 0.0;
        f[9] = rhoL*CL_v*vxL;
        f[10] = rhoL*SL_v*vxL;
    } else if (SR <= 0.0) {
        f[0] = rhoR*vxR;
        f[1] = rhoR*vxR*vxR + ptR - Bx2;
        f[2] = rhoR*vyR*vxR - Bx_f*ByR;
        f[3] = rhoR*vzR*vxR - Bx_f*BzR;
        f[4] = (ch > 0.0) ? psi_f : 0.0;
        f[5] = ByR*vxR - Bx_f*vyR;
        f[6] = BzR*vxR - Bx_f*vzR;
        f[7] = (ER + ptR)*vxR - Bx_f*vBR;
        f[8] = (ch > 0.0) ? Bx_f*ch*ch : 0.0;
        f[9] = rhoR*CR_v*vxR;
        f[10] = rhoR*SR_v*vxR;
    } else {
        double FL[11], UL_c[11], ULS_c[11];
        FL[0] = rhoL*vxL;
        FL[1] = rhoL*vxL*vxL + ptL - Bx2;
        FL[2] = rhoL*vyL*vxL - Bx_f*ByL;
        FL[3] = rhoL*vzL*vxL - Bx_f*BzL;
        FL[4] = (ch > 0.0) ? psi_f : 0.0;
        FL[5] = ByL*vxL - Bx_f*vyL;
        FL[6] = BzL*vxL - Bx_f*vzL;
        FL[7] = (EL + ptL)*vxL - Bx_f*vBL;
        FL[8] = (ch > 0.0) ? Bx_f*ch*ch : 0.0;
        FL[9] = rhoL*CL_v*vxL;
        FL[10] = rhoL*SL_v*vxL;

        UL_c[0]=rhoL; UL_c[1]=rhoL*vxL; UL_c[2]=rhoL*vyL; UL_c[3]=rhoL*vzL;
        UL_c[4]=Bx_f; UL_c[5]=ByL; UL_c[6]=BzL; UL_c[7]=EL; UL_c[8]=psiL; UL_c[9]=rhoL*CL_v; UL_c[10]=rhoL*SL_v;

        ULS_c[0]=rhoLS; ULS_c[1]=rhoLS*SM; ULS_c[2]=rhoLS*vyLS; ULS_c[3]=rhoLS*vzLS;
        ULS_c[4]=Bx_f; ULS_c[5]=ByLS; ULS_c[6]=BzLS; ULS_c[7]=ELS; ULS_c[8]=psi_f; ULS_c[9]=rhoLS*CL_v; ULS_c[10]=rhoLS*SL_v;

        double FLS[11];
        for (int v=0; v<11; v++) FLS[v] = FL[v] + SL*(ULS_c[v] - UL_c[v]);

        double FR[11], UR_c[11], URS_c[11];
        FR[0] = rhoR*vxR;
        FR[1] = rhoR*vxR*vxR + ptR - Bx2;
        FR[2] = rhoR*vyR*vxR - Bx_f*ByR;
        FR[3] = rhoR*vzR*vxR - Bx_f*BzR;
        FR[4] = (ch > 0.0) ? psi_f : 0.0;
        FR[5] = ByR*vxR - Bx_f*vyR;
        FR[6] = BzR*vxR - Bx_f*vzR;
        FR[7] = (ER + ptR)*vxR - Bx_f*vBR;
        FR[8] = (ch > 0.0) ? Bx_f*ch*ch : 0.0;
        FR[9] = rhoR*CR_v*vxR;
        FR[10] = rhoR*SR_v*vxR;

        UR_c[0]=rhoR; UR_c[1]=rhoR*vxR; UR_c[2]=rhoR*vyR; UR_c[3]=rhoR*vzR;
        UR_c[4]=Bx_f; UR_c[5]=ByR; UR_c[6]=BzR; UR_c[7]=ER; UR_c[8]=psiR; UR_c[9]=rhoR*CR_v; UR_c[10]=rhoR*SR_v;

        URS_c[0]=rhoRS; URS_c[1]=rhoRS*SM; URS_c[2]=rhoRS*vyRS; URS_c[3]=rhoRS*vzRS;
        URS_c[4]=Bx_f; URS_c[5]=ByRS; URS_c[6]=BzRS; URS_c[7]=ERS; URS_c[8]=psi_f; URS_c[9]=rhoRS*CR_v; URS_c[10]=rhoRS*SR_v;

        double FRS[11];
        for (int v=0; v<11; v++) FRS[v] = FR[v] + SR*(URS_c[v] - UR_c[v]);

        double ULDS[11], URDS[11];
        ULDS[0]=rhoLS; ULDS[1]=rhoLS*SM; ULDS[2]=rhoLS*vyDS; ULDS[3]=rhoLS*vzDS;
        ULDS[4]=Bx_f; ULDS[5]=ByDS; ULDS[6]=BzDS; ULDS[7]=ELDS; ULDS[8]=psi_f; ULDS[9]=rhoLS*CL_v; ULDS[10]=rhoLS*SL_v;

        URDS[0]=rhoRS; URDS[1]=rhoRS*SM; URDS[2]=rhoRS*vyDS; URDS[3]=rhoRS*vzDS;
        URDS[4]=Bx_f; URDS[5]=ByDS; URDS[6]=BzDS; URDS[7]=ERDS; URDS[8]=psi_f; URDS[9]=rhoRS*CR_v; URDS[10]=rhoRS*SR_v;

        double FLDS[11], FRDS[11];
        for (int v=0; v<11; v++) FLDS[v] = FLS[v] + SAL*(ULDS[v] - ULS_c[v]);
        for (int v=0; v<11; v++) FRDS[v] = FRS[v] + SAR*(URDS[v] - URS_c[v]);

        if (SAL >= 0.0) {
            for (int v=0; v<11; v++) f[v] = FLS[v];
        } else if (SM >= 0.0) {
            for (int v=0; v<11; v++) f[v] = FLDS[v];
        } else if (SAR >= 0.0) {
            for (int v=0; v<11; v++) f[v] = FRDS[v];
        } else {
            for (int v=0; v<11; v++) f[v] = FRS[v];
        }

        if (ch > 0.0) {
            f[4] = psi_f;
            f[8] = ch*ch*Bx_f;
        }
    }

    for (int v = 0; v < 11; v++) {
        int gv = v;
        if (v == 1) gv = c1; else if (v == 2) gv = c2; else if (v == 3) gv = c3;
        else if (v == 4) gv = c1 + 3; else if (v == 5) gv = c2 + 3; else if (v == 6) gv = c3 + 3;
        F[gv*n_faces + idx] = f[v];
    }
}
