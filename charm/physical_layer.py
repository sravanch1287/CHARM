"""
charm_phy.py — Pure-NumPy port of the CHARM MATLAB physical layer.

Replaces the matlab.engine call
    CUDU_RL_minCU_assoc_matlab_PA_v1_debug_v1_oncluster(association, x, PA)
used inside the RL environment, so training no longer needs MATLAB.

Ported from (read directly, not inferred):
  CUDU_RL_minCU_assoc_matlab_PA_v1_debug_v1_oncluster.m  (top-level)
  calc_channel_matrix_new.m, leakage_precoding_new_EPA_newcode_v1.m,
  poweralloc_PA.m / DPA1 / DPA2 / CPA, computeslnr.m,
  Ant3GPP.m, ArrayWithAxes.m  (URA steering + azelaxes + global2localcoord)

Validation target (baseline_results_20260512_013108.mat, exhaustive SLAPP):
  deployed CUs {9,1,8}, DUs 1..8, min-path-loss assoc, PA=1
  -> per-DU rates [8.403,10.013,8.403,8.403,10.013,10.013,10.013,10.013], sum 75.2747
"""
import numpy as np
import os

# ----------------------------- constants (from the dispatcher) --------------
FC      = 140e9
C_LIGHT = 299792458.0
LAMBDA  = C_LIGHT / FC
PTX_DBM = 23.0
P_TX_LIN = 10 ** (0.1 * (PTX_DBM - 30))
KB      = 1.380649e-23
KT_DBM  = 10 * np.log10(KB * 290) + 30        # -174 dBm/Hz
NOISE_FIG = 5.0
N0_LIN  = 10 ** (0.1 * (KT_DBM + NOISE_FIG - 30))
BW      = 2e9
NPTS    = 256
BW_LOSS = 0.15
MAX_SE  = 5.89
NANT_CU = (16, 16)
NANT_DU = (16, 16)
N_CU    = NANT_CU[0] * NANT_CU[1]
M_DU    = NANT_DU[0] * NANT_DU[1]
ELGAIN  = 0
ICU_LOS = np.array([9, 1, 8, 10, 5, 4, 6, 2, 3, 7, 11])   # fixed CU priority order

# ----------------------------- geometry helpers -----------------------------
def _deg2rad(d): return np.deg2rad(d)

def dir_vec(az_deg, el_deg):
    """Unit vector for (az,el) in degrees. u = [cos el cos az, cos el sin az, sin el]."""
    az = np.deg2rad(az_deg); el = np.deg2rad(el_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])

def azelaxes(az_deg, el_deg):
    """MATLAB azelaxes(az,el): columns = local x (radial), y (az-hat), z (el-hat)."""
    az = np.deg2rad(az_deg); el = np.deg2rad(el_deg)
    ax = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    ay = np.array([-np.sin(az), np.cos(az), 0.0])
    az_ = np.array([-np.sin(el)*np.cos(az), -np.sin(el)*np.sin(az), np.cos(el)])
    return np.column_stack([ax, ay, az_])

def global2localcoord_ss(az_deg, el_deg, local_axes):
    """Convert global (az,el,r=1) to local spherical (azL,elL) deg, given local axes (cols)."""
    g = dir_vec(az_deg, el_deg)                 # 3 x npath  (broadcast)
    l = local_axes.T @ g                        # express in local basis
    azL = np.degrees(np.arctan2(l[1], l[0]))
    elL = np.degrees(np.arctan2(l[2], np.hypot(l[0], l[1])))
    return azL, elL

# ----------------------------- URA array ------------------------------------
def ura_positions(nant, dspacing_lambda=0.5):
    """phased.URA(Size,spacing,'ArrayNormal','x'): elements in y-z plane, x=0.
    Positions in metres (columns are elements). Ordering is internally consistent
    (rate is invariant to element ordering)."""
    nrow, ncol = nant
    d = dspacing_lambda * LAMBDA
    zc = (np.arange(nrow) - (nrow - 1) / 2.0) * d
    yc = (np.arange(ncol) - (ncol - 1) / 2.0) * d
    Y, Z = np.meshgrid(yc, zc)                   # nrow x ncol
    pos = np.vstack([np.zeros(Y.size), Y.ravel(), Z.ravel()])   # 3 x N
    return pos

_POS_CU = ura_positions(NANT_CU)
_POS_DU = ura_positions(NANT_DU)

def steering(pos, az_deg, el_deg):
    """Steering vectors for local angles (arrays). Returns N x npath complex.
    sv = exp(1j 2pi/lambda pos . u(az,el)). Sign convention is irrelevant to the
    rate because TX/RX use the same convention."""
    az = np.atleast_1d(az_deg); el = np.atleast_1d(el_deg)
    u = np.vstack([np.cos(np.deg2rad(el))*np.cos(np.deg2rad(az)),
                   np.cos(np.deg2rad(el))*np.sin(np.deg2rad(az)),
                   np.sin(np.deg2rad(el))])       # 3 x npath
    phase = 2 * np.pi / LAMBDA * (pos.T @ u)      # N x npath
    return np.exp(1j * phase)

# ----------------------------- 3GPP element gain ----------------------------
_EL_HFBW = 65.0; _SLAV = 30.0; _AZ_FIXED = 3.0
def _ant3gpp_gainmax():
    el = np.linspace(-90, 90, 100)
    gain = (-np.minimum(12 * (el / _EL_HFBW) ** 2, _SLAV) + _AZ_FIXED) - _AZ_FIXED
    dirlin = 10 ** (0.1 * gain)
    cosel = np.cos(np.deg2rad(el))
    g = np.mean(dirlin * cosel) / np.mean(cosel)
    return -10 * np.log10(g)
_GAINMAX = _ant3gpp_gainmax()

def ant3gpp_gain(az_deg, el_deg):
    """Fixed-azimuth 3GPP element directivity (dBi) at local angles."""
    A_EV = -np.minimum(12 * (np.atleast_1d(el_deg) / _EL_HFBW) ** 2, _SLAV)
    return _GAINMAX + A_EV + _AZ_FIXED

def array_step(pos, boresight_az, boresight_el, path_az, path_el):
    """Replicates ArrayWithAxes.step: local-angle steering vectors + element gains."""
    axes = azelaxes(boresight_az, boresight_el)
    azL, elL = global2localcoord_ss(path_az, path_el, axes)
    u = steering(pos, azL, elL)                  # N x npath
    g = ant3gpp_gain(azL, elL)                   # npath (dBi)
    return u, g

# ----------------------------- channel matrix -------------------------------
def calc_channel_matrix(utx, urx, effgain_lin, dly, phase_deg, npts=NPTS,
                        legacy_matlab=False):
    """Port of calc_channel_matrix_new.m. Only the mid-band bin H(:,:,NPTS/2) is used
    downstream (the SINR loop runs at npts=1), so compute just that bin -> ~256x faster
    and identical result."""
    # The audited model evaluates the representative narrowband channel at fc.
    # Legacy mode reproduces the MATLAB NPTS/2 slice and its raw Phase_deg use.
    f_mid = (FC + np.linspace(-BW / 2, BW / 2, npts)[npts // 2 - 1]
             if legacy_matlab else FC)
    npaths = len(dly)
    M, N = urx.shape[0], utx.shape[0]
    H = np.zeros((M, N), dtype=complex)
    for k in range(npaths):
        phase_cycles = phase_deg[k] if legacy_matlab else phase_deg[k] / 360.0
        ph = np.exp(2 * np.pi * 1j * ((-dly[k] * f_mid) + phase_cycles))
        H += np.outer(urx[:, k], utx[:, k]) * effgain_lin[k] * ph
    H *= np.sqrt(M * N / npaths)
    return H

# ----------------------------- SLNR precoder --------------------------------
def leakage_precoding(h_k_hat, N0pw, Pk):
    """Port of leakage_precoding_new_EPA_newcode_v1.m.
    h_k_hat: K x Ntx (effective post-combining channels), returns Ntx x K precoders."""
    K, Ntx = h_k_hat.shape
    w_tx = np.zeros((Ntx, K), dtype=complex)
    for k in range(K):
        Htil = np.zeros((Ntx, Ntx), dtype=complex)
        for j in range(K):
            if j != k:
                Htil += np.outer(h_k_hat[j].conj(), h_k_hat[j])
        # Principal eigenvector of Psi^{-1}(h^H h) is Psi^{-1} h^H normalized
        # (h^H h is rank-1; paper Sec. V). One linear solve instead of eig -> much faster.
        Psi = (N0pw / Pk[k]) * np.eye(Ntx) + Htil
        v = np.linalg.solve(Psi, h_k_hat[k].conj())
        w_tx[:, k] = v / np.linalg.norm(v)
    return w_tx

# ----------------------------- power allocation -----------------------------
def poweralloc_slapp(H_hat, wr_hat, w_tx, npw, SE=MAX_SE,
                     legacy_abs=False, return_diagnostics=False):
    """poweralloc_PA.m (SLAPP / PA=1)."""
    K = H_hat.shape[2]; n_k = (2 ** SE - 1) * npw
    palloc = np.zeros(K)
    denominators = np.zeros(K)
    for k in range(K):
        A_k = np.abs(wr_hat[:, k] @ H_hat[:, :, k] @ w_tx[:, k]) ** 2   # MATLAB .' (no conj)
        B_k = 0.0
        for l in range(K):
            if l != k:
                B_k += np.abs(wr_hat[:, l] @ H_hat[:, :, l] @ w_tx[:, k]) ** 2
        denominators[k] = A_k - (2 ** SE - 1) * B_k
        if legacy_abs:
            palloc[k] = n_k / max(np.abs(denominators[k]), np.finfo(float).tiny)
        elif denominators[k] > np.finfo(float).eps:
            palloc[k] = n_k / denominators[k]
    target_power = palloc.copy()
    if palloc.sum() > 0:
        allocated = palloc * P_TX_LIN / palloc.sum()
    else:
        allocated = np.full(K, P_TX_LIN / K)
    diagnostics = {
        "denominators": denominators,
        "finite_target_feasible": denominators > np.finfo(float).eps,
        "target_power_w": target_power,
        "budget_infeasible": bool(target_power.sum() > P_TX_LIN),
        "under_target_after_normalization": (
            (denominators > np.finfo(float).eps) & (allocated + 1e-15 < target_power)
        ),
    }
    return (allocated, diagnostics) if return_diagnostics else allocated

def poweralloc_dpa1(H_hat):
    K = H_hat.shape[2]
    qinv = np.array([1.0 / np.real(np.trace(H_hat[:, :, k].conj().T @ H_hat[:, :, k])) for k in range(K)])
    return np.abs(P_TX_LIN * qinv / qinv.sum())

def poweralloc_dpa2(H_hat, wr_hat, w_tx, npw, legacy_matlab=False):
    K = H_hat.shape[2]; Pdu = np.full(K, P_TX_LIN / K)
    slnr = np.zeros(K)
    for k in range(K):
        sig_p = Pdu[k] * np.abs(wr_hat[:, k] @ H_hat[:, :, k] @ w_tx[:, k]) ** 2   # MATLAB .' (no conj)
        if legacy_matlab:
            sig_l = sum(Pdu[k] * np.abs(wr_hat[:, k] @ H_hat[:, :, l] @ w_tx[:, k]) ** 2
                        for l in range(K) if l != k)
        else:
            sig_l = sum(Pdu[k] * np.abs(wr_hat[:, l] @ H_hat[:, :, l] @ w_tx[:, k]) ** 2
                        for l in range(K) if l != k)
        slnr[k] = sig_p / (sig_l + npw)
    linv = 1.0 / slnr
    return P_TX_LIN * linv / linv.sum()

# ----------------------------- scenario loader ------------------------------
class Scenario:
    """Loads pathTable + node positions once; precomputes LOS-based DU selection."""
    def __init__(self, data_dir):
        import csv
        with open(os.path.join(data_dir, "pathTable.csv"), newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.txid = np.array([int(r["txid"]) for r in rows])
        self.rxid = np.array([int(r["rxid"]) for r in rows])
        self.los  = np.array([int(r["LOS"]) for r in rows])
        self.pl   = np.array([float(r["pathloss_db"]) for r in rows])
        self.aod_el = 90 - np.array([float(r["doa_theta_deg"]) for r in rows])
        self.aod_az = np.array([float(r["doa_phi_deg"]) for r in rows])
        self.aoa_el = 90 - np.array([float(r["aoa_theta_deg"]) for r in rows])
        self.aoa_az = np.array([float(r["aoa_phi_deg"]) for r in rows])
        self.delay = np.array([float(r["toa_sec"]) for r in rows])
        self.phase = np.array([float(r["Phase_deg"]) for r in rows])
        self.ntx = int(self.txid.max()); self.nrx = int(self.rxid.max())
        losMat = np.zeros((self.nrx, self.ntx))
        for i in range(len(rows)):
            if self.los[i]:
                losMat[self.rxid[i] - 1, self.txid[i] - 1] = 1
        self.losMat = losMat
        self.legacy_matlab = os.environ.get("CHARM_PHY_LEGACY", "0") == "1"
        self.last_diagnostics = []

    def compute_rates(self, cu_txids, du_rxids, assoc, PA):
        """Core physical-layer evaluation.
        cu_txids : 1-based CU tx ids that are active (len = n active CUs)
        du_rxids : 1-based DU rx ids being served (len = D)
        assoc    : for each DU, index into cu_txids of its serving CU
        PA       : 1=SLAPP, 2=DPA1, 3=DPA2, 4=EPA
        returns  : (sinrs, rates_gbps)
        """
        ncu = len(cu_txids); D = len(du_rxids)
        npw = BW * N0_LIN
        # --- CU boresight = mean AoD direction over its served DUs (min-PL path) ---
        du_bestpath = {}
        for j, rx in enumerate(du_rxids):
            itx = cu_txids[assoc[j]]
            I = np.where((self.rxid == rx) & (self.txid == itx))[0]
            im = I[np.argmin(self.pl[I])]
            du_bestpath[j] = (I, im, itx)
        angCU = np.zeros((ncu, 2))               # (el, az) per CU
        for i in range(ncu):
            members = [j for j in range(D) if assoc[j] == i]
            X = []
            for j in members:
                _, im, _ = du_bestpath[j]
                X.append(dir_vec(self.aod_az[im], self.aod_el[im]))
            xmean = np.mean(np.array(X), axis=0)
            az = np.degrees(np.arctan2(xmean[1], xmean[0]))
            el = np.degrees(np.arctan2(xmean[2], np.hypot(xmean[0], xmean[1])))
            angCU[i] = (el, az)
        # --- per-DU channel, RX combiner ---
        H_all = [None] * D; wrx_all = [None] * D
        for j, rx in enumerate(du_rxids):
            I, im, itx = du_bestpath[j]
            icu = assoc[j]
            aod_az = self.aod_az[I]; aod_el = self.aod_el[I]
            aoa_az = self.aoa_az[I]; aoa_el = self.aoa_el[I]
            dly = self.delay[I]; ph = self.phase[I]; pl = self.pl[I]
            utx, gCU = array_step(_POS_CU, angCU[icu, 1], angCU[icu, 0], aod_az, aod_el)
            urx, gDU = array_step(_POS_DU, self.aoa_az[im], self.aoa_el[im], aoa_az, aoa_el)
            effgain_db = gCU + gDU - pl
            effgain_lin = 10 ** (0.05 * effgain_db)
            H = calc_channel_matrix(
                utx, urx, effgain_lin, dly, ph, legacy_matlab=self.legacy_matlab
            )
            # RX combiner = conj(strongest-path steering)/norm
            imloc = int(np.argmin(pl))
            wrx = np.conj(urx[:, imloc]); wrx /= np.linalg.norm(wrx)
            H_all[j] = H; wrx_all[j] = wrx
        # --- per-CU: precode, power alloc, SINR/rate ---
        sinrs = np.zeros(D); rates = np.zeros(D)
        self.last_diagnostics = []
        for i in range(ncu):
            members = [j for j in range(D) if assoc[j] == i]
            K = len(members)
            H_hat = np.stack([H_all[j] for j in members], axis=2)      # M x N x K
            wr_hat = np.column_stack([wrx_all[j] for j in members])    # M x K
            h_k = np.array([wr_hat[:, k].conj() @ H_hat[:, :, k] for k in range(K)])  # K x N
            Pepa = np.full(K, P_TX_LIN / K)
            w_tx = leakage_precoding(h_k, npw, Pepa)
            if PA == 1:
                p, pa_diag = poweralloc_slapp(
                    H_hat, wr_hat, w_tx, npw,
                    legacy_abs=self.legacy_matlab, return_diagnostics=True
                )
                pa_diag.update({"cu_slot": i, "members": members})
                self.last_diagnostics.append(pa_diag)
            elif PA == 2:
                p = poweralloc_dpa1(H_hat)
            elif PA == 3:
                p = poweralloc_dpa2(
                    H_hat, wr_hat, w_tx, npw, legacy_matlab=self.legacy_matlab
                )
            else:
                p = np.full(K, P_TX_LIN / K)
            for k in range(K):
                # MATLAB uses .' (plain transpose) here on wr_hat (== conj(urx)); no extra conj.
                sig_p = p[k] * np.abs(wr_hat[:, k] @ H_hat[:, :, k] @ w_tx[:, k]) ** 2
                sig_i = sum(p[l] * np.abs(wr_hat[:, k] @ H_hat[:, :, k] @ w_tx[:, l]) ** 2
                            for l in range(K) if l != k)
                s = sig_p / (sig_i + BW * N0_LIN)
                sinrs[members[k]] = s
                rates[members[k]] = BW * (1 - BW_LOSS) * min(np.log2(1 + s), MAX_SE) / 1e9
        return sinrs, rates


    _cache = None

    def dispatch(self, association_action, PA):
        # memoize on (association, PA): the physical layer is deterministic, so
        # repeated configs during training return instantly.
        if self._cache is None:
            self._cache = {}
        key = (tuple(int(v) for v in np.asarray(association_action).ravel()), int(PA))
        hit = self._cache.get(key)
        if hit is not None:
            out, diagnostics = hit
            self.last_diagnostics = diagnostics
            return out
        out = self._dispatch_impl(association_action, PA)
        diagnostics = [
            {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
            for d in self.last_diagnostics
        ]
        self._cache[key] = (out, diagnostics)
        return out

    def _dispatch_impl(self, association_action, PA):
        """Drop-in for CUDU_RL_..._oncluster(association_action, action_mask, PA).
        Mirrors the MATLAB entry point exactly:
          - action_mask (x) is IGNORED (as in the .m); active CUs come from unique(U)
          - active CU tx-ids  = ICU_LOS[unique(association)]        (lines 374-376)
          - served DUs        = the D least-LOS DUs, mink order     (line 518)
        Returns (sinrs, rates_gbps) for the D served DUs.
        """
        U = np.asarray(association_action, dtype=int)      # 0-based, length D
        D = len(U)
        if D == 0 or np.any(U < 0) or np.any(U >= len(ICU_LOS)):
            raise ValueError("Association indices must be in [0, 10] and non-empty")

        # served DUs = D DUs with fewest LOS links (mink(sum(losMat,2), D))
        los_count = self.losMat.sum(axis=1)
        du_index = np.argsort(los_count, kind="stable")[:D]   # 0-based DU indices
        du_rxids = (du_index + 1).tolist()                     # 1-based rx ids

        # active CUs = ICU_LOS[unique(U)]  (unique() returns sorted values, as in MATLAB)
        uniq = sorted(set(U.tolist()))
        cu_txids = [int(ICU_LOS[v]) for v in uniq]              # physical CU tx-ids
        # remap each DU's association value to its slot index into cu_txids
        assoc_slots = [uniq.index(int(v)) for v in U]

        return self.compute_rates(cu_txids, du_rxids, assoc_slots, PA)


def min_pathloss_assoc(scn, cu_txids, du_rxids):
    """Associate each DU to the active CU with lowest omni path loss (baseline rule)."""
    assoc = np.zeros(len(du_rxids), dtype=int)
    for j, rx in enumerate(du_rxids):
        best = None
        for ci, itx in enumerate(cu_txids):
            I = np.where((scn.rxid == rx) & (scn.txid == itx))[0]
            if len(I) == 0:
                continue
            # omni: sum of linear rx power across paths -> min pl
            tot = np.sum(10 ** (-0.1 * scn.pl[I]))
            plomni = -10 * np.log10(tot)
            if best is None or plomni < best[1]:
                best = (ci, plomni)
        if best is None:
            raise ValueError(f"No ray-tracing path for DU rxid={rx} to requested CUs")
        assoc[j] = best[0]
    return assoc
