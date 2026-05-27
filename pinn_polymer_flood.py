"""
PINN for Polymer Flood Production Forecasting
==============================================
Architecture   : Meng et al. SPE-218863-MS — time embedding, BN, Dropout
BL physics     : Almajid & Abu-Alsaud SPE-203033-MS — auto-diff PDE residual,
                 Adam → L-BFGS optimizer sequence
Polymer PINN   : Liu et al. Physics of Fluids 37 036622 (2025)
                   • PINN-1: one network → (Sw, Cp_out) simultaneously
                   • Simplified polymer eq. Sw·∂Cp/∂T + fw·∂Cp/∂X = 0  (Eq.22)
                   • Cubic viscosity  μw(Cp) = μwi(1+r·Cp+s·Cp²+t·Cp³)  (Eq.1)
                   • Artificial viscosity  ε·∂²Sw/∂T²  for training stability
                   • Latin Hypercube collocation points  (Table I)
                   • Multi-weight loss  L=ω1·LSw + ω2·LCp + ω3·Ldata  (Eq.8)
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

DATA_DIR = r'C:\Users\faust\OneDrive\Desktop\PINN\CMG\TRAINING 2\Excel data'

# ==============================================================================
# 1.  PHYSICAL PRIORS
# ==============================================================================
SWC      = 0.30
SOR      = 0.20
NW_INIT  = 3.0
NO_INIT  = 2.2
KRW_INIT = 0.10
KRO_INIT = 1.00
MU_O     = 1650.0
MU_WI    = 1.0          # pure-water (no polymer) viscosity [cp]
FWM      = 0.12
SW_INIT  = SWC + FWM * (1.0 - SWC - SOR)

# ==============================================================================
# 2.  DATA PIPELINE
# ==============================================================================
def load_wide(path):
    df = pd.read_csv(path, parse_dates=['time']).set_index('time').sort_index()
    df.columns = (df.columns
                  .str.replace('-', '_')
                  .str.replace(r'case_0*(\d+)', r'case_\1', regex=True))
    return df

print("[DATA] Loading CSVs …")
wc_df = load_wide(os.path.join(DATA_DIR, 'Water cut.csv'))
op_df = load_wide(os.path.join(DATA_DIR, 'Oil Production.csv'))
cp_df = load_wide(os.path.join(DATA_DIR, 'Polymer concentration.csv'))

idx   = wc_df.index.intersection(op_df.index).intersection(cp_df.index)
wc_df, op_df, cp_df = wc_df.loc[idx], op_df.loc[idx], cp_df.loc[idx]

cases   = [c for c in wc_df.columns if c in op_df.columns and c in cp_df.columns]
N_T     = len(idx)
T_SPAN  = float(N_T - 1)
DT_DAYS = float((idx[1] - idx[0]).days) if N_T > 1 else 13.33
OIL_MAX = float(op_df[cases].values.max())

t_norm = np.arange(N_T, dtype=np.float32) / T_SPAN      # [0, 1]

# Polymer inlet concentration per case [ppm] — constant within each case
poly_ppm  = cp_df.iloc[0]
POLY_MIN  = float(poly_ppm.min())    # 13.33 ppm
POLY_MAX  = float(poly_ppm.max())    # 706.67 ppm

def build_dataset(case_list):
    """
    X : (N_cases × N_T, 2)  →  [t_norm, Cpi_norm]
    Y : (N_cases × N_T, 2)  →  [water_cut, oil_rate_norm]
    """
    X_parts, Y_parts = [], []
    for c in case_list:
        cpi_n = (float(poly_ppm[c]) - POLY_MIN) / (POLY_MAX - POLY_MIN)
        wc_   = np.clip(wc_df[c].values[:N_T], 0.0, 1.0).astype('f4')
        op_   = np.clip(op_df[c].values[:N_T], 0.0, None).astype('f4') / OIL_MAX
        X_parts.append(np.column_stack([t_norm,
                                        np.full(N_T, cpi_n, 'f4')]))
        Y_parts.append(np.column_stack([wc_, op_]))
    return np.vstack(X_parts).astype('f4'), np.vstack(Y_parts).astype('f4')

np.random.seed(42)
shuffled    = np.random.permutation(cases)
train_cases = sorted(shuffled[:40].tolist(), key=lambda c: int(c.split('_')[1]))
val_cases   = sorted(shuffled[40:48].tolist(), key=lambda c: int(c.split('_')[1]))
test_cases  = sorted(shuffled[48:].tolist(),   key=lambda c: int(c.split('_')[1]))

X_tr, Y_tr = build_dataset(train_cases)
X_va, Y_va = build_dataset(val_cases)
X_te, Y_te = build_dataset(test_cases)

print(f"[SPLIT] Train={len(train_cases)} | Val={len(val_cases)} | Test={len(test_cases)}")
print(f"        X_tr={X_tr.shape}  Y_tr={Y_tr.shape}")

# ==============================================================================
# 3.  LATIN HYPERCUBE COLLOCATION POINTS  (Liu et al. Table I)
#     Physics residual is evaluated on these points, not only on data points.
#     20 000 interior  +  1 000 IC points  (Table I)
# ==============================================================================
def latin_hypercube(n, d=2, seed=0):
    """LHS in [0,1]^d — (Liu et al. use this for collocation, Table I)."""
    rng = np.random.default_rng(seed)
    result = np.zeros((n, d), dtype='f4')
    for j in range(d):
        perm = rng.permutation(n)
        result[:, j] = (perm + rng.uniform(size=n)) / n
    return result

N_COLLOC = 20_000
N_IC     = 1_000

# Interior collocation: (T, Cpi_norm) ∈ [0,1]×[0,1]
X_col = latin_hypercube(N_COLLOC, d=2, seed=1).astype('f4')  # shape (N_COLLOC, 2)

# IC collocation: T=0, Cpi_norm varies
X_ic  = np.column_stack([
    np.zeros(N_IC, 'f4'),
    latin_hypercube(N_IC, d=1, seed=2)[:, 0]
]).astype('f4')                                               # shape (N_IC, 2)

print(f"[COLLOC] Interior={N_COLLOC}  IC={N_IC}  (Liu et al. Table I)")

# ==============================================================================
# 4.  TRAINABLE PHYSICAL PARAMETERS
#     — Corey exponents, kr end-points        (Almajid Paper 2, Case 3)
#     — Cubic viscosity r, s, t               (Liu et al. Eq. 1)
#     — Pore-volume scale vp, oil-rate qsc    (Meng et al. Paper 1, Table 1)
#     All stored in log-space (→ always positive).
# ==============================================================================
log_nw    = tf.Variable(np.log(NW_INIT),  dtype='f4', name='log_nw')
log_no    = tf.Variable(np.log(NO_INIT),  dtype='f4', name='log_no')
log_krw   = tf.Variable(np.log(KRW_INIT), dtype='f4', name='log_krw')
log_kro   = tf.Variable(np.log(KRO_INIT), dtype='f4', name='log_kro')

# Cubic viscosity  μw(Cp) = MU_WI·(1 + r·Cp + s·Cp² + t·Cp³)  (Liu et al. Eq.1)
# r, s, t are raw (unconstrained) — positivity enforced via softplus
log_r  = tf.Variable(1.0,  dtype='f4', name='log_r')
log_s  = tf.Variable(0.0,  dtype='f4', name='log_s')
log_t  = tf.Variable(-1.0, dtype='f4', name='log_t')

# Pore-volume scale (BL ODE time scale)
log_vp    = tf.Variable(0.0, dtype='f4', name='log_vp')
# Polymer transport time scale
log_vp_cp = tf.Variable(0.0, dtype='f4', name='log_vp_cp')
# Oil-rate scaling factor
log_qsc   = tf.Variable(0.0, dtype='f4', name='log_qsc')

phys_vars = [log_nw, log_no, log_krw, log_kro,
             log_r, log_s, log_t,
             log_vp, log_vp_cp, log_qsc]

def get_phys():
    nw    = tf.exp(log_nw)
    no    = tf.exp(log_no)
    krw   = tf.exp(log_krw)
    kro   = tf.exp(log_kro)
    r_v   = tf.nn.softplus(log_r)       # r ≥ 0
    s_v   = tf.nn.softplus(log_s)       # s ≥ 0
    t_v   = tf.nn.softplus(log_t)       # t ≥ 0
    vp    = tf.exp(log_vp)
    vp_cp = tf.exp(log_vp_cp)
    qsc   = tf.exp(log_qsc)
    return nw, no, krw, kro, r_v, s_v, t_v, vp, vp_cp, qsc

# ==============================================================================
# 5.  FRACTIONAL FLOW — CUBIC POLYMER VISCOSITY  (Liu et al. Eq. 1, 3)
# ==============================================================================
_SWC  = tf.constant(SWC,         dtype='f4')
_SOR  = tf.constant(SOR,         dtype='f4')
_DENOM= tf.constant(1.-SWC-SOR,  dtype='f4')
_MUO  = tf.constant(MU_O,        dtype='f4')
_MUWI = tf.constant(MU_WI,       dtype='f4')

def fractional_flow(Sw_norm, Cp_out_norm):
    """
    Sw_norm   ∈ [0,1] → physical Sw = Swc + Sw_norm·(1-Swc-Sor)
    Cp_out_norm ∈ [0,1] → actual Cp_out in [0, Cpi_max]
                          used to compute μw(Cp) via cubic model (Liu et al. Eq.1)
    Returns fw, Sw.
    """
    nw, no, krw, kro, r_v, s_v, t_v, *_ = get_phys()

    Sw = _SWC + Sw_norm * _DENOM                              # physical Sw
    Se = tf.clip_by_value((Sw - _SWC) / (_DENOM + 1e-8), 0., 1.)

    krw_f = krw * tf.pow(Se + 1e-8, nw)                      # Almajid Eq. 4
    kro_f = kro * tf.pow(1. - Se + 1e-8, no)                 # Almajid Eq. 5

    # Cubic viscosity (Liu et al. Eq. 1)
    Cp    = Cp_out_norm                                       # ∈ [0,1] normalised
    mu_w  = _MUWI * (1. + r_v*Cp + s_v*Cp**2 + t_v*Cp**3)

    lam_w = krw_f / (mu_w  + 1e-8)
    lam_o = kro_f / (_MUO  + 1e-8)
    fw    = lam_w / (lam_w + lam_o + 1e-8)                   # Liu et al. Eq. 3
    return fw, Sw

# ==============================================================================
# 6.  NETWORK ARCHITECTURES
# ==============================================================================
def build_pinn_network():
    """
    PINN-1  (Liu et al. Fig. 2a):
      — ONE network estimates both Sw and Cp_out simultaneously.
      — Proven superior to two separate networks (PINN-2) in Liu et al.
      — Architecture: time embedding → concat Cpi_norm → 3×(FC+BN+Tanh+Dropout)
      — Outputs: [Sw_norm, Cp_out_norm]  both in [0,1] via sigmoid
    """
    t_inp   = keras.Input(shape=(1,), name='time')
    cpi_inp = keras.Input(shape=(1,), name='cpi_norm')

    # Time embedding — Meng et al. Fig. 5
    t_emb = keras.layers.Dense(64, kernel_initializer='glorot_normal')(t_inp)
    t_emb = keras.layers.Activation('tanh')(t_emb)

    x = keras.layers.Concatenate()([t_emb, cpi_inp])

    for units in [128, 128, 64]:
        x = keras.layers.Dense(units, kernel_initializer='glorot_normal')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('tanh')(x)
        x = keras.layers.Dropout(0.1)(x)

    # Two sigmoid outputs: Sw_norm and Cp_out_norm ∈ [0,1]
    out = keras.layers.Dense(2, activation='sigmoid', name='Sw_Cp')(x)
    return keras.Model(inputs=[t_inp, cpi_inp], outputs=out, name='PINN1')


def build_nn_network():
    """Pure data-driven NN — same capacity as PINN for fair comparison."""
    inp = keras.Input(shape=(2,))
    x   = keras.layers.Dense(64, kernel_initializer='glorot_normal')(inp)
    x   = keras.layers.BatchNormalization()(x)
    x   = keras.layers.Activation('tanh')(x)
    for units in [128, 128, 64]:
        x = keras.layers.Dense(units, kernel_initializer='glorot_normal')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('tanh')(x)
        x = keras.layers.Dropout(0.1)(x)
    out = keras.layers.Dense(2, activation='sigmoid')(x)
    return keras.Model(inp, out, name='PureNN')

# ==============================================================================
# 7.  LOSS FUNCTIONS  (Liu et al. Eq. 8-10; Almajid Eq. 9-10; Meng et al. Eq. 9)
#
#  L = ω1·LSw  +  ω2·LCp  +  ω3·Ldata          Liu et al. Eq. 8
#
#  LSw = ω11·R_BL  +  ω12·IC_Sw
#  LCp = ω21·R_Cp  +  ω22·IC_Cp
#
#  R_BL   : 0-D Buckley-Leverett ODE residual + artificial viscosity  (Eq. 9,16)
#             dSw_norm/dT = vp·(1-fw)/(1-Swc-Sor) + ε·d²Sw_norm/dT²
#  R_Cp   : Simplified polymer eq. (Liu et al. Eq. 22), 0-D form:
#             Sw·dCp_out/dT + fw·(Cp_out - Cpi_norm)·vp_cp = 0
#  IC_Sw  : Sw(T=0) = SW_INIT                    (Liu et al. Eq. 5 / Almajid Eq.6)
#  IC_Cp  : Cp_out(T=0) = 0                      (Liu et al. Eq. 5)
#  Ldata  : MSE(fw→wc) + MSE(qo→oil)             (Meng et al. Eq. 9)
# ==============================================================================

# Liu et al. loss weights  (Eq. 8)
W1, W2, W3    = 1.0, 1.0, 1.0    # ω1, ω2, ω3
W11, W12      = 1.0, 5.0         # ω11 (BL PDE), ω12 (IC Sw)
W21, W22      = 1.0, 5.0         # ω21 (polymer PDE), ω22 (IC Cp)

# Artificial viscosity coefficient  (Liu et al. Example 3, ε=1e-3 optimal)
EPS_AV = 1e-3

_SW_INIT_NORM = tf.constant((SW_INIT - SWC) / (1. - SWC - SOR), dtype='f4')


def pinn_losses(model, x_data, y_data,
                x_col=None, x_ic=None, training=True):
    """
    Parameters
    ----------
    x_data  : data-point inputs  (N_data, 2)  [t_norm, Cpi_norm]
    y_data  : observed outputs   (N_data, 2)  [water_cut, oil_norm]
    x_col   : interior LHS collocation (N_col, 2)   — physics residual
    x_ic    : IC collocation           (N_ic, 2)    — IC residual
    """
    if x_col is None:
        x_col = X_col
    if x_ic is None:
        x_ic  = X_ic

    # ── (a)  Physics residuals on LHS collocation points ─────────────────────
    t_col  = tf.convert_to_tensor(x_col[:, 0:1], dtype='f4')
    cpi_col= tf.convert_to_tensor(x_col[:, 1:2], dtype='f4')

    # Nested tapes for 1st and 2nd derivatives  (Liu et al. artificial viscosity)
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(t_col)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(t_col)
            out_col     = model([t_col, cpi_col], training=False)
            Sw_col      = out_col[:, 0:1]
            Cp_out_col  = out_col[:, 1:2]
            fw_col, _   = fractional_flow(Sw_col, Cp_out_col)
        dSw_dT  = tape1.gradient(Sw_col,     t_col)    # ∂Sw_norm/∂T
        dCp_dT  = tape1.gradient(Cp_out_col, t_col)    # ∂Cp_out_norm/∂T
        del tape1
    d2Sw_dT2 = tape2.gradient(dSw_dT, t_col)           # ∂²Sw_norm/∂T²
    del tape2

    _, _, _, _, _, _, _, vp, vp_cp, _ = get_phys()

    # BL residual + artificial viscosity  (Liu et al. Eq.9 + ε·∂²Sw/∂T²)
    rhs_BL   = vp * (1. - fw_col) / (_DENOM + 1e-8)
    R_BL     = dSw_dT - rhs_BL
    if d2Sw_dT2 is not None:
        R_BL = R_BL - EPS_AV * d2Sw_dT2                # artificial viscosity

    # Simplified polymer eq.  Sw·∂Cp/∂T + fw·(Cp_out - Cpi)·vp_cp = 0
    # 0-D form of Liu et al. Eq. 22: Sw·∂Cp/∂T + fw·∂Cp/∂X = 0
    # where ∂Cp/∂X ≈ (Cp_out - Cpi_norm)·vp_cp  (finite-difference lumped approx.)
    R_Cp     = Sw_col * dCp_dT + fw_col * (Cp_out_col - cpi_col) * vp_cp

    L_PDE_Sw = tf.reduce_mean(tf.square(R_BL))
    L_PDE_Cp = tf.reduce_mean(tf.square(R_Cp))

    # ── (b)  Initial condition residuals  (Liu et al. Eq. 5) ─────────────────
    t_ic   = tf.convert_to_tensor(x_ic[:, 0:1], dtype='f4')   # T = 0
    cpi_ic = tf.convert_to_tensor(x_ic[:, 1:2], dtype='f4')

    out_ic    = model([t_ic, cpi_ic], training=False)
    Sw_ic     = out_ic[:, 0:1]
    Cp_out_ic = out_ic[:, 1:2]

    # Sw(T=0) = SW_INIT  →  Sw_norm(T=0) = _SW_INIT_NORM
    L_IC_Sw   = tf.reduce_mean(tf.square(Sw_ic - _SW_INIT_NORM))
    # Cp_out(T=0) = 0  (no polymer at outlet at start)
    L_IC_Cp   = tf.reduce_mean(tf.square(Cp_out_ic))

    # ── (c)  Data loss on observed points  (Meng et al. Eq. 9) ───────────────
    t_d   = tf.convert_to_tensor(x_data[:, 0:1], dtype='f4')
    cpi_d = tf.convert_to_tensor(x_data[:, 1:2], dtype='f4')

    out_d     = model([t_d, cpi_d], training=training)
    Sw_d      = out_d[:, 0:1]
    Cp_out_d  = out_d[:, 1:2]
    fw_d, _   = fractional_flow(Sw_d, Cp_out_d)

    _, _, _, _, _, _, _, _, _, qsc = get_phys()
    qo_d      = qsc * (1. - fw_d)

    wc_obs    = tf.convert_to_tensor(y_data[:, 0:1], dtype='f4')
    oil_obs   = tf.convert_to_tensor(y_data[:, 1:2], dtype='f4')
    L_data    = (tf.reduce_mean(tf.square(fw_d - wc_obs)) +
                 tf.reduce_mean(tf.square(qo_d - oil_obs)))

    # ── (d)  Combine  (Liu et al. Eq. 8) ──────────────────────────────────────
    L_Sw    = W11 * L_PDE_Sw + W12 * L_IC_Sw
    L_Cp    = W21 * L_PDE_Cp + W22 * L_IC_Cp
    L_total = W1 * L_Sw + W2 * L_Cp + W3 * L_data

    return L_total, L_data, L_PDE_Sw, L_PDE_Cp, L_IC_Sw, L_IC_Cp


def nn_loss(model, x, y, training=True):
    yp = model(x, training=training)
    return tf.reduce_mean(tf.square(yp - y))

# ==============================================================================
# 8.  FLAT-WEIGHT HELPERS FOR L-BFGS
# ==============================================================================
def flat_weights(variables):
    return tf.concat([tf.reshape(v, [-1]) for v in variables], axis=0)

def assign_flat(flat_w, variables):
    idx = 0
    for v in variables:
        n = v.shape.num_elements()
        v.assign(tf.reshape(flat_w[idx:idx+n], v.shape))
        idx += n

# ==============================================================================
# 9.  TRAINING
# ==============================================================================
EPOCHS_ADAM  = 500
BATCH_SIZE   = 2048
LR           = 1e-3                # Liu et al. Table I: lr=10⁻³
LBFGS_ITER   = 500                 # Almajid Paper 2 Table 2


def train_pinn(model, tag='PINN'):
    all_vars = model.trainable_variables + phys_vars
    opt = keras.optimizers.Adam(learning_rate=LR, weight_decay=1e-4)

    ds = (tf.data.Dataset.from_tensor_slices((X_tr, Y_tr))
          .shuffle(80_000).batch(BATCH_SIZE))

    hist = {'tr_data': [], 'tr_pde_sw': [], 'tr_pde_cp': [],
            'tr_ic_sw': [], 'tr_ic_cp': [], 'va': []}
    best_val, best_w = np.inf, None

    print(f"\n[Stage 1 – Adam]  {tag}  ({EPOCHS_ADAM} epochs) …")
    for ep in range(1, EPOCHS_ADAM + 1):
        # Physics weight ramp-up — avoids early gradient conflict (Meng et al.)
        w_phy = min(1.0, ep / 200.0)
        ep_losses = {k: [] for k in hist if k.startswith('tr')}

        for xb, yb in ds:
            with tf.GradientTape() as tape:
                L_tot, Ld, Lpde_sw, Lpde_cp, Lic_sw, Lic_cp = pinn_losses(
                    model, xb, yb, training=True)
                # Apply physics ramp to PDE+IC terms
                L_scaled = (W3 * tf.stop_gradient(Ld) +
                            w_phy * (W1*(W11*Lpde_sw + W12*Lic_sw) +
                                     W2*(W21*Lpde_cp + W22*Lic_cp)) +
                            Ld)
            grads = tape.gradient(L_scaled, all_vars)
            grads = [tf.clip_by_norm(g, 1.0) if g is not None else g
                     for g in grads]
            opt.apply_gradients(zip(grads, all_vars))
            ep_losses['tr_data'].append(float(Ld))
            ep_losses['tr_pde_sw'].append(float(Lpde_sw))
            ep_losses['tr_pde_cp'].append(float(Lpde_cp))
            ep_losses['tr_ic_sw'].append(float(Lic_sw))
            ep_losses['tr_ic_cp'].append(float(Lic_cp))

        # Validation data loss only
        _, Ld_va, *_ = pinn_losses(model, X_va, Y_va, training=False)
        val_loss = float(Ld_va)

        for k in ep_losses:
            hist[k].append(np.mean(ep_losses[k]))
        hist['va'].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_w   = model.get_weights()

        if ep % 100 == 0 or ep == 1:
            nw_, no_, _, _, r_, s_, t_, vp_, vp_cp_, qs_ = [
                float(tf.nn.softplus(p)) if 'log_r' in p.name
                or 'log_s' in p.name or 'log_t' in p.name
                else float(tf.exp(p)) for p in phys_vars]
            print(f"  Ep {ep:4d} | Ldata={hist['tr_data'][-1]:.5f} "
                  f"| BL={hist['tr_pde_sw'][-1]:.5f} "
                  f"| Cp={hist['tr_pde_cp'][-1]:.5f} | Val={val_loss:.5f} "
                  f"| nw={nw_:.2f} r={r_:.2f} s={s_:.2f} t={t_:.2f}")

    model.set_weights(best_w)

    # ── Stage 2 : L-BFGS  (Almajid Paper 2, Table 2) ─────────────────────────
    print(f"\n[Stage 2 – L-BFGS]  {tag}  (max {LBFGS_ITER} iters) …")

    def lbfgs_obj(flat_w):
        assign_flat(flat_w, all_vars)
        with tf.GradientTape() as tape:
            L_tot, Ld, Lpde_sw, Lpde_cp, Lic_sw, Lic_cp = pinn_losses(
                model, X_tr, Y_tr, training=False)
        grads     = tape.gradient(L_tot, all_vars)
        flat_grad = tf.concat(
            [tf.reshape(g if g is not None else tf.zeros_like(v), [-1])
             for g, v in zip(grads, all_vars)], axis=0)
        return L_tot, flat_grad

    try:
        res = tfp.optimizer.lbfgs_minimize(
            lbfgs_obj,
            initial_position=flat_weights(all_vars),
            max_iterations=LBFGS_ITER,
            num_correction_pairs=50,
            tolerance=1e-8)
        assign_flat(res.position, all_vars)
        print(f"  Converged={res.converged.numpy()}  "
              f"Iterations={res.num_iterations.numpy()}")
    except Exception as exc:
        print(f"  L-BFGS error: {exc}")

    return hist


def train_nn(model, tag='Pure NN'):
    opt = keras.optimizers.Adam(learning_rate=LR, weight_decay=1e-4)
    ds  = (tf.data.Dataset.from_tensor_slices((X_tr, Y_tr))
           .shuffle(80_000).batch(BATCH_SIZE))
    tr_h, va_h = [], []
    best_val, best_w = np.inf, None

    print(f"\n[Stage 1 – Adam]  {tag}  ({EPOCHS_ADAM} epochs) …")
    for ep in range(1, EPOCHS_ADAM + 1):
        ep_l = []
        for xb, yb in ds:
            with tf.GradientTape() as tape:
                loss = nn_loss(model, xb, yb, training=True)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            ep_l.append(float(loss))
        val_loss = float(nn_loss(model, X_va, Y_va, training=False))
        tr_h.append(np.mean(ep_l)); va_h.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss; best_w = model.get_weights()
        if ep % 100 == 0 or ep == 1:
            print(f"  Ep {ep:4d} | Train={tr_h[-1]:.5f} | Val={val_loss:.5f}")

    model.set_weights(best_w)

    print(f"\n[Stage 2 – L-BFGS]  {tag}  (max {LBFGS_ITER} iters) …")

    def lbfgs_obj(flat_w):
        assign_flat(flat_w, model.trainable_variables)
        with tf.GradientTape() as tape:
            loss = nn_loss(model, X_tr, Y_tr, training=False)
        grads     = tape.gradient(loss, model.trainable_variables)
        flat_grad = tf.concat(
            [tf.reshape(g if g is not None else tf.zeros_like(v), [-1])
             for g, v in zip(grads, model.trainable_variables)], axis=0)
        return loss, flat_grad

    try:
        res = tfp.optimizer.lbfgs_minimize(
            lbfgs_obj,
            initial_position=flat_weights(model.trainable_variables),
            max_iterations=LBFGS_ITER,
            num_correction_pairs=50, tolerance=1e-8)
        assign_flat(res.position, model.trainable_variables)
        print(f"  Converged={res.converged.numpy()}")
    except Exception as exc:
        print(f"  L-BFGS error: {exc}")

    return {'tr': np.array(tr_h), 'va': np.array(va_h)}

# ==============================================================================
# 10.  BUILD & TRAIN
# ==============================================================================
print('\n[BUILD] Initializing models …')
pinn_model = build_pinn_network()
nn_model   = build_nn_network()
pinn_model.summary()

pinn_hist = train_pinn(pinn_model, 'PINN')
nn_hist   = train_nn(nn_model,     'Pure NN')

# Report learned physics
nw_l, no_l, krw_l, kro_l, r_l, s_l, t_l, vp_l, vp_cp_l, qsc_l = (
    [float(tf.exp(p)) for p in [log_nw, log_no, log_krw, log_kro,
                                 log_vp, log_vp_cp, log_qsc]] +
    [float(tf.nn.softplus(p)) for p in [log_r, log_s, log_t]])
# fix: reorder
nw_l, no_l, krw_l, kro_l = [float(tf.exp(p))
                              for p in [log_nw, log_no, log_krw, log_kro]]
r_l, s_l, t_l = [float(tf.nn.softplus(p)) for p in [log_r, log_s, log_t]]
vp_l, vp_cp_l, qsc_l = [float(tf.exp(p))
                          for p in [log_vp, log_vp_cp, log_qsc]]

print("\n[PHYSICS] Learned parameters:")
print(f"  Corey nw    = {nw_l:.4f}  (prior {NW_INIT})")
print(f"  Corey no    = {no_l:.4f}  (prior {NO_INIT})")
print(f"  krw_max     = {krw_l:.4f}  (prior {KRW_INIT})")
print(f"  kro_max     = {kro_l:.4f}  (prior {KRO_INIT})")
print(f"  Cubic visc  r={r_l:.4f}  s={s_l:.4f}  t={t_l:.4f}")
print(f"  μw(Cp)      = {MU_WI}·(1 + {r_l:.2f}·Cp + {s_l:.2f}·Cp² + {t_l:.2f}·Cp³)")
print(f"  vp_scale    = {vp_l:.4f}")
print(f"  vp_cp_scale = {vp_cp_l:.4f}")
print(f"  q_scale     = {qsc_l:.4f}")

# ==============================================================================
# 11.  PREDICTION HELPER
# ==============================================================================
def pinn_predict(X):
    """Returns [fw (water cut), oil_rate_norm] from the PINN physics path."""
    t_in   = tf.constant(X[:, 0:1], dtype='f4')
    cpi_in = tf.constant(X[:, 1:2], dtype='f4')
    out    = pinn_model([t_in, cpi_in], training=False)
    Sw_n   = out[:, 0:1]
    Cp_n   = out[:, 1:2]
    fw, _  = fractional_flow(Sw_n, Cp_n)
    _, _, _, _, _, _, _, _, _, qsc = get_phys()
    return np.concatenate([fw.numpy(), (qsc*(1.-fw)).numpy()], axis=1)

# ==============================================================================
# 12.  METRICS  (Paper 1 Table 3 format)
# ==============================================================================
nn_preds   = nn_model(X_te, training=False).numpy()
pinn_preds = pinn_predict(X_te)

def metrics(y_true, y_pred):
    mask = y_true > 1e-3
    mape = np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return r2, rmse, mape

r2_wc_nn,  rmse_wc_nn,  mape_wc_nn  = metrics(Y_te[:,0], nn_preds[:,0])
r2_wc_pi,  rmse_wc_pi,  mape_wc_pi  = metrics(Y_te[:,0], pinn_preds[:,0])
r2_op_nn,  rmse_op_nn,  mape_op_nn  = metrics(Y_te[:,1], nn_preds[:,1])
r2_op_pi,  rmse_op_pi,  mape_op_pi  = metrics(Y_te[:,1], pinn_preds[:,1])

_, Ld_tr_pi, *_ = pinn_losses(pinn_model, X_tr, Y_tr, training=False)
_, Ld_va_pi, *_ = pinn_losses(pinn_model, X_va, Y_va, training=False)
_, Ld_te_pi, *_ = pinn_losses(pinn_model, X_te, Y_te, training=False)
Ld_tr_nn = float(nn_loss(nn_model, X_tr, Y_tr, training=False))
Ld_va_nn = float(nn_loss(nn_model, X_va, Y_va, training=False))
Ld_te_nn = float(nn_loss(nn_model, X_te, Y_te, training=False))

print("\n" + "="*80)
print("           STATISTICAL EVALUATION METRICS  (Meng et al. Table 3 format)")
print("="*80)
print(f"{'Metric':<18} {'Pure NN':>16} {'Proposed PINN':>16}")
print("-"*80)
print(f"{'Train L_D':<18} {Ld_tr_nn:>16.6f} {float(Ld_tr_pi):>16.6f}")
print(f"{'Test  L_D':<18} {Ld_te_nn:>16.6f} {float(Ld_te_pi):>16.6f}")
print(f"{'Val   L_D':<18} {Ld_va_nn:>16.6f} {float(Ld_va_pi):>16.6f}")
print("="*80)
print(f"\n{'Metric':<14} {'NN WC':>12} {'PINN WC':>12} "
      f"{'NN Oil':>12} {'PINN Oil':>12}")
print("-"*64)
print(f"{'R²':<14} {r2_wc_nn:>12.4f} {r2_wc_pi:>12.4f} "
      f"{r2_op_nn:>12.4f} {r2_op_pi:>12.4f}")
print(f"{'RMSE':<14} {rmse_wc_nn:>12.6f} {rmse_wc_pi:>12.6f} "
      f"{rmse_op_nn:>12.6f} {rmse_op_pi:>12.6f}")
print(f"{'MAPE (%)':<14} {mape_wc_nn:>12.2f} {mape_wc_pi:>12.2f} "
      f"{mape_op_nn:>12.2f} {mape_op_pi:>12.2f}")
print("="*80)

# ==============================================================================
# 13.  FIGURES
# ==============================================================================
years_ax = t_norm * T_SPAN * DT_DAYS / 365.25

# ── Fig 1: Loss sub-terms  (Liu et al. Fig. 3 / Meng et al. Fig. 7) ──────────
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))

ax = axes1[0]
ax.semilogy(nn_hist['tr'], color='#2196F3', lw=1.5, label='Train L_D')
ax.semilogy(nn_hist['va'], color='#e74c3c',  lw=1.5, label='Val L_D', alpha=.85)
ax.set_title('(a) Pure Data-driven NN', fontweight='bold')

ax = axes1[1]
ax.semilogy(pinn_hist['tr_data'],   color='#2196F3', lw=1.5, label='Train L_D (data)')
ax.semilogy(pinn_hist['va'],        color='#e74c3c',  lw=1.5, label='Val L_D',   alpha=.85)
ax.semilogy(pinn_hist['tr_pde_sw'], color='#4CAF50',  lw=1.5, label='L_PDE Sw (BL)', ls='--')
ax.semilogy(pinn_hist['tr_pde_cp'], color='#FF9800',  lw=1.5, label='L_PDE Cp (polymer)', ls=':')
ax.semilogy(pinn_hist['tr_ic_sw'],  color='#9C27B0',  lw=1.0, label='L_IC Sw', ls='-.')
ax.semilogy(pinn_hist['tr_ic_cp'],  color='#795548',  lw=1.0, label='L_IC Cp', ls='-.')
ax.set_title('(b) Proposed PINN (Liu et al. Fig. 3)', fontweight='bold')

for ax in axes1:
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log scale)')
    ax.legend(fontsize=8); ax.grid(True, which='both', alpha=.3)
fig1.suptitle('Loss function and sub-terms during training', fontweight='bold')
fig1.tight_layout()
plt.savefig('fig1_losses.png', dpi=150, bbox_inches='tight'); plt.show()

# ── Fig 2: Cumulative oil crossplot  (Meng et al. Fig. 10) ───────────────────
def cumulative_oil(case_list, pred_fn=None):
    out = []
    for c in case_list:
        cpi_n = (float(poly_ppm[c]) - POLY_MIN) / (POLY_MAX - POLY_MIN)
        if pred_fn is None:
            oil = np.clip(op_df[c].values[:N_T], 0, None)
        else:
            Xc  = np.column_stack([t_norm,
                                    np.full(N_T, cpi_n, 'f4')]).astype('f4')
            oil = pred_fn(Xc)[:, 1] * OIL_MAX
        out.append(np.sum(oil) * DT_DAYS)
    return np.array(out)

act_tr = cumulative_oil(train_cases)
act_va = cumulative_oil(val_cases)
act_te = cumulative_oil(test_cases)
nn_fn  = lambda X: nn_model(X, training=False).numpy()

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
for ax, fn, sub, ttl in [
    (axes2[0], nn_fn,      '(a)', 'Standard Data-driven NN'),
    (axes2[1], pinn_predict,'(b)', 'Proposed PINN'),
]:
    ptr = cumulative_oil(train_cases, fn)
    pva = cumulative_oil(val_cases,   fn)
    pte = cumulative_oil(test_cases,  fn)
    ax.scatter(act_tr, ptr, c='#2196F3', s=45, alpha=.85, label='Train',      zorder=4)
    ax.scatter(act_va, pva, c='#4CAF50', s=45, alpha=.85, label='Validation', zorder=4, marker='s')
    ax.scatter(act_te, pte, c='#FF9800', s=70, alpha=.95, label='Test',       zorder=5, marker='^')
    all_a = np.concatenate([act_tr, act_va, act_te])
    all_p = np.concatenate([ptr, pva, pte])
    lo = min(all_a.min(), all_p.min())*.97
    hi = max(all_a.max(), all_p.max())*1.03
    ax.plot([lo,hi],[lo,hi],'k--',lw=1.5,label='Ideal 1:1',zorder=3)
    ax.set_xlim([lo,hi]); ax.set_ylim([lo,hi])
    r2s = f"R²  {r2_score(act_tr,ptr):.3f} Train\n    {r2_score(act_va,pva):.3f} Val\n    {r2_score(act_te,pte):.3f} Test"
    ax.text(0.04,.96,r2s,transform=ax.transAxes,fontsize=9,va='top',
            bbox=dict(boxstyle='round',fc='white',alpha=.85))
    ax.set_xlabel('Actual Cumulative Oil (bbl)')
    ax.set_ylabel('Predicted Cumulative Oil (bbl)')
    ax.set_title(f'{sub} {ttl}', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=.3)
fig2.suptitle('Predicted vs. Actual Cumulative Oil Production', fontweight='bold')
fig2.tight_layout()
plt.savefig('fig2_crossplot.png', dpi=150, bbox_inches='tight'); plt.show()

# ── Fig 3: Production match for test cases  (Meng et al. Fig. 9) ─────────────
fig3, axes3 = plt.subplots(len(test_cases), 4,
                            figsize=(18, 3.2*len(test_cases)), sharex=True)
for i, c in enumerate(test_cases):
    cpi_n = (float(poly_ppm[c]) - POLY_MIN) / (POLY_MAX - POLY_MIN)
    Xc    = np.column_stack([t_norm, np.full(N_T,cpi_n,'f4')]).astype('f4')
    nn_p  = nn_model(Xc, training=False).numpy()
    pi_p  = pinn_predict(Xc)
    twc   = wc_df[c].values[:N_T]
    top   = op_df[c].values[:N_T]
    cpi_v = float(poly_ppm[c])

    for col, pred, col_idx in [(nn_p, 'NN', 0), (pi_p, 'PINN', 1)]:
        axes3[i, col_idx].plot(years_ax, twc, 'k', lw=2, label='CMG STARS')
        axes3[i, col_idx].plot(years_ax, pred[:,0],
                               '#e74c3c' if col_idx==0 else '#2980b9',
                               lw=1.5, ls='--' if col_idx==0 else '-',
                               label=col)
        axes3[i, col_idx].set_ylim([0,1]); axes3[i, col_idx].grid(True, alpha=.2)
        axes3[i, col_idx+2].plot(years_ax, top, 'k', lw=2)
        axes3[i, col_idx+2].plot(years_ax, pred[:,1]*OIL_MAX,
                                  '#e74c3c' if col_idx==0 else '#2980b9',
                                  lw=1.5, ls='--' if col_idx==0 else '-')
        axes3[i, col_idx+2].grid(True, alpha=.2)

    axes3[i,0].set_ylabel(f'{c}\n({cpi_v:.0f} ppm)\nWater Cut')
    axes3[i,2].set_ylabel('Oil Rate (bbl/d)')
    if i == 0:
        for ax,ttl in zip(axes3[0],['NN — Water Cut','PINN — Water Cut',
                                     'NN — Oil Rate','PINN — Oil Rate']):
            ax.set_title(ttl, fontweight='bold', fontsize=10)
        axes3[0,0].legend(prop={'size':7}, loc='lower right')
        axes3[0,1].legend(prop={'size':7}, loc='lower right')

for ax in axes3[-1]:
    ax.set_xlabel('Time (Years)')
fig3.tight_layout()
plt.savefig('fig3_match.png', dpi=150, bbox_inches='tight'); plt.show()

# ── Fig 4: Learned fractional flow curves  (interpretability) ────────────────
fig4, ax4 = plt.subplots(figsize=(8, 5))
Sw_vals     = np.linspace(SWC, 1.-SOR, 200, dtype='f4')
Sw_norm_arr = tf.constant(((Sw_vals-SWC)/(1.-SWC-SOR)).reshape(-1,1))
colors      = plt.cm.plasma(np.linspace(0.1, 0.9, 7))
for col, cpi_v in zip(colors, np.linspace(POLY_MIN, POLY_MAX, 7)):
    cpi_n = (cpi_v - POLY_MIN)/(POLY_MAX - POLY_MIN)
    Cp_arr = tf.constant(np.full((200,1), cpi_n, 'f4'))
    fw_v, _ = fractional_flow(Sw_norm_arr, Cp_arr)
    ax4.plot(Sw_vals, fw_v.numpy().flatten(), color=col, lw=2,
             label=f'{cpi_v:.0f} ppm')
ax4.set_xlabel('Water Saturation $S_w$', fontsize=11)
ax4.set_ylabel('Fractional Flow $f_w$',  fontsize=11)
ax4.set_title('Learned Fractional Flow Curves — Cubic Viscosity Model\n'
              '(Liu et al. Eq. 1-3: higher polymer → lower fw → more oil)',
              fontweight='bold', fontsize=10)
ax4.legend(title='Polymer Conc.', fontsize=8); ax4.grid(True, alpha=.3)
fig4.tight_layout()
plt.savefig('fig4_fw_curves.png', dpi=150, bbox_inches='tight'); plt.show()

# ── Fig 5: Sw and Cp_out time-series (Liu et al. Fig. 4 analog) ──────────────
fig5, axes5 = plt.subplots(2, len(test_cases),
                            figsize=(4*len(test_cases), 7), sharex=True)
for i, c in enumerate(test_cases):
    cpi_n = (float(poly_ppm[c]) - POLY_MIN)/(POLY_MAX - POLY_MIN)
    Xc    = np.column_stack([t_norm, np.full(N_T,cpi_n,'f4')]).astype('f4')
    t_tf  = tf.constant(Xc[:,0:1]); cpi_tf = tf.constant(Xc[:,1:2])
    out   = pinn_model([t_tf, cpi_tf], training=False).numpy()

    axes5[0,i].plot(years_ax, out[:,0], '#2980b9', lw=2, label='$S_w$ norm')
    axes5[0,i].axhline(float(_SW_INIT_NORM), color='k', ls='--', lw=1,
                        label='$S_{w,init}$')
    axes5[0,i].set_ylim([0,1]); axes5[0,i].grid(True, alpha=.2)
    axes5[0,i].set_title(f'{c}\n({float(poly_ppm[c]):.0f} ppm)', fontsize=9)

    axes5[1,i].plot(years_ax, out[:,1], '#e74c3c', lw=2, label='$C_{p,out}$ norm')
    axes5[1,i].axhline(cpi_n, color='k', ls='--', lw=1, label='$C_{pi}$ (inlet)')
    axes5[1,i].set_ylim([0,1]); axes5[1,i].grid(True, alpha=.2)
    axes5[1,i].set_xlabel('Time (Years)')

axes5[0,0].set_ylabel('$S_w$ norm (PINN output)')
axes5[1,0].set_ylabel('$C_{p,out}$ norm (PINN output)')
axes5[0,0].legend(fontsize=8); axes5[1,0].legend(fontsize=8)
fig5.suptitle('PINN internal states: $S_w$ and $C_{p,out}$ for test cases\n'
              '(Liu et al. Fig. 4 analog — unobserved quantities inferred from data)',
              fontweight='bold')
fig5.tight_layout()
plt.savefig('fig5_internal_states.png', dpi=150, bbox_inches='tight'); plt.show()

print("\n[DONE]  All figures saved.")
