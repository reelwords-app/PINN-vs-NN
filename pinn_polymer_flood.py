"""
PINN for Polymer Flood Production Forecasting — 1D Spatial Model
=================================================================
Physics     : Liu et al. Physics of Fluids 37 036622 (2025)
               • 1D BL eq:     ∂Sw/∂T + vp·∂fw/∂X = 0              (Eq. 9)
               • Polymer eq:   ∂(Sw·Cp)/∂T + vp·∂(fw·Cp)/∂X = 0  (Eq. 4)
               • Cubic visc:   μw(Cp) = μwi(1+r·Cp+s·Cp²+t·Cp³)   (Eq. 1)
               • PINN-1:       one network → (Sw, Cp) simultaneously
               • LHS 3-D:      (X, T, Cpi) collocation              (Table I)
               • Multi-weight: L = ω1·LSw + ω2·LCp + ω3·Ldata      (Eq. 8)
Architecture: Meng et al. SPE-218863-MS — space/time embedding, BN, Dropout
Optimizer   : Almajid & Abu-Alsaud SPE-203033-MS — Adam → L-BFGS

Geometry    : Pelican Lake HP-6 Pilot  P1-I1-P2-I2-P3
               L = 175 m  injector-producer spacing
               X = x / 175 m  ∈ [0, 1]
               X = 0 : Injector  Sw(0,T)=1-Sor   Cp(0,T)=Cpi
               X = 1 : Producer  fw(1,T) = observed WC  (CMG data)

vp meaning  : vp = q·T_ref / (A·φ·L)  — dimensionless pore volumes injected
               q  = injection rate [bbl/d]
               A  = well length × reservoir thickness [m²]
               φ  = porosity
               L  = 175 m
               vp is computed exactly from injection files (not trainable)
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
SWC      = 0.23          # connate water saturation (Swr, CMG Corey curve — paper Sec. 2.3)
SOR      = 0.20          # residual oil saturation  (Sro, paper Sec. 2.3)
NW       = 3.0           # Corey exponent water     (paper Sec. 2.3 — fixed, not trainable)
NO       = 2.2           # Corey exponent oil       (paper Sec. 2.3 — fixed, not trainable)
KRW_MAX  = 0.10          # endpoint rel-perm water  (paper Sec. 2.3 — fixed, not trainable)
KRO_MAX  = 1.00          # endpoint rel-perm oil    (paper Sec. 2.3 — fixed, not trainable)
MU_O     = 1650.0        # oil viscosity [cp]       (paper Table 2)
MU_WI    = 1.0           # pure-water viscosity [cp]
FWM      = 0.12          # mobile water fraction    (paper Sec. 2.5, calibrated)
SW_INIT  = 0.36          # initial water saturation (Swinitial, paper Sec. 2.5 — calibrated)

# Pelican Lake HP-6 geometry
L_INJE   = 175.0         # injector-producer spacing [m]
L_WELL   = 1400.0        # horizontal well length [m]
H_RES    = 14.434692     # reservoir thickness [ft]
PHI      = 0.312         # porosity
BBL_TO_M3 = 0.158987     # bbl → m³

# ==============================================================================
# 2.  DATA PIPELINE
#     Observations are all at X=1 (producer well).
#     X_tr columns: [X_norm=1, T_norm, Cpi_norm]
#     Y_tr columns: [water_cut, oil_rate_norm]
# ==============================================================================
def load_wide(path):
    df = pd.read_csv(path, parse_dates=['time']).set_index('time').sort_index()
    df.columns = (df.columns
                  .str.replace('-', '_')
                  .str.replace(r'case_0*(\d+)', r'case_\1', regex=True))
    return df

print("[DATA] Loading CSVs …")
wc_df  = load_wide(os.path.join(DATA_DIR, 'Water cut.csv'))
op_df  = load_wide(os.path.join(DATA_DIR, 'Oil Production.csv'))
cp_df  = load_wide(os.path.join(DATA_DIR, 'Polymer concentration.csv'))
inj1_df= load_wide(os.path.join(DATA_DIR, 'Injection rate inj1.csv'))
inj2_df= load_wide(os.path.join(DATA_DIR, 'Injection rate inj2.csv'))

idx   = wc_df.index.intersection(op_df.index).intersection(cp_df.index)
wc_df, op_df, cp_df = wc_df.loc[idx], op_df.loc[idx], cp_df.loc[idx]

cases   = [c for c in wc_df.columns if c in op_df.columns and c in cp_df.columns]
N_T     = len(idx)
T_SPAN  = float(N_T - 1)
DT_DAYS = float((idx[1] - idx[0]).days) if N_T > 1 else 13.33
OIL_MAX = float(op_df[cases].values.max())

# ── Compute exact vp = q·T_ref / (A·φ·L)  (paper formula, not trainable) ────
# All 53 cases have identical injection rates — use case_1, ignore zero rows
Q_I1_MEAN  = float(inj1_df['case_1'][inj1_df['case_1'] > 0].mean())  # bbl/day
Q_I2_MEAN  = float(inj2_df['case_1'][inj2_df['case_1'] > 0].mean())  # bbl/day
# Each injector splits equally to its two adjacent producers (Voronoi symmetry)
Q_PER_PAIR = (Q_I1_MEAN + Q_I2_MEAN) / 4.0                          # bbl/day

H_M        = H_RES * 0.3048                     # ft → m  = 4.401 m
A_M2       = L_WELL * H_M                       # cross-section [m²] = 6161 m²
Q_M3       = Q_PER_PAIR * BBL_TO_M3             # bbl/day → m³/day
T_REF_DAYS = T_SPAN * DT_DAYS                   # total simulation duration [days]
VP_EXACT   = float(Q_M3 * T_REF_DAYS / (A_M2 * PHI * L_INJE))  # dimensionless

print(f"[VP]  Q_I1={Q_I1_MEAN:.1f} bbl/d  Q_I2={Q_I2_MEAN:.1f} bbl/d  "
      f"q/pair={Q_PER_PAIR:.1f} bbl/d")
print(f"      A={A_M2:.0f} m²  φ={PHI}  L={L_INJE} m  T_ref={T_REF_DAYS:.0f} d")
print(f"      vp = q·T_ref/(A·φ·L) = {VP_EXACT:.4f}  (exact, not trainable)")

t_norm   = np.arange(N_T, dtype=np.float32) / T_SPAN   # T ∈ [0, 1]

# Inlet polymer concentration per case [ppm] — first row = constant injected value
poly_ppm = cp_df.iloc[0]
POLY_MIN = float(poly_ppm.min())
POLY_MAX = float(poly_ppm.max())


def build_dataset(case_list):
    """
    Returns
    -------
    X : (N_cases × N_T, 3)  →  [X_norm=1, T_norm, Cpi_norm]
        X_norm=1 because all observations are at the producer (X=1).
    Y : (N_cases × N_T, 2)  →  [water_cut, oil_rate_norm]
    """
    X_parts, Y_parts = [], []
    for c in case_list:
        cpi_n = (float(poly_ppm[c]) - POLY_MIN) / (POLY_MAX - POLY_MIN)
        wc_   = np.clip(wc_df[c].values[:N_T], 0.0, 1.0).astype('f4')
        op_   = np.clip(op_df[c].values[:N_T], 0.0, None).astype('f4') / OIL_MAX
        X_parts.append(np.column_stack([
            np.ones(N_T,  'f4'),            # X = 1  (producer)
            t_norm,                          # T ∈ [0, 1]
            np.full(N_T, cpi_n, 'f4'),      # Cpi_norm ∈ [0, 1]
        ]))
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
print(f"        X_tr={X_tr.shape}  columns=[X=1, T, Cpi]")

# ==============================================================================
# 3.  LATIN HYPERCUBE COLLOCATION  (Liu et al. Table I)
#     3-D sampling in (X, T, Cpi) space.
#
#     Interior  : X∈(0,1), T∈(0,1), Cpi∈[0,1]  — enforce both PDEs
#     IC        : T=0, X∈[0,1], Cpi∈[0,1]       — initial conditions
#     Injector  : X=0, T∈[0,1], Cpi∈[0,1]       — injector boundary conditions
# ==============================================================================
def latin_hypercube(n, d=2, seed=0):
    """Latin Hypercube Sampling in [0,1]^d."""
    rng    = np.random.default_rng(seed)
    result = np.zeros((n, d), dtype='f4')
    for j in range(d):
        perm = rng.permutation(n)
        result[:, j] = (perm + rng.uniform(size=n)) / n
    return result


N_COLLOC = 5_000
N_IC     = 500
N_BC     = 500

# Interior: columns [X, T, Cpi]  all ∈ [0,1]
X_col = latin_hypercube(N_COLLOC, d=3, seed=1).astype('f4')

# IC: T=0, columns [X, T=0, Cpi]
_ic   = latin_hypercube(N_IC, d=2, seed=2)
X_ic  = np.column_stack([
    _ic[:, 0],                # X ∈ [0, 1]
    np.zeros(N_IC, 'f4'),    # T = 0
    _ic[:, 1],                # Cpi ∈ [0, 1]
]).astype('f4')

# Injector BC: X=0, columns [X=0, T, Cpi]
_bc   = latin_hypercube(N_BC, d=2, seed=3)
X_bc  = np.column_stack([
    np.zeros(N_BC, 'f4'),    # X = 0  (injector)
    _bc[:, 0],                # T ∈ [0, 1]
    _bc[:, 1],                # Cpi ∈ [0, 1]
]).astype('f4')

print(f"[COLLOC] Interior={N_COLLOC} | IC={N_IC} | InjectorBC={N_BC}")

# ==============================================================================
# 4.  PHYSICAL PARAMETERS
#     Corey exponents and endpoints are KNOWN from paper Sec. 2.3 → fixed constants.
#     Cubic viscosity coefficients (r, s, t) and oil-rate scale (qsc) are unknown
#     and must be learned from production data → trainable variables.
#
#     vp = q·T_ref/(A·φ·L) is computed EXACTLY from injection files → fixed constant.
# ==============================================================================

# Fixed constants — known from CMG model (paper Sec. 2.3)
_NW  = tf.constant(NW,      dtype='float32')   # Corey exponent water
_NO  = tf.constant(NO,      dtype='float32')   # Corey exponent oil
_KRW = tf.constant(KRW_MAX, dtype='float32')   # endpoint krw
_KRO = tf.constant(KRO_MAX, dtype='float32')   # endpoint kro

# vp as a fixed TF constant — exact value from injection rate files + geometry
_VP = tf.constant(VP_EXACT, dtype='float32')

# Trainable parameters — only those NOT known from the paper
# Cubic viscosity coefficients  μw(Cp) = μwi(1+r·Cp+s·Cp²+t·Cp³)
log_r   = tf.Variable(1.0,  dtype='float32', name='log_r')
log_s   = tf.Variable(0.0,  dtype='float32', name='log_s')
log_t   = tf.Variable(-1.0, dtype='float32', name='log_t')

# Oil-rate scaling factor (accounts for unit conversion and well geometry)
log_qsc = tf.Variable(0.0, dtype='float32', name='log_qsc')

phys_vars = [log_r, log_s, log_t, log_qsc]


def get_phys():
    r_v = tf.nn.softplus(log_r)
    s_v = tf.nn.softplus(log_s)
    t_v = tf.nn.softplus(log_t)
    qsc = tf.exp(log_qsc)
    return r_v, s_v, t_v, qsc

# ==============================================================================
# 5.  FRACTIONAL FLOW — CUBIC POLYMER VISCOSITY  (Liu et al. Eq. 1, 3)
# ==============================================================================
_SWC   = tf.constant(SWC,         dtype='float32')
_DENOM = tf.constant(1.-SWC-SOR,  dtype='float32')
_MUO   = tf.constant(MU_O,        dtype='float32')
_MUWI  = tf.constant(MU_WI,       dtype='float32')


def fractional_flow(Sw_norm, Cp_norm):
    """
    Sw_norm ∈ [0,1] → physical Sw = Swc + Sw_norm·(1-Swc-Sor)
    Cp_norm ∈ [0,1] → normalised polymer concentration
    Returns fw (fractional flow) and physical Sw.
    """
    r_v, s_v, t_v, *_ = get_phys()

    Sw    = _SWC + Sw_norm * _DENOM
    Se    = tf.clip_by_value((Sw - _SWC) / (_DENOM + 1e-8), 0., 1.)

    krw_f = _KRW * tf.pow(Se + 1e-8,       _NW)   # Corey krw (paper Sec. 2.3)
    kro_f = _KRO * tf.pow(1. - Se + 1e-8,  _NO)   # Corey kro (paper Sec. 2.3)

    # Liu et al. Eq. 1 — cubic polymer viscosity
    mu_w  = _MUWI * (1. + r_v*Cp_norm + s_v*Cp_norm**2 + t_v*Cp_norm**3)

    lam_w = krw_f / (mu_w + 1e-8)
    lam_o = kro_f / (_MUO + 1e-8)
    fw    = lam_w / (lam_w + lam_o + 1e-8)        # Liu et al. Eq. 3
    return fw, Sw

# ==============================================================================
# 6.  NETWORK ARCHITECTURES
# ==============================================================================
def build_pinn_network():
    """
    PINN-1  (Liu et al. Fig. 2a + Meng et al. Fig. 5)
    Inputs  : [X_norm, T_norm, Cpi_norm]  — spatial + time + polymer param
    Outputs : [Sw_norm(X,T), Cp_norm(X,T)] — field quantities via sigmoid ∈ [0,1]

    Separate space and time embeddings allow the network to learn
    independent representations before coupling them.
    """
    x_inp   = keras.Input(shape=(1,), name='x_norm')
    t_inp   = keras.Input(shape=(1,), name='t_norm')
    cpi_inp = keras.Input(shape=(1,), name='cpi_norm')

    # Space embedding
    x_emb = keras.layers.Dense(64, kernel_initializer='glorot_normal')(x_inp)
    x_emb = keras.layers.Activation('tanh')(x_emb)

    # Time embedding  (Meng et al. Fig. 5)
    t_emb = keras.layers.Dense(64, kernel_initializer='glorot_normal')(t_inp)
    t_emb = keras.layers.Activation('tanh')(t_emb)

    z = keras.layers.Concatenate()([x_emb, t_emb, cpi_inp])

    for units in [128, 128, 64]:
        z = keras.layers.Dense(units, kernel_initializer='glorot_normal')(z)
        z = keras.layers.BatchNormalization()(z)
        z = keras.layers.Activation('tanh')(z)
        z = keras.layers.Dropout(0.1)(z)

    out = keras.layers.Dense(2, activation='sigmoid', name='Sw_Cp')(z)
    return keras.Model(inputs=[x_inp, t_inp, cpi_inp], outputs=out, name='PINN1_1D')


def build_nn_network():
    """Pure data-driven NN — same capacity as PINN, no physics."""
    inp = keras.Input(shape=(3,))   # [X=1, T, Cpi]
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
# 7.  LOSS FUNCTIONS  (Liu et al. Eq. 8-10, exact 1-D physics)
#
#  BL equation   (Liu et al. Eq. 9):
#    ∂Sw/∂T + vp·∂fw/∂X = 0
#
#  Polymer equation  (Liu et al. Eq. 4, full product rule):
#    ∂(Sw·Cp)/∂T + vp·∂(fw·Cp)/∂X = 0
#
#  Initial conditions  (Liu et al. Eq. 5):
#    Sw(X, 0) = Swi   for all X ∈ [0,1]
#    Cp(X, 0) = 0     for all X ∈ [0,1]
#
#  Injector boundary conditions  (Liu et al. Eq. 5):
#    Sw(0, T) = 1 - Sor  →  Sw_norm(0,T) = 1.0
#    Cp(0, T) = Cpi_norm
#
#  Data loss at producer X=1  (Meng et al. Eq. 9):
#    fw(1,T,Cpi) vs observed water cut
#    qo(1,T,Cpi) vs observed oil rate
#
#  Combined  (Liu et al. Eq. 8):
#    L = W1·LSw + W2·LCp + W3·Ldata
#    LSw = W11·L_BL  + W12·L_IC_Sw + W13·L_BC_Sw
#    LCp = W21·L_Cp  + W22·L_IC_Cp + W23·L_BC_Cp
# ==============================================================================
W1,  W2,  W3  = 1.0, 1.0, 1.0
W11, W12, W13 = 1.0, 5.0, 5.0   # BL PDE | IC Sw | BC Sw (injector)
W21, W22, W23 = 1.0, 5.0, 5.0   # Cp PDE | IC Cp | BC Cp (injector)


_SW_INIT_NORM = tf.constant((SW_INIT - SWC) / (1. - SWC - SOR), dtype='float32')
_SW_INJ_NORM  = tf.constant(1.0, dtype='float32')   # Sw(0,T)=1-Sor → norm=1.0


def pinn_losses(model, x_data, y_data,
                x_col=None, x_ic=None, x_bc=None, training=True):
    """
    Parameters
    ----------
    x_data : (N, 3)     [X=1, T, Cpi]   — producer observations
    y_data : (N, 2)     [water_cut, oil_norm]
    x_col  : (N_col, 3) [X, T, Cpi]     — interior LHS collocation (PDE)
    x_ic   : (N_ic,  3) [X, T=0, Cpi]   — initial condition points
    x_bc   : (N_bc,  3) [X=0, T, Cpi]   — injector boundary points

    Returns
    -------
    L_total, L_data, L_BL, L_Cp_pde, L_IC_Sw, L_IC_Cp, L_BC_Sw, L_BC_Cp
    """
    if x_col is None: x_col = X_col
    if x_ic  is None: x_ic  = X_ic
    if x_bc  is None: x_bc  = X_bc

    # ── (a) Interior PDE residuals on LHS collocation ─────────────────────────
    X_c   = tf.convert_to_tensor(x_col[:, 0:1], dtype='float32')  # spatial
    T_c   = tf.convert_to_tensor(x_col[:, 1:2], dtype='float32')  # time
    Cpi_c = tf.convert_to_tensor(x_col[:, 2:3], dtype='float32')  # polymer

    # Single GradientTape — first derivatives ∂/∂X and ∂/∂T only.
    # Artificial viscosity (ε·∂²Sw/∂X²) is dropped: the nested tape that
    # computes the second derivative is the main training bottleneck.
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch([X_c, T_c])
        out_c   = model([X_c, T_c, Cpi_c], training=False)
        Sw_c    = out_c[:, 0:1]
        Cp_c    = out_c[:, 1:2]
        fw_c, _ = fractional_flow(Sw_c, Cp_c)
        SwCp_c  = Sw_c * Cp_c
        fwCp_c  = fw_c * Cp_c

    dSw_dT   = tape1.gradient(Sw_c,   T_c)   # ∂Sw/∂T
    dfw_dX   = tape1.gradient(fw_c,   X_c)   # ∂fw/∂X
    dSwCp_dT = tape1.gradient(SwCp_c, T_c)   # ∂(Sw·Cp)/∂T
    dfwCp_dX = tape1.gradient(fwCp_c, X_c)   # ∂(fw·Cp)/∂X
    del tape1

    _, _, _, qsc = get_phys()

    # BL equation  (Liu et al. Eq. 9)
    # ∂Sw/∂T + vp·∂fw/∂X = 0
    # vp = q·T_ref/(A·φ·L) — exact value from injection rates (not trainable)
    R_BL = dSw_dT + _VP * dfw_dX

    # Full polymer equation  (Liu et al. Eq. 4)
    # ∂(Sw·Cp)/∂T + vp·∂(fw·Cp)/∂X = 0
    R_Cp = dSwCp_dT + _VP * dfwCp_dX

    L_BL     = tf.reduce_mean(tf.square(R_BL))
    L_Cp_pde = tf.reduce_mean(tf.square(R_Cp))

    # ── (b) Initial conditions: T=0, all X  (Liu et al. Eq. 5) ───────────────
    X_ic_tf  = tf.convert_to_tensor(x_ic[:, 0:1], dtype='float32')
    T_ic_tf  = tf.convert_to_tensor(x_ic[:, 1:2], dtype='float32')  # = 0
    C_ic_tf  = tf.convert_to_tensor(x_ic[:, 2:3], dtype='float32')

    out_ic   = model([X_ic_tf, T_ic_tf, C_ic_tf], training=False)
    Sw_ic    = out_ic[:, 0:1]
    Cp_ic    = out_ic[:, 1:2]

    L_IC_Sw  = tf.reduce_mean(tf.square(Sw_ic - _SW_INIT_NORM))  # Sw(X,0)=Swi
    L_IC_Cp  = tf.reduce_mean(tf.square(Cp_ic))                   # Cp(X,0)=0

    # ── (c) Injector BC: X=0, all T  (Liu et al. Eq. 5) ──────────────────────
    X_bc_tf  = tf.convert_to_tensor(x_bc[:, 0:1], dtype='float32')  # = 0
    T_bc_tf  = tf.convert_to_tensor(x_bc[:, 1:2], dtype='float32')
    C_bc_tf  = tf.convert_to_tensor(x_bc[:, 2:3], dtype='float32')

    out_bc   = model([X_bc_tf, T_bc_tf, C_bc_tf], training=False)
    Sw_bc    = out_bc[:, 0:1]
    Cp_bc    = out_bc[:, 1:2]

    # Sw(0,T) = 1-Sor  →  Sw_norm = 1.0
    L_BC_Sw  = tf.reduce_mean(tf.square(Sw_bc - _SW_INJ_NORM))
    # Cp(0,T) = Cpi_norm
    L_BC_Cp  = tf.reduce_mean(tf.square(Cp_bc - C_bc_tf))

    # ── (d) Data loss at producer X=1  (Meng et al. Eq. 9) ───────────────────
    X_d   = tf.convert_to_tensor(x_data[:, 0:1], dtype='float32')  # = 1
    T_d   = tf.convert_to_tensor(x_data[:, 1:2], dtype='float32')
    C_d   = tf.convert_to_tensor(x_data[:, 2:3], dtype='float32')

    _, _, _, qsc = get_phys()
    out_d    = model([X_d, T_d, C_d], training=training)
    Sw_d     = out_d[:, 0:1]
    Cp_d     = out_d[:, 1:2]
    fw_d, _  = fractional_flow(Sw_d, Cp_d)
    qo_d     = qsc * (1. - fw_d)

    wc_obs   = tf.convert_to_tensor(y_data[:, 0:1], dtype='float32')
    oil_obs  = tf.convert_to_tensor(y_data[:, 1:2], dtype='float32')
    L_data   = (tf.reduce_mean(tf.square(fw_d - wc_obs)) +
                tf.reduce_mean(tf.square(qo_d - oil_obs)))

    # ── (e) Combine  (Liu et al. Eq. 8) ───────────────────────────────────────
    L_Sw    = W11*L_BL     + W12*L_IC_Sw + W13*L_BC_Sw
    L_Cp    = W21*L_Cp_pde + W22*L_IC_Cp + W23*L_BC_Cp
    L_total = W1*L_Sw + W2*L_Cp + W3*L_data

    return L_total, L_data, L_BL, L_Cp_pde, L_IC_Sw, L_IC_Cp, L_BC_Sw, L_BC_Cp


def nn_loss(model, x, y, training=True):
    yp = model(x, training=training)
    return tf.reduce_mean(tf.square(yp - y))

# ==============================================================================
# 8.  FLAT-WEIGHT HELPERS FOR L-BFGS  (Almajid Paper 2)
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
EPOCHS_ADAM = 500
BATCH_SIZE  = 2048
LR          = 1e-3      # Liu et al. Table I
LBFGS_ITER  = 500       # Almajid Paper 2 Table 2


def train_pinn(model, tag='PINN-1D'):
    all_vars = model.trainable_variables + phys_vars
    opt = keras.optimizers.Adam(learning_rate=LR, weight_decay=1e-4)

    ds = (tf.data.Dataset.from_tensor_slices((X_tr, Y_tr))
          .shuffle(80_000).batch(BATCH_SIZE))

    hist = {k: [] for k in ['tr_data', 'tr_bl', 'tr_cp',
                              'tr_ic_sw', 'tr_ic_cp',
                              'tr_bc_sw', 'tr_bc_cp', 'va']}
    best_val, best_w = np.inf, None

    print(f"\n[Stage 1 – Adam]  {tag}  ({EPOCHS_ADAM} epochs) …")
    for ep in range(1, EPOCHS_ADAM + 1):
        # Physics weight ramp-up over first 200 epochs (Meng et al.)
        # Prevents physics gradients from dominating before data loss converges
        w_phy = min(1.0, ep / 200.0)
        ep_l  = {k: [] for k in hist if k.startswith('tr')}

        for xb, yb in ds:
            with tf.GradientTape() as tape:
                (L_tot, Ld, Lbl, Lcp,
                 Lic_sw, Lic_cp, Lbc_sw, Lbc_cp) = pinn_losses(
                    model, xb, yb, training=True)
                L_scaled = (W3 * Ld +
                            w_phy * (
                                W1*(W11*Lbl    + W12*Lic_sw + W13*Lbc_sw) +
                                W2*(W21*Lcp    + W22*Lic_cp + W23*Lbc_cp)))
            grads = tape.gradient(L_scaled, all_vars)
            grads = [tf.clip_by_norm(g, 1.0) if g is not None else g
                     for g in grads]
            opt.apply_gradients(zip(grads, all_vars))

            ep_l['tr_data'].append(float(Ld))
            ep_l['tr_bl'].append(float(Lbl))
            ep_l['tr_cp'].append(float(Lcp))
            ep_l['tr_ic_sw'].append(float(Lic_sw))
            ep_l['tr_ic_cp'].append(float(Lic_cp))
            ep_l['tr_bc_sw'].append(float(Lbc_sw))
            ep_l['tr_bc_cp'].append(float(Lbc_cp))

        _, Ld_va, *_ = pinn_losses(model, X_va, Y_va, training=False)
        val_loss = float(Ld_va)
        for k in ep_l:
            hist[k].append(np.mean(ep_l[k]))
        hist['va'].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_w   = model.get_weights()

        if ep % 100 == 0 or ep == 1:
            r_  = float(tf.nn.softplus(log_r))
            s_  = float(tf.nn.softplus(log_s))
            t_  = float(tf.nn.softplus(log_t))
            print(f"  Ep {ep:4d} | Ldata={hist['tr_data'][-1]:.5f} "
                  f"| BL={hist['tr_bl'][-1]:.5f} "
                  f"| Cp={hist['tr_cp'][-1]:.5f} | Val={val_loss:.5f} "
                  f"| r={r_:.2f} s={s_:.2f} t={t_:.2f}")

    model.set_weights(best_w)

    # ── Stage 2: L-BFGS  (Almajid Paper 2, Table 2) ──────────────────────────
    print(f"\n[Stage 2 – L-BFGS]  {tag}  (max {LBFGS_ITER} iters) …")

    def lbfgs_obj(flat_w):
        assign_flat(flat_w, all_vars)
        with tf.GradientTape() as tape:
            L_tot, *_ = pinn_losses(model, X_tr, Y_tr, training=False)
        grads = tape.gradient(L_tot, all_vars)
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
        grads = tape.gradient(loss, model.trainable_variables)
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

pinn_hist = train_pinn(pinn_model, 'PINN-1D')
nn_hist   = train_nn(nn_model,     'Pure NN')

# Report physics parameters
r_l   = float(tf.nn.softplus(log_r))
s_l   = float(tf.nn.softplus(log_s))
t_l   = float(tf.nn.softplus(log_t))
qsc_l = float(tf.exp(log_qsc))

print("\n[PHYSICS] Parameters:")
print(f"  vp (EXACT) = {VP_EXACT:.4f}  ← q·T_ref/(A·φ·L), not trainable")
print(f"               q={Q_PER_PAIR:.1f} bbl/d | A={A_M2:.0f} m² | "
      f"φ={PHI} | L={L_INJE} m | T={T_REF_DAYS:.0f} d")
print(f"  Corey nw   = {NW}  (FIXED — paper Sec. 2.3)")
print(f"  Corey no   = {NO}  (FIXED — paper Sec. 2.3)")
print(f"  krw_max    = {KRW_MAX}  (FIXED — paper Sec. 2.3)")
print(f"  kro_max    = {KRO_MAX}  (FIXED — paper Sec. 2.3)")
print(f"  Cubic r    = {r_l:.4f}  (learned)")
print(f"  Cubic s    = {s_l:.4f}  (learned)")
print(f"  Cubic t    = {t_l:.4f}  (learned)")
print(f"  μw(Cp) = {MU_WI}·(1 + {r_l:.2f}·Cp + {s_l:.2f}·Cp² + {t_l:.2f}·Cp³)")
print(f"  q_scale    = {qsc_l:.4f}  (learned)")

# ==============================================================================
# 11.  PREDICTION HELPERS
# ==============================================================================
def pinn_predict(X_arr):
    """
    X_arr : (N, 3)  [X_norm, T_norm, Cpi_norm]
    Returns (N, 2)  [fw = water cut, oil_rate_norm]
    Producer predictions: pass X_norm=1.
    """
    X_tf  = tf.constant(X_arr[:, 0:1], dtype='float32')
    T_tf  = tf.constant(X_arr[:, 1:2], dtype='float32')
    C_tf  = tf.constant(X_arr[:, 2:3], dtype='float32')
    out   = pinn_model([X_tf, T_tf, C_tf], training=False)
    Sw_n  = out[:, 0:1]
    Cp_n  = out[:, 1:2]
    fw, _ = fractional_flow(Sw_n, Cp_n)
    _, _, _, qsc = get_phys()
    return np.concatenate([fw.numpy(), (qsc*(1.-fw)).numpy()], axis=1)


def pinn_field(x_norm_arr, t_norm_arr, cpi_n):
    """
    Evaluate PINN on a 2-D (X, T) grid for one polymer concentration.
    Returns Sw_grid and Cp_grid, shape (len(t_norm_arr), len(x_norm_arr)).
    Used for Fig 5 spatial profile plots.
    """
    XX, TT = np.meshgrid(x_norm_arr, t_norm_arr)
    X_flat = XX.ravel().astype('f4')
    T_flat = TT.ravel().astype('f4')
    C_flat = np.full_like(X_flat, cpi_n)

    out = pinn_model(
        [X_flat[:, None], T_flat[:, None], C_flat[:, None]],
        training=False).numpy()
    Sw_grid = out[:, 0].reshape(len(t_norm_arr), len(x_norm_arr))
    Cp_grid = out[:, 1].reshape(len(t_norm_arr), len(x_norm_arr))
    return Sw_grid, Cp_grid

# ==============================================================================
# 12.  METRICS  (Meng et al. Table 3 format)
# ==============================================================================
nn_preds   = nn_model(X_te, training=False).numpy()
pinn_preds = pinn_predict(X_te)


def metrics(y_true, y_pred):
    mask = y_true > 1e-3
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
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
print(f"{'Val   L_D':<18} {Ld_va_nn:>16.6f} {float(Ld_va_pi):>16.6f}")
print(f"{'Test  L_D':<18} {Ld_te_nn:>16.6f} {float(Ld_te_pi):>16.6f}")
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
x_phys   = np.linspace(0, L_INJE, 100, dtype=np.float32)  # [0, 175 m]
x_norm_g = x_phys / L_INJE                                  # [0, 1]

# ── Fig 1: All loss sub-terms  (Liu et al. Fig. 3) ───────────────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(15, 5))

ax = axes1[0]
ax.semilogy(nn_hist['tr'], color='#2196F3', lw=1.5, label='Train L_D')
ax.semilogy(nn_hist['va'], color='#e74c3c', lw=1.5, label='Val L_D', alpha=.85)
ax.set_title('(a) Pure Data-driven NN', fontweight='bold')

ax = axes1[1]
ax.semilogy(pinn_hist['tr_data'],  color='#2196F3', lw=1.5,
            label='$L_{data}$ (producer X=1)')
ax.semilogy(pinn_hist['va'],       color='#e74c3c', lw=1.5,
            label='Val $L_{data}$', alpha=.85)
ax.semilogy(pinn_hist['tr_bl'],    color='#4CAF50', lw=1.5,
            label='$L_{BL}$  ∂Sw/∂T + vp·∂fw/∂X', ls='--')
ax.semilogy(pinn_hist['tr_cp'],    color='#FF9800', lw=1.5,
            label='$L_{Cp}$  ∂(Sw·Cp)/∂T + vp·∂(fw·Cp)/∂X', ls=':')
ax.semilogy(pinn_hist['tr_ic_sw'], color='#9C27B0', lw=1.0,
            label='$L_{IC,Sw}$  Sw(X,0)=Swi', ls='-.')
ax.semilogy(pinn_hist['tr_ic_cp'], color='#795548', lw=1.0,
            label='$L_{IC,Cp}$  Cp(X,0)=0', ls='-.')
ax.semilogy(pinn_hist['tr_bc_sw'], color='#00BCD4', lw=1.0,
            label='$L_{BC,Sw}$  Sw(0,T)=1−Sor', ls='-.')
ax.semilogy(pinn_hist['tr_bc_cp'], color='#F44336', lw=1.0,
            label='$L_{BC,Cp}$  Cp(0,T)=Cpi', ls='-.')
ax.set_title('(b) 1-D Spatial PINN — All Loss Sub-terms', fontweight='bold')

for ax in axes1:
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log scale)')
    ax.legend(fontsize=7); ax.grid(True, which='both', alpha=.3)
fig1.suptitle(
    'Training losses  |  BL: ∂Sw/∂T + vp·∂fw/∂X = 0  '
    '|  Polymer: ∂(Sw·Cp)/∂T + vp·∂(fw·Cp)/∂X = 0',
    fontweight='bold', fontsize=10)
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
            Xc  = np.column_stack([
                np.ones(N_T,  'f4'),
                t_norm,
                np.full(N_T, cpi_n, 'f4'),
            ]).astype('f4')
            oil = pred_fn(Xc)[:, 1] * OIL_MAX
        out.append(np.sum(oil) * DT_DAYS)
    return np.array(out)


act_tr = cumulative_oil(train_cases)
act_va = cumulative_oil(val_cases)
act_te = cumulative_oil(test_cases)
nn_fn  = lambda X: nn_model(X, training=False).numpy()

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
for ax, fn, sub, ttl in [
    (axes2[0], nn_fn,        '(a)', 'Standard NN'),
    (axes2[1], pinn_predict,  '(b)', 'Proposed 1-D PINN'),
]:
    ptr = cumulative_oil(train_cases, fn)
    pva = cumulative_oil(val_cases,   fn)
    pte = cumulative_oil(test_cases,  fn)
    ax.scatter(act_tr, ptr, c='#2196F3', s=45, alpha=.85, label='Train',      zorder=4)
    ax.scatter(act_va, pva, c='#4CAF50', s=45, alpha=.85, label='Validation', zorder=4, marker='s')
    ax.scatter(act_te, pte, c='#FF9800', s=70, alpha=.95, label='Test',       zorder=5, marker='^')
    all_a = np.concatenate([act_tr, act_va, act_te])
    all_p = np.concatenate([ptr, pva, pte])
    lo = min(all_a.min(), all_p.min()) * .97
    hi = max(all_a.max(), all_p.max()) * 1.03
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='Ideal 1:1', zorder=3)
    ax.set_xlim([lo, hi]); ax.set_ylim([lo, hi])
    r2s = (f"R²  {r2_score(act_tr,ptr):.3f} Train\n"
           f"    {r2_score(act_va,pva):.3f} Val\n"
           f"    {r2_score(act_te,pte):.3f} Test")
    ax.text(0.04, .96, r2s, transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', fc='white', alpha=.85))
    ax.set_xlabel('Actual Cumulative Oil (bbl)')
    ax.set_ylabel('Predicted Cumulative Oil (bbl)')
    ax.set_title(f'{sub} {ttl}', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=.3)
fig2.suptitle('Predicted vs. Actual Cumulative Oil Production', fontweight='bold')
fig2.tight_layout()
plt.savefig('fig2_crossplot.png', dpi=150, bbox_inches='tight'); plt.show()

# ── Fig 3: Production match — test cases  (Meng et al. Fig. 9) ───────────────
fig3, axes3 = plt.subplots(len(test_cases), 4,
                            figsize=(18, 3.2*len(test_cases)), sharex=True)
if len(test_cases) == 1:
    axes3 = axes3[np.newaxis, :]

for i, c in enumerate(test_cases):
    cpi_n = (float(poly_ppm[c]) - POLY_MIN) / (POLY_MAX - POLY_MIN)
    Xc    = np.column_stack([np.ones(N_T, 'f4'), t_norm,
                              np.full(N_T, cpi_n, 'f4')]).astype('f4')
    nn_p  = nn_model(Xc, training=False).numpy()
    pi_p  = pinn_predict(Xc)
    twc   = wc_df[c].values[:N_T]
    top   = op_df[c].values[:N_T]

    for col_idx, (pred, lbl) in enumerate([(nn_p, 'NN'), (pi_p, 'PINN')]):
        clr = '#e74c3c' if col_idx == 0 else '#2980b9'
        ls  = '--'      if col_idx == 0 else '-'
        axes3[i, col_idx].plot(years_ax, twc, 'k', lw=2, label='CMG STARS')
        axes3[i, col_idx].plot(years_ax, pred[:, 0], clr, lw=1.5, ls=ls, label=lbl)
        axes3[i, col_idx].set_ylim([0, 1])
        axes3[i, col_idx].grid(True, alpha=.2)
        axes3[i, col_idx+2].plot(years_ax, top, 'k', lw=2)
        axes3[i, col_idx+2].plot(years_ax, pred[:, 1]*OIL_MAX, clr, lw=1.5, ls=ls)
        axes3[i, col_idx+2].grid(True, alpha=.2)

    axes3[i, 0].set_ylabel(f'{c}\n({float(poly_ppm[c]):.0f} ppm)\nWater Cut')
    axes3[i, 2].set_ylabel('Oil Rate (bbl/d)')
    if i == 0:
        for ax, ttl in zip(axes3[0], ['NN — WC', 'PINN — WC',
                                        'NN — Oil', 'PINN — Oil']):
            ax.set_title(ttl, fontweight='bold', fontsize=10)
        axes3[0, 0].legend(prop={'size': 7}, loc='lower right')
        axes3[0, 1].legend(prop={'size': 7}, loc='lower right')

for ax in axes3[-1]:
    ax.set_xlabel('Time (Years)')
fig3.tight_layout()
plt.savefig('fig3_match.png', dpi=150, bbox_inches='tight'); plt.show()

# ── Fig 4: Learned fractional flow curves  (interpretability) ────────────────
fig4, ax4 = plt.subplots(figsize=(8, 5))
Sw_vals     = np.linspace(SWC, 1.-SOR, 200, dtype='float32')
Sw_norm_arr = tf.constant(((Sw_vals - SWC) / (1.-SWC-SOR)).reshape(-1, 1))
colors4     = plt.cm.plasma(np.linspace(0.1, 0.9, 7))
for col, cpi_v in zip(colors4, np.linspace(POLY_MIN, POLY_MAX, 7)):
    cpi_n  = (cpi_v - POLY_MIN) / (POLY_MAX - POLY_MIN)
    Cp_arr = tf.constant(np.full((200, 1), cpi_n, 'f4'))
    fw_v, _ = fractional_flow(Sw_norm_arr, Cp_arr)
    ax4.plot(Sw_vals, fw_v.numpy().flatten(), color=col, lw=2,
             label=f'{cpi_v:.0f} ppm')
ax4.set_xlabel('Water Saturation $S_w$', fontsize=11)
ax4.set_ylabel('Fractional Flow $f_w$',  fontsize=11)
ax4.set_title(
    'Learned Fractional Flow Curves — Cubic Polymer Viscosity Model\n'
    '$\\mu_w(C_p) = \\mu_{wi}(1 + rC_p + sC_p^2 + tC_p^3)$  '
    '[Liu et al. Eq. 1-3]\n'
    f'r={r_l:.2f}  s={s_l:.2f}  t={t_l:.2f}  '
    f'(higher polymer → lower $f_w$ → more oil)',
    fontweight='bold', fontsize=9)
ax4.legend(title='Polymer Conc.', fontsize=8)
ax4.grid(True, alpha=.3)
fig4.tight_layout()
plt.savefig('fig4_fw_curves.png', dpi=150, bbox_inches='tight'); plt.show()

# ── Fig 5: Spatial profiles Sw(x,t) and Cp(x,t)  (Liu et al. Fig. 4) ────────
# Shows the BL saturation shock front moving from injector (x=0) to producer
# (x=175m). This is ONLY possible with the 1-D spatial PINN — the 0-D model
# cannot produce this figure.
T_snaps  = [0.10, 0.25, 0.50, 0.75, 1.00]
colors5  = plt.cm.viridis(np.linspace(0.1, 0.9, len(T_snaps)))
n_show   = min(len(test_cases), 3)

fig5, axes5 = plt.subplots(2, n_show, figsize=(5*n_show, 8))
if n_show == 1:
    axes5 = axes5[:, np.newaxis]

for i, c in enumerate(test_cases[:n_show]):
    cpi_n = (float(poly_ppm[c]) - POLY_MIN) / (POLY_MAX - POLY_MIN)
    cpi_v = float(poly_ppm[c])
    Sw_grid, Cp_grid = pinn_field(x_norm_g, np.array(T_snaps, 'f4'), cpi_n)

    for j, (T_s, col) in enumerate(zip(T_snaps, colors5)):
        t_yr = T_s * T_SPAN * DT_DAYS / 365.25
        axes5[0, i].plot(x_phys, Sw_grid[j], color=col, lw=2,
                          label=f'{t_yr:.1f} yr')
        axes5[1, i].plot(x_phys, Cp_grid[j], color=col, lw=2,
                          label=f'{t_yr:.1f} yr')

    for row in range(2):
        axes5[row, i].axvline(0,      color='red',  ls=':', lw=1.5,
                               label='Injector (x=0 m)')
        axes5[row, i].axvline(L_INJE, color='blue', ls=':', lw=1.5,
                               label='Producer (x=175 m)')
        axes5[row, i].set_xlabel('Distance from Injector (m)')
        axes5[row, i].set_xlim([0, L_INJE])
        axes5[row, i].grid(True, alpha=.3)

    axes5[0, i].axhline(float(_SW_INIT_NORM), color='k', ls='--', lw=1,
                         label='$S_{wi}$ norm')
    axes5[0, i].set_ylim([0, 1.05])
    axes5[0, i].set_ylabel('$S_w$ (normalised)')
    axes5[0, i].set_title(f'{c} — {cpi_v:.0f} ppm', fontweight='bold')

    axes5[1, i].axhline(cpi_n, color='k', ls='--', lw=1,
                         label='$C_{pi}$ (inlet)')
    axes5[1, i].set_ylim([0, 1.05])
    axes5[1, i].set_ylabel('$C_p$ (normalised)')

axes5[0, 0].legend(fontsize=7, loc='lower right')
axes5[1, 0].legend(fontsize=7, loc='lower right')
fig5.suptitle(
    'Spatial profiles $S_w(x,t)$ and $C_p(x,t)$ — BL shock front propagation\n'
    'Injector (x=0 m) → Producer (x=175 m)  |  Liu et al. Fig. 4 analog\n'
    'Physics inferred from producer data only — spatial field from PDE constraints',
    fontweight='bold', fontsize=10)
fig5.tight_layout()
plt.savefig('fig5_spatial_profiles.png', dpi=150, bbox_inches='tight'); plt.show()

print("\n[DONE]  All 5 figures saved.")
print(f"\n[SUMMARY]")
print(f"  Spatial domain  : x ∈ [0, {L_INJE:.0f} m]  →  X ∈ [0, 1]")
print(f"  Injector BC     : Sw(0,T)=1-Sor  Cp(0,T)=Cpi")
print(f"  Producer data   : X=1  water cut + oil rate from CMG STARS")
print(f"  vp (exact)      : {VP_EXACT:.4f}  = q·T_ref/(A·φ·L)")
print(f"  BL equation     : ∂Sw/∂T + {VP_EXACT:.3f}·∂fw/∂X = 0")
print(f"  Polymer eq      : ∂(Sw·Cp)/∂T + {VP_EXACT:.3f}·∂(fw·Cp)/∂X = 0")
