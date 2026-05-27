"""
PINN for Polymer Flood Production Forecasting
==============================================
Architecture  : Meng et al. SPE-218863-MS (time embedding, BN, Dropout)
Physics loss  : Almajid & Abu-Alsaud SPE-203033-MS (BL residual via auto-diff)
Trainable phys: nw, no, krw_max, kro_max, polymer viscosity α, pore-vol scale
Optimizer     : Adam (with physics ramp-up) → L-BFGS  (Paper 2, Table 2)
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

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = r'C:\Users\faust\OneDrive\Desktop\PINN\CMG\TRAINING 2\Excel data'

# ==============================================================================
# 1.  PRIOR PHYSICAL CONSTANTS  (some become trainable in PINN)
# ==============================================================================
SWC      = 0.30    # connate water saturation
SOR      = 0.20    # residual oil saturation
NW_INIT  = 3.0     # Corey exponent – water  (Paper 2, Eq. 4)
NO_INIT  = 2.2     # Corey exponent – oil    (Paper 2, Eq. 5)
KRW_INIT = 0.10    # end-point krw           (Paper 2, Eq. 4)
KRO_INIT = 1.00    # end-point kro           (Paper 2, Eq. 5)
MU_O     = 1650.0  # oil viscosity [cp]
MU_W0    = 1.0     # pure-water viscosity [cp]

# Initial average water saturation (before polymer injection)
FWM      = 0.12    # fraction of movable water already present
SW_INIT  = SWC + FWM * (1.0 - SWC - SOR)

# ==============================================================================
# 2.  DATA PIPELINE
# ==============================================================================
def load_wide(path):
    """Load a wide CSV (rows = timesteps, cols = cases); fix column names."""
    df = pd.read_csv(path, parse_dates=['time']).set_index('time').sort_index()
    df.columns = (df.columns
                  .str.replace('-', '_')
                  .str.replace(r'case_0*(\d+)', r'case_\1', regex=True))
    return df

print("[DATA] Loading CSVs …")
wc_df = load_wide(os.path.join(DATA_DIR, 'Water cut.csv'))
op_df = load_wide(os.path.join(DATA_DIR, 'Oil Production.csv'))
cp_df = load_wide(os.path.join(DATA_DIR, 'Polymer concentration.csv'))  # ppm

idx   = wc_df.index.intersection(op_df.index).intersection(cp_df.index)
wc_df, op_df, cp_df = wc_df.loc[idx], op_df.loc[idx], cp_df.loc[idx]

cases   = [c for c in wc_df.columns if c in op_df.columns and c in cp_df.columns]
N_T     = len(idx)
T_SPAN  = float(N_T - 1)                                     # last step index
DT_DAYS = float((idx[1] - idx[0]).days) if N_T > 1 else 13.33
OIL_MAX = float(op_df[cases].values.max())

# Normalized time axis common to all cases
t_norm = np.arange(N_T, dtype=np.float32) / T_SPAN           # [0, 1]

# Polymer concentration per case (ppm) — constant within each case
poly_ppm  = cp_df.iloc[0]                                     # Series: case → ppm
POLY_MIN  = float(poly_ppm.min())                             # 13.33 ppm
POLY_MAX  = float(poly_ppm.max())                             # 706.67 ppm

def build_dataset(case_list):
    """
    Returns
        X : (n_cases × N_T, 2)  columns = [t_norm, cp_norm]
        Y : (n_cases × N_T, 2)  columns = [water_cut, oil_rate_norm]
    """
    X_parts, Y_parts = [], []
    for c in case_list:
        pc_norm = (float(poly_ppm[c]) - POLY_MIN) / (POLY_MAX - POLY_MIN)
        wc_     = np.clip(wc_df[c].values[:N_T], 0.0, 1.0).astype('f4')
        op_     = np.clip(op_df[c].values[:N_T], 0.0, None).astype('f4') / OIL_MAX
        X_parts.append(np.column_stack([t_norm,
                                        np.full(N_T, pc_norm, 'f4')]))
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
print(f"        X shape={X_tr.shape}  Y shape={Y_tr.shape}")

# ==============================================================================
# 3.  TRAINABLE PHYSICAL PARAMETERS  (Paper 1 Table 1, Paper 2 Case 3)
#     Stored in log-space so they stay positive during gradient updates.
# ==============================================================================
log_nw      = tf.Variable(np.log(NW_INIT),  dtype=tf.float32, name='log_nw')
log_no      = tf.Variable(np.log(NO_INIT),  dtype=tf.float32, name='log_no')
log_krw     = tf.Variable(np.log(KRW_INIT), dtype=tf.float32, name='log_krw')
log_kro     = tf.Variable(np.log(KRO_INIT), dtype=tf.float32, name='log_kro')
# Polymer viscosity model: mu_w(cp_norm) = MU_W0 * exp(alpha * cp_norm)
# alpha = 0 → pure water; alpha > 0 → polymer thickens water
log_alpha   = tf.Variable(0.5, dtype=tf.float32, name='log_alpha')
# Pore-volume scale = total dimensionless pore volumes injected over simulation
# dSw_norm/dt_norm = vp_scale * (1 - fw) / (1 - SWC - SOR)
log_vp      = tf.Variable(0.0, dtype=tf.float32, name='log_vp')
# Oil-rate scale: qo_norm = q_scale * (1 - fw)
log_qscale  = tf.Variable(0.0, dtype=tf.float32, name='log_qscale')

phys_vars = [log_nw, log_no, log_krw, log_kro, log_alpha, log_vp, log_qscale]

def get_phys():
    return (tf.exp(log_nw),  tf.exp(log_no),
            tf.exp(log_krw), tf.exp(log_kro),
            tf.exp(log_alpha), tf.exp(log_vp), tf.exp(log_qscale))

# ==============================================================================
# 4.  FRACTIONAL FLOW FROM PREDICTED Sw  (Corey + polymer viscosity)
#     Following Paper 2, Eq. 3-5  and  Paper 1, Eq. 6-8
# ==============================================================================
_SWC  = tf.constant(SWC,           dtype=tf.float32)
_SOR  = tf.constant(SOR,           dtype=tf.float32)
_DENOM= tf.constant(1.0-SWC-SOR,   dtype=tf.float32)
_MUO  = tf.constant(MU_O,          dtype=tf.float32)
_MUW0 = tf.constant(MU_W0,         dtype=tf.float32)

def fractional_flow(Sw_norm, cp_norm):
    """
    Compute fw and physical Sw from the network's normalized Sw output.

    Sw_norm ∈ [0,1]  →  Sw = Swc + Sw_norm*(1-Swc-Sor)
    Polymer model    :  mu_w = MU_W0 * exp(alpha * cp_norm)   [cp]
    """
    nw, no, krw, kro, alpha, vp, qsc = get_phys()

    Sw    = _SWC + Sw_norm * _DENOM                              # physical Sw
    Se    = tf.clip_by_value((Sw - _SWC) / (_DENOM + 1e-8),
                             0.0, 1.0)                           # effective sat.

    krw_f = krw * tf.pow(Se + 1e-8, nw)                         # Paper 2 Eq.4
    kro_f = kro * tf.pow(1.0 - Se + 1e-8, no)                   # Paper 2 Eq.5
    mu_w  = _MUW0 * tf.exp(alpha * cp_norm)                     # polymer visc.

    lam_w = krw_f / (mu_w + 1e-8)
    lam_o = kro_f / (_MUO + 1e-8)
    fw    = lam_w / (lam_w + lam_o + 1e-8)                      # Paper 2 Eq.3

    return fw, Sw

# ==============================================================================
# 5.  NETWORK ARCHITECTURES
# ==============================================================================
def build_pinn_network():
    """
    Paper 1 Fig. 5: embed time → concat control input → 2×(FC+BN+Tanh+Dropout).
    Output: Sw_norm ∈ [0,1]  (intermediate physical variable, not directly fw)
    """
    t_inp  = keras.Input(shape=(1,), name='time')
    cp_inp = keras.Input(shape=(1,), name='cp_norm')

    # Time embedding (Paper 1: initial FC on time before concat)
    t_emb = keras.layers.Dense(64, kernel_initializer='glorot_normal')(t_inp)
    t_emb = keras.layers.Activation('tanh')(t_emb)

    x = keras.layers.Concatenate()([t_emb, cp_inp])

    # Two identical blocks: FC + BatchNorm + Tanh + Dropout  (Paper 1 Fig. 5)
    for units in [128, 128, 64]:
        x = keras.layers.Dense(units, kernel_initializer='glorot_normal')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('tanh')(x)
        x = keras.layers.Dropout(0.1)(x)

    # Sigmoid output → Sw_norm ∈ [0,1]
    sw_out = keras.layers.Dense(1, activation='sigmoid', name='Sw_norm')(x)
    return keras.Model(inputs=[t_inp, cp_inp], outputs=sw_out, name='PINN')


def build_nn_network():
    """Pure data-driven NN — same depth/width as PINN for fair comparison."""
    inp = keras.Input(shape=(2,))
    x   = keras.layers.Dense(64,  kernel_initializer='glorot_normal')(inp)
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
# 6.  LOSS FUNCTIONS  (Paper 1 Eq. 9–11;  Paper 2 Eq. 9–10)
# ==============================================================================
SW_INIT_NORM = tf.constant((SW_INIT - SWC) / (1.0 - SWC - SOR), dtype=tf.float32)

def pinn_losses(model, x_batch, y_batch, training=True):
    """
    Three-component loss (Paper 1 Eq. 11):
        L = L_data  +  λ_phys * L_phys  +  λ_ic * L_ic

    L_data  – MSE between physics-computed fw and observed water cut,
              plus MSE between physics-computed qo_norm and observed oil rate.
              (Paper 1 Eq. 9)

    L_phys  – Residual of 0-D Buckley-Leverett ODE (auto-diff, Paper 2 Eq. 9):
                  dSw_norm/dt_norm = vp_scale*(1-fw)/(1-Swc-Sor)
              This enforces conservation of water in the lumped reservoir volume.

    L_ic    – Initial condition: Sw(t=0) = SW_INIT  (Paper 2 Eq. 6)
    """
    # Convert whole batch to tensor first; slicing a tensor yields a tensor,
    # which tape.watch() requires (slicing a numpy array yields another ndarray).
    x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
    t_in    = x_batch[:, 0:1]
    cp_in   = x_batch[:, 1:2]
    _, _, _, _, _, vp, qsc = get_phys()

    # Auto-diff: track t so we can compute dSw/dt  (Paper 2 methodology)
    with tf.GradientTape() as tape:
        tape.watch(t_in)
        Sw_norm = model([t_in, cp_in], training=training)
        fw, _   = fractional_flow(Sw_norm, cp_in)

    dSw_norm_dt = tape.gradient(Sw_norm, t_in)          # d(Sw_norm)/d(t_norm)

    # ── Physics residual  (Paper 2 Eq. 9 / Paper 1 Eq. 10) ──────────────────
    # 0-D water balance: phi*Vp * dSw/dt = Q_inj*(1-fw)
    # In normalized time:
    #   dSw_norm/dt_norm = vp_scale * (1-fw) / (1-Swc-Sor)
    phys_rhs      = vp * (1.0 - fw) / (_DENOM + 1e-8)
    L_phys        = tf.reduce_mean(tf.square(dSw_norm_dt - phys_rhs))

    # ── Initial condition loss  (Paper 2 Eq. 6) ──────────────────────────────
    ic_mask = tf.cast(t_in < (2.0 / T_SPAN), tf.float32)
    L_ic    = tf.reduce_mean(tf.square((Sw_norm - SW_INIT_NORM) * ic_mask))

    # ── Data loss  (Paper 1 Eq. 9) ───────────────────────────────────────────
    wc_obs  = y_batch[:, 0:1]
    oil_obs = y_batch[:, 1:2]
    # Oil rate: qo ∝ (1-fw) * q_total  →  normalized by qsc
    qo_pred = qsc * (1.0 - fw)
    L_data  = (tf.reduce_mean(tf.square(fw     - wc_obs)) +
               tf.reduce_mean(tf.square(qo_pred - oil_obs)))

    return L_data, L_phys, L_ic


def nn_loss(model, x_batch, y_batch, training=True):
    yp = model(x_batch, training=training)
    return tf.reduce_mean(tf.square(yp - y_batch))

# ==============================================================================
# 7.  FLAT-WEIGHT HELPERS  (needed for correct L-BFGS gradient)
# ==============================================================================
def flat_weights(variables):
    return tf.concat([tf.reshape(v, [-1]) for v in variables], axis=0)

def assign_flat(flat_w, variables):
    idx = 0
    for v in variables:
        n = v.shape.num_elements()
        v.assign(tf.reshape(flat_w[idx:idx + n], v.shape))
        idx += n

# ==============================================================================
# 8.  TRAINING
# ==============================================================================
EPOCHS_ADAM = 500          # Paper 1: 100 000; reduced for demonstration
BATCH_SIZE  = 2048
LR          = 0.001        # Paper 1: 0.005; Paper 2: 0.001
LAMBDA_PHYS = 0.5          # Paper 1: λ=0.5 (3-D case)
LAMBDA_IC   = 10.0         # strong IC enforcement
LBFGS_ITER  = 500          # Paper 2: switches to L-BFGS after Adam


def train_pinn(model, tag='PINN'):
    all_vars = model.trainable_variables + phys_vars
    opt = keras.optimizers.Adam(learning_rate=LR, weight_decay=1e-4)
    ds  = (tf.data.Dataset.from_tensor_slices((X_tr, Y_tr))
           .shuffle(80000).batch(BATCH_SIZE))

    tr_data_h, tr_phys_h, va_h = [], [], []
    best_val, best_w = np.inf, None

    print(f"\n[Stage 1 – Adam]  {tag}  ({EPOCHS_ADAM} epochs) …")
    for ep in range(1, EPOCHS_ADAM + 1):
        # Physics weight ramp-up: avoids early gradient conflict (Paper 1 strategy)
        lam = LAMBDA_PHYS * min(1.0, ep / 200.0)

        ep_data, ep_phys = [], []
        for xb, yb in ds:
            with tf.GradientTape() as tape:
                Ld, Lp, Lic = pinn_losses(model, xb, yb, training=True)
                total = Ld + lam * Lp + LAMBDA_IC * Lic
            grads = tape.gradient(total, all_vars)
            grads = [tf.clip_by_norm(g, 1.0) if g is not None else g
                     for g in grads]
            opt.apply_gradients(zip(grads, all_vars))
            ep_data.append(float(Ld)); ep_phys.append(float(Lp))

        # Validation (data loss only, same as Paper 1 Table 3)
        Ld_va, _, _ = pinn_losses(model, X_va, Y_va, training=False)
        va_loss = float(Ld_va)

        tr_data_h.append(np.mean(ep_data))
        tr_phys_h.append(np.mean(ep_phys))
        va_h.append(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            best_w   = model.get_weights()

        if ep % 100 == 0 or ep == 1:
            nw_v, no_v, _, _, a_v, vp_v, qs_v = [float(tf.exp(p)) for p in phys_vars]
            print(f"  Ep {ep:4d} | L_data={tr_data_h[-1]:.5f} "
                  f"| L_phys={tr_phys_h[-1]:.5f} | Val={va_loss:.5f} "
                  f"| nw={nw_v:.2f} no={no_v:.2f} α={a_v:.3f} vp={vp_v:.3f}")

    model.set_weights(best_w)

    # ── Stage 2: L-BFGS  (Paper 2, Table 2) ─────────────────────────────────
    print(f"\n[Stage 2 – L-BFGS]  {tag}  (max {LBFGS_ITER} iters) …")

    def lbfgs_obj(flat_w):
        assign_flat(flat_w, all_vars)
        with tf.GradientTape() as tape:
            Ld, Lp, Lic = pinn_losses(model, X_tr, Y_tr, training=False)
            total = Ld + LAMBDA_PHYS * Lp + LAMBDA_IC * Lic
        # Gradient w.r.t. each trainable variable → flatten into one vector
        grads     = tape.gradient(total, all_vars)
        flat_grad = tf.concat(
            [tf.reshape(g if g is not None else tf.zeros_like(v), [-1])
             for g, v in zip(grads, all_vars)], axis=0)
        return total, flat_grad

    try:
        res = tfp.optimizer.lbfgs_minimize(
            lbfgs_obj,
            initial_position=flat_weights(all_vars),
            max_iterations=LBFGS_ITER,
            num_correction_pairs=50,
            tolerance=1e-8)
        assign_flat(res.position, all_vars)
        print(f"  Converged: {res.converged.numpy()} | "
              f"Iterations: {res.num_iterations.numpy()}")
    except Exception as exc:
        print(f"  L-BFGS error: {exc}")

    return np.array(tr_data_h), np.array(tr_phys_h), np.array(va_h)


def train_nn(model, tag='Pure NN'):
    opt = keras.optimizers.Adam(learning_rate=LR, weight_decay=1e-4)
    ds  = (tf.data.Dataset.from_tensor_slices((X_tr, Y_tr))
           .shuffle(80000).batch(BATCH_SIZE))

    tr_h, va_h = [], []
    best_val, best_w = np.inf, None

    print(f"\n[Stage 1 – Adam]  {tag}  ({EPOCHS_ADAM} epochs) …")
    for ep in range(1, EPOCHS_ADAM + 1):
        ep_losses = []
        for xb, yb in ds:
            with tf.GradientTape() as tape:
                loss = nn_loss(model, xb, yb, training=True)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            ep_losses.append(float(loss))

        va_loss = float(nn_loss(model, X_va, Y_va, training=False))
        tr_h.append(np.mean(ep_losses)); va_h.append(va_loss)

        if va_loss < best_val:
            best_val = va_loss; best_w = model.get_weights()

        if ep % 100 == 0 or ep == 1:
            print(f"  Ep {ep:4d} | Train={tr_h[-1]:.5f} | Val={va_loss:.5f}")

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
            num_correction_pairs=50,
            tolerance=1e-8)
        assign_flat(res.position, model.trainable_variables)
        print(f"  Converged: {res.converged.numpy()}")
    except Exception as exc:
        print(f"  L-BFGS error: {exc}")

    return np.array(tr_h), np.array(va_h)

# ==============================================================================
# 9.  BUILD & RUN
# ==============================================================================
print('\n[BUILD] Initializing models …')
pinn_model = build_pinn_network()
nn_model   = build_nn_network()

pinn_model.summary()

pn_data_h, pn_phys_h, pn_va_h = train_pinn(pinn_model, 'PINN')
nn_tr_h,   nn_va_h             = train_nn(nn_model,    'Pure NN')

# Report learned physical parameters
nw_l, no_l, krw_l, kro_l, alpha_l, vp_l, qs_l = [
    float(tf.exp(p)) for p in phys_vars]
print("\n[PHYSICS] Learned parameters after training:")
print(f"  Corey nw      = {nw_l:.4f}   (prior: {NW_INIT})")
print(f"  Corey no      = {no_l:.4f}   (prior: {NO_INIT})")
print(f"  krw_max       = {krw_l:.4f}   (prior: {KRW_INIT})")
print(f"  kro_max       = {kro_l:.4f}   (prior: {KRO_INIT})")
print(f"  Polymer α     = {alpha_l:.4f}  (mu_w = {MU_W0}·exp(α·cp_norm))")
print(f"  VP scale      = {vp_l:.4f}   (dimensionless PV injected)")
print(f"  Oil q-scale   = {qs_l:.4f}")

# ==============================================================================
# 10.  PREDICTION HELPER
# ==============================================================================
def pinn_predict(X):
    """Return (water_cut, oil_rate_norm) using physics equations."""
    t_in  = tf.constant(X[:, 0:1])
    cp_in = tf.constant(X[:, 1:2])
    Sw_norm = pinn_model([t_in, cp_in], training=False)
    fw, _   = fractional_flow(Sw_norm, cp_in)
    _, _, _, _, _, _, qsc = get_phys()
    qo_norm = qsc * (1.0 - fw)
    return np.concatenate([fw.numpy(), qo_norm.numpy()], axis=1)

# ==============================================================================
# 11.  METRICS  (Paper 1 Table 3 format)
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

# Compute train/val losses for Table 3 equivalent
Ld_tr_pi, Lp_tr_pi, _ = pinn_losses(pinn_model, X_tr, Y_tr, training=False)
Ld_va_pi, Lp_va_pi, _ = pinn_losses(pinn_model, X_va, Y_va, training=False)
Ld_te_pi, Lp_te_pi, _ = pinn_losses(pinn_model, X_te, Y_te, training=False)
Ld_tr_nn = float(nn_loss(nn_model, X_tr, Y_tr, training=False))
Ld_va_nn = float(nn_loss(nn_model, X_va, Y_va, training=False))
Ld_te_nn = float(nn_loss(nn_model, X_te, Y_te, training=False))

print("\n" + "="*80)
print("           STATISTICAL EVALUATION METRICS  (Paper 1 Table 3 format)")
print("="*80)
print(f"{'Metric':<18} {'Pure NN':>16} {'Proposed PINN':>16}")
print("-"*80)
print(f"{'Train L_D':<18} {Ld_tr_nn:>16.6f} {float(Ld_tr_pi):>16.6f}")
print(f"{'Test  L_D':<18} {Ld_te_nn:>16.6f} {float(Ld_te_pi):>16.6f}")
print(f"{'Val   L_D':<18} {Ld_va_nn:>16.6f} {float(Ld_va_pi):>16.6f}")
print(f"{'Train L_P':<18} {'—':>16} {float(Lp_tr_pi):>16.6f}")
print(f"{'Test  L_P':<18} {'—':>16} {float(Lp_te_pi):>16.6f}")
print(f"{'Val   L_P':<18} {'—':>16} {float(Lp_va_pi):>16.6f}")
print("="*80)
print(f"\n{'Metric':<14} {'NN Water Cut':>14} {'PINN Water Cut':>16} "
      f"{'NN Oil Rate':>13} {'PINN Oil Rate':>15}")
print("-"*80)
print(f"{'R²':<14} {r2_wc_nn:>14.4f} {r2_wc_pi:>16.4f} "
      f"{r2_op_nn:>13.4f} {r2_op_pi:>15.4f}")
print(f"{'RMSE':<14} {rmse_wc_nn:>14.6f} {rmse_wc_pi:>16.6f} "
      f"{rmse_op_nn:>13.6f} {rmse_op_pi:>15.6f}")
print(f"{'MAPE (%)':<14} {mape_wc_nn:>14.2f} {mape_wc_pi:>16.2f} "
      f"{mape_op_nn:>13.2f} {mape_op_pi:>15.2f}")
print("="*80)

# ==============================================================================
# 12.  FIGURES
# ==============================================================================
years_ax = t_norm * T_SPAN * DT_DAYS / 365.25

# ── Fig 1: Training & validation losses  (Paper 1 Fig. 7) ────────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5))

axes1[0].semilogy(nn_tr_h,  color='#2196F3', lw=1.5, label='Train L_D')
axes1[0].semilogy(nn_va_h,  color='#e74c3c',  lw=1.5, label='Val L_D', alpha=0.85)
axes1[0].set_title('(a) Pure Data-driven NN', fontweight='bold')

axes1[1].semilogy(pn_data_h, color='#2196F3', lw=1.5, label='Train L_D (data)')
axes1[1].semilogy(pn_va_h,   color='#e74c3c',  lw=1.5, label='Val L_D',   alpha=0.85)
axes1[1].semilogy(pn_phys_h, color='#4CAF50',  lw=1.5, label='Train L_P (physics)', ls='--')
axes1[1].set_title('(b) Proposed PINN', fontweight='bold')

for ax in axes1:
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (MSE, log scale)')
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)
fig1.suptitle('Training and validation losses of the adopted methods.',
              fontweight='bold')
fig1.tight_layout()
plt.savefig('fig1_losses.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Fig 2: Cumulative oil crossplot  (Paper 1 Fig. 10) ───────────────────────
def cumulative_oil(case_list, pred_fn=None):
    out = []
    for c in case_list:
        pc_n = (float(poly_ppm[c]) - POLY_MIN) / (POLY_MAX - POLY_MIN)
        if pred_fn is None:
            oil = np.clip(op_df[c].values[:N_T], 0, None)
        else:
            Xc  = np.column_stack([t_norm,
                                    np.full(N_T, pc_n, 'f4')]).astype('f4')
            oil = pred_fn(Xc)[:, 1] * OIL_MAX
        out.append(np.sum(oil) * DT_DAYS)
    return np.array(out)

act_tr = cumulative_oil(train_cases)
act_va = cumulative_oil(val_cases)
act_te = cumulative_oil(test_cases)

nn_fn   = lambda X: nn_model(X, training=False).numpy()
pinn_fn = pinn_predict

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
for ax, fn, sub, title in [
    (axes2[0], nn_fn,   '(a)', 'Standard Data-driven NN'),
    (axes2[1], pinn_fn, '(b)', 'Proposed PINN'),
]:
    ptr = cumulative_oil(train_cases, fn)
    pva = cumulative_oil(val_cases,   fn)
    pte = cumulative_oil(test_cases,  fn)

    ax.scatter(act_tr, ptr, c='#2196F3', s=45, alpha=0.85, label='Train',      zorder=4)
    ax.scatter(act_va, pva, c='#4CAF50', s=45, alpha=0.85, label='Validation', zorder=4, marker='s')
    ax.scatter(act_te, pte, c='#FF9800', s=70, alpha=0.95, label='Test',       zorder=5, marker='^')

    all_a = np.concatenate([act_tr, act_va, act_te])
    all_p = np.concatenate([ptr, pva, pte])
    lo = min(all_a.min(), all_p.min()) * 0.97
    hi = max(all_a.max(), all_p.max()) * 1.03
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='Ideal 1:1', zorder=3)
    ax.set_xlim([lo, hi]); ax.set_ylim([lo, hi])

    r2_tr = r2_score(act_tr, ptr)
    r2_va = r2_score(act_va, pva)
    r2_te = r2_score(act_te, pte)
    ax.text(0.04, 0.96,
            f'R²  {r2_tr:.3f}  Train\n    {r2_va:.3f}  Validation\n    {r2_te:.3f}  Test',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', fc='white', alpha=0.85))
    ax.set_xlabel('Actual Cumulative Oil (bbl)')
    ax.set_ylabel('Predicted Cumulative Oil (bbl)')
    ax.set_title(f'{sub} {title}', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

fig2.suptitle('Predicted vs. Actual Cumulative Oil Production', fontweight='bold')
fig2.tight_layout()
plt.savefig('fig2_crossplot.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Fig 3: Production match for test cases  (Paper 1 Fig. 9) ─────────────────
fig3, axes3 = plt.subplots(len(test_cases), 4,
                            figsize=(18, 3.2 * len(test_cases)), sharex=True)
col_titles = ['NN — Water Cut', 'PINN — Water Cut',
              'NN — Oil Rate (bbl/d)', 'PINN — Oil Rate (bbl/d)']

for i, c in enumerate(test_cases):
    pc_n = (float(poly_ppm[c]) - POLY_MIN) / (POLY_MAX - POLY_MIN)
    Xc   = np.column_stack([t_norm, np.full(N_T, pc_n, 'f4')]).astype('f4')

    nn_p   = nn_model(Xc, training=False).numpy()
    pinn_p = pinn_predict(Xc)
    true_wc = wc_df[c].values[:N_T]
    true_op = op_df[c].values[:N_T]
    cp_ppm_val = float(poly_ppm[c])

    # Water cut — NN
    axes3[i,0].plot(years_ax, true_wc,   'k',       lw=2,   label='CMG STARS')
    axes3[i,0].plot(years_ax, nn_p[:,0], '#e74c3c', lw=1.5, ls='--', label='NN')
    axes3[i,0].set_ylim([0, 1])
    axes3[i,0].set_ylabel(f'{c}\n({cp_ppm_val:.0f} ppm)\nWater Cut')
    axes3[i,0].grid(True, alpha=0.2)

    # Water cut — PINN
    axes3[i,1].plot(years_ax, true_wc,    'k',       lw=2)
    axes3[i,1].plot(years_ax, pinn_p[:,0],'#2980b9', lw=1.5, label='PINN')
    axes3[i,1].set_ylim([0, 1])
    axes3[i,1].grid(True, alpha=0.2)

    # Oil rate — NN
    axes3[i,2].plot(years_ax, true_op,               'k',       lw=2)
    axes3[i,2].plot(years_ax, nn_p[:,1]   * OIL_MAX, '#e74c3c', lw=1.5, ls='--')
    axes3[i,2].set_ylabel('Oil Rate (bbl/d)')
    axes3[i,2].grid(True, alpha=0.2)

    # Oil rate — PINN
    axes3[i,3].plot(years_ax, true_op,               'k',       lw=2)
    axes3[i,3].plot(years_ax, pinn_p[:,1] * OIL_MAX, '#2980b9', lw=1.5)
    axes3[i,3].grid(True, alpha=0.2)

    if i == 0:
        for ax, ttl in zip(axes3[0], col_titles):
            ax.set_title(ttl, fontweight='bold', fontsize=10)
        axes3[0,0].legend(prop={'size': 7}, loc='lower right')
        axes3[0,1].legend(prop={'size': 7}, loc='lower right')

for ax in axes3[-1]:
    ax.set_xlabel('Time (Years)')

fig3.tight_layout()
plt.savefig('fig3_production_match.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Fig 4: Fractional flow curves — physical interpretability  (Paper 1 Fig. 8 analog)
fig4, ax4 = plt.subplots(figsize=(8, 5))

Sw_vals     = np.linspace(SWC, 1.0 - SOR, 200).astype('f4')
Sw_norm_arr = tf.constant(((Sw_vals - SWC) / (1.0 - SWC - SOR)).reshape(-1, 1))
cp_ppm_plot = np.linspace(POLY_MIN, POLY_MAX, 7)
colors      = plt.cm.plasma(np.linspace(0.1, 0.9, len(cp_ppm_plot)))

for col, cp_val in zip(colors, cp_ppm_plot):
    cp_n_arr = tf.constant(
        np.full((200, 1), (cp_val - POLY_MIN) / (POLY_MAX - POLY_MIN), 'f4'))
    fw_vals, _ = fractional_flow(Sw_norm_arr, cp_n_arr)
    ax4.plot(Sw_vals, fw_vals.numpy().flatten(),
             color=col, lw=2, label=f'{cp_val:.0f} ppm')

ax4.set_xlabel('Water Saturation  $S_w$', fontsize=11)
ax4.set_ylabel('Fractional Flow  $f_w$',  fontsize=11)
ax4.set_title('Learned Fractional Flow Curves at Different Polymer Concentrations\n'
              '(Physical Interpretability — higher polymer → lower fw → more oil)',
              fontweight='bold', fontsize=10)
ax4.legend(title='Polymer Conc.', fontsize=8, loc='upper left')
ax4.grid(True, alpha=0.3)
fig4.tight_layout()
plt.savefig('fig4_fw_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n[DONE]  All figures saved.")
