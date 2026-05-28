"""
pinn_polymer_datafree.py
========================
Data-Free 1D Polymer Flood PINN — Pelican Lake HP-6

Follows EXACTLY the methodology of:
  Liu et al. (2025) Phys. Fluids 37, 036622
  "Physics-informed neural network (PINNs) for convection equations
   in polymer flooding reservoirs"

Governing equations (Liu et al. Eqs. 2, 4, 22):
  ∂Sw/∂t  +  (q / φA) · ∂fw/∂x  =  0                   (Eq. 2 — BL)
  ∂(Sw·Cp)/∂t  +  (q / φA) · ∂(fw·Cp)/∂x  =  0         (Eq. 4 — polymer)
  → simplified (Eq. 22): Sw·∂Cp/∂t  +  fw·∂Cp/∂x  =  0

Fractional flow (Eq. 3):
  fw(Sw, Cp) = [krw(Sw)/μw(Cp)] / [krw(Sw)/μw(Cp)  +  kro(Sw)/μo]
             = krw(Sw) / [krw(Sw)  +  M · kro(Sw)]
  where  M = μw(Cp)/μo

Polymer viscosity (Eq. 1):
  μw(Cp) = μwi · (1 + r·Cp + s·Cp² + t·Cp³)

Coordinate transform (so coefficient in normalized PDEs = 1):
  X = x / L,   T = t / t_ref   with  t_ref = φ·A·L / q
  → ∂Sw/∂T + ∂fw/∂X  =  0  (coefficient exactly 1)
  → Sw·∂Cp/∂T + fw·∂Cp/∂X  =  0

For non-convex fw (this case uses cubic μw → Example 3 regime of Liu et al.),
artificial viscosity is added (Liu et al. Fig. 7–8):
  ∂Sw/∂T + ∂fw/∂X  −  ε·∂²Sw/∂X²  =  0
  Sw·∂Cp/∂T + fw·∂Cp/∂X  −  ε·∂²Cp/∂X²  =  0

Loss function (Liu et al. Eqs. 8–10, ω3=0 → data-free):
  L(θ) = ω1·LSw + ω2·LCp
  LSw = ω11·L_pde_Sw + ω12·L_ic_Sw + ω13·L_bc_Sw
  LCp = ω21·L_pde_Cp + ω22·L_ic_Cp + ω23·L_bc_Cp

Network: PINN-1 (Liu et al. Fig. 2a) — single MLP (x,t)→(Sw,Cp)

IC (T=0, all X):   Sw = Swi = 0.36,   Cp = 0
BC (X=0, all T):   Sw = 1−Sor,        Cp = Cpi  (injection concentration)

No labeled production data used.
"""

import os, time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy.stats import qmc

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  RESERVOIR & FLUID PARAMETERS  (from paper — all fixed)
# ═══════════════════════════════════════════════════════════════════════════════
# Water saturation limits
Swi   = 0.30    # irreducible water saturation  (Delaplace 2013 — 1D BL model)
Sor   = 0.20    # residual oil saturation        (paper Sec. 2.3)

# Corey relative permeabilities (Liu et al. use piecewise-linear; we use Corey
# with exponents from Manuscript_02 Sec. 2.3 — physically equivalent form)
nw        = 3.0   # Corey exponent water
no        = 2.2   # Corey exponent oil
krw_max   = 0.10  # endpoint krw  (fixed, paper Sec. 2.3)
kro_max   = 1.00  # endpoint kro  (fixed, paper Sec. 2.3)

# Fluid viscosities
mu_o  = 1650.0    # oil viscosity [cp]  (paper Table 2)
mu_wi = 1.0       # pure-water viscosity [cp]

# Polymer viscosity coefficients (Liu et al. Eq. 1):  μw = μwi(1+r·Cp+s·Cp²+t·Cp³)
r_vis = 3.0
s_vis = 2.0
t_vis = 1.0

# Initial water saturation at T=0 (from paper Sec. 2.5, FWM=0.12)
Swi_mobile = 0.12 * (1.0 - Swi - Sor)   # FWM × (1-Swi-Sor)
Sw_init    = Swi + Swi_mobile             # = 0.36

# Injection polymer concentration (normalisation reference)
Cpi_phys = 0.5    # physical injection conc [kg/m³]

# ─── Pelican Lake HP-6 geometry  (for coordinate transform t_ref) ───────────
q   = 95.0                          # injection rate [m³/day]
phi = 0.312                         # porosity
H   = 14.434692 * 0.3048            # net pay [m]  (14.43 ft)
Lw  = 1400.0                        # lateral well length [m]
A   = H * Lw                        # cross-sectional area [m²]
L   = 175.0                         # injector–producer distance [m]

# Time reference: t_ref = φ·A·L / q  (Liu et al. Example 1 approach)
# ensures the coefficient  (q/φA)·(t_ref/L) = 1  in the normalised PDEs
t_ref = phi * A * L / q            # ≈ 3540 days  (1 pore-volume injection time)

# ─── Saturation normalisation (network output ∈ [0,1]) ─────────────────────
#   Sw_norm = (Sw − Swi) / (1 − Swi − Sor)
Sw_mobile_range = 1.0 - Swi - Sor          # = 0.50
Sw_norm_ic  = (Sw_init - Swi) / Sw_mobile_range   # ≈ 0.12
Sw_norm_bc  = (1.0 - Sor  - Swi) / Sw_mobile_range  # = 1.00
Cp_norm_bc  = 1.0   # Cp / Cpi_phys at X=0

# ─── Artificial viscosity (Liu et al. Example 3, Fig. 7–8) ──────────────────
epsilon = 1e-3    # ε=1e-3 gives best accuracy for non-convex fw (Table V)

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  TRAINING SETTINGS  (Liu et al. Table IV — Example 3 regime)
# ═══════════════════════════════════════════════════════════════════════════════
N_m   = 10_000   # number of interior collocation points (N_m in paper)
N_IC  =  1_000   # initial-condition sample points
N_BC  =  1_000   # boundary-condition sample points

N_iter = 20_000  # total Adam iterations  (Liu et al. Table IV)
lr_0   = 1e-3    # initial learning rate  (Liu et al. Table I)
lr_1   = 1e-4    # reduced LR at halfway  (matches Table IV approach)

# Loss weights (Liu et al. Eqs. 8–10: ω1,ω2,ω11…ω23)
omega1,  omega2  = 1.0, 1.0
omega11, omega12, omega13 = 1.0, 10.0, 10.0   # Sw: PDE | IC | BC
omega21, omega22, omega23 = 1.0, 10.0, 10.0   # Cp: PDE | IC | BC

SEED = 42
tf.random.set_seed(SEED);  np.random.seed(SEED)

# ─── Print summary ────────────────────────────────────────────────────────────
print("=" * 60)
print("Data-Free PINN  |  Pelican Lake HP-6  |  Liu et al. 2025")
print("=" * 60)
print(f"  q = {q} m³/day,  φ = {phi},  A = {A:.1f} m²,  L = {L} m")
print(f"  t_ref = φ·A·L/q = {t_ref:.1f} days  ({t_ref/365:.2f} yr)")
print(f"  (q/φA)·(t_ref/L) = {(q/(phi*A))*(t_ref/L):.4f}  (= 1 by construction)")
print(f"  Swi = {Swi},  Sor = {Sor},  Sw_init = {Sw_init:.4f}")
print(f"  Sw_norm  IC={Sw_norm_ic:.4f}  BC={Sw_norm_bc:.4f}")
print(f"  epsilon  = {epsilon}  (artificial viscosity)")
print(f"  N_m={N_m}, N_IC={N_IC}, N_BC={N_BC},  N_iter={N_iter}")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  TF CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
def _c(v): return tf.constant(float(v), dtype=tf.float32)

_Swi  = _c(Swi);   _Sr = _c(Sor);   _Smob = _c(Sw_mobile_range)
_krw  = _c(krw_max); _kro = _c(kro_max)
_nw   = _c(nw);    _no  = _c(no)
_muo  = _c(mu_o);  _muwi = _c(mu_wi)
_r    = _c(r_vis); _s   = _c(s_vis); _t = _c(t_vis)
_eps  = _c(epsilon)
_Sw0  = _c(Sw_norm_ic)   # IC target
_SwBC = _c(Sw_norm_bc)   # BC target
_CpBC = _c(Cp_norm_bc)   # BC target

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  FRACTIONAL FLOW   fw(Sw, Cp)  — Liu et al. Eq. 3
# ═══════════════════════════════════════════════════════════════════════════════
@tf.function
def fractional_flow(Sw_norm, Cp_norm):
    """
    Eq. 3 of Liu et al.:
      fw = [krw/μw(Cp)] / [krw/μw(Cp) + kro/μo]
         = krw / [krw + M·kro]   where  M = μw(Cp)/μo
    Inputs are the normalised network outputs ∈ (0,1).
    """
    Sw  = _Swi + Sw_norm * _Smob
    Se  = tf.clip_by_value((Sw - _Swi) / (_Smob + 1e-8), 0.0, 1.0)
    krw = _krw * tf.pow(Se + 1e-8, _nw)           # Corey krw
    kro = _kro * tf.pow(1.0 - Se + 1e-8, _no)     # Corey kro
    mu_w = _muwi * (1.0 + _r*Cp_norm + _s*Cp_norm**2 + _t*Cp_norm**3)  # Eq. 1
    M    = mu_w / (_muo + 1e-8)                    # viscosity ratio
    fw   = krw / (krw + M * kro + 1e-8)            # Eq. 3
    return fw

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  PINN-1 NETWORK   (Liu et al. Fig. 2a — single MLP for Sw and Cp)
# ═══════════════════════════════════════════════════════════════════════════════
def build_pinn1(n_hidden: int = 8, n_neurons: int = 64):
    """
    PINN-1 (Liu et al. Fig. 2a):
      Input  : (X, T) ∈ [0,1]²   (normalised spatial-temporal coordinates)
      Hidden : tanh,  glorot_normal initialisation
      Output : (Sw_norm, Cp_norm) ∈ (0,1)  via sigmoid   (physical bounds enforced)
    """
    inp = keras.Input(shape=(2,), name='xt')
    z   = inp
    for _ in range(n_hidden):
        z = keras.layers.Dense(n_neurons, activation='tanh',
                               kernel_initializer='glorot_normal')(z)
    out = keras.layers.Dense(2, activation='sigmoid', name='Sw_Cp')(z)
    return keras.Model(inp, out, name='PINN1')

model     = build_pinn1()
optimizer = keras.optimizers.Adam(learning_rate=lr_0)
model.summary()

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  COLLOCATION POINTS  — Latin Hypercube Sampling
#     (Liu et al.: "Latin hypercube sampling was conducted on both t∈[0,1] and
#      x∈[0,1] using 20,000 random sample points")
# ═══════════════════════════════════════════════════════════════════════════════
lhs    = qmc.LatinHypercube(d=2, seed=SEED)
XT_m   = lhs.random(N_m).astype(np.float32)    # interior PDE points

X_ic   = np.random.uniform(0, 1, N_IC).astype(np.float32)
XT_ic  = np.column_stack([X_ic, np.zeros(N_IC, np.float32)])   # T=0

T_bc   = np.random.uniform(0, 1, N_BC).astype(np.float32)
XT_bc  = np.column_stack([np.zeros(N_BC, np.float32), T_bc])   # X=0

# Non-trainable Variables so the gradient tapes can watch them
X_col = tf.Variable(XT_m[:, 0:1], trainable=False, dtype=tf.float32)
T_col = tf.Variable(XT_m[:, 1:2], trainable=False, dtype=tf.float32)
XT_ic_tf = tf.constant(XT_ic)
XT_bc_tf = tf.constant(XT_bc)

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  TRAINING STEP
# ═══════════════════════════════════════════════════════════════════════════════
@tf.function
def train_step():
    """
    Three nested gradient tapes:
      g_param  — outer tape watching θ (model weights) for the Adam update
      g2       — middle tape watching X,T for 2nd-order spatial derivatives (ε term)
      g1       — inner tape watching X,T for 1st-order derivatives in BL & Cp PDEs
    """
    with tf.GradientTape() as g_param:

        # ── PDE residuals at interior collocation points ─────────────────
        with tf.GradientTape(persistent=True) as g2:
            g2.watch([X_col, T_col])
            with tf.GradientTape(persistent=True) as g1:
                g1.watch([X_col, T_col])
                out   = model(tf.concat([X_col, T_col], axis=1), training=True)
                Sw    = out[:, 0:1]
                Cp    = out[:, 1:2]
                fw    = fractional_flow(Sw, Cp)
            # 1st-order derivatives
            dSw_dX = g1.gradient(Sw, X_col)
            dSw_dT = g1.gradient(Sw, T_col)
            dfw_dX = g1.gradient(fw, X_col)
            dCp_dX = g1.gradient(Cp, X_col)
            dCp_dT = g1.gradient(Cp, T_col)
            del g1
        # 2nd-order spatial derivatives
        d2Sw_dX2 = g2.gradient(dSw_dX, X_col)
        d2Cp_dX2 = g2.gradient(dCp_dX, X_col)
        del g2

        # BL PDE residual (Eq. 2 + ε stabilisation, Liu et al. Eq. 9):
        #   ∂Sw/∂T + ∂fw/∂X − ε·∂²Sw/∂X² = 0   (coefficient = 1 after normalization)
        R_Sw = dSw_dT + dfw_dX - _eps * d2Sw_dX2

        # Polymer PDE residual (simplified Eq. 22 + ε, Liu et al. Eq. 10):
        #   Sw·∂Cp/∂T + fw·∂Cp/∂X − ε·∂²Cp/∂X² = 0
        R_Cp = Sw * dCp_dT + fw * dCp_dX - _eps * d2Cp_dX2

        L_pde_Sw = tf.reduce_mean(tf.square(R_Sw))   # Eq. 9 first term
        L_pde_Cp = tf.reduce_mean(tf.square(R_Cp))   # Eq. 10 first term

        # ── Initial condition   T=0  (Eq. 9 second term, Eq. 10 second term) ─
        out_ic  = model(XT_ic_tf, training=True)
        L_ic_Sw = tf.reduce_mean(tf.square(out_ic[:, 0:1] - _Sw0))    # Sw=Swi
        L_ic_Cp = tf.reduce_mean(tf.square(out_ic[:, 1:2]))            # Cp=0

        # ── Boundary condition   X=0  (Eq. 9 third term, Eq. 10 third term) ──
        out_bc  = model(XT_bc_tf, training=True)
        L_bc_Sw = tf.reduce_mean(tf.square(out_bc[:, 0:1] - _SwBC))   # Sw=1-Sor
        L_bc_Cp = tf.reduce_mean(tf.square(out_bc[:, 1:2] - _CpBC))   # Cp=Cpi

        # ── Total loss (Liu et al. Eq. 8, ω3=0 data-free) ───────────────
        L_Sw    = omega11*L_pde_Sw + omega12*L_ic_Sw + omega13*L_bc_Sw
        L_Cp    = omega21*L_pde_Cp + omega22*L_ic_Cp + omega23*L_bc_Cp
        L_total = omega1*L_Sw + omega2*L_Cp

    grads = g_param.gradient(L_total, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return L_total, L_pde_Sw, L_pde_Cp, L_ic_Sw, L_ic_Cp, L_bc_Sw, L_bc_Cp

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════
keys = ['total', 'pde_Sw', 'pde_Cp', 'ic_Sw', 'ic_Cp', 'bc_Sw', 'bc_Cp']
hist = {k: [] for k in keys}
t_start = time.time()

print("\nTraining  (Liu et al. 2025 — Example 3 setup)...")
for it in range(1, N_iter + 1):
    vals = [v.numpy() for v in train_step()]
    for k, v in zip(keys, vals):
        hist[k].append(v)

    if it % 500 == 0:
        elapsed  = time.time() - t_start
        eta_min  = elapsed / it * (N_iter - it) / 60
        print(f"  [{it:6d}/{N_iter}]  "
              f"L={vals[0]:.3e}  BL={vals[1]:.3e}  Cp={vals[2]:.3e}  "
              f"IC_Sw={vals[3]:.3e}  BC_Cp={vals[6]:.3e}  ETA {eta_min:.1f}min")

    # Reduce LR at halfway (Table IV, Example 3 approach)
    if it == N_iter // 2:
        optimizer.learning_rate.assign(lr_1)
        print(f"\n  ── lr reduced: {lr_0:.0e} → {lr_1:.0e} at iteration {it} ──\n")

total_min = (time.time() - t_start) / 60
print(f"\nDone in {total_min:.1f} min  ({total_min*60/N_iter:.2f} s/iter)\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 9.  SAVE
# ═══════════════════════════════════════════════════════════════════════════════
os.makedirs('models', exist_ok=True)
model.save('models/pinn_datafree.keras')
np.save('models/pinn_datafree_history.npy', hist)
print("Saved: models/pinn_datafree.keras")

# ═══════════════════════════════════════════════════════════════════════════════
# 10.  PREDICTION ON UNIFORM GRID
# ═══════════════════════════════════════════════════════════════════════════════
N_plt   = 200
X_pts   = np.linspace(0, 1, N_plt, dtype=np.float32)
T_pts   = np.linspace(0, 1, N_plt, dtype=np.float32)
XX, TT  = np.meshgrid(X_pts, T_pts)
XT_grid = np.stack([XX.ravel(), TT.ravel()], axis=1)
pred    = model.predict(XT_grid, batch_size=4000, verbose=0)

Sw_grid = Swi + pred[:, 0].reshape(N_plt, N_plt) * Sw_mobile_range  # physical Sw
Cp_grid = pred[:, 1].reshape(N_plt, N_plt) * Cpi_phys                # physical Cp [kg/m³]

# fractional flow at producer (X=1) over time → water-cut proxy
XT_prod = np.column_stack([np.ones(N_plt, np.float32), X_pts])
p_prod  = model.predict(XT_prod, batch_size=N_plt, verbose=0)
Sw_p    = tf.constant(p_prod[:, 0:1], tf.float32)
Cp_p    = tf.constant(p_prod[:, 1:2], tf.float32)
fw_prod = fractional_flow(Sw_p, Cp_p).numpy().ravel()
t_days  = X_pts * t_ref   # dimensional time axis [days]

# ═══════════════════════════════════════════════════════════════════════════════
# 11.  PLOTS  (Liu et al. Fig. 3 / 4 / 6 / 8 style)
# ═══════════════════════════════════════════════════════════════════════════════
SNAPS  = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
COLORS = ['navy', 'royalblue', 'deepskyblue', 'green', 'orange', 'red']

# ── Fig A: solution maps + profiles ─────────────────────────────────────────
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(
    "Data-Free PINN — 1D Polymer Flood  (Pelican Lake HP-6)\n"
    "Liu et al. (2025) PINN-1,  Eq. 22 simplified Cp,  ε=1e-3  (Example 3 regime)",
    fontsize=11)

# Sw(x,t) map
im0 = ax[0,0].contourf(X_pts*L, T_pts*t_ref, Sw_grid, 50, cmap='RdYlBu_r')
plt.colorbar(im0, ax=ax[0,0], label='Sw')
ax[0,0].set_xlabel('x  [m]'); ax[0,0].set_ylabel('t  [days]')
ax[0,0].set_title('Sw(x, t)')

# Cp(x,t) map
im1 = ax[0,1].contourf(X_pts*L, T_pts*t_ref, Cp_grid, 50, cmap='viridis')
plt.colorbar(im1, ax=ax[0,1], label='Cp  [kg/m³]')
ax[0,1].set_xlabel('x  [m]'); ax[0,1].set_ylabel('t  [days]')
ax[0,1].set_title('Cp(x, t)')

# Water-cut at producer
ax[0,2].plot(t_days, fw_prod, 'b', lw=2, label='PINN (data-free)')
ax[0,2].set_xlabel('t  [days]'); ax[0,2].set_ylabel('fw  (water-cut at x=L)')
ax[0,2].set_title('Water-Cut at Producer'); ax[0,2].set_ylim([0, 1.05])
ax[0,2].legend(fontsize=9)

# Sw spatial profiles (Fig. 4 / 6 / 8 style)
for frac, col in zip(SNAPS, COLORS):
    idx = int(frac * (N_plt - 1))
    ax[1,0].plot(X_pts, Sw_grid[idx, :], color=col,
                 label=f'T={frac:.1f}  ({frac*t_ref:.0f}d)')
ax[1,0].axhline(Sw_init, color='k', lw=0.8, ls=':', label=f'Swi={Sw_init:.2f}')
ax[1,0].set_xlabel('X  (normalised)'); ax[1,0].set_ylabel('Sw')
ax[1,0].set_title('Sw at T snapshots'); ax[1,0].legend(fontsize=7)
ax[1,0].set_ylim([Swi - 0.02, 1 - Sor + 0.05])

# Cp spatial profiles
for frac, col in zip(SNAPS, COLORS):
    idx = int(frac * (N_plt - 1))
    ax[1,1].plot(X_pts, Cp_grid[idx, :], color=col,
                 label=f'T={frac:.1f}  ({frac*t_ref:.0f}d)')
ax[1,1].set_xlabel('X  (normalised)'); ax[1,1].set_ylabel('Cp  [kg/m³]')
ax[1,1].set_title('Cp at T snapshots'); ax[1,1].legend(fontsize=7)

# IC verification
pred_ic = model.predict(XT_ic_tf.numpy(), verbose=0)
ax[1,2].scatter(XT_ic[:, 0], Swi + pred_ic[:, 0]*Sw_mobile_range,
                s=4, alpha=0.5, label='Sw_pred')
ax[1,2].axhline(Sw_init, color='r', lw=1.5, label=f'Target {Sw_init:.2f}')
ax[1,2].set_xlabel('X'); ax[1,2].set_ylabel('Sw at T=0')
ax[1,2].set_title('IC enforcement  (Sw, T=0)'); ax[1,2].legend(fontsize=9)

plt.tight_layout()
plt.savefig('pinn_datafree_solution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: pinn_datafree_solution.png")

# ── Fig B: loss curves  (Fig. 3 / 5 / 7 style) ──────────────────────────────
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Loss History — Data-Free PINN  (Liu et al. 2025, Fig. 3 style)',
             fontsize=11)
its = range(1, N_iter + 1)

ax[0,0].semilogy(its, hist['total'], 'k', lw=1)
ax[0,0].set_title('Total Loss  L(θ)'); ax[0,0].set_xlabel('Iteration')

ax[0,1].semilogy(its, hist['pde_Sw'], label='BL  (Sw)', lw=1)
ax[0,1].semilogy(its, hist['pde_Cp'], label='Polymer  (Cp)', lw=1)
ax[0,1].set_title('PDE Loss'); ax[0,1].legend(); ax[0,1].set_xlabel('Iteration')

ax[1,0].semilogy(its, hist['bc_Sw'], label='Sw', lw=1)
ax[1,0].semilogy(its, hist['bc_Cp'], label='Cp', lw=1)
ax[1,0].set_title('Boundary Condition Loss'); ax[1,0].legend(); ax[1,0].set_xlabel('Iteration')

ax[1,1].semilogy(its, hist['ic_Sw'], label='Sw', lw=1)
ax[1,1].semilogy(its, hist['ic_Cp'], label='Cp', lw=1)
ax[1,1].set_title('Initial Condition Loss'); ax[1,1].legend(); ax[1,1].set_xlabel('Iteration')

plt.tight_layout()
plt.savefig('pinn_datafree_loss.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: pinn_datafree_loss.png")

# ─── Terminal summary ────────────────────────────────────────────────────────
print("\n── Final loss (last iteration) ──")
for k in keys:
    print(f"  {k:12s}: {hist[k][-1]:.4e}")
print(f"\nNetwork: {model.count_params():,} parameters  ({8} hidden layers × 64 neurons)")
print("Done.")
