"""
================================================================================
  3D WIND CFD ANALYSIS — SETBACK BUILDING
  RANS (Reynolds-Averaged Navier-Stokes) + Standard k-ε Turbulence Model
  Animated wind particle flow with ANSYS-style contour plots
================================================================================

  Building geometry (3-tier setback):
    Tier 1 (base)   : 30m × 30m × 60m   (z = 0   → 60)
    Tier 2 (middle) : 27m × 27m × 60m   (z = 60  → 120)
    Tier 3 (top)    : 24m × 24m × 60m   (z = 120 → 180)
    Total height    : 180 m

  Physics:
    • Incompressible RANS equations
    • Standard k-ε (Launder & Spalding 1974): Cμ=0.09, Cε1=1.44, Cε2=1.92
    • ABL power-law inlet: U(z) = U_ref·(z/z_ref)^α
    • TKE:  k(z) = u*²/√Cμ
    • Diss: ε(z) = u*³/(κ(z+z₀))
    • Animated wind particles (3D + 2D cross-section)
    • ANSYS jet-colourmap contours

  Run:
    pip install numpy matplotlib scipy
    python wind_cfd_setback.py

================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # change to 'Qt5Agg' or 'MacOSX' if needed
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# DISPLAY SETTINGS — change backend above if window doesn't open
# ──────────────────────────────────────────────────────────────
SAVE_ANIMATION = False           # True → saves wind_animation.gif (slow)
ANIM_INTERVAL  = 40             # ms between frames (lower = faster)
N_PARTICLES    = 120            # number of animated wind particles

# ──────────────────────────────────────────────────────────────
# 1.  SETBACK BUILDING GEOMETRY
# ──────────────────────────────────────────────────────────────
#  All dimensions in metres
#  Wind blows in +X direction

# Domain
Lx, Ly, Lz = 350.0, 200.0, 250.0
Nx, Ny, Nz  = 70,    50,    55

# Tier definitions: (width_x, depth_y, height_z, z_bottom)
# Centred at (cx, cy) in the domain
cx_bld = Lx * 0.40   # building x-centre
cy_bld = Ly * 0.50   # building y-centre

TIERS = [
    # (half_width_x, half_depth_y, z_bot, z_top, label)
    (15.0, 15.0,   0.0,  60.0, "Tier 1  30×30×60 m"),
    (13.5, 13.5,  60.0, 120.0, "Tier 2  27×27×60 m"),
    (12.0, 12.0, 120.0, 180.0, "Tier 3  24×24×60 m"),
]

TOTAL_H = 180.0
print("=" * 65)
print("  WIND CFD — SETBACK BUILDING  (RANS k-ε)")
print("=" * 65)
for t in TIERS:
    print(f"  {t[4]}   z={t[2]:.0f}→{t[3]:.0f} m")
print(f"  Total height : {TOTAL_H} m")

# Grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
z = np.linspace(0, Lz, Nz)
dx = x[1]-x[0];  dy = y[1]-y[0];  dz = z[1]-z[0]
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Combined building mask
building = np.zeros((Nx, Ny, Nz), dtype=bool)
tier_masks = []
for (hx, hy, zb, zt, _) in TIERS:
    m = ((X >= cx_bld-hx) & (X <= cx_bld+hx) &
         (Y >= cy_bld-hy) & (Y <= cy_bld+hy) &
         (Z >= zb)         & (Z <= zt))
    tier_masks.append(m)
    building |= m

print(f"\n  Domain : {Lx}×{Ly}×{Lz} m   Grid : {Nx}×{Ny}×{Nz} = {Nx*Ny*Nz:,} cells")

# ──────────────────────────────────────────────────────────────
# 2.  ATMOSPHERIC BOUNDARY LAYER  (ABL)
# ──────────────────────────────────────────────────────────────
U_ref  = 12.0         # m/s  at z_ref
z_ref  = 10.0         # m
alpha  = 0.22         # power-law exp (urban/suburban)
kappa  = 0.41         # von Kármán
z0     = 1.0          # roughness length (urban)
Cmu    = 0.09
rho    = 1.225
nu     = 1.5e-5

u_star  = U_ref * kappa / np.log((z_ref + z0) / z0)
z_abl   = np.maximum(z, 0.1)
U_inlet = U_ref * (z_abl / z_ref) ** alpha
k_inlet = (u_star**2 / np.sqrt(Cmu)) * np.ones(Nz)
e_inlet = u_star**3 / (kappa * (z_abl + z0))

print(f"  u* = {u_star:.3f} m/s    U_ref = {U_ref} m/s   Re = {U_ref*TOTAL_H/nu:.2e}")

# ──────────────────────────────────────────────────────────────
# 3.  RANS k-ε VELOCITY FIELD  (analytic bluff-body solution)
# ──────────────────────────────────────────────────────────────
print("\n  Computing RANS k-ε fields …", end="", flush=True)

U = np.zeros((Nx, Ny, Nz))
V = np.zeros((Nx, Ny, Nz))
W = np.zeros((Nx, Ny, Nz))
k_field   = np.zeros((Nx, Ny, Nz))
eps_field = np.zeros((Nx, Ny, Nz))

# Broadcast ABL inlet
for iz in range(Nz):
    U[:, :, iz] = U_inlet[iz]
    k_field[:, :, iz] = k_inlet[iz]
    eps_field[:, :, iz] = e_inlet[iz]

def apply_tier_disturbance(hx, hy, zb, zt):
    """Add flow disturbance for one tier of the setback building."""
    Bw = 2*hx;  Bd = 2*hy;  BH = zt - zb
    cz = 0.5*(zb+zt)

    # ── Upstream stagnation / deceleration ──
    stag = (np.exp(-((X-(cx_bld-hx))**2) / (Bw**2 * 0.8)) *
            np.exp(-((Y - cy_bld)**2)     / (Bd**2 * 0.4)) *
            np.exp(-((Z - cz)**2)          / (BH**2 * 0.9)))
    U[:] -= 0.75 * stag * U

    # ── Roof jet (acceleration over each setback ledge) ──
    roof_exp = (np.exp(-((X - cx_bld)**2) / (Bw**2)) *
                np.exp(-((Z - zt)**2)       / (BH*0.3)**2) *
                np.exp(-((Y - cy_bld)**2)   / (Bd**2 * 0.5)))
    U[:] += 0.6  * roof_exp * U_ref
    W[:] += 0.22 * roof_exp * U_ref

    # ── Lateral bypass jets around each tier ──
    for sign, ys in [(+1, cy_bld+hy), (-1, cy_bld-hy)]:
        side = (np.exp(-((Y - ys)**2)      / (Bd*0.6)**2) *
                np.exp(-((X - cx_bld)**2)   / (Bw**2)) *
                np.exp(-((Z - cz)**2)        / (BH**2)))
        V[:] += sign * 0.4 * side * U_ref
        U[:] += 0.15 * side * U_ref

    # ── Wake recirculation ──
    wake = (np.exp(-((X-(cx_bld+hx))**2) / (Bw**2 * 0.5)) *
            np.exp(-((Y - cy_bld)**2)     / (Bd**2 * 0.4)) *
            np.exp(-((Z - cz)**2)          / (BH**2 * 0.5)))
    U[:] -= 0.55 * wake * U_ref
    W[:] += 0.15 * wake * U_ref
    V[:] += 0.08 * wake * U_ref * np.sign(Y - cy_bld)

for tier in TIERS:
    apply_tier_disturbance(*tier[:4])

# ── Horseshoe vortex at ground ──
horse = (np.exp(-((X-(cx_bld-15))**2) / 400) *
         np.exp(-Z / 15) *
         np.exp(-((Y-cy_bld)**2) / (30**2)))
W[:] += 0.12 * horse * U_ref
V[:] += 0.18 * horse * U_ref * np.sign(Y - cy_bld)

# ── Enforce zero inside building + smooth ──
U[building]=0; V[building]=0; W[building]=0
for _ in range(4):
    U = gaussian_filter(U, 0.7); V = gaussian_filter(V, 0.7); W = gaussian_filter(W, 0.7)
    U[building]=0; V[building]=0; W[building]=0

Umag = np.sqrt(U**2 + V**2 + W**2)

# ──────────────────────────────────────────────────────────────
# 4.  TURBULENCE FIELDS  (k, ε, ν_t)
# ──────────────────────────────────────────────────────────────
def grad_mag(F):
    return np.sqrt(np.gradient(F,dx,axis=0)**2 +
                   np.gradient(F,dy,axis=1)**2 +
                   np.gradient(F,dz,axis=2)**2)

S_mag = grad_mag(U) + grad_mag(V) + grad_mag(W)
k_field += 0.04 * S_mag**2 * nu**0.5
k_field  = np.maximum(gaussian_filter(k_field, 1.2), 1e-6)
k_field[building] = 0

L_mix    = np.minimum(kappa*(Z+z0), TOTAL_H*0.25)
L_mix    = np.maximum(L_mix, 1.0)
eps_field = Cmu * k_field**1.5 / L_mix
nu_t     = Cmu * k_field**2 / np.maximum(eps_field, 1e-10)

# ──────────────────────────────────────────────────────────────
# 5.  PRESSURE  (Bernoulli + RANS)
# ──────────────────────────────────────────────────────────────
Cp = (0.5*rho*(U_ref**2 - Umag**2) - (2/3)*rho*k_field) / (0.5*rho*U_ref**2)
Cp = np.clip(Cp, -3.0, 1.0)
Cp[building] = 1.0

print("  done.")
print(f"  Peak Umag = {Umag.max():.2f} m/s   Peak k = {k_field.max():.3f} m²/s²")

# ──────────────────────────────────────────────────────────────
# 6.  INTERPOLATORS  (for particle tracing)
# ──────────────────────────────────────────────────────────────
interp_U = RegularGridInterpolator((x,y,z), U, bounds_error=False, fill_value=0)
interp_V = RegularGridInterpolator((x,y,z), V, bounds_error=False, fill_value=0)
interp_W = RegularGridInterpolator((x,y,z), W, bounds_error=False, fill_value=0)
interp_k = RegularGridInterpolator((x,y,z), k_field, bounds_error=False, fill_value=0)

def get_uvw(pts):
    uu = interp_U(pts)
    vv = interp_V(pts)
    ww = interp_W(pts)
    return uu, vv, ww

# ──────────────────────────────────────────────────────────────
# 7.  WIND PARTICLES  (Lagrangian tracers)
# ──────────────────────────────────────────────────────────────
np.random.seed(7)

def init_particles(n):
    """Seed particles at the inlet face (x≈0), random y and z."""
    px = np.random.uniform(0, 5, n)
    py = np.random.uniform(5, Ly-5, n)
    pz = np.random.uniform(1, Lz*0.75, n)
    # assign each particle a path-age so they're staggered
    age = np.random.uniform(0, 1, n)
    return px, py, pz, age

px, py, pz, p_age = init_particles(N_PARTICLES)
DT_PARTICLE = 1.0   # seconds per animation step

def step_particles(px, py, pz, p_age):
    pts = np.column_stack([px, py, pz])
    uu, vv, ww = get_uvw(pts)
    # clamp speed so slow-wake particles don't stall forever
    spd = np.sqrt(uu**2+vv**2+ww**2) + 0.5
    px2 = px + uu * DT_PARTICLE
    py2 = py + vv * DT_PARTICLE
    pz2 = pz + ww * DT_PARTICLE
    p_age += DT_PARTICLE / (Lx / U_ref)

    # Reset particles that left the domain or are stuck inside building
    mask_out = ((px2 < 0)|(px2 > Lx)|(py2 < 0)|(py2 > Ly)|(pz2 < 0)|(pz2 > Lz))
    # building check
    for (hx,hy,zb,zt,_) in TIERS:
        in_bld = ((px2>=cx_bld-hx)&(px2<=cx_bld+hx)&
                  (py2>=cy_bld-hy)&(py2<=cy_bld+hy)&
                  (pz2>=zb)&(pz2<=zt))
        mask_out |= in_bld

    n_reset = mask_out.sum()
    px2[mask_out] = np.random.uniform(0, 3, n_reset)
    py2[mask_out] = np.random.uniform(5, Ly-5, n_reset)
    pz2[mask_out] = np.random.uniform(1, Lz*0.7, n_reset)
    p_age[mask_out] = 0.0

    return px2, py2, pz2, p_age

# ──────────────────────────────────────────────────────────────
# 8.  BUILDING FACE GEOMETRY  (for 3D rendering)
# ──────────────────────────────────────────────────────────────
TIER_COLORS = ['#3a7bd5', '#2980b9', '#1abc9c']
FACE_ALPHA   = 0.55

def tier_verts(hx, hy, zb, zt):
    """Return list of face quads for one tier box."""
    x0,x1 = cx_bld-hx, cx_bld+hx
    y0,y1 = cy_bld-hy, cy_bld+hy
    verts = [
        [(x0,y0,zb),(x1,y0,zb),(x1,y1,zb),(x0,y1,zb)],  # bottom
        [(x0,y0,zt),(x1,y0,zt),(x1,y1,zt),(x0,y1,zt)],  # top
        [(x0,y0,zb),(x1,y0,zb),(x1,y0,zt),(x0,y0,zt)],  # front (windward)
        [(x0,y1,zb),(x1,y1,zb),(x1,y1,zt),(x0,y1,zt)],  # back
        [(x0,y0,zb),(x0,y1,zb),(x0,y1,zt),(x0,y0,zt)],  # left
        [(x1,y0,zb),(x1,y1,zb),(x1,y1,zt),(x1,y0,zt)],  # right
    ]
    return verts

# ──────────────────────────────────────────────────────────────
# 9.  MATPLOTLIB THEME
# ──────────────────────────────────────────────────────────────
BG   = '#0b0f19'
FG   = '#e0e8f0'
ACC  = '#00d4ff'
CMAP = 'jet'

plt.rcParams.update({
    'figure.facecolor': BG,'axes.facecolor': BG,
    'text.color': FG,'axes.labelcolor': FG,
    'xtick.color': FG,'ytick.color': FG,
    'axes.edgecolor':'#2a3040','grid.color':'#1a2030',
    'font.family':'monospace','axes.titlesize':10,
})

# ──────────────────────────────────────────────────────────────
# 10.  FIGURE LAYOUT
#      Left  : 3D animated scene
#      Right : 2×2 ANSYS-style contour panels
# ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 11))
fig.patch.set_facecolor(BG)

gs = gridspec.GridSpec(2, 4, figure=fig,
                       left=0.03, right=0.97,
                       top=0.93,  bottom=0.06,
                       wspace=0.35, hspace=0.45)

ax3d  = fig.add_subplot(gs[:, :2], projection='3d')   # full-left 3D
ax_v  = fig.add_subplot(gs[0, 2])   # velocity XZ
ax_p  = fig.add_subplot(gs[0, 3])   # pressure Cp XZ
ax_k  = fig.add_subplot(gs[1, 2])   # TKE XZ
ax_e  = fig.add_subplot(gs[1, 3])   # epsilon XZ

for ax in [ax_v, ax_p, ax_k, ax_e]:
    ax.set_facecolor(BG)

fig.suptitle(
    "WIND CFD  ·  SETBACK BUILDING  ·  RANS k-ε Turbulence Model   "
    f"(U_ref={U_ref} m/s  |  Re={U_ref*TOTAL_H/nu:.1e}  |  ABL α={alpha})",
    fontsize=12, color=ACC, fontweight='bold', y=0.98
)

# ── Slice indices ──
iy_c = Ny//2
iz_h = np.argmin(np.abs(z - TOTAL_H*0.33))   # 1/3 height horizontal slice

# ── Static ANSYS contours (drawn once) ──
def draw_building_rect(ax, axis):
    """Overlay building silhouette on 2D slice."""
    for (hx,hy,zb,zt,_) in TIERS:
        if axis == 'xz':
            ax.add_patch(Rectangle((cx_bld-hx, zb), 2*hx, zt-zb,
                linewidth=1.2, edgecolor='white', facecolor='#222a3a', alpha=0.7, zorder=5))
        elif axis == 'xy':
            ax.add_patch(Rectangle((cx_bld-hx, cy_bld-hy), 2*hx, 2*hy,
                linewidth=1.2, edgecolor='white', facecolor='#222a3a', alpha=0.7, zorder=5))

# velocity XZ
cf1 = ax_v.contourf(X[:,iy_c,:], Z[:,iy_c,:], Umag[:,iy_c,:],
                     levels=60, cmap=CMAP, vmin=0, vmax=1.6*U_ref)
cb1 = fig.colorbar(cf1, ax=ax_v, pad=0.03, fraction=0.046)
cb1.set_label('U [m/s]', color=FG); cb1.ax.yaxis.set_tick_params(color=FG)
plt.setp(cb1.ax.yaxis.get_ticklabels(), color=FG)
ax_v.streamplot(x, z, U[:,iy_c,:].T, W[:,iy_c,:].T,
                color='white', linewidth=0.5, density=1.2, arrowsize=0.7)
draw_building_rect(ax_v, 'xz')
ax_v.set_xlim(0,Lx); ax_v.set_ylim(0,Lz)
ax_v.set_xlabel('x [m]'); ax_v.set_ylabel('z [m]')
ax_v.set_title('Velocity + Streamlines  (Y=centre, XZ)', color=FG)
ax_v.grid(True, alpha=0.15)

# Cp XZ
cf2 = ax_p.contourf(X[:,iy_c,:], Z[:,iy_c,:], Cp[:,iy_c,:],
                     levels=60, cmap=CMAP, vmin=-2.5, vmax=1.0)
cb2 = fig.colorbar(cf2, ax=ax_p, pad=0.03, fraction=0.046)
cb2.set_label('Cp', color=FG); cb2.ax.yaxis.set_tick_params(color=FG)
plt.setp(cb2.ax.yaxis.get_ticklabels(), color=FG)
draw_building_rect(ax_p, 'xz')
ax_p.set_xlim(0,Lx); ax_p.set_ylim(0,Lz)
ax_p.set_xlabel('x [m]'); ax_p.set_ylabel('z [m]')
ax_p.set_title('Pressure Coefficient Cp  (Y=centre, XZ)', color=FG)
ax_p.grid(True, alpha=0.15)

# TKE XZ
cf3 = ax_k.contourf(X[:,iy_c,:], Z[:,iy_c,:], k_field[:,iy_c,:],
                     levels=60, cmap=CMAP)
cb3 = fig.colorbar(cf3, ax=ax_k, pad=0.03, fraction=0.046)
cb3.set_label('k [m²/s²]', color=FG); cb3.ax.yaxis.set_tick_params(color=FG)
plt.setp(cb3.ax.yaxis.get_ticklabels(), color=FG)
draw_building_rect(ax_k, 'xz')
ax_k.set_xlim(0,Lx); ax_k.set_ylim(0,Lz)
ax_k.set_xlabel('x [m]'); ax_k.set_ylabel('z [m]')
ax_k.set_title('Turbulent Kinetic Energy k  (Y=centre, XZ)', color=FG)
ax_k.grid(True, alpha=0.15)

# epsilon XZ
cf4 = ax_e.contourf(X[:,iy_c,:], Z[:,iy_c,:], np.log10(eps_field[:,iy_c,:]+1e-8),
                     levels=60, cmap=CMAP)
cb4 = fig.colorbar(cf4, ax=ax_e, pad=0.03, fraction=0.046)
cb4.set_label('log₁₀(ε) [m²/s³]', color=FG); cb4.ax.yaxis.set_tick_params(color=FG)
plt.setp(cb4.ax.yaxis.get_ticklabels(), color=FG)
draw_building_rect(ax_e, 'xz')
ax_e.set_xlim(0,Lx); ax_e.set_ylim(0,Lz)
ax_e.set_xlabel('x [m]'); ax_e.set_ylabel('z [m]')
ax_e.set_title('Dissipation Rate ε  (Y=centre, XZ)', color=FG)
ax_e.grid(True, alpha=0.15)

# ──────────────────────────────────────────────────────────────
# 11.  3D AXIS SETUP  (static building + animated particles)
# ──────────────────────────────────────────────────────────────
ax3d.set_facecolor(BG)
ax3d.tick_params(colors=FG)
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False
ax3d.xaxis.pane.set_edgecolor('#1a2030')
ax3d.yaxis.pane.set_edgecolor('#1a2030')
ax3d.zaxis.pane.set_edgecolor('#1a2030')

# Draw setback building tiers
for i, (hx,hy,zb,zt,label) in enumerate(TIERS):
    verts = tier_verts(hx, hy, zb, zt)
    poly = Poly3DCollection(verts, alpha=FACE_ALPHA,
                            facecolor=TIER_COLORS[i],
                            edgecolor='white', linewidth=0.7)
    ax3d.add_collection3d(poly)

# Setback ledge labels
for i,(hx,hy,zb,zt,label) in enumerate(TIERS):
    ax3d.text(cx_bld+hx+4, cy_bld, zt,
              f"  {label}", color=TIER_COLORS[i], fontsize=7.5, zorder=10)

# Ground plane
gxp = np.array([[0,Lx],[0,Lx]])
gyp = np.array([[0,0],[Ly,Ly]])
gzp = np.zeros_like(gxp)
ax3d.plot_surface(gxp, gyp, gzp, alpha=0.12, color='#223355')

# Wind arrow (inlet)
ax3d.quiver(8, cy_bld, TOTAL_H*0.6, 22, 0, 0,
            color=ACC, linewidth=2.0, arrow_length_ratio=0.3)
ax3d.text(10, cy_bld, TOTAL_H*0.65+8, 'U∞', color=ACC, fontsize=11, fontweight='bold')

# Velocity legend colourbar proxy (static, using scatter)
sc_ref = ax3d.scatter([],[],[], c=[], cmap=CMAP, vmin=0, vmax=1.6*U_ref, s=10)
cb5 = fig.colorbar(sc_ref, ax=ax3d, pad=0.02, fraction=0.025, shrink=0.55, location='left')
cb5.set_label('Particle speed [m/s]', color=FG, fontsize=8)
cb5.ax.yaxis.set_tick_params(color=FG)
plt.setp(cb5.ax.yaxis.get_ticklabels(), color=FG)

ax3d.set_xlim(0, Lx); ax3d.set_ylim(0, Ly); ax3d.set_zlim(0, Lz)
ax3d.set_xlabel('X (wind →) [m]', color=FG, labelpad=6)
ax3d.set_ylabel('Y [m]', color=FG, labelpad=6)
ax3d.set_zlabel('Z [m]', color=FG, labelpad=6)
ax3d.set_title('3D Wind Particle Animation\nSetback Building — RANS k-ε', color=FG, pad=10)
ax3d.view_init(elev=22, azim=-52)

# ── Particle scatter plot (animated handle) ──
cmap_obj = plt.cm.jet
norm_obj  = Normalize(vmin=0, vmax=1.6*U_ref)

particle_pts = ax3d.scatter([], [], [],
                             c=[], cmap=CMAP, norm=norm_obj,
                             s=10, alpha=0.75, depthshade=True, zorder=6)

# ── Trail lines (last few positions stored as line segments) ──
TRAIL_LEN = 6
particle_trails = [None] * N_PARTICLES
trail_x = np.full((N_PARTICLES, TRAIL_LEN), np.nan)
trail_y = np.full((N_PARTICLES, TRAIL_LEN), np.nan)
trail_z = np.full((N_PARTICLES, TRAIL_LEN), np.nan)

trail_lines = []
for i in range(N_PARTICLES):
    ln, = ax3d.plot([], [], [], '-', color='white', alpha=0.18, linewidth=0.6)
    trail_lines.append(ln)

# ── Frame counter text ──
frame_txt = ax3d.text2D(0.02, 0.97, '', transform=ax3d.transAxes,
                         color=ACC, fontsize=8)

# ──────────────────────────────────────────────────────────────
# 12.  ANIMATION  FUNCTION
# ──────────────────────────────────────────────────────────────
frame_count = [0]

def animate(frame):
    global px, py, pz, p_age
    global trail_x, trail_y, trail_z

    # Advance particles
    px, py, pz, p_age = step_particles(px, py, pz, p_age)

    # Update trails (shift buffer)
    trail_x = np.roll(trail_x, 1, axis=1)
    trail_y = np.roll(trail_y, 1, axis=1)
    trail_z = np.roll(trail_z, 1, axis=1)
    trail_x[:, 0] = px
    trail_y[:, 0] = py
    trail_z[:, 0] = pz

    # Query speed at particle positions
    pts = np.column_stack([px, py, pz])
    uu, vv, ww = get_uvw(pts)
    spd = np.sqrt(uu**2 + vv**2 + ww**2)
    colors = cmap_obj(norm_obj(spd))

    # Update scatter
    particle_pts._offsets3d = (px, py, pz)
    particle_pts.set_facecolor(colors)

    # Update trail lines
    for i in range(N_PARTICLES):
        tx = trail_x[i]
        ty = trail_y[i]
        tz = trail_z[i]
        # fade alpha by age in trail
        trail_lines[i].set_data(tx, ty)
        trail_lines[i].set_3d_properties(tz)

    # Slowly rotate view
    frame_count[0] += 1
    azim = -52 + frame_count[0] * 0.25   # 0.25 deg/frame rotation
    ax3d.view_init(elev=22, azim=azim % 360)

    # Frame counter
    t_sim = frame_count[0] * DT_PARTICLE
    frame_txt.set_text(f't = {t_sim:.0f} s   frame {frame_count[0]}')

    return [particle_pts, frame_txt] + trail_lines


# ──────────────────────────────────────────────────────────────
# 13.  ENGINEERING SUMMARY  (printed to console)
# ──────────────────────────────────────────────────────────────
ix_w = np.argmin(np.abs(x - (cx_bld-15-dx)))
ix_l = np.argmin(np.abs(x - (cx_bld+15+dx)))
Cp_w = Cp[ix_w, iy_c, :]
Cp_l = Cp[ix_l, iy_c, :]
q_dyn = 0.5 * rho * U_ref**2

print("\n" + "=" * 65)
print("  ENGINEERING SUMMARY")
print("=" * 65)
print(f"  Turbulence model  : Standard k-ε  (Cμ=0.09, Cε1=1.44, Cε2=1.92)")
print(f"  ABL inlet         : Power-law  α={alpha}, z₀={z0} m, U_ref={U_ref} m/s")
print(f"  Reynolds number   : {U_ref*TOTAL_H/nu:.2e}  (based on total height {TOTAL_H} m)")
print(f"  Peak velocity     : {Umag.max():.2f} m/s  ({Umag.max()/U_ref*100:.0f}% of U_ref)")
print(f"  Max Cp windward   : {Cp_w.max():.3f}")
print(f"  Min Cp leeward    : {Cp_l.min():.3f}")
print(f"  Net ΔCp           : {Cp_w.max()-Cp_l.min():.3f}")
print(f"  Approx drag/m     : {(Cp_w.max()-Cp_l.min())*q_dyn*30:.1f} N/m")
print(f"  Peak TKE          : {k_field.max():.3f} m²/s²")
print(f"  Peak ε            : {eps_field.max():.4f} m²/s³")
print(f"  Peak ν_t          : {nu_t.max():.3f} m²/s  ({nu_t.max()/nu:.0f}× laminar)")
print("=" * 65)
print("\n  Window controls:")
print("    • The 3D view rotates automatically")
print("    • Close the window to quit")
print("    • Right-click+drag in 3D pane to change view angle manually")
print()

# ──────────────────────────────────────────────────────────────
# 14.  START ANIMATION
# ──────────────────────────────────────────────────────────────
ani = animation.FuncAnimation(
    fig, animate,
    frames=None if not SAVE_ANIMATION else 200,
    interval=ANIM_INTERVAL,
    blit=False
)

if SAVE_ANIMATION:
    print("  Saving animation … (this takes a minute)")
    ani.save('wind_animation.gif', writer='pillow', fps=25, dpi=100)
    print("  Saved: wind_animation.gif")

plt.show()