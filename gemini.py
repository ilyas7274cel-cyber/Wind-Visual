import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import warnings

warnings.filterwarnings("ignore")

# ==============================================================
# GLOBAL CONFIGURATION 
# ==============================================================
# Physics & Environment
U_REF         = 15.0         # Reference wind speed at 10m (m/s)
Z_REF         = 10.0         # Reference height (m)
ALPHA         = 0.22         # ABL Power-law exponent
RHO           = 1.225        # Air density (kg/m^3)

# Domain Dimensions
LX, LY, LZ    = 350.0, 200.0, 250.0
NX, NY, NZ    = 60, 50, 50   # Grid resolution

# Building Geometry (Setback Tiers)
CX, CY        = LX * 0.35, LY * 0.5  
TIERS = [
    (15.0, 15.0, 0.0, 60.0,   "Tier 1 (Base)"),
    (12.5, 12.5, 60.0, 120.0,  "Tier 2 (Mid)"),
    (10.0, 10.0, 120.0, 180.0, "Tier 3 (Top)")
]

# Visualization Settings
N_PARTICLES   = 300          # Number of 3D wind tracers
TRAIL_LEN     = 8            # Length of 3D trails
ANIM_INTERVAL = 40           # Frame delay (ms)
BG_COLOR      = '#0b0f19'
TEXT_COLOR    = '#ffffff'
CMAP          = 'turbo'      # Professional heatmap colors

# ==============================================================
# 1. GRID & FLOW SOLVER (RANS Approximation)
# ==============================================================
print("Computing 3D Flow Field & Pressure Data...")
x = np.linspace(0, LX, NX)
y = np.linspace(0, LY, NY)
z = np.linspace(0, LZ, NZ)
dx, dy, dz = x[1]-x[0], y[1]-y[0], z[1]-z[0]
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

building_mask = np.zeros((NX, NY, NZ), dtype=bool)
for (hx, hy, zb, zt, _) in TIERS:
    m = ((X >= CX-hx) & (X <= CX+hx) & (Y >= CY-hy) & (Y <= CY+hy) & (Z >= zb) & (Z <= zt))
    building_mask |= m

U_inlet = U_REF * (np.maximum(z, 0.1) / Z_REF)**ALPHA
U = np.zeros((NX, NY, NZ))
V = np.zeros((NX, NY, NZ))
W = np.zeros((NX, NY, NZ))

for iz in range(NZ): U[:, :, iz] = U_inlet[iz]

# Apply Physics Disturbances (Stagnation, Wake, Roof Jets)
for (hx, hy, zb, zt, _) in TIERS:
    # Front Stagnation pushes air UP and SIDEWAYS
    stag = (np.exp(-((X-(CX-hx))**2)/(hx*2)**2) * np.exp(-((Y-CY)**2)/hy**2) * np.exp(-((Z-(zb+zt)/2)**2)/(zt-zb)**2))
    U -= 0.8 * stag * U
    V += 0.3 * stag * U_REF * np.sign(Y - CY)
    W += 0.4 * stag * U_REF 
    
    # Wake pulls air back and down
    wake = (np.exp(-((X-(CX+hx+30))**2)/(hx*4)**2) * np.exp(-((Y-CY)**2)/hy**2) * np.exp(-((Z-(zb+zt)/2)**2)/(zt-zb)**2))
    U -= 0.6 * wake * U_REF
    V -= 0.15 * wake * U_REF * np.sign(Y - CY)
    W -= 0.15 * wake * U_REF

U[building_mask] = 0; V[building_mask] = 0; W[building_mask] = 0
U = gaussian_filter(U, 0.8); V = gaussian_filter(V, 0.8); W = gaussian_filter(W, 0.8)
Umag = np.sqrt(U**2 + V**2 + W**2)

q_dyn = 0.5 * RHO * U_REF**2
Cp = (0.5 * RHO * (U_REF**2 - Umag**2)) / q_dyn
Cp = np.clip(Cp, -2.5, 1.2)
Cp[building_mask] = 1.0

# Interpolators for 3D Particles
interp_U = RegularGridInterpolator((x, y, z), U, bounds_error=False, fill_value=U_REF)
interp_V = RegularGridInterpolator((x, y, z), V, bounds_error=False, fill_value=0)
interp_W = RegularGridInterpolator((x, y, z), W, bounds_error=False, fill_value=0)

# ==============================================================
# 2. PARTICLE ENGINE (3D Streamlines)
# ==============================================================
def reset_particles(n):
    """Seed particles in a structured inlet grid."""
    px = np.random.uniform(-10, 10, n)
    py = np.random.uniform(20, LY-20, n)
    pz = np.random.uniform(5, LZ*0.8, n)
    return px, py, pz

px, py, pz = reset_particles(N_PARTICLES)
trail_x = np.tile(px[:, None], (1, TRAIL_LEN))
trail_y = np.tile(py[:, None], (1, TRAIL_LEN))
trail_z = np.tile(pz[:, None], (1, TRAIL_LEN))

# ==============================================================
# 3. VISUALIZATION & UI SETUP
# ==============================================================
plt.rcParams.update({'text.color': TEXT_COLOR, 'axes.labelcolor': TEXT_COLOR, 'axes.edgecolor': '#334155'})
fig = plt.figure(figsize=(18, 10), facecolor=BG_COLOR)
gs = gridspec.GridSpec(2, 2, width_ratios=[1.4, 1], wspace=0.15, hspace=0.35)

ax3d = fig.add_subplot(gs[:, 0], projection='3d', facecolor=BG_COLOR)
ax_vel = fig.add_subplot(gs[0, 1], facecolor=BG_COLOR)
ax_cp = fig.add_subplot(gs[1, 1], facecolor=BG_COLOR)

# Normalization Scales
norm_u = Normalize(vmin=0, vmax=U_REF * 1.4)
norm_cp = Normalize(vmin=-2.5, vmax=1.2)
cmap_obj = plt.get_cmap(CMAP)

# --- 3D Scene Setup ---
ax3d.view_init(elev=20, azim=-55)
ax3d.set_xlim(0, LX); ax3d.set_ylim(0, LY); ax3d.set_zlim(0, LZ)
ax3d.set_xlabel('Wind Direction (X) [m]'); ax3d.set_ylabel('Width (Y) [m]'); ax3d.set_zlabel('Height (Z) [m]')
ax3d.set_title("3D Volumetric Wind Flow Simulation", color='#00f2ff', fontsize=14, pad=10)
ax3d.xaxis.pane.fill = False; ax3d.yaxis.pane.fill = False; ax3d.zaxis.pane.fill = False
ax3d.grid(color='#1e293b', alpha=0.5)

# Draw Building
for (hx, hy, zb, zt, _) in TIERS:
    x0, x1 = CX-hx, CX+hx; y0, y1 = CY-hy, CY+hy
    verts = [[(x0,y0,zb),(x1,y0,zb),(x1,y1,zb),(x0,y1,zb)], [(x0,y0,zt),(x1,y0,zt),(x1,y1,zt),(x0,y1,zt)],
             [(x0,y0,zb),(x1,y0,zb),(x1,y0,zt),(x0,y0,zt)], [(x0,y1,zb),(x1,y1,zb),(x1,y1,zt),(x0,y1,zt)],
             [(x0,y0,zb),(x0,y1,zb),(x0,y1,zt),(x0,y0,zt)], [(x1,y0,zb),(x1,y1,zb),(x1,y1,zt),(x1,y0,zt)]]
    ax3d.add_collection3d(Poly3DCollection(verts, alpha=0.5, facecolor='#1e293b', edgecolor='#38bdf8', linewidth=1))

# Initialize 3D Particles & Trails
scatter_heads = ax3d.scatter(px, py, pz, c='white', s=15, alpha=1.0, zorder=5)
trail_lines = [ax3d.plot([], [], [], '-', alpha=0.4, linewidth=1.5)[0] for _ in range(N_PARTICLES)]

# 3D Colorbar
sm_3d = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm_u)
cb_3d = fig.colorbar(sm_3d, ax=ax3d, shrink=0.5, pad=0.05)
cb_3d.set_label('Wind Velocity [m/s]', color=TEXT_COLOR)
cb_3d.ax.yaxis.set_tick_params(color=TEXT_COLOR)

# --- 2D Subplot Setup (Mid-Y plane) ---
iy = NY // 2
X_slice = X[:, iy, :]; Z_slice = Z[:, iy, :]
U_slice = Umag[:, iy, :]; Cp_slice = Cp[:, iy, :]

# Velocity Panel
cf_vel = ax_vel.contourf(x, z, U_slice.T, levels=40, cmap=CMAP, norm=norm_u)
ax_vel.set_title("Mid-Plane Velocity Magnitude (m/s)", color=TEXT_COLOR)
ax_vel.set_xlabel("X [m]"); ax_vel.set_ylabel("Z [m]")
cb_vel = fig.colorbar(cf_vel, ax=ax_vel)
cb_vel.set_label('U [m/s]', color=TEXT_COLOR)

# Pressure Panel
cf_cp = ax_cp.contourf(x, z, Cp_slice.T, levels=40, cmap=CMAP, norm=norm_cp)
ax_cp.set_title("Mid-Plane Pressure Coefficient (Cp)", color=TEXT_COLOR)
ax_cp.set_xlabel("X [m]"); ax_cp.set_ylabel("Z [m]")
cb_cp = fig.colorbar(cf_cp, ax=ax_cp)
cb_cp.set_label('Cp', color=TEXT_COLOR)

# Draw building silhouette on 2D plots
for (hx, hy, zb, zt, _) in TIERS:
    ax_vel.add_patch(Rectangle((CX-hx, zb), hx*2, zt-zb, facecolor='black', edgecolor='cyan', alpha=0.8))
    ax_cp.add_patch(Rectangle((CX-hx, zb), hx*2, zt-zb, facecolor='black', edgecolor='cyan', alpha=0.8))

for cb in [cb_vel, cb_cp]: cb.ax.yaxis.set_tick_params(color=TEXT_COLOR); plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)


# ==============================================================
# 4. ANIMATION LOOP
# ==============================================================
def update(frame):
    global px, py, pz, trail_x, trail_y, trail_z
    
    # Interpolate 3D Velocities
    pts = np.column_stack([px, py, pz])
    u_p = interp_U(pts)
    v_p = interp_V(pts)
    w_p = interp_W(pts)
    
    # Calculate Particle Speed for Colors
    spd = np.sqrt(u_p**2 + v_p**2 + w_p**2)
    colors = cmap_obj(norm_u(spd))
    
    # Update Positions
    dt = 0.8
    px += u_p * dt
    py += v_p * dt
    pz += w_p * dt
    
    # Boundary / Building Collision Reset
    out_bounds = (px > LX) | (px < -15) | (py < 0) | (py > LY) | (pz > LZ) | (pz < 0)
    for (hx, hy, zb, zt, _) in TIERS:
        in_bld = ((px >= CX-hx) & (px <= CX+hx) & (py >= CY-hy) & (py <= CY+hy) & (pz >= zb) & (pz <= zt))
        out_bounds |= in_bld

    n_reset = np.sum(out_bounds)
    if n_reset > 0:
        rx, ry, rz = reset_particles(n_reset)
        px[out_bounds] = rx; py[out_bounds] = ry; pz[out_bounds] = rz
        
        # Snap trails to new position so they don't stretch across the screen
        trail_x[out_bounds, :] = rx[:, None]
        trail_y[out_bounds, :] = ry[:, None]
        trail_z[out_bounds, :] = rz[:, None]

    # Shift Trail History
    trail_x = np.roll(trail_x, 1, axis=1); trail_x[:, 0] = px
    trail_y = np.roll(trail_y, 1, axis=1); trail_y[:, 0] = py
    trail_z = np.roll(trail_z, 1, axis=1); trail_z[:, 0] = pz

    # Update Scatter Points (Heads)
    scatter_heads._offsets3d = (px, py, pz)
    scatter_heads.set_facecolors(colors)

    # Update Trail Lines
    for i in range(N_PARTICLES):
        trail_lines[i].set_data(trail_x[i], trail_y[i])
        trail_lines[i].set_3d_properties(trail_z[i])
        trail_lines[i].set_color(colors[i])

    # Very slow camera rotation to show off the 3D aspect
    ax3d.view_init(elev=20, azim=-55 + frame * 0.1)

    return [scatter_heads] + trail_lines

ani = animation.FuncAnimation(fig, update, frames=500, interval=ANIM_INTERVAL, blit=False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()