import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
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
NX, NY, NZ    = 80, 60, 60   # Grid resolution

# Building Geometry (Setback Tiers)
CX, CY        = LX * 0.35, LY * 0.5  
TIERS = [
    (15.0, 15.0, 0.0, 60.0,   "Tier 1 (Base)"),
    (12.5, 12.5, 60.0, 120.0,  "Tier 2 (Mid)"),
    (10.0, 10.0, 120.0, 180.0, "Tier 3 (Top)")
]

# Visualization Settings
ANIM_INTERVAL = 60           # Frame delay (ms)
BG_COLOR      = '#0b0f19'
TEXT_COLOR    = '#f8fafc'
CMAP          = 'turbo'      # Professional heatmap colors

# ==============================================================
# 1. GRID & FLOW SOLVER (RANS Approximation)
# ==============================================================
print("Computing Flow Field, Stress, and Force Data...")
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

# Apply Physics Disturbances
for (hx, hy, zb, zt, _) in TIERS:
    stag = (np.exp(-((X-(CX-hx))**2)/(hx*2)**2) * np.exp(-((Y-CY)**2)/hy**2) * np.exp(-((Z-(zb+zt)/2)**2)/(zt-zb)**2))
    U -= 0.8 * stag * U
    wake = (np.exp(-((X-(CX+hx+30))**2)/(hx*5)**2) * np.exp(-((Y-CY)**2)/hy**3) * np.exp(-((Z-(zb+zt)/2)**2)/(zt-zb)**2))
    U -= 0.6 * wake * U_REF

U[building_mask] = 0
U = gaussian_filter(U, 0.8)
Umag = np.sqrt(U**2 + V**2 + W**2)

# Calculate Pressure Coefficient (Cp) and Physical Wind Stress (Pascals)
q_dyn = 0.5 * RHO * U_REF**2
Cp = (0.5 * RHO * (U_REF**2 - Umag**2)) / q_dyn
Cp = np.clip(Cp, -2.5, 1.2)
Cp[building_mask] = 1.0

Wind_Stress = Cp * q_dyn  # Pressure in Pascals (N/m^2)

# Calculate 1D Cumulative Force Profile (kN/m vs Height)
force_z = np.zeros(NZ)
total_base_shear = 0

for iz in range(NZ):
    z_val = z[iz]
    current_tier = None
    for (hx, hy, zb, zt, _) in TIERS:
        if zb <= z_val <= zt:
            current_tier = (hx, hy)
            break
            
    if current_tier:
        hx, hy = current_tier
        ix_w = np.argmin(np.abs(x - (CX - hx)))
        ix_l = np.argmin(np.abs(x - (CX + hx)))
        
        # Mask for the Y-width of the building at this tier
        iy_mask = (y >= CY - hy) & (y <= CY + hy)
        
        # Extract stress on windward and leeward faces
        P_w = Wind_Stress[ix_w, iy_mask, iz]
        P_l = Wind_Stress[ix_l, iy_mask, iz]
        
        # Integrate Delta Pressure across the width of the building (dy)
        force_per_meter = np.trapz(P_w - P_l, y[iy_mask]) / 1000.0  # Convert N/m to kN/m
        force_z[iz] = force_per_meter
        total_base_shear += force_per_meter * dz

# ==============================================================
# 2. VISUALIZATION & UI SETUP
# ==============================================================
plt.rcParams.update({
    'text.color': TEXT_COLOR, 
    'axes.labelcolor': TEXT_COLOR, 
    'axes.edgecolor': '#334155',
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR
})

fig = plt.figure(figsize=(18, 10), facecolor=BG_COLOR)
fig.suptitle(f"WIND LOAD ANALYSIS DASHBOARD  |  U_ref = {U_REF} m/s  |  Base Shear = {total_base_shear:,.0f} kN", 
             fontsize=16, fontweight='bold', color='#00f2ff', y=0.96)

gs = gridspec.GridSpec(2, 2, wspace=0.15, hspace=0.35, left=0.05, right=0.95, top=0.88, bottom=0.08)

ax_vel    = fig.add_subplot(gs[0, 0], facecolor=BG_COLOR)
ax_cp     = fig.add_subplot(gs[0, 1], facecolor=BG_COLOR)
ax_stress = fig.add_subplot(gs[1, 0], facecolor=BG_COLOR)
ax_force  = fig.add_subplot(gs[1, 1], facecolor=BG_COLOR)

# Normalization Scales for consistent coloring during animation
norm_u      = Normalize(vmin=0, vmax=U_REF * 1.3)
norm_cp     = Normalize(vmin=-2.5, vmax=1.2)
norm_stress = Normalize(vmin=-2.5 * q_dyn, vmax=1.2 * q_dyn)

# Define Colorbars (Drawn once to prevent duplication)
sm_vel = plt.cm.ScalarMappable(cmap=CMAP, norm=norm_u)
cb_vel = fig.colorbar(sm_vel, ax=ax_vel, pad=0.02)
cb_vel.set_label('Velocity [m/s]', color=TEXT_COLOR)

sm_cp = plt.cm.ScalarMappable(cmap=CMAP, norm=norm_cp)
cb_cp = fig.colorbar(sm_cp, ax=ax_cp, pad=0.02)
cb_cp.set_label('Cp', color=TEXT_COLOR)

sm_stress = plt.cm.ScalarMappable(cmap=CMAP, norm=norm_stress)
cb_stress = fig.colorbar(sm_stress, ax=ax_stress, pad=0.02)
cb_stress.set_label('Stress [Pascals]', color=TEXT_COLOR)

for cb in [cb_vel, cb_cp, cb_stress]:
    cb.ax.yaxis.set_tick_params(color=TEXT_COLOR)

# Initialize Contour Holders
cset_vel = None; cset_cp = None; cset_stress = None

# Set up the Static Force Profile Chart
ax_force.plot(force_z, z, color='#00f2ff', linewidth=3, zorder=5)
ax_force.fill_betweenx(z, 0, force_z, color='#00f2ff', alpha=0.2, zorder=4)
ax_force.set_title("Total Building Drag Force Profile", color='#f8fafc', fontsize=12, pad=10)
ax_force.set_xlabel("Drag Force per meter height [kN/m]")
ax_force.set_ylabel("Elevation (Z) [m]")
ax_force.set_ylim(0, LZ)
ax_force.set_xlim(0, max(force_z) * 1.2)
ax_force.grid(color='#1e293b', alpha=0.6, linestyle='--')

# Draw horizontal lines for tier boundaries on Force plot
for (_, _, zb, zt, label) in TIERS:
    ax_force.axhline(zt, color='#475569', linestyle=':', linewidth=1.5, zorder=2)
    if max(force_z) > 0:
        ax_force.text(max(force_z)*0.8, (zb+zt)/2, label, color='#94a3b8', va='center', ha='left')

# Text Element for dynamic slice info
slice_text = fig.text(0.5, 0.92, '', fontsize=12, fontfamily='monospace', color='#cbd5e1', ha='center')

# ==============================================================
# 3. ANIMATION LOOP (Sweeping Heatmap Slice)
# ==============================================================
def update(frame):
    global cset_vel, cset_cp, cset_stress
    
    # Calculate Sweeping Y-Index (Oscillates back and forth across building width)
    iy = int(NY/2 + (NY/4) * np.sin(frame * 0.08))
    y_pos = y[iy]
    
    # Remove previous contours & patches to prevent memory leak
    for ax in [ax_vel, ax_cp, ax_stress]:
        for c in ax.collections: c.remove()
        [p.remove() for p in reversed(ax.patches)]
    
    # Extract 2D slices
    X_slice = X[:, iy, :]
    Z_slice = Z[:, iy, :]
    
    # Redraw Contours
    cset_vel    = ax_vel.contourf(X_slice, Z_slice, Umag[:, iy, :], levels=40, cmap=CMAP, norm=norm_u)
    cset_cp     = ax_cp.contourf(X_slice, Z_slice, Cp[:, iy, :], levels=40, cmap=CMAP, norm=norm_cp)
    cset_stress = ax_stress.contourf(X_slice, Z_slice, Wind_Stress[:, iy, :], levels=40, cmap=CMAP, norm=norm_stress)
    
    # Format Axes & Draw Building Silhouettes
    for ax, title in zip([ax_vel, ax_cp, ax_stress], 
                         ["Velocity Magnitude", "Pressure Coefficient (Cp)", "Wind Stress / Pressure"]):
        ax.set_title(title, color=TEXT_COLOR, fontsize=12, pad=10)
        ax.set_xlabel("X [m]"); ax.set_ylabel("Z [m]")
        
        # Draw building slice (only if the sweeping plane is intersecting the building)
        for (hx, hy, zb, zt, _) in TIERS:
            if (CY - hy) <= y_pos <= (CY + hy):
                ax.add_patch(Rectangle((CX-hx, zb), hx*2, zt-zb, facecolor='#0f172a', edgecolor='#38bdf8', lw=1.5, alpha=0.9))

    slice_text.set_text(f"Scanning Cross-Section: Y = {y_pos:.1f}m")

    return []

ani = animation.FuncAnimation(fig, update, frames=300, interval=ANIM_INTERVAL, blit=False)
plt.show()