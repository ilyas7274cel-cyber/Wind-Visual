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
U_REF         = 60.0         # Reference wind speed at 10m (m/s)
Z_REF         = 10.0         # Reference height (m)
ALPHA         = 0.22         # ABL Power-law exponent
RHO           = 1.225        # Air density (kg/m^3)
Q_DYN         = 0.5 * RHO * U_REF**2  # Dynamic Pressure

# Domain Dimensions
LX, LY, LZ    = 400.0, 200.0, 250.0
NX, NY, NZ    = 80, 50, 60   # Grid resolution

# Building Geometry (Setback Tiers)
CX, CY        = LX * 0.35, LY * 0.5  
TIERS = [
    (15.0, 15.0, 0.0, 120.0,   "Tier 1 (Base)"),
    (12.5, 12.5, 120.0, 150.0,  "Tier 2 (Mid)"),
    (10.0, 10.0, 150.0, 180.0, "Tier 3 (Top)")
]

# Visualization Settings
ANIM_INTERVAL = 40           # Frame delay (ms)
BG_COLOR      = '#0b0f19'
TEXT_COLOR    = '#f8fafc'
CMAP          = 'turbo'      # Professional heatmap colors

# ==============================================================
# 1. GRID & BASE FLOW SOLVER (Steady State)
# ==============================================================
print("Computing Static Flow Field and Building Mask...")
x = np.linspace(0, LX, NX)
y = np.linspace(0, LY, NY)
z = np.linspace(0, LZ, NZ)
dx, dy, dz = x[1]-x[0], y[1]-y[0], z[1]-z[0]
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Create Static Building Mask
building_mask = np.zeros((NX, NY, NZ), dtype=bool)
for (hx, hy, zb, zt, _) in TIERS:
    m = ((X >= CX-hx) & (X <= CX+hx) & (Y >= CY-hy) & (Y <= CY+hy) & (Z >= zb) & (Z <= zt))
    building_mask |= m

# Base Wind Field
U_inlet = U_REF * (np.maximum(z, 0.1) / Z_REF)**ALPHA
U = np.zeros((NX, NY, NZ))
for iz in range(NZ): U[:, :, iz] = U_inlet[iz]

# Apply Static Physics Disturbances (Stagnation & Wake)
for (hx, hy, zb, zt, _) in TIERS:
    stag = (np.exp(-((X-(CX-hx))**2)/(hx*2)**2) * np.exp(-((Y-CY)**2)/hy**2) * np.exp(-((Z-(zb+zt)/2)**2)/(zt-zb)**2))
    U -= 0.8 * stag * U
    wake = (np.exp(-((X-(CX+hx+30))**2)/(hx*5)**2) * np.exp(-((Y-CY)**2)/hy**3) * np.exp(-((Z-(zb+zt)/2)**2)/(zt-zb)**2))
    U -= 0.6 * wake * U_REF

U[building_mask] = 0
U = gaussian_filter(U, 0.8)
Umag_base = np.abs(U)

# Extract 2D Slice EXACTLY at the center
iy = NY // 2
X_slice = X[:, iy, :]
Z_slice = Z[:, iy, :]
U_slice = Umag_base[:, iy, :]
B_mask_slice = building_mask[:, iy, :]

# Calculate Pressures
Cp_base = (0.5 * RHO * (U_REF**2 - Umag_base**2)) / Q_DYN
Cp_base = np.clip(Cp_base, -2.5, 1.2)
Stress_base = Cp_base * Q_DYN

# -----------------------------------------------------------
# CORRECTED DRAG FORCE ALGORITHM (Offset Probing)
# -----------------------------------------------------------
force_z = np.zeros(NZ)
total_base_shear = 0

for iz in range(NZ):
    z_val = z[iz]
    for (hx, hy, zb, zt, _) in TIERS:
        if zb <= z_val < zt:
            # Probe 1.5 grid cells OUTSIDE the geometry to prevent reading internal solid data
            ix_w = np.argmin(np.abs(x - (CX - hx - dx*1.5))) 
            ix_l = np.argmin(np.abs(x - (CX + hx + dx*1.5))) 
            
            iy_mask = (y >= CY - hy) & (y <= CY + hy)
            
            P_w = Stress_base[ix_w, iy_mask, iz]
            P_l = Stress_base[ix_l, iy_mask, iz]
            
            # Integrate the pressure differential across the width of the tier
            force_per_meter = np.trapz(P_w - P_l, y[iy_mask]) / 1000.0  
            force_z[iz] = max(force_per_meter, 0) # Ensure no negative drag anomalies
            total_base_shear += force_z[iz] * dz
            break

# ==============================================================
# 2. VISUALIZATION & UI SETUP
# ==============================================================
plt.rcParams.update({
    'text.color': TEXT_COLOR, 'axes.labelcolor': TEXT_COLOR, 
    'axes.edgecolor': '#334155', 'xtick.color': TEXT_COLOR, 'ytick.color': TEXT_COLOR
})

fig = plt.figure(figsize=(18, 10), facecolor=BG_COLOR)
fig.suptitle(f"WIND LOAD ANALYSIS  |  U_ref = {U_REF} m/s  |  Total Base Shear = {total_base_shear:,.0f} kN", 
             fontsize=16, fontweight='bold', color='#00f2ff', y=0.96)

gs = gridspec.GridSpec(2, 2, wspace=0.15, hspace=0.35, left=0.05, right=0.95, top=0.88, bottom=0.08)

ax_vel    = fig.add_subplot(gs[0, 0], facecolor=BG_COLOR)
ax_cp     = fig.add_subplot(gs[0, 1], facecolor=BG_COLOR)
ax_stress = fig.add_subplot(gs[1, 0], facecolor=BG_COLOR)
ax_force  = fig.add_subplot(gs[1, 1], facecolor=BG_COLOR)

norm_u      = Normalize(vmin=0, vmax=U_REF * 1.35)
norm_cp     = Normalize(vmin=-2.5, vmax=1.2)
norm_stress = Normalize(vmin=-2.5 * Q_DYN, vmax=1.2 * Q_DYN)

cb_vel = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=norm_u), ax=ax_vel, pad=0.02)
cb_vel.set_label('Wind Velocity [m/s]', color=TEXT_COLOR)
cb_cp = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=norm_cp), ax=ax_cp, pad=0.02)
cb_cp.set_label('Pressure Coefficient (Cp)', color=TEXT_COLOR)
cb_stress = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=norm_stress), ax=ax_stress, pad=0.02)
cb_stress.set_label('Wind Stress [Pascals]', color=TEXT_COLOR)

for cb in [cb_vel, cb_cp, cb_stress]: cb.ax.yaxis.set_tick_params(color=TEXT_COLOR)

# Setup 4th Panel (Static Force Profile)
ax_force.plot(force_z, z, color='#00f2ff', linewidth=3)
ax_force.fill_betweenx(z, 0, force_z, color='#00f2ff', alpha=0.2)
ax_force.set_title("Total Building Drag Force Profile (Corrected)", color='#f8fafc', fontsize=12, pad=10)
ax_force.set_xlabel("Drag Force per meter height [kN/m]")
ax_force.set_ylabel("Elevation (Z) [m]")
ax_force.set_ylim(0, LZ)
ax_force.set_xlim(0, max(force_z) * 1.2)
ax_force.grid(color='#1e293b', alpha=0.6, linestyle='--')
for (_, _, zb, zt, label) in TIERS:
    ax_force.axhline(zt, color='#475569', linestyle=':', linewidth=1.5)
    if max(force_z) > 0: ax_force.text(max(force_z)*0.8, (zb+zt)/2, label, color='#94a3b8', va='center')

# ==============================================================
# 3. ANIMATION LOOP (Traveling Wind Gusts left-to-right)
# ==============================================================
def update(frame):
    for ax in [ax_vel, ax_cp, ax_stress]:
        for c in ax.collections: c.remove()
        [p.remove() for p in reversed(ax.patches)]
    
    # Mathematical Traveling Wave
    gust_wave = 0.15 * np.sin(0.04 * X_slice - 0.3 * frame) + 0.05 * np.sin(0.1 * X_slice - 0.5 * frame)
    
    U_anim = U_slice * (1.0 + gust_wave)
    U_anim[B_mask_slice] = 0 
    
    Cp_anim = (0.5 * RHO * (U_REF**2 - U_anim**2)) / Q_DYN
    Cp_anim = np.clip(Cp_anim, -2.5, 1.2)
    Cp_anim[B_mask_slice] = 1.0  
    
    Stress_anim = Cp_anim * Q_DYN
    
    ax_vel.contourf(X_slice, Z_slice, U_anim, levels=45, cmap=CMAP, norm=norm_u)
    ax_cp.contourf(X_slice, Z_slice, Cp_anim, levels=45, cmap=CMAP, norm=norm_cp)
    ax_stress.contourf(X_slice, Z_slice, Stress_anim, levels=45, cmap=CMAP, norm=norm_stress)
    
    for ax, title in zip([ax_vel, ax_cp, ax_stress], ["Velocity Magnitude", "Pressure Coefficient (Cp)", "Wind Stress (Pascals)"]):
        ax.set_title(title, color=TEXT_COLOR, fontsize=12, pad=10)
        ax.set_xlabel("X [m]"); ax.set_ylabel("Z [m]")
        for (hx, hy, zb, zt, _) in TIERS:
            ax.add_patch(Rectangle((CX-hx, zb), hx*2, zt-zb, facecolor='#0f172a', edgecolor='#38bdf8', lw=1.5, alpha=1.0))

    return []

# ani = animation.FuncAnimation(fig, update, frames=300, interval=ANIM_INTERVAL, blit=False)
# plt.show()

ani = animation.FuncAnimation(
    fig,
    update,
    frames=300,
    interval=ANIM_INTERVAL,
    blit=False
)

# -------------------------------
# SAVE AS GIF
# -------------------------------
print("Saving animation as GIF...")


import os
i = 0
while os.path.exists(f"wind_cfd_animation_{i}.gif"):
    i += 1

ani.save(
    f"wind_cfd_animation_{i}.gif",
    writer="pillow",
    fps=25,
    dpi=100
)

print("Saved successfully: wind_cfd_animation.gif")

plt.show()