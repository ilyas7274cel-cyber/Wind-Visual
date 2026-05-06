import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
import os
import csv
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
Q_DYN         = 0.5 * RHO * U_REF**2  # Nominal Dynamic Pressure

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
N_FRAMES      = 300
ANIM_INTERVAL = 40           # Frame delay (ms)
BG_COLOR      = '#0b0f19'
TEXT_COLOR    = '#f8fafc'
CMAP          = 'turbo'      # Professional heatmap colors

# ==============================================================
# 1. 3D SOLVER & BUILDING MASKS
# ==============================================================
print("1/4 Computing Static 3D Flow Field...")
x = np.linspace(0, LX, NX); y = np.linspace(0, LY, NY); z = np.linspace(0, LZ, NZ)
dx, dy, dz = x[1]-x[0], y[1]-y[0], z[1]-z[0]
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

building_mask = np.zeros((NX, NY, NZ), dtype=bool)
for (hx, hy, zb, zt, _) in TIERS:
    m = ((X >= CX-hx) & (X <= CX+hx) & (Y >= CY-hy) & (Y <= CY+hy) & (Z >= zb) & (Z <= zt))
    building_mask |= m

U_inlet = U_REF * (np.maximum(z, 0.1) / Z_REF)**ALPHA
U = np.zeros((NX, NY, NZ))
for iz in range(NZ): U[:, :, iz] = U_inlet[iz]

for (hx, hy, zb, zt, _) in TIERS:
    stag = (np.exp(-((X-(CX-hx))**2)/(hx*2)**2) * np.exp(-((Y-CY)**2)/hy**2) * np.exp(-((Z-(zb+zt)/2)**2)/(zt-zb)**2))
    U -= 0.8 * stag * U
    wake = (np.exp(-((X-(CX+hx+30))**2)/(hx*5)**2) * np.exp(-((Y-CY)**2)/hy**3) * np.exp(-((Z-(zb+zt)/2)**2)/(zt-zb)**2))
    U -= 0.6 * wake * U_REF

U[building_mask] = 0; U = gaussian_filter(U, 0.8); Umag_base = np.abs(U)

# Pre-Extract 2D Slice for rendering
iy = NY // 2
X_slice = X[:, iy, :]; Z_slice = Z[:, iy, :]
U_slice = Umag_base[:, iy, :]; B_mask_slice = building_mask[:, iy, :]

Cp_base = (0.5 * RHO * (U_REF**2 - Umag_base**2)) / Q_DYN
Cp_base = np.clip(Cp_base, -2.5, 1.2)
Stress_base = Cp_base * Q_DYN

# ==============================================================
# 2. EXACT SURFACE INTEGRALS (Forces & Coefficients)
# ==============================================================
print("2/4 Calculating Precise Aerodynamic Surface Integrals...")

# Identify absolute surface boundary cells mathematically
windward_mask = ~building_mask & np.roll(building_mask, shift=-1, axis=0)
leeward_mask  = ~building_mask & np.roll(building_mask, shift=1, axis=0)
roof_mask     = ~building_mask & np.roll(building_mask, shift=1, axis=2)

# Ignore ground boundaries
windward_mask[:,:,0] = False; leeward_mask[:,:,0] = False; roof_mask[:,:,0] = False

# Calculate Base Static Forces
A_ref = np.sum(windward_mask) * dy * dz
L_ref = 180.0  # Total Height

F_D_static = np.sum(Stress_base[windward_mask]) * dy * dz - np.sum(Stress_base[leeward_mask]) * dy * dz
F_L_static = np.sum(-Stress_base[roof_mask]) * dx * dy  # Suction pulls UP (positive lift)
M_static   = np.sum(Stress_base[windward_mask] * Z[windward_mask]) * dy * dz - np.sum(Stress_base[leeward_mask] * Z[leeward_mask]) * dy * dz

C_D_static = F_D_static / (Q_DYN * A_ref)
C_L_static = F_L_static / (Q_DYN * A_ref)
C_M_static = M_static / (Q_DYN * A_ref * L_ref)

# Compute 1D Force Profile for Chart (Z vs kN/m)
force_z = np.zeros(NZ)
for iz in range(NZ):
    z_mask = windward_mask[:,:,iz]
    l_mask = leeward_mask[:,:,iz]
    f_z = np.sum(Stress_base[:,:,iz][z_mask]) * dy - np.sum(Stress_base[:,:,iz][l_mask]) * dy
    force_z[iz] = max(f_z / 1000.0, 0)

# ==============================================================
# 3. GENERATE TIME-SERIES ARRAYS & EXPORT CSV
# ==============================================================
print("3/4 Generating Time-Series Data and Exporting CSV...")

time_arr = np.arange(N_FRAMES) * (ANIM_INTERVAL / 1000.0)
# Instantaneous Global Gust Multiplier based on the wave crossing the building's center
gust_wave_center = 0.15 * np.sin(0.04 * CX - 0.3 * np.arange(N_FRAMES)) + 0.05 * np.sin(0.1 * CX - 0.5 * np.arange(N_FRAMES))
gust_factor = (1.0 + gust_wave_center)**2

# Dynamic Arrays
CD_arr = C_D_static * gust_factor
CL_arr = C_L_static * gust_factor
CM_arr = C_M_static * gust_factor
FD_arr = (F_D_static / 1000.0) * gust_factor  # kN
FL_arr = (F_L_static / 1000.0) * gust_factor  # kN
M_arr  = (M_static / 1000.0) * gust_factor    # kN-m

# Export CSV
csv_filename = "wind_cfd_data_export.csv"
i = 0
while os.path.exists(f"wind_cfd_data_export_{i}.csv"): i += 1
csv_filename = f"wind_cfd_data_export_{i}.csv"

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame', 'Time_s', 'Gust_Factor', 'Drag_kN', 'Lift_kN', 'Moment_kNm', 'C_D', 'C_L', 'C_M'])
    for idx in range(N_FRAMES):
        writer.writerow([idx, time_arr[idx], gust_factor[idx], FD_arr[idx], FL_arr[idx], M_arr[idx], CD_arr[idx], CL_arr[idx], CM_arr[idx]])

print(f"    --> Saved exact frame-by-frame data to: {csv_filename}")

# ==============================================================
# 4. 6-PANEL UI DASHBOARD SETUP
# ==============================================================
print("4/4 Initializing UI Rendering Engine...")
plt.rcParams.update({'text.color': TEXT_COLOR, 'axes.labelcolor': TEXT_COLOR, 'axes.edgecolor': '#334155', 'xtick.color': TEXT_COLOR, 'ytick.color': TEXT_COLOR})

fig = plt.figure(figsize=(24, 11), facecolor=BG_COLOR)
fig.suptitle(f"CFD WIND LOAD ANALYSIS  |  U_ref = {U_REF} m/s  |  A_ref = {A_ref:,.0f} m²", fontsize=18, fontweight='bold', color='#00f2ff', y=0.97)

gs = gridspec.GridSpec(2, 3, wspace=0.2, hspace=0.35, left=0.04, right=0.96, top=0.90, bottom=0.08)

ax_vel    = fig.add_subplot(gs[0, 0], facecolor=BG_COLOR)
ax_cp     = fig.add_subplot(gs[0, 1], facecolor=BG_COLOR)
ax_stress = fig.add_subplot(gs[0, 2], facecolor=BG_COLOR)
ax_forcez = fig.add_subplot(gs[1, 0], facecolor=BG_COLOR)
ax_coef   = fig.add_subplot(gs[1, 1], facecolor=BG_COLOR)
ax_totf   = fig.add_subplot(gs[1, 2], facecolor=BG_COLOR)

norm_u      = Normalize(vmin=0, vmax=U_REF * 1.35)
norm_cp     = Normalize(vmin=-2.5, vmax=1.2)
norm_stress = Normalize(vmin=-2.5 * Q_DYN, vmax=1.2 * Q_DYN)

# --- Bottom Left: Static Drag Profile ---
ax_forcez.plot(force_z, z, color='#00f2ff', linewidth=3)
ax_forcez.fill_betweenx(z, 0, force_z, color='#00f2ff', alpha=0.2)
ax_forcez.set_title("Base Drag Force Profile", color='#f8fafc')
ax_forcez.set_xlabel("Drag [kN/m]"); ax_forcez.set_ylabel("Z [m]")
ax_forcez.set_ylim(0, LZ); ax_forcez.set_xlim(0, max(force_z) * 1.2)
ax_forcez.grid(color='#1e293b', alpha=0.6, linestyle='--')
for (_, _, zb, zt, label) in TIERS: ax_forcez.axhline(zt, color='#475569', linestyle=':')

# --- Bottom Mid: Aerodynamic Coefficients ---
ax_coef.set_title("Live Aerodynamic Coefficients", color='#f8fafc')
ax_coef.set_xlabel("Time [s]"); ax_coef.set_ylabel("Coefficient Value")
ax_coef.set_xlim(0, time_arr[-1])
ax_coef.set_ylim(min(CL_arr)*1.5, max(CD_arr)*1.5)
ax_coef.grid(color='#1e293b', alpha=0.6, linestyle='--')

line_cd, = ax_coef.plot([], [], color='#ff4757', lw=2, label='Drag (Cd)')
line_cl, = ax_coef.plot([], [], color='#2ed573', lw=2, label='Lift (Cl)')
line_cm, = ax_coef.plot([], [], color='#1e90ff', lw=2, label='Moment (Cm)')
ax_coef.legend(loc='upper right', facecolor=BG_COLOR, edgecolor='#334155', labelcolor=TEXT_COLOR)

# --- Bottom Right: Global Forces ---
ax_totf.set_title("Instantaneous Global Forces [kN]", color='#f8fafc')
ax_totf.set_xlabel("Time [s]"); ax_totf.set_ylabel("Force [kN]")
ax_totf.set_xlim(0, time_arr[-1])
ax_totf.set_ylim(min(FL_arr)*1.5, max(FD_arr)*1.2)
ax_totf.grid(color='#1e293b', alpha=0.6, linestyle='--')

line_fd, = ax_totf.plot([], [], color='#ff7f50', lw=2.5, label='Total Drag [kN]')
line_fl, = ax_totf.plot([], [], color='#7bed9f', lw=2.5, label='Total Lift [kN]')
ax_totf.legend(loc='center right', facecolor=BG_COLOR, edgecolor='#334155', labelcolor=TEXT_COLOR)

# ==============================================================
# 5. ANIMATION LOOP
# ==============================================================
def update(frame):
    for ax in [ax_vel, ax_cp, ax_stress]:
        for c in ax.collections: c.remove()
        [p.remove() for p in reversed(ax.patches)]
    
    # Apply gust to 2D slice matrices
    gust_wave = 0.15 * np.sin(0.04 * X_slice - 0.3 * frame) + 0.05 * np.sin(0.1 * X_slice - 0.5 * frame)
    U_anim = U_slice * (1.0 + gust_wave)
    U_anim[B_mask_slice] = 0 
    
    Cp_anim = (0.5 * RHO * (U_REF**2 - U_anim**2)) / Q_DYN
    Cp_anim = np.clip(Cp_anim, -2.5, 1.2); Cp_anim[B_mask_slice] = 1.0  
    Stress_anim = Cp_anim * Q_DYN
    
    # Redraw Heatmaps
    ax_vel.contourf(X_slice, Z_slice, U_anim, levels=45, cmap=CMAP, norm=norm_u)
    ax_cp.contourf(X_slice, Z_slice, Cp_anim, levels=45, cmap=CMAP, norm=norm_cp)
    ax_stress.contourf(X_slice, Z_slice, Stress_anim, levels=45, cmap=CMAP, norm=norm_stress)
    
    for ax, title in zip([ax_vel, ax_cp, ax_stress], ["Velocity Magnitude [m/s]", "Pressure Coefficient (Cp)", "Wind Stress Distribution [Pa]"]):
        ax.set_title(title, color=TEXT_COLOR, fontsize=12, pad=10)
        for (hx, hy, zb, zt, _) in TIERS: ax.add_patch(Rectangle((CX-hx, zb), hx*2, zt-zb, facecolor='#0f172a', edgecolor='#38bdf8', lw=1.5, alpha=1.0))

    # Update Line Charts efficiently
    line_cd.set_data(time_arr[:frame+1], CD_arr[:frame+1])
    line_cl.set_data(time_arr[:frame+1], CL_arr[:frame+1])
    line_cm.set_data(time_arr[:frame+1], CM_arr[:frame+1])
    
    line_fd.set_data(time_arr[:frame+1], FD_arr[:frame+1])
    line_fl.set_data(time_arr[:frame+1], FL_arr[:frame+1])

    return [line_cd, line_cl, line_cm, line_fd, line_fl]

ani = animation.FuncAnimation(fig, update, frames=N_FRAMES, interval=ANIM_INTERVAL, blit=False)

# -------------------------------
# SAVE AS GIF
# -------------------------------
print("Rendering animation... This will take a moment.")
gif_filename = f"wind_cfd_animation_{i}.gif"
ani.save(gif_filename, writer="pillow", fps=25, dpi=100)
print(f"    --> Saved visual animation to: {gif_filename}")

plt.show()