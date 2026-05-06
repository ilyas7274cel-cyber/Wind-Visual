import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

# -------------------------------
# DOMAIN
# -------------------------------
Lx, Ly, Lz = 200, 120, 120
Nx, Ny, Nz = 60, 40, 40

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
z = np.linspace(0, Lz, Nz)

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# -------------------------------
# BUILDING (simple cuboid)
# -------------------------------
cx, cy = Lx * 0.4, Ly * 0.5
bx, by, bz = 15, 15, 80

building = ((X > cx-bx) & (X < cx+bx) &
            (Y > cy-by) & (Y < cy+by) &
            (Z < bz))

# -------------------------------
# FLOW FIELD (improved analytic RANS-style)
# -------------------------------
U_ref = 10.0

U = np.ones_like(X) * U_ref
V = np.zeros_like(U)
W = np.zeros_like(U)

# Wake + curvature effects
wake = np.exp(-((X-(cx+bx))**2)/800 - ((Y-cy)**2)/400 - ((Z-40)**2)/900)
U -= 5 * wake
W += 2 * wake

# Side flow
side = np.exp(-((Y-(cy+by))**2)/200) + np.exp(-((Y-(cy-by))**2)/200)
V += 2 * side

# Smooth field
U = gaussian_filter(U, 1.2)
V = gaussian_filter(V, 1.2)
W = gaussian_filter(W, 1.2)

U[building] = 0
V[building] = 0
W[building] = 0

Umag = np.sqrt(U**2 + V**2 + W**2)
Umag = np.clip(Umag, 0, 2*U_ref)

# -------------------------------
# TURBULENCE (k-epsilon approx)
# -------------------------------
k_field = 0.05 * Umag**2
eps_field = 0.1 * k_field

k_field = np.clip(k_field, 1e-6, 20)
eps_field = np.clip(eps_field, 1e-6, 50)

# -------------------------------
# INTERPOLATORS
# -------------------------------
interp_U = RegularGridInterpolator((x,y,z), U, bounds_error=False, fill_value=0)
interp_V = RegularGridInterpolator((x,y,z), V, bounds_error=False, fill_value=0)
interp_W = RegularGridInterpolator((x,y,z), W, bounds_error=False, fill_value=0)

def get_vel(p):
    u = interp_U(p)
    v = interp_V(p)
    w = interp_W(p)
    return np.column_stack([u,v,w])

# -------------------------------
# RK4 PARTICLE INTEGRATION
# -------------------------------
def rk4(p, dt):
    k1 = get_vel(p)
    k2 = get_vel(p + 0.5*dt*k1)
    k3 = get_vel(p + 0.5*dt*k2)
    k4 = get_vel(p + dt*k3)
    return p + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

# -------------------------------
# PARTICLES
# -------------------------------
N = 150
px = np.random.uniform(0, 5, N)
py = np.random.uniform(10, Ly-10, N)
pz = np.random.uniform(5, Lz-10, N)

trail_len = 12
trail_x = np.full((N, trail_len), np.nan)
trail_y = np.full((N, trail_len), np.nan)
trail_z = np.full((N, trail_len), np.nan)

# -------------------------------
# FIGURE
# -------------------------------
fig = plt.figure(figsize=(14,8))
ax = fig.add_subplot(111, projection='3d')

ax.set_facecolor('#0a0f1a')
fig.patch.set_facecolor('#0a0f1a')

# Building
ax.bar3d(cx-bx, cy-by, 0, 2*bx, 2*by, bz,
         color='cyan', alpha=0.3)

sc = ax.scatter([],[],[], c=[], cmap='jet', s=10)

lines = []
for _ in range(N):
    l, = ax.plot([],[],[], color='white', alpha=0.2)
    lines.append(l)

ax.set_xlim(0,Lx)
ax.set_ylim(0,Ly)
ax.set_zlim(0,Lz)

# -------------------------------
# ANIMATION
# -------------------------------
def update(frame):
    global px, py, pz
    global trail_x, trail_y, trail_z

    pts = np.column_stack([px,py,pz])

    # RK4 + substeps
    for _ in range(4):
        pts = rk4(pts, 0.5)

    px, py, pz = pts[:,0], pts[:,1], pts[:,2]

    # reset particles
    mask = (px>Lx)|(py<0)|(py>Ly)|(pz<0)|(pz>Lz)
    px[mask] = np.random.uniform(0,5,np.sum(mask))
    py[mask] = np.random.uniform(10,Ly-10,np.sum(mask))
    pz[mask] = np.random.uniform(5,Lz-10,np.sum(mask))

    # update trails
    trail_x = np.roll(trail_x,1,axis=1)
    trail_y = np.roll(trail_y,1,axis=1)
    trail_z = np.roll(trail_z,1,axis=1)

    trail_x[:,0] = px
    trail_y[:,0] = py
    trail_z[:,0] = pz

    # smooth trails
    trail_x[:] = gaussian_filter(trail_x,0.7)
    trail_y[:] = gaussian_filter(trail_y,0.7)
    trail_z[:] = gaussian_filter(trail_z,0.7)

    # velocity coloring
    vel = get_vel(pts)
    speed = np.linalg.norm(vel,axis=1)

    sc._offsets3d = (px,py,pz)
    sc.set_array(speed)
    sc.set_sizes(10 + 30*(speed/U_ref))

    # update lines
    for i,l in enumerate(lines):
        l.set_data(trail_x[i], trail_y[i])
        l.set_3d_properties(trail_z[i])

    ax.view_init(elev=25, azim=frame*0.4)

    return [sc] + lines

ani = animation.FuncAnimation(fig, update, interval=40)

plt.show()