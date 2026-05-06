import numpy as np
import pyvista as pv

# -------------------------------
# DOMAIN
# -------------------------------
nx, ny, nz = 50, 25, 25
Lx, Ly, Lz = 10.0, 5.0, 5.0

dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
dt = 0.001  # time step (VERY IMPORTANT)

# Fields
u = np.ones((nx, ny, nz)) * 5.0
v = np.zeros_like(u)
w = np.zeros_like(u)
p = np.zeros_like(u)

k = np.ones_like(u) * 0.05
eps = np.ones_like(u) * 0.01

rho = 1.225
nu = 1.5e-5

# Relaxation factors (CRITICAL)
alpha_u = 0.5
alpha_k = 0.5
alpha_eps = 0.5

# -------------------------------
# BUILDING
# -------------------------------
solid = np.zeros_like(u, dtype=bool)
solid[20:30, 10:15, 0:15] = True

# -------------------------------
# k-epsilon constants
# -------------------------------
C_mu = 0.09
C1 = 1.44
C2 = 1.92

nt = 200

for t in range(nt):

    u_old = u.copy()

    # Turbulent viscosity (SAFE)
    nu_t = C_mu * (k**2 / (eps + 1e-5))
    nu_t = np.clip(nu_t, 0, 1.0)

    nu_eff = nu + nu_t

    # -------------------------------
    # Momentum (STABLE FORM)
    # -------------------------------
    convection = (u_old[1:-1,1:-1,1:-1] *
                 (u_old[1:-1,1:-1,1:-1] - u_old[:-2,1:-1,1:-1]) / dx)

    diffusion = nu_eff[1:-1,1:-1,1:-1] * (
        (u_old[2:,1:-1,1:-1] - 2*u_old[1:-1,1:-1,1:-1] + u_old[:-2,1:-1,1:-1]) / dx**2
    )

    u_new = u_old[1:-1,1:-1,1:-1] + dt * (-convection + diffusion)

    # Under-relaxation
    u[1:-1,1:-1,1:-1] = (1-alpha_u)*u_old[1:-1,1:-1,1:-1] + alpha_u*u_new

    # -------------------------------
    # Turbulence equations (STABLE)
    # -------------------------------
    production = nu_t * (u**2)

    k_new = k + dt * (production - eps)
    eps_new = eps + dt * (C1*production - C2*eps)

    # Under-relaxation
    k = (1-alpha_k)*k + alpha_k*k_new
    eps = (1-alpha_eps)*eps + alpha_eps*eps_new

    # CLIPPING (VERY IMPORTANT)
    k = np.clip(k, 1e-6, 10)
    eps = np.clip(eps, 1e-6, 10)

    # -------------------------------
    # Boundary conditions
    # -------------------------------
    u[0,:,:] = 5.0
    u[-1,:,:] = u[-2,:,:]

    u[solid] = 0
    v[solid] = 0
    w[solid] = 0

    if t % 20 == 0:
        print(f"Step {t}/{nt}")

# -------------------------------
# VISUALIZATION
# -------------------------------
grid = pv.ImageData()

grid.dimensions = np.array(u.shape) + 1
grid.spacing = (dx, dy, dz)
grid.origin = (0, 0, 0)

vel_mag = np.sqrt(u**2 + v**2 + w**2)

grid.cell_data["velocity"] = vel_mag.flatten(order="F")
grid.cell_data["pressure"] = p.flatten(order="F")

plotter = pv.Plotter()
plotter.add_volume(grid, scalars="velocity", opacity="sigmoid")
plotter.add_axes()
plotter.show()