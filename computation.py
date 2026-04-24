import numpy as np
import cupy as cp

xp = cp

sim_ratio = 0.57
e_mass = np.array(9.1093835611e-31, dtype=np.float64)
c = np.array(299792458, dtype=np.float64)
e_vel_x = 0.01*c
#e_vel_x = 0
e_vel_y = 0
hbar = np.array(1.054571817e-34, dtype=np.float64)
k_0_x = e_vel_x*e_mass/hbar
k_0_y = e_vel_y*e_mass/hbar
k_0 = xp.hypot(k_0_x, k_0_y)
e_wavelength = 2*np.pi/k_0
sigma = np.array(2e-9, dtype=np.float64)
cell_spacing = sigma/(20.0*4.0)
L = 12*sigma
delta_t = e_mass*cell_spacing**2/(10*hbar) # 8
sim_dims = np.array([5*L, 5*L*sim_ratio], dtype=np.float64)
x_i = sim_dims[0]/4
#x_i = sim_dims[0]/2
y_i = sim_dims[1]/2

num_cells = [int(sim_dims[0]*cell_spacing), int(sim_dims[1]/cell_spacing)]

width_cells = np.linspace(0, sim_dims[0], num_cells[0])
height_cells = np.linspace(0, sim_dims[1], num_cells[1])
A, B = np.meshgrid(width_cells, height_cells)
cell_pos = np.stack([A, B], axis=-1)

# Initial Values
psi = np.exp((1j)*(k_0_x*(cell_pos[..., 0] - x_i) + k_0_y*(cell_pos[..., 1] - y_i)) - ((cell_pos[..., 0] - x_i)**2 + (cell_pos[..., 1] - y_i)**2)/(0.2*sigma**2))
psi = xp.asarray(psi, dtype=xp.float64)

# Integrate
psi_prob_int = (xp.abs(psi)**2)
psi_prob_int = xp.sum(psi_prob_int)
psi_prob_int *= xp.asarray(cell_spacing, dtype=xp.float64)**2

# Normalize wavefunction
psi /= xp.sqrt(psi_prob_int)

V_real = xp.zeros_like(psi)

x = xp.arange(num_cells[0])
y = xp.arange(num_cells[1])

X, Y = xp.meshgrid(x, y)
mask = ((num_cells[0]/2 - 100 < X) & (X < num_cells[0]/2 + 100))
mask &= ((Y < num_cells[1]/2 - 300) | (Y > num_cells[1]/2 + 300) | ((Y < num_cells[1]/2 + 100) & (Y > num_cells[1]/2 - 100)))
infinite_P = ~mask

dx = xp.minimum(x, num_cells[0] - x - 1)
dy = xp.minimum(y, num_cells[1] - y - 1)

DX, DY = xp.meshgrid(dx, dy)
dist = xp.minimum(DX, DY)
width = 3.5*sigma / cell_spacing
width = xp.asarray(width, dtype=xp.float64)
mask = xp.clip((width - dist)/width, 0, 1)
E = 0.5 * e_mass * (e_vel_x**2 + e_vel_y**2)
E = xp.asarray(E, dtype=xp.float64)
W = 1.5*E*mask**4
V = V_real - 1j*W

kx = xp.fft.fftfreq(int(sim_dims[0]/cell_spacing), d=cell_spacing)*2*xp.pi
ky = xp.fft.fftfreq(int(sim_dims[1]/cell_spacing), d=cell_spacing)*2*xp.pi
KX, KY = xp.meshgrid(kx, ky)
k_squared = KX**2 + KY**2

psi_prob = xp.square(xp.abs(psi))
psi_vis = xp.log1p(psi_prob)
max_vis = xp.max(psi_vis)

while True:
    psi = psi*xp.exp(-1j*V*delta_t/hbar)
    psi *= infinite_P

    psi_hat = xp.fft.fft2(psi)
    psi_hat = psi_hat*xp.exp(-1j*hbar*k_squared*delta_t/(2*e_mass))
    psi = xp.fft.ifft2(psi_hat)
    psi *= infinite_P

    psi_prob = xp.square(xp.abs(psi))
    psi_vis = xp.log1p(psi_prob)
    psi_vis /= max_vis
    psi_vis = xp.power(psi_vis, 2.0)
    psi_vis = xp.clip(psi_vis, 0, 1.0)