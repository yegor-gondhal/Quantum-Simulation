import numpy as np
import cupy as cp
import time

xp = cp
np_prec = np.float64
xp_prec = xp.float64
xp_complex = xp.complex128

e_mass = np.array(9.1093835611e-31, dtype=np_prec)
c = np.array(299792458, dtype=np_prec)
e_vel_x = 0.01*c
#e_vel_x = 0
e_vel_y = 0
hbar = np.array(1.054571817e-34, dtype=np_prec)
k_0_x = e_vel_x*e_mass/hbar
k_0_y = e_vel_y*e_mass/hbar
k_0 = xp.hypot(k_0_x, k_0_y)
e_wavelength = 2*np.pi/k_0
sigma = np.array(2e-9, dtype=np_prec)
L = 12*sigma
num_cells = [7776, 4374]
#num_cells = [4608, 2592]
sim_ratio = num_cells[1]/num_cells[0]
sim_dims = np.array([10*L, 10*L*sim_ratio], dtype=np_prec)
cell_spacing = sim_dims[0]/num_cells[0]
delta_t = e_mass*cell_spacing**2/(15*hbar)
x_i = sim_dims[0]/4
y_i = sim_dims[1]/2

width_cells = np.linspace(0, sim_dims[0], num_cells[0])
height_cells = np.linspace(0, sim_dims[1], num_cells[1])
A, B = np.meshgrid(width_cells, height_cells)
cell_pos = np.stack([A, B], axis=-1)

# Initial Values
psi = np.exp((1j)*(k_0_x*(cell_pos[..., 0] - x_i) + k_0_y*(cell_pos[..., 1] - y_i)) - ((cell_pos[..., 0] - x_i)**2 + (cell_pos[..., 1] - y_i)**2)/(0.2*sigma**2))
psi = xp.asarray(psi, dtype=xp_complex)

# Integrate
psi_prob_int = (xp.abs(psi)**2)
psi_prob_int = xp.sum(psi_prob_int)
psi_prob_int *= xp.asarray(cell_spacing, dtype=xp_prec)**2

# Normalize wavefunction
psi /= xp.sqrt(psi_prob_int)

V_real = xp.zeros_like(psi)

E = 0.5 * e_mass * (e_vel_x**2 + e_vel_y**2) + e_mass * c**2
E = xp.asarray(E, dtype=xp_prec)

x = xp.arange(num_cells[0])
y = xp.arange(num_cells[1])

X, Y = xp.meshgrid(x, y)
mask = ((num_cells[0]/2 - 100 < X) & (X < num_cells[0]/2 + 100))
mask &= ((Y < num_cells[1]/2 - 60) | (Y > num_cells[1]/2 + 60) | ((Y < num_cells[1]/2 + 20) & (Y > num_cells[1]/2 - 20)))
infinite_P = ~mask
V_real[mask] = -1j * 100 * E

dx = xp.minimum(x, num_cells[0] - x - 1)
dy = xp.minimum(y, num_cells[1] - y - 1)

DX, DY = xp.meshgrid(dx, dy)
dist = xp.minimum(DX, DY)
width = 3.5*sigma / cell_spacing
width = xp.asarray(width, dtype=xp_prec)
mask = xp.clip((width - dist)/width, 0, 1)
W = 1*E*mask**4
V = V_real - 1j*W

kx = xp.fft.fftfreq(num_cells[0], d=cell_spacing)*2*xp.pi
ky = xp.fft.fftfreq(num_cells[1], d=cell_spacing)*2*xp.pi
KX, KY = xp.meshgrid(kx, ky)
k_squared = KX**2 + KY**2

psi_prob = xp.square(xp.abs(psi))
psi_vis = xp.log1p(psi_prob)
max_vis = xp.max(psi_vis)

delta_t = xp.asarray(delta_t,dtype=xp_prec)
hbar = xp.asarray(hbar, dtype=xp_prec)


first_po = xp.exp(-1j * V * delta_t / hbar)
first_ki = xp.exp(-1j * hbar * k_squared * delta_t / (2 * e_mass))
def first_order(psi):
    psi *= first_po
    #psi *= infinite_P

    psi_hat = xp.fft.fft2(psi)
    psi_hat *= first_ki
    psi = xp.fft.ifft2(psi_hat)
    #psi *= infinite_P

    return psi

second_po = xp.exp(-1j * V * delta_t / (2*hbar))
def second_order(psi):
    psi *= second_po
    #psi *= infinite_P

    psi_hat = xp.fft.fft2(psi)
    psi_hat *= first_ki
    psi = xp.fft.ifft2(psi_hat)
    #psi *= infinite_P

    psi *= second_po
    #psi *= infinite_P

    return psi

a = xp.power(2, 1/3)
p = 1/(2 - a)
q = -a/(2 - a)
third_po_p = xp.exp(-1j * V * delta_t * p / (2*hbar))
third_po_q = xp.exp(-1j * V * delta_t * q / (2*hbar))
third_ki_p = xp.exp(-1j * hbar * k_squared * delta_t * p / (2 * e_mass))
third_ki_q = xp.exp(-1j * hbar * k_squared * delta_t * q / (2 * e_mass))
def fourth_order(psi):
    psi *= third_po_p
    #psi *= infinite_P

    psi_hat = xp.fft.fft2(psi)
    psi_hat *= third_ki_p
    psi = xp.fft.ifft2(psi_hat)
    #psi *= infinite_P

    psi *= third_po_p
    #psi *= infinite_P

    psi *= third_po_q
    #psi *= infinite_P

    psi_hat = xp.fft.fft2(psi)
    psi_hat *= third_ki_q
    psi = xp.fft.ifft2(psi_hat)
    #psi *= infinite_P

    psi *= third_po_q
    #psi *= infinite_P

    psi *= third_po_p
    #psi *= infinite_P

    psi_hat = xp.fft.fft2(psi)
    psi_hat *= third_ki_p
    psi = xp.fft.ifft2(psi_hat)
    #psi *= infinite_P

    psi *= third_po_p
    #psi *= infinite_P

    return psi

num_frames_saved = 8000 #8000
start_offset = 1000 # 1000
frame = 0
save_every = 10
buffer_counter = 0
if num_frames_saved < 25:
    buffer_size = num_frames_saved
else:
    buffer_size = 25
write_index = 0
H, W = psi.shape
buffer = xp.zeros((buffer_size, H, W), dtype=xp.float16)
output = np.lib.format.open_memmap(
    "psi_vis_output.npy",
    mode="w+",
    dtype=np.float16,
    shape=(num_frames_saved, H, W)
)
t1 = time.time()
while frame < (save_every*(num_frames_saved+start_offset)):

    if frame%100 == 0:
        t2 = time.time()
        print(100*frame/(save_every*(num_frames_saved+start_offset)), "%")
        print(t2-t1, "\n")

    psi = fourth_order(psi)

    if frame % save_every == 0:
        if frame/save_every < start_offset:
            frame += 1
            continue
        psi_prob = xp.square(xp.abs(psi))
        psi_vis = xp.log1p(psi_prob)
        psi_vis /= max_vis
        psi_vis = xp.power(psi_vis, 2.0)
        psi_vis = xp.clip(psi_vis, 0, 1.0)
        save_frame = int(frame/save_every)
        buffer[save_frame%buffer_size] = psi_vis.astype(xp.float16)
        buffer_counter += 1
        if buffer_counter == buffer_size:
            output[write_index:write_index+buffer_size] = xp.asnumpy(buffer)
            write_index += buffer_size
            buffer_counter = 0

    frame += 1

t3 = time.time()
print("Total Time: ", t3-t1)
np.savez(
    "sim_params.npz",
    sim_dims=sim_dims,
    H=H,
    W=W,
    max_frames=output.shape[0]
)