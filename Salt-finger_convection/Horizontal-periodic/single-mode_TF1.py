import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3

# Parameters
kx, ky = 4, 0   # wavenumber
Lz = 1.         # computational domain
Nz = 128        # number of points
dealias = 3/2   # scaling factor

Ra = 1e5        # Rayleigh number
Pr = 7.0       # Prandtl number
tau = 0.01      # diffusivity ratio = ks/kt
Rp = 40.0       # density ratio

# Bases
coords = d3.Coordinate('z')
dist = d3.Distributor(coords, dtype=np.complex128)

# define the coordinate system
zbasis = d3.Chebyshev(coords, size=Nz, bounds=(0, Lz), dealias=dealias)

# define fields for single-mode equations
barU0   = dist.Field(name='barU0', bases=(zbasis)) 
hatu    = dist.Field(name='hatu', bases=(zbasis)) 
hatw    = dist.Field(name='hatw', bases=(zbasis)) 
hatzeta = dist.Field(name='hatzeta', bases=(zbasis)) 
barT0   = dist.Field(name='barT0', bases=(zbasis)) 
hatT    = dist.Field(name='hatT', bases=(zbasis)) 
barS0   = dist.Field(name='barS0', bases=(zbasis)) 
hatS    = dist.Field(name='hatS', bases=(zbasis)) 

tau_barU0_1   = dist.Field(name='tau_barU0_1')
tau_barU0_2   = dist.Field(name='tau_barU0_2')
tau_hatw_1    = dist.Field(name='tau_hatw_1')
tau_hatw_2    = dist.Field(name='tau_hatw_2')
tau_hatw_3    = dist.Field(name='tau_hatw_3')
tau_hatw_4    = dist.Field(name='tau_hatw_4')
tau_hatzeta_1 = dist.Field(name='tau_hatzeta_1') 
tau_hatzeta_2 = dist.Field(name='tau_hatzeta_2')
tau_barT0_1   = dist.Field(name='tau_barT0_1') 
tau_barT0_2   = dist.Field(name='tau_barT0_2')
tau_hatT_1    = dist.Field(name='tau_hatT_1')
tau_hatT_2    = dist.Field(name='tau_hatT_2')
tau_barS0_1   = dist.Field(name='tau_barS0_1')
tau_barS0_2   = dist.Field(name='tau_barS0_2')
tau_hatS_1    = dist.Field(name='tau_hatS_1')
tau_hatS_2    = dist.Field(name='tau_hatS_2')

# Substitutions
z = dist.local_grids(zbasis) # get coordinate arrays in horizontal and vertical directions

i = 1j
conj = lambda A: np.conj(A)
dz = lambda A: d3.Differentiate(A, coords) 

lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)

kx2ky2 = kx*kx+ky*ky
hatnabla2 = lambda A: (dz(dz(A))-kx2ky2*A)
hatnabla2perp = (-kx2ky2)
hatnabla4 = lambda A: (dz(dz(dz(dz(A)))) - 2*kx2ky2*dz(dz(A)) + kx2ky2*kx2ky2*A)

# Problem
problem = d3.IVP([barU0,tau_barU0_1,tau_barU0_2,
                  hatu,
                  hatw,tau_hatw_1,tau_hatw_2,tau_hatw_3,tau_hatw_4,
                  hatzeta,tau_hatzeta_1,tau_hatzeta_2,
                  barS0,tau_barS0_1,tau_barS0_2,
                  hatS,tau_hatS_1,tau_hatS_2,
                  barT0,tau_barT0_1,tau_barT0_2,
                  hatT,tau_hatT_1,tau_hatT_2
                  ], namespace=locals())

# Tau polynomials
tau_basis = zbasis.derivative_basis(2)
p1 = dist.Field(bases=tau_basis)
p2 = dist.Field(bases=tau_basis)
p3 = dist.Field(bases=tau_basis)
p4 = dist.Field(bases=tau_basis)
p1['c'][-1] = 1
p2['c'][-2] = 2
p3['c'][-3] = 3
p4['c'][-4] = 4

problem.add_equation("dt(hatnabla2(hatw)) - Pr*hatnabla4(hatw) - Pr*hatnabla2perp*Ra*(hatT-(1./Rp)*hatS) +  tau_hatw_1*p1+ tau_hatw_2*p2+tau_hatw_3*p3+ tau_hatw_4*p4 = - i*kx*barU0*hatnabla2(hatw) + i*kx*dz(dz(barU0))*hatw")
problem.add_equation("dt(hatzeta) - Pr*hatnabla2(hatzeta) + tau_hatzeta_1*p1+ tau_hatzeta_2*p2= - i*kx*barU0*hatzeta")
problem.add_equation("dt(hatT) - hatnabla2(hatT) + hatw + tau_hatT_1*p1+ tau_hatT_2*p2= - i*kx*barU0*hatT - hatw*dz(barT0)")
problem.add_equation("dt(hatS) - tau*hatnabla2(hatS) + hatw + tau_hatS_1*p1+ tau_hatS_2*p2= - i*kx*barU0*hatS - hatw*dz(barS0)")

problem.add_equation("dt(barU0) - Pr*dz(dz(barU0)) + tau_barU0_1*p1+ tau_barU0_2*p2= - dz(conj(hatw)*hatu+hatw*conj(hatu))")
problem.add_equation("dt(barT0) - dz(dz(barT0)) + tau_barT0_1*p1+ tau_barT0_2*p2= - dz(conj(hatw)*hatT+hatw*conj(hatT))")
problem.add_equation("dt(barS0) - tau*dz(dz(barS0)) + tau_barS0_1*p1+ tau_barS0_2*p2= - dz(conj(hatw)*hatS+hatw*conj(hatS))")

problem.add_equation("hatu = i*kx*dz(hatw)/kx2ky2 - i*ky*hatzeta/kx2ky2")


# Boundary conditions
problem.add_equation("barS0(z=0) = 0")
problem.add_equation("barS0(z=Lz) = 0")
problem.add_equation("hatS(z=0) = 0")
problem.add_equation("hatS(z=Lz) = 0")

problem.add_equation("barT0(z=0) = 0")
problem.add_equation("barT0(z=Lz) = 0")
problem.add_equation("hatT(z=0) = 0")
problem.add_equation("hatT(z=Lz) = 0")

problem.add_equation("barU0(z=0) = 0")
problem.add_equation("barU0(z=Lz) = 0")
problem.add_equation("hatw(z=0) = 0")
problem.add_equation("hatw(z=Lz) = 0")
problem.add_equation("hatzeta(z=0) = 0")
problem.add_equation("hatzeta(z=Lz) = 0")
problem.add_equation("dz(hatw)(z=0) = 0") 
problem.add_equation("dz(hatw)(z=Lz) = 0") 

stop_sim_time = 301 # Stopping criteria
# timestepper = d3.RK443 # 3rd-order 4-stage DIRK+ERK scheme [Ascher 1997 sec 2.8] doi 10.1016/S0168-9274(97)00056-1
timestepper = d3.RK222
# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time


barU0.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
hatu.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
hatw.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
hatzeta.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
barS0.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
hatS.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
barT0.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
hatT.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise


# Analysis
import glob
import os
sim_name = "singlemode_tf1"
if os.path.exists(sim_name):
    dir_path = '/'+sim_name
    file_pattern = sim_name+"s*.h5" # pattern for file names to be deleted
    file_paths = glob.glob(os.path.join(dir_path, file_pattern)) # get a list of file paths using the glob module
    # loop over each file path and delete the file
    for file_path in file_paths:
        os.remove(file_path)

dataset = solver.evaluator.add_file_handler(sim_name, sim_dt=50.0, max_writes=1000)
dataset.add_task(barU0, name='barU0')
dataset.add_task(hatu, name='hatu')
dataset.add_task(hatw, name='hatw')
dataset.add_task(barS0, name='barS0')
dataset.add_task(hatS, name='hatS')
dataset.add_task(barT0, name='barT0')
dataset.add_task(hatT, name='hatT')

# Setup storage
barS0.change_scales(1)
barS0_list = [np.copy(barS0['g'])]
t_list = [solver.sim_time]
# Main loop
print('Starting main loop')
timestep = 0.05
while solver.proceed:
    solver.step(timestep)
    if solver.iteration % 10 == 0:
        barS0.change_scales(1)
        barS0_list.append(np.copy(barS0['g']))
        t_list.append(solver.sim_time)
    if solver.iteration % 1000 == 0:
        print('Completed iteration {}'.format(solver.iteration))
        # print(barS0['g'])

# Convert storage lists to arrays
barS0_array = np.array(barS0_list)
t_array = np.array(t_list)


# Plot solution
plt.figure(figsize=(6, 7), dpi=100)
plt.pcolormesh(z, t_array, np.real(barS0_array), shading='nearest')
plt.colorbar()
plt.xlabel('z')
plt.ylabel('t')
plt.title(r'$\bar{S}_0(t)$')
plt.tight_layout()
plt.savefig("BarS0.png", dpi=300, bbox_inches="tight")
plt.show()