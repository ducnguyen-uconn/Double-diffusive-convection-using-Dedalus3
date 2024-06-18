#!/usr/bin/env python
# coding: utf-8

# # Salt-finger convection based on Single-mode equation using Dedalus 3
# 
# This notebook presents a Python code for simulating salt-finger convections in works by [Chang Liu 2022 JFM](https://doi.org/10.1017/jfm.2022.865) by using [Dedalus framework](https://dedalus-project.org/) which performs the spectral method. In particular, we consider a fliud between two infinitely long parallel plates with a distance $h$. The temperature ($T_*$) and salinity $S_*$ at these two plates are maintained at constant values with the top plate maintained at a higher temperature and salinity. The subscript $∗$ denotes a dimensional variable. Variables are non-dimensionalized based on the thermal diffusivity $\kappa_T$
# 
# \begin{equation}
# T = \frac{T_*}{\Delta T}, \quad S = \frac{S_*}{\Delta S}, \quad t = \frac{t_*}{h^2/\kappa_T}, \quad \vec{u} = \frac{\vec{u}_*}{\kappa_T/h}, \quad p = \frac{p_*}{\kappa_T^2 \rho_{r*}/h^2}.
# \end{equation}
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3


# In this work, I use a computational domain of $[L_x\times L_y\times L_z]:=[2\pi/k_x\times 2\pi/k_y\times 1]$ with ($k_x$, $k_y$) representing the wavenumber pair in the horizontal directions. However, the present code only shows a 2D configuration ($k_y=0$). Fourier and Shebyshev spectral methods are used in the horizontal and vertical directions with $N_x = 128$ and $N_z = 128$, respectively. Dealiasing scaling factor is set by $3/2$. 
# 

# The system is governed by (equation 2.2 in the reference)
# 
# \begin{align}
# \nabla\cdot \mathbf{u} &= 0,\\
# \partial_t \mathbf{u}+\mathbf{u}\cdot\nabla\mathbf{u} &= Pr\nabla^2\mathbf{u} -\nabla p + PrRa_{T}(T-R_{\rho}^{-1}S)\mathbf{e}_z,\\
# \partial_t T +\mathbf{u}\cdot\nabla T + w &= \nabla^2 T,\\
# \partial_t S +\mathbf{u}\cdot\nabla S + w &= \tau\nabla^2 S.
# \end{align}
# 
# where $\mathbf{u}=(u,v,w)$, $p$, and $\mathbf{e}_z$ are the fluid velocity, pressure, unit vector in the vertical. The governing parameters include the Prandtl number, the diffusivity ratio, the density ratio and the thermal Rayleigh number defined by
# 
# \begin{equation}
# Ra := \frac{g\alpha\Delta T h^3}{\nu\kappa_T}, \quad Pr := \frac{\nu}{\kappa_T}, \quad \tau := \frac{\kappa_S}{\kappa_T}, \quad R_\rho := \frac{\alpha \Delta T}{\beta\Delta S},  
# \end{equation}
# 
# where $\nu$ is the viscosity and $\kappa_S$ is the salinity diffusivity.

# ## Single-mode equation for SFC

# This section describes the governing equation by using the single-mode equation. Following the procedure in [Herring 1963](https://doi.org/10.1175/1520-0469(1963)020%3C0325:IOPITC%3E2.0.CO;2), [Gough et al. 1975](https://doi.org/10.1017/S0022112075001188), [Gough & Toomre 1982](https://doi.org/10.1017/S0022112082003267), [Paparella & Spiegel 1999](https://doi.org/10.1063/1.869890), and [Paparella et al. 2002](https://doi.org/10.1080/03091920290029031), the single-mode ansatz is expressed by
# \begin{align}
#     S(x,y,z,t) &= \bar{S}_0(z,t) + \hat{S}(z,t) e^{i(k_x x + k_y y)} + \text{c.c.},\\
#     T(x,y,z,t) &= \bar{T}_0(z,t) + \hat{T}(z,t) e^{i(k_x x + k_y y)} + \text{c.c.},\\
#     \mathbf{u}(x,y,z,t) &= \bar{U}_0(z,t) \mathbf{e}_x + \hat{\mathbf{u}}(z,t) e^{i(k_x x + k_y y)} + \text{c.c.},\\
#     p(x,y,z,t) &= \bar{P}_0(z,t) + \hat{p}(z,t) e^{i(k_x x + k_y y)} + \text{c.c.},
# \end{align}
# with $\text{c.c.}$ representing the complex conjugate.
# Velocity vector can be decomposed into
# \begin{align}
#     u(x,y,z,t) &= \bar{U}_0(z,t) + \hat{u}(z,t) e^{i(k_x x + k_y y)} + \text{c.c.},\\
#     v(x,y,z,t) &= \hat{v}(z,t) e^{i(k_x x + k_y y)} + \text{c.c.},\\
#     w(x,y,z,t) &= \hat{w}(z,t) e^{i(k_x x + k_y y)} + \text{c.c.},
# \end{align}

# Quantities are decomposed into a horizontal averaged component marked by $\bar{.}$ and a single harmonic in the horizontal direction associated with the wavenumber pair ($k_x$,$k_y$) and characterised by the comple amplitude marked by $\hat{.}$. For 2D configuration, velocity can be decomposed into a large-scale shear $\bar{U}_{0} (z,t) \mathbf{e}_x$ and a harmonic with the same wavenumber pair ($k_x$,$k_y$). This allows generating a mean flow in the horizontal direction by assuming that the large-scale shear $\bar{\mathbf{u}}_0$ is generated in the x-direction. **In three dimensions, the large-scale shear can be oriented in principle in any horizontal direction, a possibility that is left for future study**. 

# Finally, we have new governing equations obtained from original governing equations and single-mode equations. Terms of the vertical velocity $w$ and vertical vorticity $\zeta:=\partial_y u-\partial_x v$ :
# $$
# \begin{gathered}
# \partial_t \hat{\nabla}^2 \hat{w}+\mathrm{i} k_x \bar{U}_0 \hat{\nabla}^2 \hat{w}-\mathrm{i} k_x \bar{U}_0^{\prime \prime} \hat{w}=\operatorname{Pr} \hat{\nabla}^4 \hat{w}+\operatorname{Pr} \hat{\nabla}_{\perp}^2 \operatorname{Ra}\left(\hat{T}-R_\rho^{-1} \hat{S}\right), \\
# \partial_t \hat{\zeta}+\mathrm{i} k_x \bar{U}_0 \hat{\zeta}+\mathrm{i} k_y \bar{U}_0^{\prime} \hat{w}=\operatorname{Pr} \hat{\nabla}^2 \hat{\zeta}, \\
# \partial_t \hat{T}+\mathrm{i} k_x \bar{U}_0 \hat{T}+\hat{w} \partial_z \bar{T}_0+\hat{w}=\hat{\nabla}^2 \hat{T}, \\
# \partial_t \hat{S}+\mathrm{i} k_x \bar{U}_0 \hat{S}+\hat{w} \partial_z \bar{S}_0+\hat{w}=\tau \hat{\nabla}^2 \hat{S}, \\
# \partial_t \bar{U}_0+\partial_z\left(\hat{w}^* \hat{u}+\hat{w} \hat{u}^*\right)=\operatorname{Pr} \partial_z^2 \bar{U}_0, \\
# \partial_t \bar{T}_0+\partial_z\left(\hat{w}^* \hat{T}+\hat{w} \hat{T}^*\right)=\partial_z^2 \bar{T}_0, \\
# \partial_t \bar{S}_0+\partial_z\left(\hat{w}^* \hat{S}+\hat{w} \hat{S}^*\right)=\tau \partial_z^2 \bar{S}_0, \\
# \hat{u}=\frac{\mathrm{i} k_x \partial_z \hat{w}}{k_x^2+k_y^2}-\frac{\mathrm{i} k_y \hat{\zeta}}{k_x^2+k_y^2}, \quad \hat{v}=\frac{\mathrm{i} k_y \partial_z \hat{w}}{k_x^2+k_y^2}+\frac{\mathrm{i} k_x \hat{\zeta}}{k_x^2+k_y^2},
# \end{gathered}
# $$
# where the superscript * denotes a complex conjugate and $\hat{\nabla}^2:=\partial_z^2-k_x^2-k_y^2, \hat{\nabla}_{\perp}^2:=$ $-k_x^2-k_y^2, \quad \hat{\nabla}^4:=\partial_z^4-2\left(k_x^2+k_y^2\right) \partial_z^2+\left(k_x^2+k_y^2\right)^2, \quad \bar{U}_0^{\prime}:=\partial_z \bar{U}_0$ and $\bar{U}_0^{\prime \prime}:=\partial_z^2 \bar{U}_0$. 

# As shown in above equations, we can see that single-mode equations only depend on z-direction. This reduces one dimension for system.

# In[2]:


# Parameters
kx, ky = 1, 0   # wavenumber
Lz = 1.         # computational domain
Nz = 128        # number of points
dealias = 3/2   # scaling factor

Ra = 1e5        # Rayleigh number
Pr = 0.05       # Prandtl number
tau = 0.01      # diffusivity ratio = ks/kt
Rp = 40.0       # density ratio


# To begin with, we must create basis for problem, including the mesh system based on the 2D Cartesian coordinate system and defination of field distributor with floating point in double precision.

# In[ ]:


# Bases
coords = d3.Coordinate('z')
dist = d3.Distributor(coords, dtype=np.complex128)


# There is a periodic boundary condition in horizontal direction, so we use Fourier for this direction (x axis). The no-periodic in vertical direction (z axis), the method of Chebyshev will be used there.

# In[ ]:


# define the coordinate system
zbasis = d3.Chebyshev(coords, size=Nz, bounds=(0, Lz), dealias=dealias)


# Now, we will instantiate fields which will appears within the problem using the distributor, including pressure $p$, velocity $\vec{u}$, temperature $T$, and salinity $S$.

# Following table shows representations of quantities in code.
# 
# | Variables | Representations | Notes |
# | ----------- | ----------- |  ----------- |
# | $u$ | u | Horizontal velocity component |
# | $\bar{U}_0$ | barU0 | Horizontally averaged large-scale shear |
# | $\hat{u}$ | hatu | Amplitude of horizontal velocity's harmonic component |
# | $\hat{w}$ | hatw | Amplitude of vertical velocity's harmonic component |
# | $\hat{\zeta}$ | hatzeta | Amplitude of vertical vorticity's harmonic component |
# | $T$ | T | Temperature |
# | $\bar{T}_0$ | barT0 | Horizontally averaged large-scale temperature component |
# | $\hat{T}$ | hatT | Amplitude of temperature's harmonic component|
# | $S$ | S | Sanility |
# | $\bar{S}_0$ | barS0 | Horizontally averaged large-scale sanility component |
# | $\hat{S}$ | hatS | Amplitude of sanility's harmonic component|

# In[ ]:


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


# To simply when typing and determining the equations in Dedalus3, we can define replace operators for complex operators

# In[ ]:


# Substitutions
z = dist.local_grids(zbasis) # get coordinate arrays in horizontal and vertical directions

i = 1j
conj = lambda A: np.conj(A)
dz = lambda A: d3.Differentiate(A, coords) 

lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)


# $$
# \begin{gathered}
# \hat{\nabla}^2 \hat{w} := \partial_z^2 \hat{w}-(k_x^2+k_y^2) \hat{w},\\
# \hat{\nabla}_{\perp}^2 := -k_x^2-k_y^2,\\
# \bar{U}_0^{\prime \prime}:=\partial_z^2 \bar{U}_0,\\
# \hat{\nabla}^4:=\partial_z^4-2\left(k_x^2+k_y^2\right) \partial_z^2+\left(k_x^2+k_y^2\right)^2,
# \end{gathered}
# $$

# In[ ]:


kx2ky2 = kx*kx+ky*ky
hatnabla2 = lambda A: (dz(dz(A))-kx2ky2*A)
hatnabla2perp = (-kx2ky2)
hatnabla4 = lambda A: (dz(dz(dz(dz(A)))) - 2*kx2ky2*dz(dz(A)) + kx2ky2*kx2ky2*A)


# ## Define problem's equations

# When we have all the fields for solving, we must build the IVP problem and introcude fields to the solver. At this step, we don't have full field for solving all three governing equations, so we just introduce corresponding codes to perform this.

# In[ ]:


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


# In[ ]:


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


# $$
# \partial_t \hat{\nabla}^2 \hat{w}+\mathrm{i} k_x \bar{U}_0 \hat{\nabla}^2 \hat{w}-\mathrm{i} k_x \bar{U}_0^{\prime \prime} \hat{w}=\operatorname{Pr} \hat{\nabla}^4 \hat{w}+\operatorname{Pr} \hat{\nabla}_{\perp}^2 \operatorname{Ra}\left(\hat{T}-R_\rho^{-1} \hat{S}\right), 
# $$
# To fit with Dedalus3, this equation was modified as
# $$
# \partial_t \hat{\nabla}^2 \hat{w} - \operatorname{Pr} \hat{\nabla}^4 \hat{w} - \operatorname{Pr} \hat{\nabla}_{\perp}^2 \operatorname{Ra}\left(\hat{T} + R_\rho^{-1} \hat{S}\right) = -\mathrm{i} k_x \bar{U}_0 \hat{\nabla}^2 \hat{w}+\mathrm{i} k_x \bar{U}_0^{\prime \prime} \hat{w}, 
# $$

# In[ ]:


# equation 1
problem.add_equation("dt(hatnabla2(hatw)) - Pr*hatnabla4(hatw) - Pr*hatnabla2perp*Ra*(hatT-(1./Rp)*hatS) +  tau_hatw_1*p1+ tau_hatw_2*p2+tau_hatw_3*p3+ tau_hatw_4*p4 = - i*kx*barU0*hatnabla2(hatw) + i*kx*dz(dz(barU0))*hatw")
# problem.add_equation("dt(hatnabla2(hatw)) - Pr*hatnabla4(hatw) +  tau_hatw_1*p1+ tau_hatw_2*p2 +tau_hatw_3*p3+ tau_hatw_4*p4 = - i*kx*barU0*hatnabla2(hatw) + i*kx*dz(dz(barU0))*hatw")


# $$
# \begin{gathered}
# \partial_t \hat{\zeta}+\mathrm{i} k_x \bar{U}_0 \hat{\zeta}+\mathrm{i} k_y \bar{U}_0^{\prime} \hat{w}=\operatorname{Pr} \hat{\nabla}^2 \hat{\zeta}, \\
# \partial_t \hat{T}+\mathrm{i} k_x \bar{U}_0 \hat{T}+\hat{w} \partial_z \bar{T}_0+\hat{w}=\hat{\nabla}^2 \hat{T},\\
# \partial_t \hat{S}+\mathrm{i} k_x \bar{U}_0 \hat{S}+\hat{w} \partial_z \bar{S}_0+\hat{w}=\tau \hat{\nabla}^2 \hat{S},
# \end{gathered}
# $$
# To fit with Dedalus3, this equation was modified as
# $$
# \begin{gathered}
# \partial_t \hat{\zeta} - \operatorname{Pr} \hat{\nabla}^2 \hat{\zeta}= -\mathrm{i} k_x \bar{U}_0 \hat{\zeta}-\mathrm{i} k_y \bar{U}_0^{\prime} \hat{w}, \\
# \partial_t \hat{T} - \hat{\nabla}^2 \hat{T} + \hat{w} = -\mathrm{i} k_x \bar{U}_0 \hat{T}-\hat{w} \partial_z \bar{T}_0,\\
# \partial_t \hat{S} - \tau \hat{\nabla}^2 \hat{S} + \hat{w} = -\mathrm{i} k_x \bar{U}_0 \hat{S}-\hat{w} \partial_z \bar{S}_0,
# \end{gathered}
# $$
# with $\bar{U}_0^{\prime}:=\partial_z \bar{U}_0$

# In[ ]:


# equation 2
problem.add_equation("dt(hatzeta) - Pr*hatnabla2(hatzeta) + tau_hatzeta_1*p1+ tau_hatzeta_2*p2= - i*kx*barU0*hatzeta")
# equation 3
problem.add_equation("dt(hatT) - hatnabla2(hatT) + hatw + tau_hatT_1*p1+ tau_hatT_2*p2= - i*kx*barU0*hatT - hatw*dz(barT0)")
# equation 4
problem.add_equation("dt(hatS) - tau*hatnabla2(hatS) + hatw + tau_hatS_1*p1+ tau_hatS_2*p2= - i*kx*barU0*hatS - hatw*dz(barS0)")


# $$
# \begin{gathered}
# \partial_t \bar{U}_0+\partial_z\left(\hat{w}^* \hat{u}+\hat{w} \hat{u}^*\right)=\operatorname{Pr} \partial_z^2 \bar{U}_0, \\
# \partial_t \bar{T}_0+\partial_z\left(\hat{w}^* \hat{T}+\hat{w} \hat{T}^*\right)=\partial_z^2 \bar{T}_0, \\
# \partial_t \bar{S}_0+\partial_z\left(\hat{w}^* \hat{S}+\hat{w} \hat{S}^*\right)=\tau \partial_z^2 \bar{S}_0,
# \end{gathered}
# $$

# In[ ]:


# equation 5
problem.add_equation("dt(barU0) - Pr*dz(dz(barU0)) + tau_barU0_1*p1+ tau_barU0_2*p2= - dz(conj(hatw)*hatu+hatw*conj(hatu))")
# # equation 6
problem.add_equation("dt(barT0) - dz(dz(barT0)) + tau_barT0_1*p1+ tau_barT0_2*p2= - dz(conj(hatw)*hatT+hatw*conj(hatT))")
# # equation 7
problem.add_equation("dt(barS0) - tau*dz(dz(barS0)) + tau_barS0_1*p1+ tau_barS0_2*p2= - dz(conj(hatw)*hatS+hatw*conj(hatS))")


# $$
# \begin{gathered}
# \hat{u}=\frac{\mathrm{i} k_x \partial_z \hat{w}}{k_x^2+k_y^2}-\frac{\mathrm{i} k_y \hat{\zeta}}{k_x^2+k_y^2},\\
# \hat{v}=\frac{\mathrm{i} k_y \partial_z \hat{w}}{k_x^2+k_y^2}+\frac{\mathrm{i} k_x \hat{\zeta}}{k_x^2+k_y^2},
# \end{gathered}
# $$

# In[ ]:


# equation 8
problem.add_equation("hatu = i*kx*dz(hatw)/kx2ky2 - i*ky*hatzeta/kx2ky2")


# ## Set up the boundary conditions
# 

# Researching vertically confined salt-finger convection helps people to understand the interior between two well-mixed layers. The temperature and salanity are imposed as constant variables roling boundary conditions in this simulation. The corresponding boundary conditions for the salinity and temperature are:
# $$
# \begin{gathered}
# \hat{S}(z=0, t)=\hat{S}(z=1, t)=\hat{T}(z=0, t)=\hat{T}(z=1, t) \\
# =\bar{S}_0(z=0, t)=\bar{S}_0(z=1, t)=\bar{T}_0(z=0, t)=\bar{T}_0(z=1, t)=0,
# \end{gathered}
# $$

# In[ ]:


# Boundary conditions
problem.add_equation("barS0(z=0) = 0")
problem.add_equation("barS0(z=Lz) = 0")
problem.add_equation("hatS(z=0) = 0")
problem.add_equation("hatS(z=Lz) = 0")

problem.add_equation("barT0(z=0) = 0")
problem.add_equation("barT0(z=Lz) = 0")
problem.add_equation("hatT(z=0) = 0")
problem.add_equation("hatT(z=Lz) = 0")


# 
# To approach laboratory experiments more, the no-slip boudanry condition for velocity is adopted at top and bottom planes. In some cases, we can use the stress-free velocity boundary conditions instead to understand oceanographic scenarios. The no-slip boudanry condition for velocity in present situation is defined by
# $$
# \begin{gathered}
# \hat{w}(z=0, t)=\hat{w}(z=1, t)=\partial_z \hat{w}(z=0, t)=\partial_z \hat{w}(z=1, t) \\
# =\hat{\zeta}(z=0, t)=\hat{\zeta}(z=1, t)=\bar{U}_0(z=0, t)=\bar{U}_0(z=1, t)=0 .
# \end{gathered}
# $$

# In[ ]:


problem.add_equation("barU0(z=0) = 0")
problem.add_equation("barU0(z=Lz) = 0")
problem.add_equation("hatw(z=0) = 0")
problem.add_equation("hatw(z=Lz) = 0")
problem.add_equation("hatzeta(z=0) = 0")
problem.add_equation("hatzeta(z=Lz) = 0")
problem.add_equation("dz(hatw)(z=0) = 0") 
problem.add_equation("dz(hatw)(z=Lz) = 0") 


# Note that, we have two boundary conditions for each quantity, so we must have two tau fields.

# ## Build Solver

# In[ ]:


stop_sim_time = 300 # Stopping criteria
# timestepper = d3.RK443 # 3rd-order 4-stage DIRK+ERK scheme [Ascher 1997 sec 2.8] doi 10.1016/S0168-9274(97)00056-1
timestepper = d3.RK222
# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time


# ### Set up initial conditions

# In[ ]:


barU0.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
hatu.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
hatw.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
hatzeta.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
barS0.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
hatS.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
barT0.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
hatT.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise


# In[ ]:


# Analysis
# import shutil, os
# sim_name = 'SMESFC_DNS'
# if os.path.exists(sim_name):
#     shutil.rmtree(sim_name) # remove the output directory and its previous contents

# dataset = solver.evaluator.add_file_handler(sim_name, sim_dt=1.0, max_writes=1000)
# dataset.add_task(barU0, name='barU0')
# dataset.add_task(hatu, name='hatu')
# dataset.add_task(hatw, name='hatw')
# dataset.add_task(barS0, name='barS0')
# dataset.add_task(hatS, name='hatS')
# dataset.add_task(barT0, name='barT0')
# dataset.add_task(hatT, name='hatT')


# In[ ]:


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


# In[ ]:


# Plot solution
plt.figure(figsize=(6, 7), dpi=100)
plt.pcolormesh(z, t_array, np.real(barS0_array), shading='nearest')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.title(r'$\bar{S}_0(t)$')
plt.tight_layout()
plt.savefig("BarS0.png", dpi=300, bbox_inches="tight")
plt.show()

