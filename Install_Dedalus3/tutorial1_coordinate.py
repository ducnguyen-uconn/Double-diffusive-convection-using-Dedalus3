'''
To run and plot:
    $ python3 tutorial1_coordinate.py

'''

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3

# Parameters
Lx, Lz = 1, 1 # define size of domain
Nx, Nz = 64, 64 # define number of points

# Bases: create coordinate system
coords = d3.CartesianCoordinates('x', 'z') # create name of dimensions, this case uses two dimensions x and z
dist = d3.Distributor(coords, dtype=np.float64) # define type of nodes
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0,Lx)) # create array of coordinate x
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(0,Lz)) # create array of coordinate z

# grid_normal = zbasis.global_grid(dist, scale=1).ravel()
# grid_dealias = zbasis.global_grid(dist, scale=3/2).ravel()

# Fields: create fields including velocity, pressure, viscosity, or others
p = dist.Field(name='p', bases=(xbasis,zbasis)) # pressure field
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis)) # velocity field


# Substitutions
x, z = dist.local_grids(xbasis, zbasis) # create location of x, z
ex, ez = coords.unit_vector_fields(dist) # unit vector

# plot
plt.figure(figsize=(6, 1.5), dpi=100)
# plt.plot(grid_normal, 0*grid_normal+1, 'o', markersize=5)
# plt.plot(grid_dealias, 0*grid_dealias-1, 'o', markersize=5)
plt.xlabel('x')
plt.ylabel('z')
# plt.title('tutorial1_coordinate')
# plt.ylim([-2, 2])
plt.gca().yaxis.set_ticks([])
plt.tight_layout()

plt.savefig("tutorial1_coordinate.png", dpi=200)
plt.show()