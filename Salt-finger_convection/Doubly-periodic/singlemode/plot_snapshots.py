"""
Plot 2D cartesian snapshots.

Usage:
    plot_snapshots.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dedalus.extras import plot_tools


def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings
    scale = 1.5
    dpi = 200
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'write_{:06}.png'.format(write)

    Nx, Nz = 100, 128
    kx = 8
    Lz = 1
    x = np.linspace(0,2*np.pi/kx,Nx)
    z = np.linspace(0,Lz,Nz)
    tS = np.zeros((Nx,Nz)) # define total salinity = z + S(z,t)

    # Plot writes
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            dset1 = file['tasks']['barS0']
            dset2 = file['tasks']['hatS']
            
            for i in range(Nx):
                for k in range(Nz):
                    S = np.real(dset1[index][k]) + np.real(dset2[index][k]*np.exp(1j*kx*x[i]))
                    tS[i][k]= z[k] + S

            plt.pcolormesh(x, z, tS.transpose(), shading='nearest')
            plt.colorbar()
            plt.xlabel(r'$x$')
            plt.ylabel(r'$z$')
            
            
            title = title_func(file['scales/sim_time'][index])
            plt.title(title)
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            plt.savefig(str(savepath), dpi=dpi)
            plt.clf()
    plt.close()


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)