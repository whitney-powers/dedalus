"""
Plot spherical snapshots.

Usage:
    plot_sphere.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames_sphere]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cbook import get_sample_data
#from matplotlib._png import read_png

def build_s2_coord_vertices(phi, theta):
    phi = phi.ravel()
    phi_vert = np.concatenate([phi, [2*np.pi]])
    phi_vert -= phi_vert[1] / 2
    theta = theta.ravel()
    theta_mid = (theta[:-1] + theta[1:]) / 2
    theta_vert = np.concatenate([[np.pi], theta_mid, [0]])
    return np.meshgrid(phi_vert, theta_vert, indexing='ij')


def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""
    # Plot settings
    task = 'bmid'
    cmap = plt.cm.RdBu_r
    dpi = 100
    figsize = (8, 8)
    savename_func = lambda write: 'write_{:06}.png'.format(write)
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    arr = plt.imread("pondering-my-orb-header-art.png").transpose(1,0,2)
    #arr = read_png(fn)
    # 10 is equal length of x and y axises of your surface
    stepY, stepZ = 10. / arr.shape[0], 10. / arr.shape[1]
    
    Z1 = np.arange(-5, 5, stepZ)
    Y1 = np.arange(-5, 5, stepY)
    Z1, Y1 = np.meshgrid(Z1, Y1)
    # stride args allows to determine image quality 
    # stride = 1 work slow
    ax.plot_surface(-1, 12/6.75*Y1, -Z1, rstride=1, cstride=1, facecolors=arr)
    #ax.plot_surface(-1, 12/6.75*Y1, -Z1, facecolors=arr)

    # Plot writes
    with h5py.File(filename, mode='r') as file:
        dset = file['tasks'][task]
        phi = dset.dims[1][0][:].ravel()
        theta = dset.dims[2][0][:].ravel()
        phi_vert, theta_vert = build_s2_coord_vertices(phi+.25, theta+.25)
        x = 2.5*(np.sin(theta_vert) * np.cos(phi_vert))
        y = 2.5*(np.sin(theta_vert) * np.sin(phi_vert))+2.5
        z = 2.5*(np.cos(theta_vert))-2
        norm = matplotlib.colors.Normalize(0, 1)
        for index in range(start, start+count):
            data_slices = (index, slice(None), slice(None), 0)
            data = dset[data_slices]
            fc = cmap(norm(data))
            fc[:, theta.size//2, :] = [0,0,0,1]  # black equator
            if index == start:
                surf = ax.plot_surface(x, y, z, facecolors=fc, cstride=1, rstride=1, linewidth=0, antialiased=False, shade=False, zorder=3)
                ax.set_box_aspect((1,1.25,0.675))
                #ax.set_box_aspect((1,1,1))
                #ax.set_xlim(-0.7, 0.7)
                #ax.set_ylim(-0.7, 0.7)
                #ax.set_zlim(-0.7, 0.7)
                ax.azim=0
                ax.elev=0
                ax.axis('off')
                
            else:
                surf.set_facecolors(fc.reshape(fc.size//4, 4))
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
    plt.close(fig)


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

