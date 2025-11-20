import os
from pathlib import Path
import scipy.stats as st
from scipy.special import erf
from skspatial.objects import Plane, Points

import numpy as np
from matplotlib import pyplot as plt

from pydeep_sim.rough_surface.rough_surface import RoughSurface


def delete_content(directory):
    """
    Recursively deletes files in all the subfolders of a specified directory
    """

    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return

    for item in os.listdir(directory):
        print(item)
        item_path = os.path.join(directory, item)

        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
                print(f"Deleted file: {item_path}")
            elif os.path.isdir(item_path):
                # Recursively delete contents of subdirectory
                delete_content(item_path)
                os.rmdir(item_path)
                print(f"Deleted folder: {item_path}")
        except Exception as e:
            print(f"Failed to delete {item_path}. Reason: {e}")


def patches_generation(
    H,
    n_iter,
    surf_id=0,
    path_to_patches="./surface_patches",
    path_to_surface="./surface_database",
    file_tail="RMD",
    N=128,
    l=1.0,
    std0=0.09,
    verbose=False,
):

    path_to_patches = Path(path_to_patches)
    path_to_surface = Path(path_to_surface)

    assert int(N % np.sqrt(n_iter)) == 0
    resolution = int(np.log2(N / np.sqrt(n_iter)))
    if verbose:
        print(
            "{0} points, repeated {1} times on each side.".format(
                2**resolution, int(np.sqrt(n_iter))
            )
        )

    path_to_dir = path_to_patches / "niter_{0:02d}".format(n_iter)
    path_to_dir.mkdir(parents=True, exist_ok=True)

    # Generate the patches:
    path_to_topology = {}
    for k in range(n_iter):
        topology = RoughSurface(
            path_to_dir,
            file_tail + "_{0}_{1:03d}".format(k, int(N / np.sqrt(n_iter))),
            resolution,
            H,
            std0,
            n_iter,
            l,
        )
        path_to_topology[k] = topology.generate_surface_RMD()

    # Import the patch surfaces into a dictionary object
    z_raw = {}
    h_max = []
    for k in range(n_iter):
        z_raw[k] = np.loadtxt(
            path_to_topology[k], delimiter=";", usecols=range(2**resolution + 1)
        )
        z_raw[k] = z_raw[k][:-1, :-1]
        h_max.append(np.max(z_raw[k]))

    # -1- ######################################################################
    #
    # Remove average slope from all the patch surfaces, because the reviewer
    # asks so.
    #
    ############################################################################

    dx = l/2**resolution
    dy = dx
    x = np.linspace(dx/2, l-dx/2, 2**resolution)
    y = np.linspace(dy/2, l-dy/2, 2**resolution)
    X, Y = np.meshgrid(x, y)

    for k in range(n_iter):

        # Rearrange surface points to perform plane fitting:
        surface_array = Points(
            np.vstack((X.ravel(), Y.ravel(), z_raw[k].ravel())).T)
        fitting_plane = Plane.best_fit(surface_array)

        # Coefficients of the fitting plane a*x+b*y+c*z+d == 0
        (a, b, c, d) = fitting_plane.cartesian()

        # Subtraction of fitting plane from original surface:
        z_raw[k] = z_raw[k]-(-1/c*(a*X+b*Y+d))

        # By performing this operation, a small tilt remains, due to
        # linearization. This can be shown doing a new plane fitting, that
        # highlights how a residual angle is present. To obtain a rough surface
        # whose mean plane is perfectly even, a rotation must be performed
        # around the first two Euler angles of the unit normal of the mean
        # plane, expressed in spherical coordinates. Out of curiosity, this can
        # be accomplished by:

        """
        # Plane normal unit vector:
        n = fitting_plane.normal

        # Original polar and azimuth angles in spherical coordinates:
        th = np.acos(n[2])
        phi = np.atan2(n[1], n[0])

        # Rotation of -phi about z-axis:
        Rphi = np.array([[np.cos(-phi), -np.sin(-phi), 0.0],
                        [np.sin(-phi),  np.cos(-phi), 0.0],
                        [0.0,           0.0, 1.0]])

        # Rotation of -th about y-axis:
        Rth = np.array([[np.cos(-th),  0.0, np.sin(-th)],
                        [0.0,          1.0,         0.0],
                        [-np.sin(-th), 0.0, np.cos(-th)]])

        # Compund 3D rotation operation:
        R = Rth@Rphi
        surface_rot = surface_array@R.T #Extended expression is: (R@surface_rot.T).T

        fitting_plane_rot = Plane.best_fit(surface_rot)
        (a, b, c, d) = fitting_plane_rot.cartesian()
        print(th, phi, a, b, c, d)

        # In this case, the surface should be then mapped back to NxN format.
        """

    # -2- ######################################################################
    #
    # Remove mean plane from all the patch surfaces, since asperities with same
    # height from different surfaces must come into contact with same separation
    # plane.
    #
    ############################################################################

    for k in range(n_iter):
        z_raw[k] -= np.mean(z_raw[k])

    # -3- ######################################################################
    #
    # Join the patches together to generate the final patchwork surface
    #
    ############################################################################

    """
    # Small test to see if things are coming out as expected
    a = np.array([[0,0],[0,0]])
    b = np.array([[1,1],[1,1]])
    c = np.array([[2,2],[2,2]])
    d = np.array([[3,3],[3,3]])
    A_test = {0:a,1:b,2:c,3:d}
    """

    patches_per_side = int(np.sqrt(n_iter))
    z_patch_raw = np.block(
        [
            [z_raw[i * patches_per_side + j] for j in range(patches_per_side)]
            for i in range(patches_per_side)
        ]
    )

    # -4- ######################################################################
    #
    # Smooth the "fault lines" to avoid undesired "canyoning" effect
    #
    ############################################################################

    z_patch = z_patch_raw.copy()

    th = 0.8
    for i in range(N):

        i_top = n*i
        i_bottom = n*(i+1)-1

        if i_top > 0:
            z_patch[i_top, :] = th * \
                z_patch_raw[i_top, :] + (1-th)*z_patch_raw[i_top-1, :]
        if i_bottom < n*N-1:
            z_patch[i_bottom, :] = th * \
                z_patch_raw[i_bottom, :] + (1-th)*z_patch_raw[i_bottom+1, :]

    for j in range(N):

        j_left = n*j
        j_right = n*(j+1)-1

        if j_left > 0:
            z_patch[:, j_left] = th * \
                z_patch_raw[:, j_left] + (1-th)*z_patch[:, j_left-1]
        if j_right < n*N-1:
            z_patch[:, j_right] = th * \
                z_patch_raw[:, j_right] + (1-th)*z_patch_raw[:, j_right+1]

    # -5- ######################################################################
    #
    # Set minimum height to 0.0, so that lowest point is in contact without
    # exterting force.
    #
    ############################################################################

    z_patch -= np.min(z_patch)

    # delete_content(path_to_patches)

    # Save final patchwork surface and store it in surface database
    try:
        np.savetxt(
            path_to_surface,
            z_patch,
            fmt="%25.17e",
            delimiter=";",
        )
        if verbose:
            print("Final aggregated surface was created successfully")
        return z_patch
    except Exception as e:
        print(f"An error occurred: {e}")


def plot_probability_density(z, num_patches, path_to_figure):
    plt.figure(1, figsize=(10, 6))
    std_z = st.tstd(z, axis=None)
    mean_z = st.tmean(z, axis=None)
    x = np.linspace(mean_z - 5 * std_z, mean_z + 5 * std_z, 100)
    plt.plot(
        x,
        st.norm.pdf(x, mean_z, std_z),
        label=f"N($\mu=${mean_z:.3f}, $\sigma=${std_z:.3f})",
        linestyle="--",
        color="k",
    )
    plt.axvline(mean_z, linestyle=":", color="k",
                linewidth=0.75, label="$\mu$")
    plt.axvline(
        mean_z + 2 * std_z,
        linestyle="-.",
        color="k",
        linewidth=0.75,
        label="$\mu \pm 2 \sigma$",
    )
    plt.axvline(
        mean_z - 2 * std_z,
        linestyle="-.",
        color="k",
        linewidth=0.75,
    )
    hist_patch = np.histogram(np.ravel(z, order="F"), bins=500, density=True)
    plt.plot(hist_patch[1][:-1], hist_patch[0],
             label=f"Patches: {num_patches}")
    plt.legend()
    plt.grid("show")
    plt.xlabel("h (mum)")
    plt.ylabel("Probability density")
    plt.savefig(path_to_figure)
    plt.close()


def plot_cumulative_distribution(z, num_patches, path_to_figure, probability=0.1):
    plt.figure(2, figsize=(10, 6))
    std_z = st.tstd(z, axis=None)
    mean_z = st.tmean(z, axis=None)
    z_flat = np.ravel(z, order="F")
    x = np.linspace(mean_z - 5 * std_z, mean_z + 5 * std_z, 100)
    cumulated_dist = 1 / 2 * (1 + erf((x - mean_z) / (np.sqrt(2) * std_z)))
    plt.plot(
        x,
        cumulated_dist,
        label=f"Normal({mean_z:.3f}, {std_z:.3f})",
        linestyle="--",
        color="k",
    )
    hist_patch = np.histogram(np.ravel(z, order="F"), bins=500, density=True)
    cumulated_sum = (hist_patch[1][1] - hist_patch[1]
                     [0]) * np.cumsum(hist_patch[0])
    plt.plot(hist_patch[1][:-1], cumulated_sum,
             label=f"Patches: {num_patches}")
    probabilities = np.linspace(0, 1, 100)
    quantiles = np.quantile(z_flat, probabilities)
    plt.plot(quantiles, probabilities, label=f"Scipy")

    plt.axhline(y=probability, color="grey", linestyle=":", linewidth=1)
    plt.axvline(
        x=np.quantile(z_flat, probability),
        color="grey",
        linestyle=":",
        linewidth=1,
        label=f"{probability} quantile",
    )
    plt.legend()
    plt.grid("show")
    plt.xlabel("h (mum)")
    plt.ylabel("Cumulated distribution")
    plt.savefig(path_to_figure)
    plt.close()


################################################################################
# MAIN STARTS HERE
################################################################################

if __name__ == "__main__":
    std0 = 0.09
    l = 1.0
    N = 128

    # Let us fix the number of points per side of the patchwork to N=128, and the
    # number of patches to [1,4,16,64], this leaves us with [128, 64, 32, 16] points
    # per side, respectively, and resolutions of [7,6,5,4].

    n_iter = [1, 4, 16, 64]  # number of patches in total
    H = 0.75
    # unique identifier for final "big" surface (I hope this help in database generation)
    surf_id = 1

    z_surf = {}
    surf_id = 0
    path_to_database = Path("./surface_database")
    for n in n_iter:
        surf_id += 1
        final_surface_name = "topology_RMD_aggregated_{0:02d}x{1:03d}_{2:04d}".format(
            n, int(N / np.sqrt(n)), surf_id
        )
        path_to_surface = path_to_database / (final_surface_name + ".dat")
        z_surf[n] = patches_generation(
            H=H,
            n_iter=n,
            surf_id=surf_id,
            N=N,
            path_to_surface=path_to_surface,
            verbose=True,
        )
        plot_probability_density(
            z_surf[n],
            num_patches=n,
            path_to_figure=path_to_database /
            (final_surface_name + "_histogram.png"),
        )
        plot_cumulative_distribution(
            z_surf[n],
            num_patches=n,
            path_to_figure=path_to_database /
            (final_surface_name + "_cdf.png"),
        )
