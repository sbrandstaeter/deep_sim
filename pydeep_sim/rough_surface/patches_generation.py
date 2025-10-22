import os
from pathlib import Path

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
):

    path_to_patches = Path(path_to_patches)
    path_to_surface = Path(path_to_surface)

    assert int(N % np.sqrt(n_iter)) == 0
    resolution = int(np.log2(N / np.sqrt(n_iter)))
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

    # Remove mean plane from all the patch surfaces, since asperities with same
    # height from different surfaces must come into contact with same separation
    # plane
    for k in range(n_iter):
        z_raw[k] -= np.mean(z_raw[k])

    """
    # Small test to see if things are coming out as expected
    a = np.array([[0,0],[0,0]])
    b = np.array([[1,1],[1,1]])
    c = np.array([[2,2],[2,2]])
    d = np.array([[3,3],[3,3]])
    A_test = {0:a,1:b,2:c,3:d}
    """

    patches_per_side = int(np.sqrt(n_iter))
    z_patch = np.block(
        [
            [z_raw[i * patches_per_side + j] for j in range(patches_per_side)]
            for i in range(patches_per_side)
        ]
    )

    delete_content(path_to_patches)

    # Save final patchwork surface and store it in surface database
    try:
        np.savetxt(
            path_to_surface,
            z_patch,
            fmt="%.17e",
            delimiter=";",
        )
        print("Final aggregated surface was created successfully")
        return z_patch, topology
    except Exception as e:
        print(f"An error occurred: {e}")


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
    surf_id = 1  # unique identifier for final "big" surface (I hope this help in database generation)

    z_surf = {}
    surf_id = 0
    for n in n_iter:
        surf_id += 1
        final_surface_name = (
            "topology_RMD_aggregated_{0:02d}x{1:03d}_{2:04d}.dat".format(
                n, int(N / np.sqrt(n)), surf_id
            )
        )
        path_to_surface = Path("./surface_database") / final_surface_name
        z_surf[n] = patches_generation(
            H=H, n_iter=n, surf_id=surf_id, N=N, path_to_surface=path_to_surface
        )
        z_surf[n] -= np.min(z_surf[n])

    """
    std_z = st.tstd(z_surf[n], axis = None)
    mean_z = st.tmean(z_surf[n], axis = None)
    z_flat = np.ravel(z_surf[n], order = 'F')
    x = np.linspace(-5*std_z,+5*std_z, 1000)
    """

    plt.figure(1, figsize=(10, 6))
    for n in n_iter:
        hist_patch = np.histogram(
            np.ravel(z_surf[n], order="F"), bins=500, density=True
        )
        plt.plot(hist_patch[1][:-1], hist_patch[0], label="Patches: {}".format(n))
        # plt.plot(x, st.norm.pdf(x, 0.0, std_z))
        # plt.axvline(3*std_z, linestyle = '-.', color = 'k',linewidth = 0.75)
        plt.legend()
        plt.grid("show")
        plt.xlabel("h (mum)")
        plt.ylabel("Probability density")
    plt.savefig("histogram.png")

    plt.figure(2, figsize=(10, 6))
    for n in n_iter:
        hist_patch = np.histogram(
            np.ravel(z_surf[n], order="F"), bins=500, density=True
        )
        cumulated_sum = (hist_patch[1][1] - hist_patch[1][0]) * np.cumsum(hist_patch[0])
        plt.plot(hist_patch[1][:-1], cumulated_sum, label="Patches: {}".format(n))
        # plt.plot(x,cumulated_dist)
        # cumulated_dist = 1/2*(1+erf((x-mean_z)/(np.sqrt(2)*std_z)))
        plt.legend()
        plt.grid("show")
        plt.xlabel("h (mum)")
        plt.ylabel("Cumulated distribution")
    plt.savefig("cumulated.png")
