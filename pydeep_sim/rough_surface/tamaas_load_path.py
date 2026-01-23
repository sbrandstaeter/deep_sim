import time

import tamaas as tm
import matplotlib.pyplot as plt
import numpy as np
from tamaas.utils import load_path
import scipy
from count_switches import count_switches

# Surface resolution (points pe rside)


def generate_surface_and_solve_pressure_driven_eff_area_load_path(
    q1=16,
    q2=64,
    hurst=0.8,
    n=512,
    L=1.0,  # Surface lateral size
    random_seed=1,
    p_target=0.1,
    num_load_steps=10,
    periodic=False,
    solver_tolerance=1e-09,  # solver tolerance, usually around 1e-9 adhesion-less contact (see https://gitlab.com/tamaas/tutorials/-/blob/master/elastic_contact.ipynb)
    scale_factor_surface=1.0,
    solve_contact_problem=True,
):

    # Grid cell area
    dx = L / n
    dA = dx**2

    # Surface generator
    sg = tm.SurfaceGeneratorFilter2D([n, n])
    sg.random_seed = random_seed

    # Define spectrum
    sg.spectrum = tm.Isopowerlaw2D()

    # Roll-off wavenumber (defines ratio between lateral size of the surface and longest wavelength. if lambda_0=2*pi/q0 ~ L the surface is not Gaussian)
    sg.spectrum.q0 = q1  # we set q0=q1 as it is not super important
    # Lowest wavenumber (defines longest wavelength of the surface as lambda_1 = 2*pi/q1)
    sg.spectrum.q1 = q1
    # Highest wavenumber (defines shortest wavelength of the surface as lambda_2 = 2*pi/q2)
    sg.spectrum.q2 = q2
    # Defines slope of the spectrum
    sg.spectrum.hurst = hurst

    # Generates the surface given the spectrum
    surface = sg.buildSurface()

    # Comment not to normalize the RMSSLope
    surface /= scale_factor_surface

    rms_slope = tm.Statistics2D.computeSpectralRMSSlope(surface)

    # Should be equal to 1 if normalized
    rms_slope_check = tm.Statistics2D.computeSpectralRMSSlope(surface)

    # Creates the model
    model = tm.ModelFactory.createModel(tm.model_type.basic_2d, [L, L], [n, n])

    # Uncomment to solve equivalent non-periodic problem:
    if not periodic:
        tm.ModelFactory.registerNonPeriodic(model, "dcfft")

    # Mechanical parameters
    model.E = 1.0

    # Initialize the solver
    solver = tm.PolonskyKeerRey(model, surface, tolerance=solver_tolerance)

    # Uncomment to solve equivalent non-periodic problem:
    if not periodic:
        solver.setIntegralOperator("dcfft")

    # Define load steps:

    loads = np.linspace(0, p_target, num_load_steps + 1)
    loads = loads[1:]

    # Solve for given load path:
    A_raw = np.zeros((len(loads),))
    A_cor = np.zeros_like(A_raw)

    Dmax = np.zeros_like(A_raw)
    Dmin = np.zeros_like(A_raw)
    Dmean = np.zeros_like(A_raw)
    run_times = np.zeros_like(A_raw)

    run_times_intermediate = np.zeros((len(loads) + 1,))

    if solve_contact_problem:
        start_time = time.time()
        run_times_intermediate[0] = start_time
        try:
            for i, model in enumerate(load_path(solver, loads)):

                solver.solve(loads[i])

                # To compute the true displacement (for non-periodic problem), one needs to re-evaluate the displacement
                if not periodic:
                    model.operators["dcfft"](model.traction, model.displacement)

                # Effective contact area (uncorrected)
                A_raw[i] = dA * len(model.traction[model.traction > 0.0])

                # Perform correction of the area according to Yastrebov:
                M = count_switches(model.traction)
                Sd = M * dx
                A_cor[i] = A_raw[i] - (np.pi - 1 + np.log(2)) / 24 * Sd * dx

                Dmin[i] = np.min(model.displacement)
                Dmean[i] = np.mean(model.displacement)
                Dmax[i] = np.max(model.displacement)
                run_times_intermediate[i + 1] = time.time()

            run_times = np.diff(run_times_intermediate)

        except Exception as e:
            print(f"Error in surface with q1:{q1}, q2:{q2}, hurst:{hurst}")
            print("Exception message:", e)

    return surface, A_raw, A_cor, loads, rms_slope, Dmin, Dmean, Dmax, run_times


def generate_surface_and_solve_pressure_driven_eff_area_individual_load_steps(
    q1=16,
    q2=64,
    hurst=0.8,
    n=512,
    L=1.0,  # Surface lateral size
    random_seed=1,
    p_target=0.1,
    num_load_steps=10,
    periodic=False,
    solver_tolerance=1e-09,  # solver tolerance, usually around 1e-9 adhesion-less contact (see https://gitlab.com/tamaas/tutorials/-/blob/master/elastic_contact.ipynb)
    scale_factor_surface=1.0,
    solve_contact_problem=True,
):
    # Grid cell area
    dx = L / n
    dA = dx**2

    # Surface generator
    sg = tm.SurfaceGeneratorFilter2D([n, n])
    sg.random_seed = random_seed

    # Define spectrum
    sg.spectrum = tm.Isopowerlaw2D()

    # Roll-off wavenumber (defines ratio between lateral size of the surface and longest wavelength. if lambda_0=2*pi/q0 ~ L the surface is not Gaussian)
    sg.spectrum.q0 = q1  # we set q0=q1 as it is not super important
    # Lowest wavenumber (defines longest wavelength of the surface as lambda_1 = 2*pi/q1)
    sg.spectrum.q1 = q1
    # Highest wavenumber (defines shortest wavelength of the surface as lambda_2 = 2*pi/q2)
    sg.spectrum.q2 = q2
    # Defines slope of the spectrum
    sg.spectrum.hurst = hurst

    # Generates the surface given the spectrum
    surface = sg.buildSurface()

    # Comment not to normalize the RMSSLope
    surface /= scale_factor_surface

    rms_slope = tm.Statistics2D.computeSpectralRMSSlope(surface)

    loads = np.linspace(0, p_target, num_load_steps + 1)
    loads = loads[1:]

    # Solve for given load path:
    A_raw = np.zeros((len(loads),))
    A_cor = np.zeros_like(A_raw)

    Dmax = np.zeros_like(A_raw)
    Dmin = np.zeros_like(A_raw)
    Dmean = np.zeros_like(A_raw)
    run_times = np.zeros_like(A_raw)
    setup_times = np.zeros_like(A_raw)

    run_times_intermediate = np.zeros((len(loads) + 1,))
    if solve_contact_problem:
        start_time = time.time()
        run_times_intermediate[0] = start_time
        try:
            for i, load in enumerate(loads):

                start_setup_time = time.time()
                # Grid cell area
                dx = L / n
                dA = dx**2

                # Creates the model
                model = tm.ModelFactory.createModel(
                    tm.model_type.basic_2d, [L, L], [n, n]
                )

                # Uncomment to solve equivalent non-periodic problem:
                if not periodic:
                    tm.ModelFactory.registerNonPeriodic(model, "dcfft")

                # Mechanical parameters
                model.E = 1.0

                # Initialize the solver
                solver = tm.PolonskyKeerRey(model, surface, tolerance=solver_tolerance)

                # Uncomment to solve equivalent non-periodic problem:
                if not periodic:
                    solver.setIntegralOperator("dcfft")

                setup_times[i] = time.time() - start_setup_time

                solver.solve(load)

                current_model = solver.model

                # To compute the true displacement (for non-periodic problem), one needs to re-evaluate the displacement
                if not periodic:
                    current_model.operators["dcfft"](
                        current_model.traction, current_model.displacement
                    )

                # Effective contact area (uncorrected)
                A_raw[i] = dA * len(
                    current_model.traction[current_model.traction > 0.0]
                )

                # Perform correction of the area according to Yastrebov:
                M = count_switches(current_model.traction)
                Sd = M * dx
                A_cor[i] = A_raw[i] - (np.pi - 1 + np.log(2)) / 24 * Sd * dx

                Dmin[i] = np.min(current_model.displacement)
                Dmean[i] = np.mean(current_model.displacement)
                Dmax[i] = np.max(current_model.displacement)
                run_times_intermediate[i + 1] = time.time()

            run_times = np.diff(run_times_intermediate)

        except Exception as e:
            print("Exception message:", e)

    return surface, A_raw, A_cor, loads, rms_slope, Dmin, Dmean, Dmax, run_times


if __name__ == "__main__":

    surface, A_raw, A_cor, loads, rms_slope, Dmin, Dmean, Dmax, run_times = (
        generate_surface_and_solve_pressure_driven_eff_area_load_path()
    )
    A_ab = np.sqrt(2 * np.pi) * loads / rms_slope
    A_pr = scipy.special.erf(np.sqrt(2) * loads / rms_slope)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(loads / rms_slope, A_raw[:], "-v", label="n=512")
    axs[1].plot(loads / rms_slope, A_cor[:], "-^", label="n=512")
    axs[0].set_title("raw data")
    axs[1].set_title("corrected data")
    for i in range(2):
        axs[i].plot(loads / rms_slope, A_ab, label="BGT")
        axs[i].plot(loads / rms_slope, A_pr, label="Persson")
        axs[i].set_xlabel("p0")
        axs[i].set_ylabel("A/A0")
        axs[i].grid(True)
        axs[i].set_xlim([0.0, 0.15])
        axs[i].set_ylim([0.0, 0.30])
        axs[i].legend(loc="lower right")
    plt.savefig("plot_load.png")

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.plot(loads / rms_slope, Dmin / np.max(surface), "-^", label="D_min")
    axs.plot(loads / rms_slope, Dmean / np.max(surface), "-^", label="D_mean")
    axs.plot(loads / rms_slope, Dmax / np.max(surface), "-<", label="D_max")
    axs.set_ylabel("D/zmax")
    axs.set_xlabel("p0")
    axs.grid(True)
    axs.legend(loc="upper right")
    axs.set_title("displacements")
    plt.savefig("plot_disp.png")
