from queens.distributions import Uniform
from queens.parameters.parameters import Parameters

HURST = Uniform(lower_bound=0.50, upper_bound=0.80)
NUM_PATCHES = Uniform(lower_bound=1, upper_bound=64)
Q1 = Uniform(lower_bound=1, upper_bound=16)
Q2 = Uniform(lower_bound=32, upper_bound=128)
RANDOM_SEED = Uniform(lower_bound=0, upper_bound=2**64 - 1)
TARGET_PRESSURE = Uniform(lower_bound=0, upper_bound=0.4)

MIRCO_ROUGH_SURFACE_PARAMETERS = Parameters(
    hurst=HURST,
    num_patches=NUM_PATCHES,
)

TAMAAS_ROUGH_SURFACE_PARAMETERS = Parameters(
    hurst=HURST,
    q1=Q1,
    q2=Q2,
    random_seed=RANDOM_SEED,
)

TAMAAS_ROUGH_SURFACE_PARAMETERS_PRESSURE = Parameters(
    hurst=HURST,
    q1=Q1,
    q2=Q2,
    random_seed=RANDOM_SEED,
    target_pressure=TARGET_PRESSURE,
)
