from queens.distributions import Uniform
from queens.parameters.parameters import Parameters

HURST = Uniform(lower_bound=0.50, upper_bound=0.80)
NUM_PATCHES = Uniform(lower_bound=1, upper_bound=64)

ROUGH_SURFACE_PARAMETERS = Parameters(
    hurst=HURST,
    num_patches=NUM_PATCHES,
)
