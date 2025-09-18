from queens.distributions import Uniform
from queens.parameters.parameters import Parameters

HURST = Uniform(lower_bound=0.50, upper_bound=0.80)
FAR_FIELD_DISPLACEMENT = Uniform(lower_bound=5.0, upper_bound=45.0)

ROUGH_SURFACE_PARAMETERS = Parameters(
    hurst=HURST, far_field_displacement=FAR_FIELD_DISPLACEMENT
)
