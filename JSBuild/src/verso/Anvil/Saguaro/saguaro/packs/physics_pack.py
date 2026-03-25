"""Physics pack."""

from .base import PackSpec

PHYSICS_PACK = PackSpec(
    name="physics_pack",
    description="Physics simulations, integrators, conservation laws, and units.",
    keywords=["physics", "integrator", "conservation", "timestep", "boundary", "units"],
    languages=["python", "cpp", "fortran"],
    file_hints=["physics", "solver", "simulation", "integrator"],
)
