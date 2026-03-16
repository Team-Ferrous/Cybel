"""Computational fluid dynamics pack."""

from .base import PackSpec

CFD_PACK = PackSpec(
    name="cfd_pack",
    description="CFD solvers, meshes, flux terms, and timestep stability heuristics.",
    keywords=["cfd", "mesh", "flux", "navier", "stokes", "courant", "pressure"],
    languages=["python", "cpp", "fortran"],
    file_hints=["cfd", "mesh", "solver", "flux", "pressure"],
)
