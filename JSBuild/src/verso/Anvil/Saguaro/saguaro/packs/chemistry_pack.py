"""Chemistry pack."""

from .base import PackSpec

CHEMISTRY_PACK = PackSpec(
    name="chemistry_pack",
    description="Molecular graphs, force fields, energy terms, and unit-sensitive chemistry code.",
    keywords=["molecule", "force field", "bond", "energy", "atom", "angstrom", "hartree"],
    languages=["python", "cpp", "fortran"],
    file_hints=["chem", "molecule", "forcefield", "energy"],
)
