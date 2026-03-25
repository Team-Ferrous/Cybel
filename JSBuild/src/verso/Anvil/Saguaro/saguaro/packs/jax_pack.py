"""JAX pack."""

from .base import PackSpec

JAX_PACK = PackSpec(
    name="jax_pack",
    description="JAX arrays, jit/vmap/pmap flows, and XLA-oriented training surfaces.",
    keywords=["jax", "jnp", "jit", "vmap", "pmap", "flax", "optax"],
    languages=["python", "cpp"],
    file_hints=["jax", "flax", "optax", "xla"],
)
