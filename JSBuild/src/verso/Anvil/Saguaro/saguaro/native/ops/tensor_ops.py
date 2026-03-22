# tensor/ops.py

TEO_FLOAT   = "float32"
TEO_FLOAT64 = "float64"
TEO_BOOL    = "bool"
TEO_COMPLEX = "complex128"
TEO_INT     = "int32"
TEO_INT64   = "int64"
TEO_STRING  = "string"
TEO_TENSOR  = "tensor"
_BACKEND    = "tensorflow"

def Tensor():
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.Tensor

def initializer(name_or_instance):
    """
    TEO wrapper for backend initializers.

    Args:
        name_or_instance: Either a string (like "zeros", "glorot_uniform") or a backend initializer instance.

    Returns:
        Backend-specific initializer.
    """
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.keras.initializers.get(name_or_instance)
    elif b == "torch":
        import torch
        # Torch doesn’t have a direct initializer object; return a callable
        if isinstance(name_or_instance, str):
            if name_or_instance.lower() == "zeros":
                return lambda shape: torch.zeros(shape)
            elif name_or_instance.lower() == "ones":
                return lambda shape: torch.ones(shape)
            elif name_or_instance.lower() in {"glorot_uniform", "xavier_uniform"}:
                import math, torch
                def fn(shape):
                    fan_in, fan_out = shape[-2], shape[-1]
                    limit = math.sqrt(6 / (fan_in + fan_out))
                    return torch.empty(shape).uniform_(-limit, limit)
                return fn
            else:
                raise RuntimeError(f"Unsupported torch initializer: {name_or_instance}")
        else:
            return name_or_instance
    elif b == "jax":
        import jax.numpy as jnp
        # Return a callable function
        return lambda shape, key=None: jnp.zeros(shape) if name_or_instance == "zeros" else jnp.ones(shape)
    elif b == "numpy":
        import numpy as np
        if name_or_instance == "zeros":
            return lambda shape: np.zeros(shape)
        elif name_or_instance == "ones":
            return lambda shape: np.ones(shape)
        else:
            raise RuntimeError(f"Unsupported numpy initializer: {name_or_instance}")
    else:
        raise RuntimeError(f"Unsupported backend: {b}")


def serialize_initializer(init):
    """
    TEO wrapper for tf.keras.initializers.serialize

    Args:
        init: Backend initializer instance

    Returns:
        Serializable dict/string representation
    """
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.keras.initializers.serialize(init)
    else:
        # For other backends, just return the callable or string name
        return init
    
def TensorSpec(shape, dtype):
    """
    TEO TensorSpec abstraction.

    Args:
        shape: Tensor shape (tuple or list)
        dtype: TEO dtype (e.g., TEO.TEO_FLOAT, TEO.TEO_INT)

    Returns:
        Backend-specific TensorSpec or placeholder
    """
    b = backend()

    if b == "tensorflow":
        import tensorflow as tf
        return tf.TensorSpec(shape=shape, dtype=TEO.dtype_map(dtype))
    elif b in {"jax", "torch", "numpy"}:
        # Other backends: return a simple tuple (shape, dtype)
        return (shape, dtype)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def orthogonal_initializer(shape, gain=1.0, dtype=None, seed=None):
    b = backend()
    
    if b == "tensorflow":
        import tensorflow as tf
        init = tf.keras.initializers.Orthogonal(gain=gain, seed=seed)
        return init(shape, dtype=dtype)
    
    elif b == "torch":
        import torch
        from torch.nn import init
        out = torch.empty(shape, dtype=dtype) if dtype is not None else torch.empty(shape)
        if seed is not None:
            torch.manual_seed(seed)
        init.orthogonal_(out, gain=gain)
        return out
    
    elif b == "jax":
        import jax.numpy as jnp
        from jax import random
        key = random.PRNGKey(seed) if seed is not None else random.PRNGKey(0)
        flat_shape = (shape[0], int(jnp.prod(jnp.array(shape[1:]))))
        a = random.normal(key, flat_shape, dtype=dtype if dtype else jnp.float32)
        u, _, v = jnp.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return gain * q.astype(dtype) if dtype else gain * q
    
    elif b == "numpy":
        import numpy as np
        rng = np.random.default_rng(seed)
        flat_shape = (shape[0], int(np.prod(shape[1:])))
        a = rng.normal(size=flat_shape).astype(dtype if dtype else np.float32)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return gain * q.astype(dtype) if dtype else gain * q
    
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
     
def scatter_nd(indices, updates, shape):
    b = backend()
    
    if b == "tensorflow":
        import tensorflow as tf
        return tf.scatter_nd(indices, updates, shape)
    
    elif b == "torch":
        import torch
        out = torch.zeros(shape, dtype=updates.dtype, device=updates.device)
        # Flatten for advanced indexing
        idx = tuple(indices.T.tolist())
        out[idx] = updates
        return out
    
    elif b == "jax":
        import jax.numpy as jnp
        out = jnp.zeros(shape, dtype=updates.dtype)
        out = out.at[tuple(indices.T)].set(updates)
        return out
    
    elif b == "numpy":
        import numpy as np
        out = np.zeros(shape, dtype=updates.dtype)
        out[tuple(indices.T)] = updates
        return out
    
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def stateless_uniform(shape, seed, minval=0.0, maxval=1.0, dtype=None):
    b = backend()
    
    if b == "tensorflow":
        import tensorflow as tf
        return tf.random.stateless_uniform(
            shape, seed=seed, minval=minval, maxval=maxval, dtype=dtype
        )
    elif b == "torch":
        import torch
        # Torch doesn't have stateless RNG by default; simulate with manual seed
        g = torch.Generator()
        g.manual_seed(seed[0] + seed[1]*2**32)  # combine two int32s like TF
        return (maxval - minval) * torch.rand(shape, generator=g, dtype=dtype) + minval
    elif b == "jax":
        import jax.numpy as jnp
        import jax.random as jr
        key = jr.PRNGKey(seed[0] ^ seed[1])  # simple combination
        return jr.uniform(key, shape, minval=minval, maxval=maxval, dtype=dtype)
    elif b == "numpy":
        import numpy as np
        np.random.seed(seed[0] ^ seed[1])
        return np.random.uniform(low=minval, high=maxval, size=shape).astype(dtype)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def truncated_normal(shape, mean=0.0, stddev=1.0):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.random.truncated_normal(shape, mean=mean, stddev=stddev)
    elif b == "torch":
        import torch
        # Torch doesn't have a direct truncated normal; approximate using normal with clamping
        tmp = torch.normal(mean=mean, std=stddev, size=shape)
        # Truncate at 2 stddev (common TF default)
        return torch.clamp(tmp, mean - 2*stddev, mean + 2*stddev)
    elif b == "jax":
        import jax.numpy as jnp
        import jax.random as jr
        key = jr.PRNGKey(0)  # you can pass a key
        tmp = jr.normal(key, shape) * stddev + mean
        return jnp.clip(tmp, mean - 2*stddev, mean + 2*stddev)
    elif b == "numpy":
        import numpy as np
        tmp = np.random.normal(loc=mean, scale=stddev, size=shape)
        return np.clip(tmp, mean - 2*stddev, mean + 2*stddev)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")

def squeeze(x, axis):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.squeeze(x, axis=axis)
    elif b == "torch":
        import torch
        return x.squeeze(dim=axis)
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.squeeze(x, axis=axis)
    elif b == "numpy":
        import numpy as np
        return np.squeeze(x, axis=axis)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")


def set_seed(seed: int):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        tf.random.set_seed(seed)
    elif b == "torch":
        import torch
        torch.manual_seed(seed)
    elif b == "jax":
        import jax
        jax.random.PRNGKey(seed)  # JAX uses keys for randomness
    elif b == "numpy":
        import numpy as np
        np.random.seed(seed)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")


def uniform_random(shape, minval=-1.0, maxval=1.0):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.random.uniform(shape, minval=minval, maxval=maxval)
    elif b == "torch":
        import torch
        return (maxval - minval) * torch.rand(*shape) + minval
    elif b == "jax":
        import jax.numpy as jnp
        import jax.random as jr
        key = jr.PRNGKey(0)  # you can pass a key if needed
        return jr.uniform(key, shape, minval=minval, maxval=maxval)
    elif b == "numpy":
        import numpy as np
        return np.random.uniform(minval, maxval, size=shape)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def ifft(x):
    """Compute inverse FFT in a backend-agnostic way."""
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        x_complex = tf.cast(x, tf.complex128)
        return tf.signal.ifft(x_complex)
    elif b == "torch":
        import torch
        x_complex = x.to(torch.complex128)
        return torch.fft.ifft(x_complex)
    elif b == "jax":
        import jax.numpy as jnp
        x_complex = x.astype(jnp.complex128)
        return jnp.fft.ifft(x_complex)
    elif b == "numpy":
        import numpy as np
        x_complex = x.astype(np.complex128)
        return np.fft.ifft(x_complex)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")


def real(x):
    """Return real part in a backend-agnostic way."""
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.math.real(x)
    elif b == "torch":
        import torch
        return x.real
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.real(x)
    elif b == "numpy":
        import numpy as np
        return np.real(x)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
class Layer:
    def __init__(self, name="layer", **kwargs):
        self.name = name
        self._weights = {}
    
    def add_weight(self, name, shape, initializer, trainable=True):
        """Backend-agnostic weight creation."""
        value = initializer(shape)
        self._weights[name] = {"value": value, "trainable": trainable}
        return value

    def get_config(self):
        return {"name": self.name}

    def call(self, *args, **kwargs):
        """Override in subclass for forward pass."""
        raise NotImplementedError
    
class TEOError(Exception):
    """Base error for TEO backend issues."""

class TEONotFoundError(TEOError):
    """Raised when a required resource or op cannot be found."""

def random_normal(shape, stddev=1.0):
    b = backend()

    if b == "tensorflow":
        import tensorflow as tf
        return tf.random.normal(shape, stddev=stddev)

    elif b == "torch":
        import torch
        return torch.randn(*shape) * stddev

    elif b == "jax":
        import jax.numpy as jnp
        import jax.random as jr
        key = jr.PRNGKey(0)
        return jr.normal(key, shape) * stddev

    elif b == "numpy":
        import numpy as np
        return np.random.normal(0, stddev, shape)

    else:
        raise RuntimeError(f"Unsupported backend: {b}")

def to_tensor(value, dtype=None):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.convert_to_tensor(value, dtype=dtype)
    elif b == "torch":
        import torch
        t = torch.tensor(value, dtype=dtype)
        return t
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.array(value, dtype=dtype)
    elif b == "numpy":
        import numpy as np
        return np.array(value, dtype=dtype)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def variable(value, trainable=True, name=None):
    b = backend()

    if b == "tensorflow":
        import tensorflow as tf
        return tf.Variable(value, trainable=trainable, name=name)

    elif b == "torch":
        import torch
        return torch.nn.Parameter(value)

    elif b == "jax":
        # JAX does not have mutable variables
        return value

    elif b == "numpy":
        return value

    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def has_op(name: str) -> bool:
    b = backend()

    if b == "tensorflow":
        import tensorflow as tf
        return name in tf.raw_ops.__dict__

    elif b == "torch":
        import torch
        return hasattr(torch.ops, name)

    elif b == "jax":
        # JAX usually doesn't expose raw ops like this
        return False

    elif b == "numpy":
        return False

    else:
        raise RuntimeError(f"Unsupported backend: {b}")

def get_op(name: str):
    b = backend()

    if b == "tensorflow":
        import tensorflow as tf
        return getattr(tf.raw_ops, name)

    elif b == "torch":
        import torch
        return getattr(torch.ops, name)

    else:
        raise RuntimeError(f"Backend {b} does not support raw ops")
    
def map_backend_error(err: Exception):
    b = backend()

    if b == "tensorflow":
        import tensorflow as tf
        if isinstance(err, tf.errors.NotFoundError):
            raise TEONotFoundError(str(err)) from err

    elif b == "torch":
        import torch
        # torch doesn't have exact equivalent but missing ops/files often raise RuntimeError
        if isinstance(err, RuntimeError) and "not found" in str(err).lower():
            raise TEONotFoundError(str(err)) from err

    elif b == "jax":
        # JAX tends to use RuntimeError or ValueError
        if isinstance(err, RuntimeError) and "not found" in str(err).lower():
            raise TEONotFoundError(str(err)) from err

    raise err

def custom_gradient(fn):
    b = backend()

    if b == "tensorflow":
        import tensorflow as tf
        return tf.custom_gradient(fn)

    elif b == "torch":
        import torch

        class TorchCustom(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                out, backward = fn(*args)
                ctx.backward_fn = backward
                ctx.save_for_backward(*args)
                return out

            @staticmethod
            def backward(ctx, grad_output):
                return ctx.backward_fn(grad_output)

        def wrapper(*args):
            return TorchCustom.apply(*args)

        return wrapper

    elif b == "jax":
        import jax

        return jax.custom_vjp(fn)

    elif b == "numpy":
        # NumPy has no autograd
        return fn

    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def dtype_map(dtype: str):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        mapping = {
            TEO_FLOAT: tf.float32,
            TEO_COMPLEX: tf.complex128,
            TEO_INT: tf.int32,
            TEO_INT64: tf.int64,
            TEO_STRING: tf.string,
            TEO_TENSOR: tf.Tensor
        }
    elif b == "torch":
        import torch
        mapping = {
            TEO_FLOAT:   torch.float32,
            TEO_COMPLEX: torch.complex128,
            TEO_INT:     torch.int32,
            TEO_INT64:   torch.int64,
            TEO_STRING:  str,
            TEO_TENSOR:  torch.Tensor
        }
    elif b == "jax":
        import jax
        import jax.numpy as jnp
        mapping = {
            TEO_FLOAT:   jnp.float32,
            TEO_COMPLEX: jnp.complex128,
            TEO_INT:     jnp.int32,
            TEO_INT64:   jnp.int64,
            TEO_STRING:  jnp.string,
            TEO_TENSOR:  jax.Array
        }
    elif b == "numpy":
        import numpy as np
        mapping = {
            TEO_FLOAT:   np.float32,
            TEO_COMPLEX: np.complex128,
            TEO_INT:     np.int32,
            TEO_INT64:   np.int64,
            TEO_STRING:  str,
            TEO_TENSOR:  np.ndarray	
        }
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    return mapping[dtype]

def range(stop):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.range(stop)
    elif b == "torch":
        import torch
        return torch.arange(stop)
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.arange(stop)
    elif b == "numpy":
        import numpy as np
        return np.arange(stop)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")

def abs(x):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.abs(x)
    elif b == "torch":
        import torch
        return torch.abs(x)
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.abs(x)
    elif b == "numpy":
        import numpy as np
        return np.abs(x)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")

def newaxis(x):
    # This one is just a syntactic helper
    # Works for all backends with slicing
    return x[..., None]  # equivalent to [:, None] or [None, :]
    
def logical_and(a, b):
    bck = backend()
    if bck == "tensorflow":
        import tensorflow as tf
        return tf.logical_and(a, b)
    elif bck == "torch":
        import torch
        return a & b
    elif bck == "jax":
        import jax.numpy as jnp
        return jnp.logical_and(a, b)
    elif bck == "numpy":
        import numpy as np
        return np.logical_and(a, b)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def band_part(x, num_lower: int, num_upper: int):
    """Return the banded part of a matrix."""
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.linalg.band_part(x, num_lower, num_upper)
    elif b == "torch":
        import torch
        # Torch has torch.tril/triu; combine for band
        lower = torch.tril(x, num_lower)
        upper = torch.triu(lower, -num_upper if num_upper != -1 else None)
        return upper
    elif b == "jax":
        import jax.numpy as jnp
        mask = jnp.tri(x.shape[-2], x.shape[-1], k=num_upper) - jnp.tri(x.shape[-2], x.shape[-1], k=-num_lower-1)
        return x * mask
    elif b == "numpy":
        import numpy as np
        mask = np.tri(x.shape[-2], x.shape[-1], k=num_upper) - np.tri(x.shape[-2], x.shape[-1], k=-num_lower-1)
        return x * mask
    else:
        raise RuntimeError(f"Unsupported backend: {b}")


def ones(shape, dtype=None):
    """Create a tensor of ones."""
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.ones(shape, dtype=dtype or tf.float32)
    elif b == "torch":
        import torch
        dt = dtype or torch.float32
        return torch.ones(shape, dtype=dt)
    elif b == "jax":
        import jax.numpy as jnp
        dt = dtype or jnp.float32
        return jnp.ones(shape, dtype=dt)
    elif b == "numpy":
        import numpy as np
        dt = dtype or np.float32
        return np.ones(shape, dtype=dt)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def name_scope(name: str):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.name_scope(name)
    elif b == "torch":
        # PyTorch has no direct equivalent, use context manager for logging/namespace
        from contextlib import nullcontext
        return nullcontext()
    elif b == "jax":
        # JAX also has no name_scope; just use nullcontext
        from contextlib import nullcontext
        return nullcontext()
    elif b == "numpy":
        from contextlib import nullcontext
        return nullcontext()
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def tile(x, multiples):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.tile(x, multiples)
    elif b == "torch":
        import torch
        return x.repeat(*multiples)
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.tile(x, multiples)
    elif b == "numpy":
        import numpy as np
        return np.tile(x, multiples)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def reverse(x, axis):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.reverse(x, axis=axis)
    elif b == "torch":
        import torch
        if isinstance(axis, int):
            axis = [axis]
        for a in axis:
            x = torch.flip(x, dims=[a])
        return x
    elif b == "jax":
        import jax.numpy as jnp
        if isinstance(axis, int):
            axis = (axis,)
        return jnp.flip(x, axis=axis)
    elif b == "numpy":
        import numpy as np
        if isinstance(axis, int):
            axis = (axis,)
        return np.flip(x, axis=axis)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def repeat(x, repeats: int, axis: int):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.repeat(x, repeats=repeats, axis=axis)
    elif b == "torch":
        import torch
        return x.repeat_interleave(repeats, dim=axis)
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.repeat(x, repeats, axis=axis)
    elif b == "numpy":
        import numpy as np
        return np.repeat(x, repeats, axis=axis)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def elu(x):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.nn.elu(x)
    elif b == "torch":
        import torch
        return torch.nn.functional.elu(x)
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.where(x > 0, x, jnp.exp(x) - 1)
    elif b == "numpy":
        import numpy as np
        return np.where(x > 0, x, np.exp(x) - 1)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")

def sigmoid(x):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.nn.sigmoid(x)
    elif b == "torch":
        import torch
        return torch.nn.functional.elu(x)
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.where(x > 0, x, jnp.exp(x) - 1)
    elif b == "numpy":
        import numpy as np
        return np.where(x > 0, x, np.exp(x) - 1)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def matmul(a, b):
    bck = backend()
    if bck == "tensorflow":
        import tensorflow as tf
        return tf.matmul(a, b)
    elif bck == "torch":
        import torch
        return torch.matmul(a, b)
    elif bck == "jax":
        import jax.numpy as jnp
        return jnp.matmul(a, b)
    elif bck == "numpy":
        import numpy as np
        return np.matmul(a, b)
    else:
        raise RuntimeError(f"Unsupported backend: {bck}")

def einsum(subscripts: str, *operands):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.einsum(subscripts, *operands)
    elif b == "torch":
        import torch
        return torch.einsum(subscripts, *operands)
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.einsum(subscripts, *operands)
    elif b == "numpy":
        import numpy as np
        return np.einsum(subscripts, *operands)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")

def softmax(x, axis: int = -1):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.nn.softmax(x, axis=axis)
    elif b == "torch":
        import torch
        import torch.nn.functional as F
        return F.softmax(x, dim=axis)
    elif b == "jax":
        import jax.numpy as jnp
        e_x = jnp.exp(x - jnp.max(x, axis=axis, keepdims=True))
        return e_x / jnp.sum(e_x, axis=axis, keepdims=True)
    elif b == "numpy":
        import numpy as np
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def expand_dims(x, axis):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.expand_dims(x, axis)
    elif b == "torch":
        import torch
        return torch.unsqueeze(x, dim=axis)
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.expand_dims(x, axis)
    elif b == "numpy":
        import numpy as np
        return np.expand_dims(x, axis)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def set_backend(name):
    global _BACKEND
    _BACKEND = name

def backend():
    return _BACKEND

def backend_version():
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return {"name": "tensorflow", "version": tf.__version__}
    elif b == "torch":
        import torch
        return {"name": "torch", "version": torch.__version__}
    elif b == "jax":
        import jax
        import jax.numpy as jnp
        return {"name": "jax", "version": jax.__version__}
    elif b == "numpy":
        import numpy as np
        return {"name": "numpy", "version": np.__version__}
    else:
        return {"name": "unknown", "version": "unknown"}

def load_custom_op(path: str):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return load_custom_op(path)
    elif b == "torch":
        import torch
        return torch.ops.load_library(path)
    elif b == "jax":
        # maybe wrap Futhark / XLA call here
        raise NotImplementedError("JAX custom ops must be defined via JIT or XLA")
    elif b == "numpy":
        # pure Python / compiled C extension
        raise NotImplementedError("NumPy custom ops must be implemented in Python or C")
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def sqrt(x):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.sqrt(x)
    elif b == "torch":
        import torch
        return torch.sqrt(x)
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.sqrt(x)
    elif b == "numpy":
        import numpy as np
        return np.sqrt(x)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")


def zeros(shape, dtype=TEO_FLOAT):
    b = backend()
    dt = dtype_map(dtype)
    if b == "tensorflow":
        import tensorflow as tf
        return tf.zeros(shape, dtype=tf.float32 if dtype == float else dt)
    elif b == "torch":
        import torch
        dt = torch.float32 if dtype == float else dt
        return torch.zeros(shape, dtype=dt)
    elif b == "jax":
        import jax.numpy as jnp
        dt = jnp.float32 if dtype == float else dt
        return jnp.zeros(shape, dtype=dt)
    elif b == "numpy":
        import numpy as np
        dt = np.float32 if dtype == float else dt
        return np.zeros(shape, dtype=dt)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")

def softplus(x):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.nn.softplus(x)
    elif b == "torch":
        import torch
        return torch.nn.functional.softplus(x)
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.log1p(jnp.exp(x))
    elif b == "numpy":
        import numpy as np
        return np.log1p(np.exp(x))
    else:
        raise RuntimeError(f"Unsupported backend: {b}")

def l2_normalize(x, axis=-1, epsilon=1e-8):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.nn.l2_normalize(x, axis=axis, epsilon=epsilon)
    elif b == "torch":
        import torch
        norm = torch.linalg.norm(x, dim=axis, keepdim=True)
        return x / (norm + epsilon)
    elif b == "jax":
        import jax.numpy as jnp
        norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
        return x / (norm + epsilon)
    elif b == "numpy":
        import numpy as np
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / (norm + epsilon)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def diag(x):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.linalg.diag(x)
    elif b == "torch":
        import torch
        return torch.diag(x)
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.diag(x)
    elif b == "numpy":
        import numpy as np
        return np.diag(x)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def softplus(x):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.nn.softplus(x)
    elif b == "torch":
        import torch
        return torch.nn.functional.softplus(x)
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.log1p(jnp.exp(x))
    elif b == "numpy":
        import numpy as np
        return np.log1p(np.exp(x))
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def full(shape, value):
    b = backend()

    if b == "tensorflow":
        import tensorflow as tf
        return tf.fill(shape, value)

    elif b == "torch":
        import torch
        return torch.full(shape, value)

    elif b == "jax":
        import jax.numpy as jnp
        return jnp.full(shape, value)

    elif b == "numpy":
        import numpy as np
        return np.full(shape, value)

    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def concat(tensors, axis=0):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.concat(tensors, axis=axis)
    elif b == "torch":
        import torch
        return torch.cat(tensors, dim=axis)
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.concatenate(tensors, axis=axis)
    elif b == "numpy":
        import numpy as np
        return np.concatenate(tensors, axis=axis)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")

def constant_initializer(value):
    """
    Backend-agnostic constant initializer.
    Produces a tensor filled with `value`.
    """
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.constant_initializer(value)
    elif b == "torch":
        import torch
        # Return a callable that takes a shape and returns a tensor
        return lambda shape, dtype=torch.float32: torch.full(shape, value, dtype=dtype)
    elif b == "jax":
        import jax.numpy as jnp
        return lambda shape, dtype=jnp.float32: jnp.full(shape, value, dtype=dtype)
    elif b == "numpy":
        import numpy as np
        return lambda shape, dtype=np.float32: np.full(shape, value, dtype=dtype)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def clip_by_value(x, min_val, max_val):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.clip_by_value(x, min_val, max_val)
    elif b == "torch":
        import torch
        return torch.clamp(x, min=min_val, max=max_val)
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.clip(x, a_min=min_val, a_max=max_val)
    elif b == "numpy":
        import numpy as np
        return np.clip(x, a_min=min_val, a_max=max_val)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def zeros_like(tensor):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.zeros_like(tensor)
    elif b == "torch":
        import torch
        return torch.zeros_like(tensor)
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.zeros_like(tensor)
    elif b == "numpy":
        import numpy as np
        return np.zeros_like(tensor)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")

def maximum(a, b):
    bck = backend()
    if bck == "tensorflow":
        import tensorflow as tf
        return tf.maximum(a, b)
    elif bck == "torch":
        import torch
        return torch.maximum(a, b)
    elif bck == "jax":
        import jax.numpy as jnp
        return jnp.maximum(a, b)
    elif bck == "numpy":
        import numpy as np
        return np.maximum(a, b)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")

def cos(x):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.cos(x)
    elif b == "torch":
        import torch
        return torch.cos(x)
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.cos(x)
    elif b == "numpy":
        import numpy as np
        return np.cos(x)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def constant(value, dtype=TEO_FLOAT):
    b = backend()
    dt = dtype_map(dtype)
    if b == "tensorflow":
        import tensorflow as tf
        #tf_dtype = tf.float32 if dtype == float else dt
        return tf.constant(value, dtype=dt)
    elif b == "torch":
        import torch
        #torch_dtype = torch.float32 if dtype == float else dt
        return torch.tensor(value, dtype=dt)
    elif b == "jax":
        import jax.numpy as jnp
        #jax_dtype = jnp.float32 if dtype == float else dt
        return jnp.array(value, dtype=dt)
    elif b == "numpy":
        import numpy as np
        #np_dtype = np.float32 if dtype == float else dt
        return np.array(value, dtype=dt)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")

def shape(x):
    """Return the shape of a tensor or array in a backend-agnostic way."""
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.shape(x)  # returns a tf.Tensor
    elif b == "torch":
        import torch
        return x.shape  # returns a tuple of ints
    elif b == "jax":
        import jax.numpy as jnp
        return x.shape  # tuple of ints
    elif b == "numpy":
        import numpy as np
        return x.shape  # tuple of ints
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def reshape(x, shape):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.reshape(x, shape)
    elif b == "torch":
        import torch
        return x.reshape(shape)
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.reshape(x, shape)
    elif b == "numpy":
        import numpy as np
        return np.reshape(x, shape)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def log(x, dtype=TEO_FLOAT):
    b = backend()
    dt = dtype_map(dtype)
    if b == "tensorflow":
        import tensorflow as tf
        return tf.math.log(tf.cast(x, dt))
    elif b == "torch":
        import torch
        return torch.log(x.to(dt))
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.log(x.astype(dt))
    elif b == "numpy":
        import numpy as np
        return np.log(x.astype(dt))
    else:
        raise RuntimeError(f"Unsupported backend: {b}")

def reduce_sum(x, axis=None, keepdims=False):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.reduce_sum(x, axis=axis, keepdims=keepdims)
    elif b == "torch":
        import torch
        return torch.sum(x, dim=axis, keepdim=keepdims)
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.sum(x, axis=axis, keepdims=keepdims)
    elif b == "numpy":
        import numpy as np
        return np.sum(x, axis=axis, keepdims=keepdims)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")

def exp(x):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.exp(x)
    elif b == "torch":
        import torch
        return torch.exp(x)
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.exp(x)
    elif b == "numpy":
        import numpy as np
        return np.exp(x)
    else:
        raise RuntimeError(f"Unsupported backend: {b}") 
    
def cast(hidden_states, cast_to):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.cast(hidden_states, cast_to) # tf.complex128
    elif b == "torch":
        import torch
        return hidden_states.to(cast_to) #torch.complex128 
    elif b == "jax":
        import jax.numpy as jnp
        return hidden_states.astype(cast_to) #jnp.complex128)
    elif b == "numpy":
        import numpy as np
        return hidden_states.astype(cast_to) #np.complex128)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def reduce_mean(x, axis):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.reduce_mean(x, axis=axis)
    elif b == "torch":
        return x.mean(dim=axis)
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.mean(x, axis=axis)
    elif b == "numpy":
        import numpy as np
        return np.mean(x, axis=axis)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def fft(x):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.signal.fft(tf.cast(x, tf.complex128))
    elif b == "torch":
        import torch
        return torch.fft.fft(x.to(torch.complex128))
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.fft.fft(x.astype(jnp.complex128))
    elif b == "numpy":
        import numpy as np
        return np.fft.fft(x.astype(np.complex128))
    else:
        raise RuntimeError(f"Unsupported backend: {b}")