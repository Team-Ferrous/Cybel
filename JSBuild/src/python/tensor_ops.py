# tensor/ops.py

TEO_FLOAT   = "float32"
TEO_COMPLEX = "complex128"
TEO_INT     = "int32"
TEO_INT64   = "int64"
TEO_STRING  = "string"
TEO_TENSOR  = "tensor"
_BACKEND    = "tensorflow"

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

def get_static_value(x):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.get_static_value(x)
    elif b == "torch":
        return x.item() if x.numel() == 1 else x
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.asarray(x)
    elif b == "numpy":
        return x

def conj(x):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.math.conj(x)
    elif b == "torch":
        import torch
        return torch.conj(x)
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.conj(x)
    elif b == "numpy":
        return np.conj(x)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")


# --------------------
# Real part
# --------------------
def real(x):
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
        return np.real(x)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")


# --------------------
# Matrix-vector multiplication
# --------------------
def matvec(A, x):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.linalg.matvec(A, x)
    elif b == "torch":
        import torch
        return torch.mv(A, x)
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.dot(A, x)
    elif b == "numpy":
        return np.dot(A, x)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
# teo_linalg.py
def qr(mat):
    """Backend-agnostic QR decomposition.

    Args:
        mat: Input 2D matrix

    Returns:
        q, r: Orthogonal/unitary matrix and upper-triangular matrix
    """
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.linalg.qr(mat)
    elif b == "torch":
        import torch
        return torch.linalg.qr(mat)  # returns namedtuple Q,R
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.linalg.qr(mat)
    elif b == "numpy":
        import numpy as np
        return np.linalg.qr(mat)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def transpose(x, perm=None):
    """Backend-agnostic transpose.

    Args:
        x: Input tensor/matrix
        perm: Optional axes permutation (None for reversing last two axes)
    """
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.transpose(x, perm=perm)
    elif b == "torch":
        import torch
        if perm is None:
            return x.T
        return x.permute(perm)
    elif b == "jax":
        import jax.numpy as jnp
        return jnp.transpose(x, axes=perm)
    elif b == "numpy":
        return np.transpose(x, axes=perm)
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    

def size(tensor, axis=None):
    """Backend-agnostic tensor size/shape.

    Args:
        tensor: Input tensor
        axis: Optional axis to get size along

    Returns:
        int or tensor of sizes
    """
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        if axis is None:
            return tf.shape(tensor)
        return tf.shape(tensor)[axis]
    elif b == "torch":
        import torch
        if axis is None:
            return torch.tensor(tensor.shape)
        return tensor.shape[axis]
    elif b == "jax":
        import jax.numpy as jnp
        if axis is None:
            return jnp.array(tensor.shape)
        return tensor.shape[axis]
    elif b == "numpy":
        import numpy as np
        if axis is None:
            return np.array(tensor.shape)
        return tensor.shape[axis]
    else:
        raise RuntimeError(f"Unsupported backend: {b}")
    
def do_not_convert(fn):
    b = backend()
    if b == "tensorflow":
        import tensorflow as tf
        return tf.autograph.experimental.do_not_convert(fn)
    else:
        return fn
    
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
        return tf.load_op_library(path)
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


def elu(x, alpha=1.0):
    bck = backend()
    if bck == "tensorflow":
        import tensorflow as tf
        return tf.nn.elu(x)
    elif bck == "torch":
        import torch
        return torch.nn.functional.elu(x, alpha=alpha)
    elif bck == "jax":
        import jax.numpy as jnp
        return jnp.where(x > 0, x, alpha * (jnp.exp(x) - 1))
    elif bck == "numpy":
        import numpy as np
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    else:
        raise RuntimeError(f"Unsupported backend: {bck}")


def softmax(x, axis=-1):
    bck = backend()
    if bck == "tensorflow":
        import tensorflow as tf
        return tf.nn.softmax(x, axis=axis)
    elif bck == "torch":
        import torch
        return torch.nn.functional.softmax(x, dim=axis)
    elif bck == "jax":
        import jax.numpy as jnp
        e_x = jnp.exp(x - jnp.max(x, axis=axis, keepdims=True))
        return e_x / jnp.sum(e_x, axis=axis, keepdims=True)
    elif bck == "numpy":
        import numpy as np
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
    else:
        raise RuntimeError(f"Unsupported backend: {bck}")
    
class TEO_LorentzianGATLayer:
    """Backend-agnostic Lorentzian Graph Attention layer using TEO."""

    def __init__(self, feature_dim: int, num_heads: int = 1, backend_name=None):
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self._backend = backend_name or backend()  # default global backend

        # Initialize weights using TEO
        self.transform_weights = random_uniform(
            (feature_dim, feature_dim), dtype=dtype_map(TEO_FLOAT)
        )
        self.transform_bias = zeros((feature_dim,), dtype=dtype_map(TEO_FLOAT))

        self.activation_weights = random_uniform(
            (feature_dim, feature_dim), dtype=dtype_map(TEO_FLOAT)
        )
        self.activation_bias = zeros((feature_dim,), dtype=dtype_map(TEO_FLOAT))

        self.output_weights = random_uniform(
            (feature_dim, feature_dim), dtype=dtype_map(TEO_FLOAT)
        )
        self.output_bias = zeros((feature_dim,), dtype=dtype_map(TEO_FLOAT))

    def call(
        self,
        node_features,
        adj_indices,
        adj_values,
        adj_dense_shape,
        attention_weights,
    ):
        """Apply Lorentzian GAT using TEO ops or fused op if available."""
        if hasattr(TEO, "fused_lorentzian_gat_op") and fused_lorentzian_gat_op is not None:
            return fused_lorentzian_gat_op(
                node_features=node_features,
                adj_indices=adj_indices,
                adj_values=adj_values,
                adj_dense_shape=adj_dense_shape,
                attention_weights=attention_weights,
                lor_transform_weights=self.transform_weights,
                lor_transform_bias=self.transform_bias,
                lor_activation_weights=self.activation_weights,
                lor_activation_bias=self.activation_bias,
                lor_output_weights=self.output_weights,
                lor_output_bias=self.output_bias,
            )

        # Otherwise, fall back to pure TEO implementation
        # (example: matmul, Lorentzian inner products, softmax)
        x = matmul(node_features, self.transform_weights) + self.transform_bias
        x = elu(x)  # or other Lorentzian-specific activation
        x = matmul(x, self.activation_weights) + self.activation_bias
        # Apply attention weights
        x = x * attention_weights[..., None]
        x = matmul(x, self.output_weights) + self.output_bias
        return x

    def get_config(self):
        return {
            "feature_dim": self.feature_dim,
            "num_heads": self.num_heads,
            "backend": self._backend,
        }