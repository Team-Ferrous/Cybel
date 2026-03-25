"""ctypes wrapper for the native C++ GGUF tokenizer."""

from __future__ import annotations

import ctypes
from contextlib import suppress

from core.native.native_ops import load_native_library

_LIB: ctypes.CDLL | None = None


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    lib = load_native_library()
    if not hasattr(lib, "anvil_native_tokenizer_create"):
        raise RuntimeError(
            "Strict native tokenizer symbols are missing from libanvil_native_ops.so"
        )
    c_char_p_p = ctypes.POINTER(ctypes.c_char_p)
    c_int_p = ctypes.POINTER(ctypes.c_int)

    lib.anvil_native_tokenizer_create.argtypes = [
        c_char_p_p,
        ctypes.c_int,
        c_char_p_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]
    lib.anvil_native_tokenizer_create.restype = ctypes.c_void_p

    lib.anvil_native_tokenizer_destroy.argtypes = [ctypes.c_void_p]
    lib.anvil_native_tokenizer_destroy.restype = None

    lib.anvil_native_tokenizer_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        c_int_p,
        ctypes.c_int,
    ]
    lib.anvil_native_tokenizer_encode.restype = ctypes.c_int

    lib.anvil_native_tokenizer_decode.argtypes = [
        ctypes.c_void_p,
        c_int_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.anvil_native_tokenizer_decode.restype = ctypes.c_int

    lib.anvil_native_tokenizer_get_suppressed_token_count.argtypes = [ctypes.c_void_p]
    lib.anvil_native_tokenizer_get_suppressed_token_count.restype = ctypes.c_int

    lib.anvil_native_tokenizer_get_suppressed_tokens.argtypes = [
        ctypes.c_void_p,
        c_int_p,
        ctypes.c_int,
    ]
    lib.anvil_native_tokenizer_get_suppressed_tokens.restype = ctypes.c_int

    _LIB = lib
    return lib


class NativeTokenizer:
    """Strict native tokenizer for production GGUF models."""

    def __init__(
        self,
        vocab_tokens: list[str],
        special_tokens: dict[str, int] | None = None,
        model_path: str | None = None,
        prefer_llama: bool | None = None,
        bpe_merges: list[str] | None = None,
        tokenizer_model: str | None = None,
        tokenizer_pre: str | None = None,
    ) -> None:
        """Initialize the native tokenizer from GGUF-extracted vocab metadata."""
        _ = model_path
        _ = prefer_llama
        self.vocab_tokens = list(vocab_tokens)
        self.vocab_size = len(self.vocab_tokens)
        self.special_tokens = dict(special_tokens or {})
        self.bos_id = int(self.special_tokens.get("bos", -1))
        self.eos_id = int(self.special_tokens.get("eos", -1))
        self.pad_id = int(self.special_tokens.get("pad", -1))
        self.unk_id = int(self.special_tokens.get("unk", -1))
        self.bpe_merges = list(bpe_merges or [])
        self.tokenizer_model = str(tokenizer_model or "")
        self.tokenizer_pre = str(tokenizer_pre or "")

        self._vocab_bytes = [str(token).encode("utf-8") for token in self.vocab_tokens]
        self._merge_bytes = [str(merge).encode("utf-8") for merge in self.bpe_merges]
        vocab_array = (ctypes.c_char_p * len(self._vocab_bytes))(*self._vocab_bytes)
        merge_array = (
            (ctypes.c_char_p * len(self._merge_bytes))(*self._merge_bytes)
            if self._merge_bytes
            else None
        )

        self._handle = _lib().anvil_native_tokenizer_create(
            vocab_array,
            len(self._vocab_bytes),
            merge_array,
            len(self._merge_bytes),
            self.bos_id,
            self.eos_id,
            self.pad_id,
            self.unk_id,
            self.tokenizer_model.encode("utf-8"),
            self.tokenizer_pre.encode("utf-8"),
        )
        if not self._handle:
            raise RuntimeError(
                "Failed to initialize native tokenizer. "
                f"model={self.tokenizer_model!r} pre={self.tokenizer_pre!r}"
            )

        count = _lib().anvil_native_tokenizer_get_suppressed_token_count(self._handle)
        if count < 0:
            raise RuntimeError(
                "Failed to query suppressed token ids from native tokenizer."
            )
        ids = (ctypes.c_int * count)()
        filled = _lib().anvil_native_tokenizer_get_suppressed_tokens(
            self._handle, ids, count
        )
        if filled < 0:
            raise RuntimeError(
                "Failed to read suppressed token ids from native tokenizer."
            )
        self.decode_skip_ids = {int(ids[idx]) for idx in range(filled)}

    def close(self) -> None:
        """Release the native tokenizer handle."""
        if getattr(self, "_handle", None):
            _lib().anvil_native_tokenizer_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:  # pragma: no cover
        """Best-effort finalizer for the native handle."""
        with suppress(Exception):
            self.close()

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode UTF-8 text to token ids via the native tokenizer."""
        payload = str(text or "").encode("utf-8")
        required = _lib().anvil_native_tokenizer_encode(
            self._handle,
            payload,
            int(bool(add_bos)),
            int(bool(add_eos)),
            None,
            0,
        )
        if required < 0:
            raise RuntimeError("Native tokenizer encode failed.")
        if required == 0:
            return []
        buffer = (ctypes.c_int * required)()
        written = _lib().anvil_native_tokenizer_encode(
            self._handle,
            payload,
            int(bool(add_bos)),
            int(bool(add_eos)),
            buffer,
            required,
        )
        if written < 0:
            raise RuntimeError("Native tokenizer encode failed.")
        return [int(buffer[idx]) for idx in range(written)]

    def decode(self, token_ids: list[int], skip_special: bool = True) -> str:
        """Decode token ids to UTF-8 text via the native tokenizer."""
        ids = [int(token_id) for token_id in token_ids]
        if ids:
            id_buffer = (ctypes.c_int * len(ids))(*ids)
            id_ptr = id_buffer
        else:
            id_buffer = None
            id_ptr = None
        required = _lib().anvil_native_tokenizer_decode(
            self._handle,
            id_ptr,
            len(ids),
            int(bool(skip_special)),
            None,
            0,
        )
        if required < 0:
            raise RuntimeError("Native tokenizer decode failed.")
        if required == 0:
            return ""
        buffer = ctypes.create_string_buffer(required + 1)
        written = _lib().anvil_native_tokenizer_decode(
            self._handle,
            id_ptr,
            len(ids),
            int(bool(skip_special)),
            buffer,
            required + 1,
        )
        if written < 0:
            raise RuntimeError("Native tokenizer decode failed.")
        return buffer.value.decode("utf-8", errors="replace")

    def batch_encode(
        self,
        texts: list[str],
        padding: bool = True,
        max_length: int | None = None,
        add_bos: bool = True,
    ) -> tuple[list[list[int]], list[list[int]] | None]:
        """Encode a batch and optionally pad to uniform length."""
        encoded = [self.encode(text, add_bos=add_bos) for text in texts]
        if max_length is not None:
            encoded = [seq[: int(max_length)] for seq in encoded]
        if not padding:
            return encoded, None

        max_len = max((len(seq) for seq in encoded), default=0)
        padded: list[list[int]] = []
        mask: list[list[int]] = []
        pad_id = self.pad_id if self.pad_id >= 0 else 0
        for seq in encoded:
            row = list(seq)
            pad_count = max(0, max_len - len(row))
            padded.append(row + [pad_id] * pad_count)
            mask.append([1] * len(row) + [0] * pad_count)
        return padded, mask

    def __len__(self) -> int:
        """Return the vocabulary size."""
        return self.vocab_size
