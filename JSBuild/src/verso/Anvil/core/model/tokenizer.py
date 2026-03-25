"""Strict native tokenizer helpers for GGUF-backed models."""

from __future__ import annotations

from functools import lru_cache
import re
from typing import Dict, List, Optional, Tuple


@lru_cache(maxsize=1)
def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b in bs:
            continue
        bs.append(b)
        cs.append(256 + n)
        n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class AnvilTokenizer:
    """Tokenizer backed directly by GGUF vocab metadata.

    The production path is fail-closed and does not route through `llama_cpp`
    or Hugging Face tokenizers.
    """

    def __init__(
        self,
        vocab_tokens: List[str],
        special_tokens: Dict[str, int] | None = None,
        model_path: Optional[str] = None,
        prefer_llama: Optional[bool] = None,
        bpe_merges: Optional[List[str]] = None,
        tokenizer_model: Optional[str] = None,
    ):
        _ = model_path
        _ = prefer_llama
        self.vocab_tokens = list(vocab_tokens)
        self.vocab_size = len(self.vocab_tokens)
        self.token_to_id: Dict[str, int] = {
            token: idx for idx, token in enumerate(self.vocab_tokens)
        }
        self.special_tokens = dict(special_tokens or {})
        self.bos_id = int(self.special_tokens.get("bos", 1))
        self.eos_id = int(self.special_tokens.get("eos", 2))
        self.pad_id = int(self.special_tokens.get("pad", 0))
        self.unk_id = int(self.special_tokens.get("unk", 0))
        self.bpe_merges = list(bpe_merges or [])
        self.tokenizer_model = str(tokenizer_model or "")
        self._use_byte_level = "gpt2" in self.tokenizer_model.lower()
        self._byte_encoder = _bytes_to_unicode() if self._use_byte_level else {}
        self._byte_decoder = (
            {value: key for key, value in self._byte_encoder.items()}
            if self._use_byte_level
            else {}
        )
        self._merge_ranks = {
            tuple(merge.split()): idx
            for idx, merge in enumerate(self.bpe_merges)
            if len(merge.split()) == 2
        }
        self._bpe_cache: Dict[str, List[str]] = {}
        self._common_cache: Dict[str, List[int]] = {}
        self._max_token_len = max((len(t) for t in self.vocab_tokens if t), default=0)
        self.control_token_ids = self._infer_control_token_ids()
        self.decode_skip_ids = {
            token_id
            for token_id in set(self.special_tokens.values()) | self.control_token_ids
            if 0 <= int(token_id) < self.vocab_size
        }
        self._special_token_text_to_id = {
            str(self.vocab_tokens[token_id]): int(token_id)
            for token_id in sorted(self.decode_skip_ids, key=int)
            if 0 <= int(token_id) < self.vocab_size and self.vocab_tokens[int(token_id)]
        }
        self._special_token_texts = sorted(
            self._special_token_text_to_id,
            key=len,
            reverse=True,
        )
        self._byte_level_pattern = re.compile(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\d+| ?[^A-Za-z\d\s]+|\s+(?!\S)|\s+"
        )

    def _infer_control_token_ids(self) -> set[int]:
        control_ids: set[int] = set()
        exact_markers = {
            "<|im_start|>",
            "<|im_end|>",
            "<|start_of_role|>",
            "<|end_of_role|>",
            "<|end_of_text|>",
            "<|assistant|>",
            "<|user|>",
            "<|system|>",
            "<bos>",
            "<eos>",
            "<pad>",
            "<unk>",
            "<s>",
            "</s>",
        }
        for token_id, token in enumerate(self.vocab_tokens):
            token_text = str(token or "")
            if token_text in exact_markers:
                control_ids.add(int(token_id))
                continue
            if token_text.startswith("<|") and token_text.endswith("|>"):
                control_ids.add(int(token_id))
        return control_ids

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> List[int]:
        if not text:
            tokens: List[int] = []
        elif text in self._common_cache:
            tokens = list(self._common_cache[text])
        else:
            tokens = self._tokenize(text)
            if len(text) < 256:
                self._common_cache[text] = list(tokens)

        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def _match_special_token(self, text: str, pos: int) -> tuple[int, int] | None:
        for token_text in self._special_token_texts:
            if text.startswith(token_text, pos):
                token_id = self._special_token_text_to_id[token_text]
                return token_id, pos + len(token_text)
        return None

    def _tokenize(self, text: str) -> List[int]:
        if self._use_byte_level:
            return self._tokenize_byte_level(text)
        tokens: List[int] = []
        pos = 0
        length = len(text)
        while pos < length:
            special_match = self._match_special_token(text, pos)
            if special_match is not None:
                token_id, pos = special_match
                tokens.append(int(token_id))
                continue

            matched = False
            max_search_len = min(length - pos, self._max_token_len)
            for span in range(max_search_len, 0, -1):
                candidate = text[pos : pos + span]
                if candidate.startswith(" "):
                    g_candidate = "Ġ" + candidate[1:]
                    token_id = self.token_to_id.get(g_candidate)
                    if token_id is not None:
                        tokens.append(int(token_id))
                        pos += span
                        matched = True
                        break
                    sp_candidate = "▁" + candidate[1:]
                    token_id = self.token_to_id.get(sp_candidate)
                    if token_id is not None:
                        tokens.append(int(token_id))
                        pos += span
                        matched = True
                        break

                token_id = self.token_to_id.get(candidate)
                if token_id is not None:
                    tokens.append(int(token_id))
                    pos += span
                    matched = True
                    break

            if matched:
                continue

            char = text[pos]
            if char == " ":
                for fallback in ("Ġ", "▁"):
                    token_id = self.token_to_id.get(fallback)
                    if token_id is not None:
                        tokens.append(int(token_id))
                        matched = True
                        break
            if not matched:
                tokens.append(int(self.token_to_id.get(char, self.unk_id)))
            pos += 1
        return tokens

    def _tokenize_byte_level(self, text: str) -> List[int]:
        tokens: List[int] = []
        pos = 0
        while pos < len(text):
            special_match = self._match_special_token(text, pos)
            if special_match is not None:
                token_id, pos = special_match
                tokens.append(int(token_id))
                continue

            match = self._byte_level_pattern.match(text, pos)
            if match is None:
                piece = text[pos]
                pos += 1
            else:
                piece = match.group(0)
                pos = match.end()
            transformed = "".join(self._byte_encoder[b] for b in piece.encode("utf-8"))
            for subword in self._apply_bpe(transformed):
                token_id = self.token_to_id.get(subword)
                if token_id is None:
                    tokens.extend(self._greedy_tokenize(subword))
                else:
                    tokens.append(int(token_id))
        return tokens

    def _greedy_tokenize(self, text: str) -> List[int]:
        tokens: List[int] = []
        pos = 0
        while pos < len(text):
            matched = False
            max_search_len = min(len(text) - pos, self._max_token_len)
            for span in range(max_search_len, 0, -1):
                candidate = text[pos : pos + span]
                token_id = self.token_to_id.get(candidate)
                if token_id is not None:
                    tokens.append(int(token_id))
                    pos += span
                    matched = True
                    break
            if not matched:
                tokens.append(self.unk_id)
                pos += 1
        return tokens

    def _apply_bpe(self, token: str) -> List[str]:
        cached = self._bpe_cache.get(token)
        if cached is not None:
            return list(cached)
        if not self._merge_ranks:
            result = [token]
            self._bpe_cache[token] = result
            return result

        word = list(token)
        while len(word) > 1:
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            ranked = [
                (self._merge_ranks[pair], pair)
                for pair in pairs
                if pair in self._merge_ranks
            ]
            if not ranked:
                break
            _, best = min(ranked, key=lambda item: item[0])
            merged: List[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    merged.append(word[i] + word[i + 1])
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged
        self._bpe_cache[token] = list(word)
        return list(word)

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        pieces: List[str] = []
        for token_id in token_ids:
            token_int = int(token_id)
            if skip_special and token_int in self.decode_skip_ids:
                continue
            if 0 <= token_int < self.vocab_size:
                token = self.vocab_tokens[token_int]
                if token.startswith("Ġ") or token.startswith("▁"):
                    pieces.append(" " + token[1:])
                elif token.startswith("<0x") and token.endswith(">"):
                    try:
                        pieces.append(chr(int(token[3:-1], 16)))
                    except Exception:
                        pieces.append(token)
                else:
                    pieces.append(token)
            else:
                pieces.append(f"<unk_{token_int}>")
        text = "".join(pieces)
        if self._use_byte_level:
            raw = bytearray()
            for char in text:
                raw.append(self._byte_decoder.get(char, ord(char) & 0xFF))
            return raw.decode("utf-8", errors="replace")
        return text[1:] if text.startswith(" ") else text

    def batch_encode(
        self,
        texts: List[str],
        padding: bool = True,
        max_length: Optional[int] = None,
        add_bos: bool = True,
    ) -> Tuple[List[List[int]], Optional[List[List[int]]]]:
        encoded = [self.encode(text, add_bos=add_bos) for text in texts]
        if max_length is not None:
            encoded = [seq[: int(max_length)] for seq in encoded]

        if not padding:
            return encoded, None

        max_len = max((len(seq) for seq in encoded), default=0)
        padded: List[List[int]] = []
        mask: List[List[int]] = []
        for seq in encoded:
            row = list(seq)
            pad_count = max(0, max_len - len(row))
            padded.append(row + [self.pad_id] * pad_count)
            mask.append([1] * len(row) + [0] * pad_count)
        return padded, mask

    def __len__(self) -> int:
        return self.vocab_size
