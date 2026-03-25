from core.native.native_tokenizer import NativeTokenizer


def test_native_tokenizer_skips_chatml_control_tokens():
    tokenizer = NativeTokenizer(
        vocab_tokens=[
            "H",
            "ello",
            " ",
            "world",
            "<|im_start|>",
            "<|im_end|>",
        ],
        special_tokens={},
        bpe_merges=["H ello"],
        tokenizer_model="gpt2",
        tokenizer_pre="qwen35",
    )

    text = tokenizer.decode([4, 0, 1, 2, 3, 5], skip_special=True)

    assert text == "Hello world"


def test_native_tokenizer_preserves_special_tokens_on_encode():
    tokenizer = NativeTokenizer(
        vocab_tokens=[
            "H",
            " ",
            "e",
            "l",
            "o",
            "w",
            "r",
            "d",
            "world",
            "<|start_of_role|>",
            "<|end_of_text|>",
        ],
        special_tokens={},
        bpe_merges=[],
        tokenizer_model="gpt2",
        tokenizer_pre="dbrx",
    )

    encoded = tokenizer.encode(
        "<|start_of_role|>Hello world<|end_of_text|>",
        add_bos=False,
        add_eos=False,
    )

    assert encoded[0] == 9
    assert encoded[-1] == 10
    assert len(encoded) > 2
