from core.model.tokenizer import AnvilTokenizer


def test_decode_skips_chat_control_tokens():
    tokenizer = AnvilTokenizer(
        vocab_tokens=[
            "<pad>",
            "<bos>",
            "<eos>",
            "<|im_start|>",
            "<|im_end|>",
            "Hello",
            "Ġworld",
        ],
        special_tokens={"pad": 0, "bos": 1, "eos": 2},
        model_path=None,
        prefer_llama=None,
        bpe_merges=None,
        tokenizer_model="bpe",
    )

    text = tokenizer.decode([1, 3, 5, 6, 4, 2], skip_special=True)

    assert text == "Hello world"


def test_decode_skips_granite_role_tokens():
    tokenizer = AnvilTokenizer(
        vocab_tokens=[
            "<pad>",
            "<bos>",
            "<eos>",
            "<|start_of_role|>",
            "<|end_of_role|>",
            "<|end_of_text|>",
            "Answer",
        ],
        special_tokens={"pad": 0, "bos": 1, "eos": 2},
        model_path=None,
        prefer_llama=None,
        bpe_merges=None,
        tokenizer_model="bpe",
    )

    text = tokenizer.decode([3, 4, 6, 5], skip_special=True)

    assert text == "Answer"
