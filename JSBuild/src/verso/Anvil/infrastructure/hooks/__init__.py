from infrastructure.hooks.base import Hook
from infrastructure.hooks.registry import HookRegistry

__all__ = [
    "AESPreVerifyHook",
    "AALClassifyHook",
    "ChronicleHook",
    "Hook",
    "HookRegistry",
]


def __getattr__(name: str):
    if name == "AESPreVerifyHook":
        from infrastructure.hooks.aes_pre_verify import AESPreVerifyHook

        return AESPreVerifyHook
    if name in {"AALClassifyHook", "ChronicleHook"}:
        from infrastructure.hooks.builtin import AALClassifyHook, ChronicleHook

        return {"AALClassifyHook": AALClassifyHook, "ChronicleHook": ChronicleHook}[name]
    raise AttributeError(name)
