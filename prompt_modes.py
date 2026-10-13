"""Checkpoint routing only: prompt-free heads cannot accept text/visual prompts.

Do not fuse a user's original/corrected model, substitute COCO labels, or assume
that a corrected checkpoint has an equivalent official prompt-free sibling.
"""
from i18n import message


PROMPT_FREE_PAIRS = {
    "yoloe-26s-seg.pt": "yoloe-26s-seg-pf.pt",
    "yoloe-26l-seg.pt": "yoloe-26l-seg-pf.pt",
    "yoloe-26x-seg.pt": "yoloe-26x-seg-pf.pt",
    "yoloe-11l-seg.pt": "yoloe-11l-seg-pf.pt",
}
PROMPTED_PAIRS = {pf: prompted for prompted, pf in PROMPT_FREE_PAIRS.items()}


def resolve_detection_model(selected, *, prompt_free=False):
    """Return an explicit sibling, or decline without silently losing corrections."""
    if prompt_free:
        if selected in PROMPTED_PAIRS:
            return selected
        if selected in PROMPT_FREE_PAIRS:
            return PROMPT_FREE_PAIRS[selected]
        raise ValueError(message(
            "所选模型没有对应的无提示版（修改版和 DINO 不会被替换）。请在模型列表选择 YOLOE 无提示版；或输入文字继续用当前模型。"))
    return PROMPTED_PAIRS.get(selected, selected)
