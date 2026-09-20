"""Compatibility shims for third-party library bugs we cannot wait for upstream to fix.

Import this module (and call the patch functions) before any
``*Processor.from_pretrained`` / ``*Tokenizer.from_pretrained`` call.
"""

import logging

logger = logging.getLogger(__name__)

_CHAT_TEMPLATE_PATCH_APPLIED = False


def patch_hub_chat_template_listing() -> None:
    """Strip the ``.jinja`` suffix that transformers forgets to strip off Hub template names.

    ``transformers.utils.hub.list_repo_templates`` has two branches. The offline
    branch returns ``Path.stem`` ("sentence_transformers"); the online branch returns
    ``entry.path.removeprefix("additional_chat_templates/")``, which keeps the
    extension ("sentence_transformers.jinja"). Both callers then rebuild the path as
    ``f"additional_chat_templates/{name}.jinja"``, so the online branch asks the Hub for
    ``sentence_transformers.jinja.jinja``. That file does not exist, ``cached_file`` is
    called with ``_raise_exceptions_for_missing_entries=False`` and returns ``None``, and
    the caller does ``open(None)``:

        TypeError: expected str, bytes or os.PathLike object, not NoneType

    Any model repo with an ``additional_chat_templates/`` folder therefore fails to load
    whenever the process can actually reach huggingface.co -- which is exactly what
    happened when tsystems/colqwen2.5-3b-multilingual-v1.0 gained
    ``additional_chat_templates/sentence_transformers.jinja``.

    Fixed upstream on transformers main (``template.removesuffix(".jinja")``) but not in
    any released 4.57.x, and 5.x is not an option while colpali-engine is pinned to
    0.3.13. Stripping the suffix here is a no-op on the offline branch and on a fixed
    transformers, so this can be deleted once the pin moves past the fix.
    """
    global _CHAT_TEMPLATE_PATCH_APPLIED
    if _CHAT_TEMPLATE_PATCH_APPLIED:
        return

    try:
        from transformers import processing_utils, tokenization_utils_base
        from transformers.utils import hub as transformers_hub
    except ImportError:  # transformers not installed / restructured -- nothing to patch
        _CHAT_TEMPLATE_PATCH_APPLIED = True
        return

    original = getattr(transformers_hub, "list_repo_templates", None)
    if original is None or getattr(original, "_morphik_suffix_patch", False):
        _CHAT_TEMPLATE_PATCH_APPLIED = True
        return

    def list_repo_templates(*args, **kwargs):
        return [name.removesuffix(".jinja") for name in original(*args, **kwargs)]

    list_repo_templates._morphik_suffix_patch = True

    # Both callers did `from .utils import list_repo_templates`, so the module-level
    # name has to be rebound in each of them, not just on the defining module.
    for module in (transformers_hub, processing_utils, tokenization_utils_base):
        if getattr(module, "list_repo_templates", None) is original:
            module.list_repo_templates = list_repo_templates

    _CHAT_TEMPLATE_PATCH_APPLIED = True
    logger.info("Applied transformers list_repo_templates .jinja-suffix patch")
