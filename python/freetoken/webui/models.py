"""Models present on this host, for the web console's profile editor. stdlib only (the daemon imports it).

A profile's model must be something ``ft serve --model`` can open: a local folder by absolute path (a
bare folder name is taken for a Hugging Face repo id and fails), or a repo id already in the Hugging
Face cache. Looks in ``~/models`` plus ``$FREETOKEN_MODELS_DIRS`` (``:``-separated) and the HF hub cache."""

from __future__ import annotations

import json
import os

WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin", ".pt")


def model_dirs(extra: list[str] | None = None) -> list[str]:
    dirs = [os.path.expanduser("~/models")]
    dirs += [d for d in os.environ.get("FREETOKEN_MODELS_DIRS", "").split(":") if d]
    dirs += list(extra or [])
    seen, out = set(), []
    for d in dirs:
        d = os.path.abspath(os.path.expanduser(d))
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _describe(folder: str) -> dict | None:
    try:
        names = os.listdir(folder)
    except OSError:
        return None
    weights = [n for n in names if n.endswith(WEIGHT_SUFFIXES)]
    if "config.json" not in names and not any(n.endswith(".gguf") for n in names):
        return None
    size = 0
    for n in weights:
        try:
            size += os.stat(os.path.join(folder, n)).st_size  # follows the HF cache's blob symlinks
        except OSError:
            pass
    info: dict = {"size_bytes": size, "weights": len(weights)}
    try:
        with open(os.path.join(folder, "config.json")) as fh:
            cfg = json.load(fh)
        info["architecture"] = (cfg.get("architectures") or [None])[0]
        info["model_type"] = cfg.get("model_type")
        quant = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config") or {}
        info["quant"] = quant.get("quant_method") or quant.get("quant_algo")
        info["max_context"] = cfg.get("max_position_embeddings") or (cfg.get("text_config") or {}).get("max_position_embeddings")
    except (OSError, ValueError, AttributeError):
        pass
    return info if weights else None


def q2_gguf_for(path: str) -> str | None:
    """The Q2_0 GGUF of this checkpoint's experts, when one sits next to it: a sibling directory
    whose name starts with the checkpoint's own (its quantization suffix dropped) and contains
    "GGUF", holding ``*Q2_0*.gguf`` (shard 1 of a split). None when there is none. The GGUF is
    not a model -- ``ft serve --model`` stays the checkpoint -- it is the expert tier the profile
    editor offers for it (``--moe-expert-gguf``, docs/gguf.md)."""
    path = os.path.normpath(path)
    base = os.path.basename(path)
    stem = base
    for suffix in ("-NVFP4", "-FP8", "-MXFP4", "-AWQ", "-GPTQ", "-bf16", "-BF16"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    parent = os.path.dirname(path)
    try:
        siblings = sorted(os.listdir(parent))
    except OSError:
        return None
    for d in siblings:
        full = os.path.join(parent, d)
        if d == base or not d.startswith(stem) or "GGUF" not in d.upper() or not os.path.isdir(full):
            continue
        for root, _, files in os.walk(full):
            hits = sorted(f for f in files if f.endswith(".gguf") and "Q2_0" in f.upper() and ("-00001-of-" in f or "-of-" not in f))
            if hits:
                return os.path.join(root, hits[0])
    return None


def resolve_model(value: str, extra_dirs: list[str] | None = None) -> str:
    """What to hand ``ft serve --model``, or ValueError explaining why this one cannot work.

    A bare folder name (``Ornith-1.5-35B-A3B-NVFP4``) is what a person types and what ft serve reads as
    a Hugging Face repo id -- it then fails at load with a 401 from the hub. Resolve it against the
    local model directories instead, and only give up when nothing there matches."""
    value = os.path.expanduser((value or "").strip()).rstrip("/")
    if not value:
        raise ValueError("model is required")
    if os.path.isabs(value):
        if not os.path.isdir(value):
            raise ValueError(f"no model folder at {value}")
        return value
    if "/" in value:  # org/name: a Hugging Face repo id, left to ft serve
        return value
    matches = [m["value"] for m in list_models(extra_dirs) if m["value"].startswith("/") and os.path.basename(m["value"]) == value]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        known = ", ".join(sorted(os.path.basename(m["value"]) for m in list_models(extra_dirs) if m["value"].startswith("/"))) or "(none)"
        raise ValueError(
            f"no model folder named {value!r}; a bare name is read as a Hugging Face repo id and fails at load. "
            f"Give the folder path or one of: {known}"
        )
    raise ValueError(f"{value!r} matches more than one model folder; give the full path")


def _hf_cache() -> str:
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.join(home, "hub")


def list_models(extra_dirs: list[str] | None = None) -> list[dict]:
    out = []
    for root in model_dirs(extra_dirs):
        try:
            entries = sorted(os.listdir(root))
        except OSError:
            continue
        for name in entries:
            path = os.path.join(root, name)
            if not os.path.isdir(path):
                continue
            info = _describe(path)
            if info:
                entry = {"value": path, "name": name, "source": root, **info}
                gguf = q2_gguf_for(path) if info.get("model_type") == "qwen4_exp" else None
                if gguf:
                    entry["expert_gguf"] = gguf  # the Q2_0 expert tier next to it (--moe-expert-gguf)
                out.append(entry)
    cache = _hf_cache()
    try:
        repos = sorted(os.listdir(cache))
    except OSError:
        repos = []
    for repo in repos:
        if not repo.startswith("models--"):
            continue
        snaps = os.path.join(cache, repo, "snapshots")
        try:
            revs = sorted(os.listdir(snaps), key=lambda r: os.stat(os.path.join(snaps, r)).st_mtime, reverse=True)
        except OSError:
            continue
        for rev in revs:
            info = _describe(os.path.join(snaps, rev))
            if info:
                repo_id = repo[len("models--"):].replace("--", "/")
                out.append({"value": repo_id, "name": repo_id, "source": "Hugging Face キャッシュ", **info})
                break
    return out
