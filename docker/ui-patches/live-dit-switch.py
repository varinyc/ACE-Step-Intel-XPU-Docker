#!/usr/bin/env python3
"""Uniform live DiT switch for ALL ACE-Step 1.5 + 1.5XL variants.

Upstream Gradio api_routes has no POST /v1/init DiT switch. This injects:

  - Full known DiT catalog (2B + 4B XL)
  - _switch_dit_model_sync with dual CPU offload for every model
  - POST /v1/init + GET /v1/models (LM helpers layered later by live-lm-reinit)

Offload defaults (Arc A770 / 16GB):
  ACESTEP_OFFLOAD_TO_CPU=true
  ACESTEP_OFFLOAD_DIT_TO_CPU=true

Every variant (turbo, base, sft, shift*, continuous, xl-*) gets the same path.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

MARKER = "# [XPU-LIVE-DIT-SWITCH]"

HELPERS = r'''
# [XPU-LIVE-DIT-SWITCH]
import asyncio
from loguru import logger

# API Key storage (set via setup_api_routes)
_api_key: Optional[str] = None
_api_key_lock = Lock()


_KNOWN_DIT_MODELS = (
    "acestep-v15-turbo",
    "acestep-v15-base",
    "acestep-v15-sft",
    "acestep-v15-turbo-shift1",
    "acestep-v15-turbo-shift3",
    "acestep-v15-turbo-continuous",
    "acestep-v15-xl-turbo",
    "acestep-v15-xl-sft",
    "acestep-v15-xl-base",
)

_init_lock = Lock()


def _env_bool_dit(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")


def _clear_accelerator_cache(tag: str = "") -> None:
    try:
        import torch
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
            logger.info(f"[memory] torch.xpu.empty_cache() {tag}")
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"[memory] torch.cuda.empty_cache() {tag}")
    except Exception as exc:
        logger.debug(f"[memory] empty_cache skipped: {exc}")


def _basename_dit(path) -> str:
    return os.path.basename(str(path).replace(chr(92), "/").rstrip("/"))


def _current_dit_name(dit_handler) -> Optional[str]:
    if dit_handler is None:
        return None
    params = getattr(dit_handler, "last_init_params", None) or {}
    config_path = params.get("config_path") or params.get("checkpoint") or ""
    if config_path:
        return _basename_dit(config_path)
    # Some builds store model name on the handler
    for attr in ("current_model_name", "_config_path", "config_path"):
        v = getattr(dit_handler, attr, None)
        if v:
            return _basename_dit(v)
    return None


def _collect_available_dit_names(dit_handler) -> List[str]:
    names = set(_KNOWN_DIT_MODELS)
    if dit_handler is not None and hasattr(dit_handler, "get_available_acestep_v15_models"):
        try:
            names.update(dit_handler.get_available_acestep_v15_models() or [])
        except Exception as exc:
            logger.warning(f"[v1/models] get_available_acestep_v15_models failed: {exc}")
    # Scan checkpoints dir for any acestep-v15-*
    ckpt = "/app/checkpoints"
    for key in ("ACESTEP_CHECKPOINTS_DIR", "ACESTEP_CHECKPOINT_DIR", "CHECKPOINT_DIR"):
        v = (os.environ.get(key) or "").strip()
        if v and os.path.isdir(v):
            ckpt = v
            break
    try:
        if os.path.isdir(ckpt):
            for entry in os.listdir(ckpt):
                if entry.startswith("acestep-v15-") and os.path.isdir(os.path.join(ckpt, entry)):
                    names.add(entry)
    except Exception:
        pass
    current = _current_dit_name(dit_handler)
    if current:
        names.add(current)
    return sorted(names)


def _resolve_dit_project_root() -> str:
    if os.path.isdir("/app/checkpoints") and os.path.isdir("/app/acestep"):
        return "/app"
    try:
        return _get_project_root()
    except Exception:
        return "/app"


def _switch_dit_model_sync(dit_handler, model_name: str) -> Dict[str, Any]:
    """Blocking DiT switch used by POST /v1/init — uniform for all 1.5 / 1.5XL."""
    if dit_handler is None:
        raise RuntimeError("DiT handler is not available")

    model_name = (model_name or "").strip()
    if not model_name:
        raise RuntimeError("model is required")
    if not model_name.startswith("acestep-v15-"):
        raise RuntimeError(f"Unsupported DiT model: {model_name}")

    current = _current_dit_name(dit_handler)
    # Dual offload for EVERY variant on Arc 16GB (override via env if needed)
    offload_to_cpu = _env_bool_dit("ACESTEP_OFFLOAD_TO_CPU", True)
    offload_dit_to_cpu = _env_bool_dit("ACESTEP_OFFLOAD_DIT_TO_CPU", True)

    if (
        current == model_name
        and getattr(dit_handler, "model", None) is not None
    ):
        _clear_accelerator_cache("(DiT already loaded, cache only)")
        return {
            "message": f"DiT '{model_name}' already loaded",
            "loaded_model": model_name,
            "switched": False,
            "offload_to_cpu": offload_to_cpu,
            "offload_dit_to_cpu": offload_dit_to_cpu,
        }

    project_root = _resolve_dit_project_root()
    device = (
        os.environ.get("ACESTEP_DEVICE")
        or os.environ.get("PYTORCH_DEVICE")
        or "xpu"
    ).strip() or "xpu"

    logger.info(
        f"[v1/init] Switching DiT: {current!r} -> {model_name!r} "
        f"(offload_to_cpu={offload_to_cpu}, offload_dit_to_cpu={offload_dit_to_cpu})"
    )
    _clear_accelerator_cache("(before DiT switch)")

    # Prefer initialize_service when available (service/API path)
    status = ""
    ok = False
    if hasattr(dit_handler, "initialize_service"):
        status, enable = dit_handler.initialize_service(
            project_root=project_root,
            config_path=model_name,
            device=device,
            use_flash_attention=False,
            compile_model=False,
            offload_to_cpu=offload_to_cpu,
            offload_dit_to_cpu=offload_dit_to_cpu,
            quantization=None,
            prefer_source=None,
        )
        ok = bool(enable)
    elif hasattr(dit_handler, "initialize"):
        # Fallback signature varies; best-effort
        try:
            status, enable = dit_handler.initialize(
                project_root=project_root,
                config_path=model_name,
                device=device,
                offload_to_cpu=offload_to_cpu,
                offload_dit_to_cpu=offload_dit_to_cpu,
            )
            ok = bool(enable)
        except TypeError:
            status, enable = dit_handler.initialize(
                project_root=project_root,
                config_path=model_name,
                device=device,
            )
            ok = bool(enable)
    else:
        raise RuntimeError("dit_handler has no initialize_service/initialize")

    if not ok:
        raise RuntimeError(status or f"Failed to initialize DiT '{model_name}'")

    # Keep env in sync so next cold start matches last pick
    os.environ["ACESTEP_CONFIG_PATH"] = model_name

    _clear_accelerator_cache("(after DiT switch)")
    loaded = _current_dit_name(dit_handler) or model_name
    logger.info(f"[v1/init] DiT loaded: {loaded}")

    msg = status or f"[OK] Model initialized successfully on {device}\nMain model: {loaded}"
    if offload_to_cpu:
        msg += "\nOffload to CPU: True"
    if offload_dit_to_cpu:
        msg += "\nOffload DiT to CPU: True"

    return {
        "message": msg,
        "loaded_model": loaded,
        "switched": True,
        "offload_to_cpu": offload_to_cpu,
        "offload_dit_to_cpu": offload_dit_to_cpu,
    }
'''

NEW_LIST = r'''
@router.get("/v1/models")
async def list_models(request: Request, _: None = Depends(verify_api_key)):
    """List available DiT models (full 1.5 + 1.5XL catalog)."""
    dit_handler = getattr(request.app.state, "dit_handler", None)
    current = _current_dit_name(dit_handler)
    available = _collect_available_dit_names(dit_handler)

    models = [
        {
            "name": name,
            "is_default": bool(current and name == current),
            "is_loaded": bool(current and name == current and dit_handler and getattr(dit_handler, "model", None) is not None),
        }
        for name in available
    ]
    models.sort(key=lambda m: (not m["is_loaded"], m["name"]))

    return _wrap_response({
        "models": models,
        "default_model": current,
    })
'''

NEW_INIT = r'''
@router.post("/v1/init")
async def init_model(request: Request, authorization: Optional[str] = Header(None)):
    """Live-switch DiT (any 1.5 / 1.5XL) without container restart.

    Always applies dual CPU offload on Arc 16GB unless env disables it.
    LM switching is layered by live-lm-reinit.py on top of this route.
    """
    content_type = (request.headers.get("content-type") or "").lower()
    if "json" in content_type:
        body = await request.json()
    else:
        try:
            form = await request.form()
            body = {k: v for k, v in form.items()}
        except Exception:
            body = {}

    verify_token_from_request(body, authorization)

    model_name = (
        body.get("model")
        or body.get("dit_model")
        or body.get("config_path")
        or ""
    )
    model_name = str(model_name).strip()
    if not model_name:
        return _wrap_response(None, code=400, error="model is required")

    dit_handler = getattr(request.app.state, "dit_handler", None)
    if dit_handler is None:
        return _wrap_response(None, code=500, error="DiT handler not available")

    if not _init_lock.acquire(blocking=False):
        return _wrap_response(
            None,
            code=409,
            error="Another model switch is already in progress; try again shortly",
        )

    try:
        result = await asyncio.to_thread(_switch_dit_model_sync, dit_handler, model_name)
        models_payload = (await list_models(request, None))["data"]  # type: ignore[index]
        if isinstance(models_payload, dict):
            result["models"] = models_payload.get("models")
            result["default_model"] = result.get("loaded_model") or models_payload.get("default_model")
        return _wrap_response(result)
    except Exception as exc:
        logger.exception(f"[v1/init] DiT switch failed model={model_name!r}")
        return _wrap_response(None, code=500, error=str(exc))
    finally:
        _init_lock.release()
'''


def _replace_function(text: str, decorator_line: str, new_block: str) -> str:
    start = text.find(decorator_line)
    if start < 0:
        raise SystemExit(f"Could not find route starting with: {decorator_line!r}")
    rest = text[start + len(decorator_line) :]
    next_router = rest.find("\n@router.")
    if next_router < 0:
        # append at end before setup functions
        end = text.find("\ndef setup_api_routes")
        if end < 0:
            end = len(text)
        return text[:start] + new_block.strip() + "\n\n" + text[end:]
    end = start + len(decorator_line) + next_router
    return text[:start] + new_block.strip() + "\n" + text[end:]


def _strip_old(text: str) -> str:
    if MARKER not in text:
        return text
    start = text.find(MARKER)
    while start > 0 and text[start - 1] in "\r\n":
        start -= 1
    # Prefer cutting to router = APIRouter if marker is before it; else to next @router
    end = text.find("router = APIRouter()", start)
    if end < 0:
        end = text.find("\n@router.", start)
        if end < 0:
            return text
    print("Stripping previous [XPU-LIVE-DIT-SWITCH] block")
    return text[:start] + "\n" + text[end:]


def main() -> None:
    paths = list(Path("/app").rglob("api_routes.py"))
    paths = [p for p in paths if "gradio" in str(p) and "api" in str(p)]
    if not paths:
        paths = list(Path(".").rglob("acestep/ui/gradio/api/api_routes.py"))
    if not paths:
        print("api_routes.py not found", file=sys.stderr)
        sys.exit(1)

    for path in paths:
        text = path.read_text()

        if MARKER in text and "_KNOWN_DIT_MODELS" in text and "_switch_dit_model_sync" in text:
            # Ensure XL names present even on older inject
            if "acestep-v15-xl-base" not in text:
                text = text.replace(
                    '"acestep-v15-turbo-continuous",',
                    '"acestep-v15-turbo-continuous",\n    "acestep-v15-xl-turbo",\n    "acestep-v15-xl-sft",\n    "acestep-v15-xl-base",',
                    1,
                )
                path.write_text(text)
                print(f"Updated known DiT list in {path}")
            else:
                print(f"Already patched: {path}")
            continue

        text = _strip_old(text)

        anchor = "router = APIRouter()"
        if anchor not in text:
            print(f"'{anchor}' not found in {path}", file=sys.stderr)
            sys.exit(1)

        text = text.replace(anchor, HELPERS.rstrip() + "\n\n" + anchor, 1)

        # Replace or insert /v1/models
        if '@router.get("/v1/models")' in text:
            text = _replace_function(text, '@router.get("/v1/models")', NEW_LIST)
        else:
            text = text.replace(anchor, anchor + "\n\n" + NEW_LIST.strip() + "\n", 1)

        # Replace or insert /v1/init
        if '@router.post("/v1/init")' in text:
            text = _replace_function(text, '@router.post("/v1/init")', NEW_INIT)
        else:
            # Insert after /v1/models
            needle = '@router.get("/v1/models")'
            # find end of list_models function — next @router
            start = text.find(needle)
            if start >= 0:
                rest = text[start + len(needle) :]
                nxt = rest.find("\n@router.")
                if nxt >= 0:
                    ins = start + len(needle) + nxt
                    text = text[:ins] + "\n\n" + NEW_INIT.strip() + "\n" + text[ins:]
                else:
                    text = text + "\n" + NEW_INIT.strip() + "\n"
            else:
                text = text + "\n" + NEW_INIT.strip() + "\n"

        try:
            compile(text, str(path), "exec")
        except SyntaxError as exc:
            raise SystemExit(f"Patched api_routes.py invalid: {exc}") from exc

        path.write_text(text)
        print(f"Patched uniform DiT live-switch into {path}")

    print("live-dit-switch complete")


if __name__ == "__main__":
    main()
