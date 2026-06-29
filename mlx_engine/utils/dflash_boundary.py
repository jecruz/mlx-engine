"""Guarded DFlash boundary and dependency probe for speculative decoding."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import importlib.util
import json
import os
import re
import socket
import subprocess
from pathlib import Path
from typing import Any, Optional

from mlx_engine.utils.dflash_snapshot import DFlashSnapshotError, load_dflash_snapshot_profile


DFLASH_ENV = "MLX_ENGINE_DFLASH"
DFLASH_TARGET_MODEL_ENV = "MLX_ENGINE_DFLASH_TARGET_MODEL"
DFLASH_DRAFTER_MODEL_ENV = "MLX_ENGINE_DFLASH_DRAFTER_MODEL"
DFLASH_MAX_DRAFT_TOKENS_ENV = "MLX_ENGINE_DFLASH_MAX_DRAFT_TOKENS"
DFLASH_ADAPTIVE_SCHEDULING_ENV = "MLX_ENGINE_DFLASH_ADAPTIVE_SCHEDULING"
DEFAULT_DFLASH_MAX_DRAFT_TOKENS = 4
DFLASH_EXPECTED_DTYPE = "bfloat16"
DFLASH_REQUIRED_TARGET_TOKENIZER_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
DFLASH_RESOURCE_PORTS = (3180, 3181, 3182, 12444)
DFLASH_LLMDYNAMIX_PORT = 12444
DFLASH_AVAILABLE_MEMORY_HEADROOM_RATIO = 0.25
DFLASH_AVAILABLE_MEMORY_HEADROOM_MIN_BYTES = 8 * 1024 * 1024 * 1024

_DFLASH_DEPENDENCY_MODULES = (
    "mlx_vlm.speculative.dflash",
    "mlx_vlm.speculative.drafters.qwen3_dflash.dflash",
)
_SUPPORTED_MODEL_MARKERS = ("qwen",)
_UNSUPPORTED_MODEL_MARKERS = ("moe", "a3b")

# LLMDYNAMIX cloud-router detection
_LLMDYNAMIX_PROCESS_MARKERS = ("llmdynamix",)
_LLMDYNAMIX_CLOUD_BACKEND_MARKERS = (
    "anthropic",
    "openai",
    "google",
    "commandcode",
    "cmd",
    "openrouter",
)
# Only mark backends that consume MLX/Metal GPU memory as local-heavy. Pure
# llama.cpp/puma.cpp backends run on CPU and do NOT contend for Metal GPU
# resources, so they should NOT block a DFlash smoke.
_LLMDYNAMIX_LOCAL_HEAVY_BACKEND_MARKERS = (
    "lm studio",
    "ollama",
    "mlx",
    "vllm",
    "swift_llm",
)
# Process names that prove a real MLX/Metal-heavy local load.
_LOCAL_MLX_METAL_PROCESS_MARKERS = (
    "mlx_engine",
    "mlx-lm",
    "mlx_lm",
    "mlx-vlm",
    "mlx_vlm",
    "lmstudio",
    "lms ",
    "/lms",
    "vllm",
    "swift_llm",
)


class ListenerClassification(str, Enum):
    """How a reserved DFlash resource port listener should be classified."""

    EMPTY = "empty"
    CLOUD_ONLY_LLMDYNAMIX = "cloud-only-llmdynamix"
    LOCAL_MLX_METAL_HEAVY = "local-mlx-metal-heavy"
    UNKNOWN_HEAVY = "unknown-heavy"


@dataclass(frozen=True, slots=True)
class ListenerEvidence:
    """Process evidence recorded for a reserved DFlash resource port."""

    port: int
    classification: ListenerClassification
    pid: Optional[int] = None
    comm: Optional[str] = None
    command: Optional[str] = None
    cloud_backend_count: int = 0
    local_heavy_backend_count: int = 0
    config_path: Optional[Path] = None
    notes: tuple[str, ...] = ()

    def is_allowed(self) -> bool:
        """True iff this listener may coexist with a real-pair DFlash smoke."""

        return self.classification in {
            ListenerClassification.EMPTY,
            ListenerClassification.CLOUD_ONLY_LLMDYNAMIX,
        }


@dataclass(frozen=True, slots=True)
class DFlashBoundaryOptions:
    """Validated opt-in state for the guarded DFlash boundary."""

    enabled: bool
    target_model_path: Path | None = None
    drafter_model_path: Path | None = None
    max_draft_tokens: int = DEFAULT_DFLASH_MAX_DRAFT_TOKENS
    adaptive_scheduling: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("dflash enabled must be a boolean")
        if self.target_model_path is not None and not isinstance(
            self.target_model_path, Path
        ):
            raise ValueError("dflash target_model_path must be a pathlib.Path")
        if self.drafter_model_path is not None and not isinstance(
            self.drafter_model_path, Path
        ):
            raise ValueError("dflash drafter_model_path must be a pathlib.Path")
        if (
            isinstance(self.max_draft_tokens, bool)
            or not isinstance(self.max_draft_tokens, int)
            or self.max_draft_tokens < 1
        ):
            raise ValueError("dflash_max_draft_tokens must be a positive integer")
        if not isinstance(self.adaptive_scheduling, bool):
            raise ValueError("dflash adaptive_scheduling must be a boolean")


@dataclass(frozen=True, slots=True)
class DFlashTargetProfile:
    """Validated local DFlash target snapshot summary."""

    model_path: Path
    config_path: Path
    tokenizer_paths: tuple[Path, ...]
    safetensors_paths: tuple[Path, ...]
    architectures: tuple[str, ...]
    model_type: str
    dtype: str
    num_hidden_layers: int
    vocab_size: int
    tokenizer_vocab_size: int


@dataclass(frozen=True, slots=True)
class DFlashReadinessReport:
    """Structured readiness report for the DFlash boundary spike."""

    enabled: bool
    dependency_available: bool
    target_family: str | None
    drafter_family: str | None
    target_profile: DFlashTargetProfile | None = None
    cache_mode_blockers: tuple[str, ...] = ()
    route_blockers: tuple[str, ...] = ()
    resource_blockers: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    listener_evidence: tuple[ListenerEvidence, ...] = ()


class DFlashUnavailableError(ValueError):
    """Raised when the DFlash boundary is opted into but not ready."""


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "")
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - defensive parsing
        raise ValueError(f"{name} must be an integer") from exc


def _coerce_path(value: str | Path | None) -> Path | None:
    if value is None or value == "":
        return None
    return value if isinstance(value, Path) else Path(value)


def resolve_dflash_options(
    dflash_toggle: bool | None,
    dflash_target_model: str | Path | None,
    dflash_drafter_model: str | Path | None,
    dflash_max_draft_tokens: int | None,
    dflash_adaptive_scheduling: bool | None = None,
) -> DFlashBoundaryOptions:
    """Resolve public DFlash kwargs/env into validated boundary options."""

    enabled = _env_flag(DFLASH_ENV) if dflash_toggle is None else dflash_toggle
    if not enabled:
        return DFlashBoundaryOptions(enabled=False)

    target_model_path = dflash_target_model
    if target_model_path is None:
        target_model_path = os.getenv(DFLASH_TARGET_MODEL_ENV, "")
    drafter_model_path = dflash_drafter_model
    if drafter_model_path is None:
        drafter_model_path = os.getenv(DFLASH_DRAFTER_MODEL_ENV, "")
    max_draft_tokens = (
        _env_int(DFLASH_MAX_DRAFT_TOKENS_ENV, DEFAULT_DFLASH_MAX_DRAFT_TOKENS)
        if dflash_max_draft_tokens is None
        else dflash_max_draft_tokens
    )
    adaptive_scheduling = (
        _env_flag(DFLASH_ADAPTIVE_SCHEDULING_ENV)
        if dflash_adaptive_scheduling is None
        else dflash_adaptive_scheduling
    )
    return DFlashBoundaryOptions(
        enabled=True,
        target_model_path=_coerce_path(target_model_path),
        drafter_model_path=_coerce_path(drafter_model_path),
        max_draft_tokens=max_draft_tokens,
        adaptive_scheduling=adaptive_scheduling,
    )


def probe_dflash_dependency() -> tuple[bool, tuple[str, ...]]:
    """Return whether the optional DFlash dependency is importable."""

    missing_modules = tuple(
        module_name
        for module_name in _DFLASH_DEPENDENCY_MODULES
        if importlib.util.find_spec(module_name) is None
    )
    return len(missing_modules) == 0, missing_modules


def _read_model_metadata(model_path: Path | None) -> dict[str, Any]:
    if model_path is None or not model_path.exists():
        return {}
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {}
    try:
        config = json.loads(config_path.read_text())
    except Exception:  # pragma: no cover - defensive probe
        return {}
    return config if isinstance(config, dict) else {}


def _classify_qwen_family(model_path: Path | None) -> str | None:
    metadata = _read_model_metadata(model_path)
    if not metadata:
        return None

    def _metadata_strings(value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            return (value,)
        if isinstance(value, list):
            return tuple(item for item in value if isinstance(item, str))
        return ()

    metadata_strings: list[str] = []
    for key in ("model_type", "architectures"):
        metadata_strings.extend(_metadata_strings(metadata.get(key)))
    for nested_key in ("text_config", "vision_config"):
        nested = metadata.get(nested_key)
        if isinstance(nested, dict):
            for key in ("model_type", "architectures"):
                metadata_strings.extend(_metadata_strings(nested.get(key)))

    corpus = " ".join(metadata_strings).lower()
    if not any(marker in corpus for marker in _SUPPORTED_MODEL_MARKERS):
        return None
    if any(marker in corpus for marker in _UNSUPPORTED_MODEL_MARKERS):
        return None
    return "qwen"


def _normalize_dtype(dtype: Any) -> str:
    normalized = str(dtype).strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return DFLASH_EXPECTED_DTYPE
    return normalized


def _collect_target_tokenizer_paths(
    model_path: Path,
    blockers: list[str],
) -> tuple[Path, ...]:
    if not model_path.exists():
        blockers.append(f"DFlash target snapshot path does not exist: {model_path}")
        return ()
    if not model_path.is_dir():
        blockers.append(f"DFlash target snapshot path is not a directory: {model_path}")
        return ()

    tokenizer_paths = tuple(
        model_path / filename for filename in DFLASH_REQUIRED_TARGET_TOKENIZER_FILES
    )
    missing_files = [path for path in tokenizer_paths if not path.exists()]
    if missing_files:
        blockers.append(
            "Missing DFlash target tokenizer/config files: "
            + ", ".join(str(path) for path in missing_files)
        )
    return tokenizer_paths


def _parse_target_profile(
    model_path: Path,
    blockers: list[str],
) -> DFlashTargetProfile | None:
    config_path = model_path / "config.json"
    tokenizer_paths = _collect_target_tokenizer_paths(model_path, blockers)
    config = _read_model_metadata(model_path)
    if not config:
        blockers.append(f"Missing or invalid DFlash target config file: {config_path}")
        return None

    if _classify_qwen_family(model_path) is None:
        blockers.append(f"DFlash target must be a Qwen-family snapshot: {model_path}")

    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not architectures or not all(
        isinstance(item, str) for item in architectures
    ):
        blockers.append("DFlash target config.architectures must be a non-empty string list")
        architectures_tuple: tuple[str, ...] = ()
    else:
        architectures_tuple = tuple(architectures)
        if not any("qwen" in item.lower() for item in architectures_tuple):
            blockers.append(
                "DFlash target config.architectures must describe a Qwen-family model"
            )

    model_type = str(config.get("model_type", "")).strip().lower()
    if not model_type.startswith("qwen"):
        blockers.append("DFlash target config.model_type must be Qwen-family")

    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        dtype_value = _normalize_dtype(text_config.get("dtype", ""))
        if dtype_value and dtype_value != DFLASH_EXPECTED_DTYPE:
            blockers.append(
                f"DFlash target text_config.dtype must be {DFLASH_EXPECTED_DTYPE!r}"
            )
        num_hidden_layers = text_config.get("num_hidden_layers")
        vocab_size = text_config.get("vocab_size", config.get("vocab_size"))
    else:
        dtype_value = ""
        num_hidden_layers = config.get("num_hidden_layers")
        vocab_size = config.get("vocab_size")

    if not isinstance(num_hidden_layers, int):
        blockers.append("DFlash target num_hidden_layers must be an integer")
        num_hidden_layers_int = -1
    else:
        num_hidden_layers_int = num_hidden_layers

    if not isinstance(vocab_size, int):
        blockers.append("DFlash target vocab_size must be an integer")
        vocab_size_int = -1
    else:
        vocab_size_int = vocab_size

    tokenizer_vocab_size = -1
    vocab_path = model_path / "vocab.json"
    if vocab_path.exists():
        try:
            tokenizer_vocab = json.loads(vocab_path.read_text())
        except json.JSONDecodeError as exc:
            blockers.append(f"Invalid JSON in DFlash target vocab file {vocab_path}: {exc.msg}")
            tokenizer_vocab = {}
        if isinstance(tokenizer_vocab, dict):
            tokenizer_vocab_size = len(tokenizer_vocab)
        else:
            blockers.append(f"DFlash target vocab file must contain a JSON object: {vocab_path}")
        if tokenizer_vocab_size != -1 and vocab_size_int != -1:
            allowed_delta = max(1024, vocab_size_int // 100)
            if tokenizer_vocab_size > vocab_size_int:
                blockers.append(
                    "DFlash target tokenizer vocab size must not exceed config.vocab_size "
                    f"({tokenizer_vocab_size} > {vocab_size_int})"
                )
            elif vocab_size_int - tokenizer_vocab_size > allowed_delta:
                blockers.append(
                    "DFlash target tokenizer vocab size must stay close to config.vocab_size "
                    f"({tokenizer_vocab_size} vs {vocab_size_int})"
                )

    if blockers:
        return None

    return DFlashTargetProfile(
        model_path=model_path,
        config_path=config_path,
        tokenizer_paths=tokenizer_paths,
        safetensors_paths=tuple(sorted(model_path.glob("*.safetensors"))),
        architectures=architectures_tuple,
        model_type=model_type,
        dtype=dtype_value or DFLASH_EXPECTED_DTYPE,
        num_hidden_layers=num_hidden_layers_int,
        vocab_size=vocab_size_int,
        tokenizer_vocab_size=tokenizer_vocab_size,
    )


def _port_is_listening(port: int) -> bool:
    """Return True if a TCP listener is bound to 127.0.0.1:port."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.05)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _lookup_listener_pid(port: int) -> Optional[int]:
    """Return the PID of the listener bound to 127.0.0.1:port via lsof."""

    try:
        result = subprocess.run(
            ["lsof", "-nP", "-iTCP:" + str(port), "-sTCP:LISTEN", "-Fpc"],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except Exception:  # pragma: no cover - defensive probe
        return None
    if result.returncode != 0 and not result.stdout.strip():
        return None
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("p") and line[1:].isdigit():
            return int(line[1:])
    return None


def _lookup_process_command(pid: int) -> tuple[Optional[str], Optional[str]]:
    """Return (comm, full_command) for a PID via ps, or (None, None)."""

    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "comm="],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except Exception:  # pragma: no cover - defensive probe
        comm = None
    else:
        comm = result.stdout.strip() if result.returncode == 0 else None
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except Exception:  # pragma: no cover - defensive probe
        return comm, None
    command = result.stdout.strip() if result.returncode == 0 else None
    return comm, command


def _is_llmdynamix_router_process(comm: Optional[str], command: Optional[str]) -> bool:
    """True iff the listener process is the LLMDYNAMIX cloud-router family."""

    haystack = " ".join(filter(None, (comm, command))).lower()
    return any(marker in haystack for marker in _LLMDYNAMIX_PROCESS_MARKERS)


def _list_llmdynamix_process_commands() -> tuple[tuple[int, str], ...]:
    """Return (pid, command) for every LLMDYNAMIX-family process on the host.

    LLMDYNAMIX may be split across a listener parent (`llmdynamix`) and a child
    engine (`llmdynamix-engine -config <path>`). We collect both because only
    the child command line carries the actual config flag.
    """

    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,command="],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:  # pragma: no cover - defensive ps probe
        return ()
    if result.returncode != 0:
        return ()
    found: list[tuple[int, str]] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        head, _, tail = line.partition(" ")
        if not head.isdigit():
            continue
        lowered_tail = tail.lower()
        if any(marker in lowered_tail for marker in _LLMDYNAMIX_PROCESS_MARKERS):
            found.append((int(head), tail))
    return tuple(found)


def _extract_llmdynamix_config_path(command: Optional[str]) -> Optional[Path]:
    """Pull a -config <path> argument out of an llmdynamix-engine command line."""

    if not command:
        return None
    match = re.search(r"-config\s+(\S+)", command)
    if not match:
        return None
    return Path(match.group(1)).expanduser()


def _count_llmdynamix_backends(config_text: str) -> tuple[int, int]:
    """Return (cloud_backend_count, local_heavy_backend_count) in a config."""

    lowered = config_text.lower()
    cloud_hits = sum(
        lowered.count(marker) for marker in _LLMDYNAMIX_CLOUD_BACKEND_MARKERS
    )
    local_hits = sum(
        lowered.count(marker) for marker in _LLMDYNAMIX_LOCAL_HEAVY_BACKEND_MARKERS
    )
    # Subtract cloud-only base-url fingerprints so CMD-API counts as cloud.
    base_url_hits = len(re.findall(r"base-url\s*:\s*\S+", lowered))
    if base_url_hits and cloud_hits:
        cloud_hits = max(cloud_hits, base_url_hits - local_hits)
    return cloud_hits, local_hits


def _extract_llmdynamix_local_backend_endpoints(
    config_text: str,
) -> tuple[tuple[str, int], ...]:
    """Return (name, port) pairs for local backends referenced by the config.

    We only care about base-URLs that resolve to a local port on 127.0.0.1,
    because that is the only place a local MLX/Metal load would actually run.
    The YAML may list ``base-url`` before or after ``name``, so we first
    collect all base-URL anchors and then match each ``name`` entry that
    follows within the same provider block.
    """

    endpoints: list[tuple[str, int]] = []
    provider_blocks = re.split(r"(?m)^\s*-\s*base-url:", config_text)
    for block in provider_blocks[1:]:
        url_match = re.match(r"\s*(\S+)", block)
        if not url_match:
            continue
        url = url_match.group(1).strip()
        parsed = re.match(r"https?://([^/:]+):(\d+)", url)
        if not parsed:
            continue
        host = parsed.group(1).lower()
        port = int(parsed.group(2))
        if host not in {"127.0.0.1", "localhost", "0.0.0.0"}:
            continue
        # Provider names appear at column 4; model entries at column 6+.
        name_match = re.search(r"(?m)^    name:\s*([^\n]+)$", block)
        if not name_match:
            continue
        name = name_match.group(1).strip().lower()
        if not any(
            marker in name for marker in _LLMDYNAMIX_LOCAL_HEAVY_BACKEND_MARKERS
        ):
            continue
        endpoints.append((name, port))
    return tuple(endpoints)


def _http_get_json(url: str) -> Optional[dict[str, Any]]:
    """Fetch a JSON document from a localhost URL with a short timeout."""

    import json as _json
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=1.0) as response:  # noqa: S310
            raw = response.read().decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover - network/IO defensive
        return None
    try:
        payload = _json.loads(raw)
    except _json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _probe_local_backend_loaded_models(
    name: str,
    port: int,
) -> tuple[bool, str]:
    """Probe a local backend port for currently-loaded MLX/Metal models.

    Returns (is_loaded, evidence_string). `is_loaded` is True only when the
    backend reports that one or more models are currently resident in memory
    (or actively serving). Empty `/api/ps` or `/v1/models` payloads indicate
    that the backend is running but holding no heavy MLX/Metal load.

    We prefer Ollama's `/api/ps` because `/v1/models` returns the entire
    library, not just loaded models. For LM Studio and ocelot we fall back to
    `/v1/models` because they don't expose a separate loaded-models endpoint
    and they only advertise models that are actually resident.
    """

    if not _port_is_listening(port):
        return False, f"{name}@{port} not listening"

    lowered_name = name.lower()
    prefer_api_ps = "ollama" in lowered_name

    if prefer_api_ps:
        url = f"http://127.0.0.1:{port}/api/ps"
        label = f"{name}@{port} /api/ps"
        payload = _http_get_json(url)
        if payload is None:
            return False, f"{name}@{port} /api/ps not reachable"
        models = payload.get("models", [])
        if isinstance(models, list) and len(models) > 0:
            return True, f"{label} reports {len(models)} loaded model(s)"
        return False, f"{label} reports 0 loaded models"

    url = f"http://127.0.0.1:{port}/v1/models"
    label = f"{name}@{port} /v1/models"
    payload = _http_get_json(url)
    if payload is None:
        return False, f"{name}@{port} /v1/models not reachable"
    data = payload.get("data", [])
    if isinstance(data, list) and len(data) > 0:
        return True, f"{label} reports {len(data)} loaded model(s)"
    return False, f"{label} reports 0 loaded models"


def _classify_llmdynamix_router(
    port: int,
    listener_pid: Optional[int],
    listener_command: Optional[str],
) -> ListenerEvidence:
    """Classify an LLMDYNAMIX listener using process + config evidence."""

    llmdynamix_processes = _list_llmdynamix_process_commands()
    if not llmdynamix_processes:
        return ListenerEvidence(
            port=port,
            classification=ListenerClassification.LOCAL_MLX_METAL_HEAVY,
            pid=listener_pid,
            command=listener_command,
            notes=("No LLMDYNAMIX-family process command line could be read",),
        )

    config_path: Optional[Path] = None
    config_holder_pid: Optional[int] = None
    for proc_pid, proc_command in llmdynamix_processes:
        candidate = _extract_llmdynamix_config_path(proc_command)
        if candidate is not None:
            config_path = candidate
            config_holder_pid = proc_pid
            break

    if config_path is None or not config_path.exists():
        return ListenerEvidence(
            port=port,
            classification=ListenerClassification.LOCAL_MLX_METAL_HEAVY,
            pid=listener_pid,
            command=listener_command,
            notes=(
                "LLMDYNAMIX config path could not be resolved from any "
                "llmdynamix-family process command line",
            ),
        )

    try:
        config_text = config_path.read_text()
    except OSError as exc:  # pragma: no cover - defensive file probe
        return ListenerEvidence(
            port=port,
            classification=ListenerClassification.LOCAL_MLX_METAL_HEAVY,
            pid=listener_pid,
            command=listener_command,
            config_path=config_path,
            notes=(f"LLMDYNAMIX config read failed: {exc}",),
        )

    cloud_count, local_count = _count_llmdynamix_backends(config_text)
    local_endpoints = _extract_llmdynamix_local_backend_endpoints(config_text)

    if cloud_count > 0 and local_count == 0:
        return ListenerEvidence(
            port=port,
            classification=ListenerClassification.CLOUD_ONLY_LLMDYNAMIX,
            pid=listener_pid,
            command=listener_command,
            cloud_backend_count=cloud_count,
            local_heavy_backend_count=0,
            config_path=config_path,
            notes=(
                f"LLMDYNAMIX config at {config_path} "
                f"(pid={config_holder_pid}) exposes only "
                f"{cloud_count} cloud backend markers and no local MLX/Metal "
                "backends",
            ),
        )

    if local_endpoints:
        loaded_evidence: list[str] = []
        for name, endpoint_port in local_endpoints:
            is_loaded, evidence_text = _probe_local_backend_loaded_models(
                name, endpoint_port
            )
            loaded_evidence.append(evidence_text)
            if is_loaded:
                return ListenerEvidence(
                    port=port,
                    classification=ListenerClassification.LOCAL_MLX_METAL_HEAVY,
                    pid=listener_pid,
                    command=listener_command,
                    cloud_backend_count=cloud_count,
                    local_heavy_backend_count=local_count,
                    config_path=config_path,
                    notes=(
                        f"LLMDYNAMIX config at {config_path} "
                        f"(pid={config_holder_pid}) routes to {name}; "
                        f"{evidence_text}",
                    ),
                )
        return ListenerEvidence(
            port=port,
            classification=ListenerClassification.CLOUD_ONLY_LLMDYNAMIX,
            pid=listener_pid,
            command=listener_command,
            cloud_backend_count=cloud_count,
            local_heavy_backend_count=local_count,
            config_path=config_path,
            notes=(
                f"LLMDYNAMIX config at {config_path} "
                f"(pid={config_holder_pid}) lists {local_count} local "
                "MLX/Metal backend markers, but live probing shows no loaded "
                "models: " + "; ".join(loaded_evidence),
            ),
        )

    if local_count > 0:
        return ListenerEvidence(
            port=port,
            classification=ListenerClassification.LOCAL_MLX_METAL_HEAVY,
            pid=listener_pid,
            command=listener_command,
            cloud_backend_count=cloud_count,
            local_heavy_backend_count=local_count,
            config_path=config_path,
            notes=(
                f"LLMDYNAMIX config at {config_path} "
                f"(pid={config_holder_pid}) routes to {local_count} local "
                "MLX/Metal backend markers",
            ),
        )
    return ListenerEvidence(
        port=port,
        classification=ListenerClassification.LOCAL_MLX_METAL_HEAVY,
        pid=listener_pid,
        command=listener_command,
        config_path=config_path,
        notes=(
            f"LLMDYNAMIX config at {config_path} has no recognized backend "
            "markers",
        ),
    )


def _classify_local_heavy_listener(
    port: int,
    comm: Optional[str],
    command: Optional[str],
) -> ListenerEvidence:
    """Classify a listener that is not LLMDYNAMIX as local-heavy."""

    pid = _lookup_listener_pid(port)
    haystack = " ".join(filter(None, (comm, command))).lower()
    if any(marker in haystack for marker in _LOCAL_MLX_METAL_PROCESS_MARKERS):
        return ListenerEvidence(
            port=port,
            classification=ListenerClassification.LOCAL_MLX_METAL_HEAVY,
            pid=pid,
            comm=comm,
            command=command,
            notes=(
                "Listener process matches a local MLX/Metal-heavy marker",
            ),
        )
    return ListenerEvidence(
        port=port,
        classification=ListenerClassification.UNKNOWN_HEAVY,
        pid=pid,
        comm=comm,
        command=command,
        notes=(
            "Listener process is not a recognized LLMDYNAMIX cloud-router",
        ),
    )


def probe_listener_evidence(port: int = DFLASH_LLMDYNAMIX_PORT) -> ListenerEvidence:
    """Return process/model evidence for the listener on the given port."""

    if not _port_is_listening(port):
        return ListenerEvidence(
            port=port,
            classification=ListenerClassification.EMPTY,
        )
    pid = _lookup_listener_pid(port)
    comm, command = (None, None)
    if pid is not None:
        comm, command = _lookup_process_command(pid)
    if _is_llmdynamix_router_process(comm, command):
        return _classify_llmdynamix_router(port, pid, command)
    return _classify_local_heavy_listener(port, comm, command)


def probe_all_listener_evidence(
    ports: tuple[int, ...] = DFLASH_RESOURCE_PORTS,
) -> tuple[ListenerEvidence, ...]:
    """Return listener evidence for every reserved DFlash resource port."""

    return tuple(probe_listener_evidence(port) for port in ports)


def build_port_blocker(evidence: ListenerEvidence) -> Optional[str]:
    """Return a human-readable blocker string for a reserved port, or None."""

    if evidence.classification == ListenerClassification.EMPTY:
        return None
    if evidence.classification == ListenerClassification.CLOUD_ONLY_LLMDYNAMIX:
        return None
    if evidence.classification == ListenerClassification.LOCAL_MLX_METAL_HEAVY:
        if evidence.pid is not None:
            return (
                f"DFlash preflight refuses to coexist with local MLX/Metal-heavy "
                f"listener on 127.0.0.1:{evidence.port} "
                f"(pid={evidence.pid}, comm={evidence.comm!r})"
            )
        return (
            f"DFlash preflight refuses to coexist with local MLX/Metal-heavy "
            f"listener on 127.0.0.1:{evidence.port}"
        )
    if evidence.pid is not None:
        return (
            f"DFlash preflight cannot classify listener on 127.0.0.1:{evidence.port} "
            f"(pid={evidence.pid}, comm={evidence.comm!r}); fail closed"
        )
    return (
        f"DFlash preflight cannot classify listener on 127.0.0.1:{evidence.port}; "
        "fail closed"
    )


def _probe_reserved_port_conflicts() -> tuple[str, ...]:
    """Return fail-closed blockers for non-empty reserved resource ports.

    Reserved ports that hold a cloud-only LLMDYNAMIX listener are NOT blockers,
    because the LLMDYNAMIX binary is a router/proxy that does not directly load
    MLX/Metal model weights. Local MLX/Metal-heavy listeners and unknown
    listeners still fail closed before any heavyweight DFlash load starts.
    """

    blockers: list[str] = []
    for evidence in probe_all_listener_evidence():
        blocker = build_port_blocker(evidence)
        if blocker is not None:
            blockers.append(blocker)
    return tuple(blockers)


def probe_reserved_listener_evidence() -> tuple[ListenerEvidence, ...]:
    """Public alias for the listener evidence used in the resource gate."""

    return probe_all_listener_evidence()


def _probe_available_memory_bytes() -> int | None:
    try:
        vm_stat = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            check=True,
        )
        page_size_proc = subprocess.run(
            ["sysctl", "-n", "hw.pagesize"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:  # pragma: no cover - macOS command availability
        return None

    try:
        page_size = int(page_size_proc.stdout.strip())
    except ValueError:  # pragma: no cover - defensive parsing
        page_size = 4096

    available_pages = 0
    for line in vm_stat.stdout.splitlines():
        match = re.match(r"Pages (free|inactive|speculative):\s+(\d+)\.", line)
        if match:
            available_pages += int(match.group(2))
    if available_pages <= 0:
        return None
    return available_pages * page_size


def _estimate_snapshot_bytes(paths: tuple[Path, ...]) -> int:
    total = 0
    for path in paths:
        try:
            total += path.stat().st_size
        except FileNotFoundError:
            continue
    return total


def _format_gib(size_in_bytes: int | None) -> str:
    if size_in_bytes is None:
        return "unknown"
    return f"{size_in_bytes / (1024 ** 3):.2f} GiB"


def probe_dflash_readiness(
    options: DFlashBoundaryOptions,
    *,
    target_resident: bool = False,
) -> DFlashReadinessReport:
    """Probe the DFlash boundary without mutating generation state.

    The optional ``target_resident`` flag selects the phase-aware memory
    accounting:

    * ``target_resident=False`` (default) — pre-load accounting used by
      ``validate_dflash_preload_compatibility`` before ``load_model`` has
      constructed the heavyweight target. Memory must cover the target bytes,
      the drafter bytes, and the configured headroom in full.
    * ``target_resident=True`` — post-load accounting used by
      ``validate_dflash_postload_compatibility`` after the Qwen-family target
      has already been loaded into the active ``ModelKit``. The target bytes
      are no longer required from residual free memory because the target is
      already resident; only the incremental drafter bytes plus headroom must
      still fit. All other fail-closed blockers (unsupported surfaces,
      listeners, dependency availability, vocab/layer matching, route and
      cache-mode checks) remain active in both phases.
    """

    blockers: list[str] = []
    dependency_available, missing_modules = probe_dflash_dependency()

    if not options.enabled:
        return DFlashReadinessReport(
            enabled=False,
            dependency_available=dependency_available,
            target_family=None,
            drafter_family=None,
            blockers=(),
        )

    target_family = _classify_qwen_family(options.target_model_path)
    drafter_family = _classify_qwen_family(options.drafter_model_path)
    target_profile = None
    resource_blockers: tuple[str, ...] = ()

    if options.target_model_path is None or options.drafter_model_path is None:
        blockers.append(
            "DFlash requires explicit Qwen-family target and drafter model paths"
        )

    if target_family is None:
        blockers.append("DFlash target model must be a Qwen-family snapshot")
    if drafter_family is None:
        blockers.append("DFlash drafter model must be a Qwen-family snapshot")

    if (
        options.target_model_path is not None
        and options.drafter_model_path is not None
        and options.target_model_path.resolve() == options.drafter_model_path.resolve()
    ):
        blockers.append("DFlash target and drafter snapshots must be distinct")

    if not dependency_available:
        blockers.append(
            "Missing optional DFlash dependency modules: "
            + ", ".join(missing_modules)
        )

    if options.target_model_path is not None:
        target_profile = _parse_target_profile(options.target_model_path, blockers)

    drafter_profile = None
    if options.drafter_model_path is not None:
        try:
            drafter_profile = load_dflash_snapshot_profile(options.drafter_model_path)
        except DFlashSnapshotError as exc:
            blockers.extend(exc.blockers)

    if target_profile is not None and drafter_profile is not None:
        if target_profile.vocab_size != drafter_profile.vocab_size:
            blockers.append(
                "DFlash target and drafter vocab sizes must match "
                f"({target_profile.vocab_size} != {drafter_profile.vocab_size})"
            )
        max_target_layer_id = max(drafter_profile.target_layer_ids)
        if target_profile.num_hidden_layers <= max_target_layer_id:
            blockers.append(
                "DFlash target does not expose every configured target layer id "
                f"(num_hidden_layers={target_profile.num_hidden_layers}, "
                f"max_target_layer_id={max_target_layer_id})"
            )

        if target_resident:
            # Post-load / create_generator phase: the target model is already
            # resident in the active ModelKit, so only the incremental drafter
            # bytes plus headroom must still fit in residual free memory.
            estimated_bytes = _estimate_snapshot_bytes(
                drafter_profile.safetensors_paths
            )
            phase_label = "post-target-load DFlash preflight"
        else:
            # Pre-load / load_model phase: target + drafter + headroom must
            # all fit in residual free memory before any heavyweight load.
            estimated_bytes = _estimate_snapshot_bytes(target_profile.safetensors_paths)
            estimated_bytes += _estimate_snapshot_bytes(
                drafter_profile.safetensors_paths
            )
            phase_label = "real-pair DFlash preflight"
        available_bytes = _probe_available_memory_bytes()
        if available_bytes is not None:
            headroom = max(
                int(estimated_bytes * DFLASH_AVAILABLE_MEMORY_HEADROOM_RATIO),
                DFLASH_AVAILABLE_MEMORY_HEADROOM_MIN_BYTES,
            )
            required_bytes = estimated_bytes + headroom
            if available_bytes < required_bytes:
                resource_blockers = (
                    f"Insufficient free memory for {phase_label}: "
                    f"need at least {_format_gib(required_bytes)}, "
                    f"found {_format_gib(available_bytes)}",
                )

    listener_evidence = probe_all_listener_evidence()
    port_blockers_list = [
        blocker
        for blocker in (
            build_port_blocker(evidence) for evidence in listener_evidence
        )
        if blocker is not None
    ]
    if port_blockers_list:
        resource_blockers = (*resource_blockers, *port_blockers_list)

    blockers.extend(resource_blockers)

    return DFlashReadinessReport(
        enabled=True,
        dependency_available=dependency_available,
        target_family=target_family,
        drafter_family=drafter_family,
        target_profile=target_profile,
        resource_blockers=resource_blockers,
        blockers=_dedupe_blockers(blockers),
        listener_evidence=listener_evidence,
    )


def validate_dflash_postload_compatibility(
    *,
    options: DFlashBoundaryOptions,
    loaded_model_path: Path,
    is_vlm_route: bool,
    vocab_only: bool,
    distributed: bool,
    max_seq_nums: int | None,
    kv_bits: int | None,
    kv_group_size: int | None,
    quantized_kv_start: int | None,
    vlm_prompt_cache_storage_root: Path | None,
    vlm_prompt_cache_min_save_tokens: int | None,
) -> DFlashReadinessReport:
    """Phase-aware DFlash preflight for the post-target-load surface.

    This is the wrapper ``create_generator`` uses after the Qwen-family target
    has already been loaded into the active ``ModelKit``. It reuses the same
    route/cache/loaded-draft-model blockers as
    :func:`validate_dflash_preload_compatibility` so that VLM, batched,
    distributed, persistent VLM cache, and quantized-KV cache combinations
    still fail closed. The only difference is the resource accounting:

    * The pre-load preflight still requires target bytes + drafter bytes +
      configured headroom from residual free memory, because no model has
      been loaded yet.
    * The post-load preflight treats the target snapshot as already
      resident, so it only requires incremental drafter bytes + headroom.
      This stops the preflight from double-counting the Qwen3.6 target that
      ``load_model`` already paid for.

    Listener evidence and all other fail-closed conditions remain identical
    between the two phases.
    """

    return validate_dflash_preload_compatibility(
        options=options,
        loaded_model_path=loaded_model_path,
        is_vlm_route=is_vlm_route,
        vocab_only=vocab_only,
        distributed=distributed,
        max_seq_nums=max_seq_nums,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantized_kv_start=quantized_kv_start,
        vlm_prompt_cache_storage_root=vlm_prompt_cache_storage_root,
        vlm_prompt_cache_min_save_tokens=vlm_prompt_cache_min_save_tokens,
        target_resident=True,
    )


def validate_dflash_preload_compatibility(
    *,
    options: DFlashBoundaryOptions,
    loaded_model_path: Path,
    is_vlm_route: bool,
    vocab_only: bool,
    distributed: bool,
    max_seq_nums: int | None,
    kv_bits: int | None,
    kv_group_size: int | None,
    quantized_kv_start: int | None,
    vlm_prompt_cache_storage_root: Path | None,
    vlm_prompt_cache_min_save_tokens: int | None,
    target_resident: bool = False,
) -> DFlashReadinessReport:
    """Fail closed before DFlash can reach heavyweight model loading.

    Set ``target_resident=True`` to switch the resource accounting to the
    post-load phase (the target snapshot is already resident in memory, so
    only the incremental drafter bytes + headroom must still fit). The
    pre-load default keeps the strict target + drafter + headroom check that
    runs before ``load_model`` constructs the heavyweight target.
    """

    readiness = probe_dflash_readiness(options, target_resident=target_resident)
    route_blockers = list(readiness.route_blockers)
    cache_mode_blockers = list(readiness.cache_mode_blockers)

    if not options.enabled:
        return readiness

    if options.target_model_path is not None and loaded_model_path.resolve() != options.target_model_path.resolve():
        route_blockers.append(
            "DFlash target model path must match the loaded model path "
            f"({loaded_model_path} != {options.target_model_path})"
        )
    if is_vlm_route:
        route_blockers.append("DFlash is only supported for sequential text generation")
    if vocab_only:
        route_blockers.append("DFlash cannot be combined with vocab_only loads yet")
    if distributed:
        route_blockers.append("DFlash cannot be combined with distributed loading yet")
    if max_seq_nums is not None and max_seq_nums > 1:
        route_blockers.append("DFlash requires the sequential route (max_seq_nums <= 1)")
    if vlm_prompt_cache_storage_root is not None:
        route_blockers.append(
            "DFlash is not compatible with persistent VLM prompt-cache storage yet"
        )
    if vlm_prompt_cache_min_save_tokens is not None:
        route_blockers.append(
            "DFlash is not compatible with persistent VLM prompt-cache admission yet"
        )

    if kv_bits is not None:
        cache_mode_blockers.append("DFlash does not support kv_bits cache mode yet")
    if kv_group_size is not None:
        cache_mode_blockers.append("DFlash does not support kv_group_size cache mode yet")
    if quantized_kv_start is not None:
        cache_mode_blockers.append(
            "DFlash does not support quantized_kv_start cache mode yet"
        )

    blockers = _dedupe_blockers(
        [
            *readiness.blockers,
            *route_blockers,
            *cache_mode_blockers,
        ]
    )
    report = DFlashReadinessReport(
        enabled=readiness.enabled,
        dependency_available=readiness.dependency_available,
        target_family=readiness.target_family,
        drafter_family=readiness.drafter_family,
        target_profile=readiness.target_profile,
        cache_mode_blockers=tuple(cache_mode_blockers),
        route_blockers=tuple(route_blockers),
        resource_blockers=readiness.resource_blockers,
        blockers=blockers,
        listener_evidence=readiness.listener_evidence,
    )
    if blockers:
        raise DFlashUnavailableError(
            build_dflash_no_go_message(report)
        )
    return report


def validate_dflash_surface_compatibility(
    *,
    enabled: bool,
    surface_label: str,
    images_b64: Optional[list[str]],
    specprefill_toggle: Optional[bool],
    speculative_decoding_toggle: Optional[bool],
    num_draft_tokens: Optional[int],
    draft_model: Any | None,
    model_kit_draft_model: Any | None = None,
) -> tuple[str, ...]:
    """Fail closed for unsupported DFlash surfaces."""

    if not enabled:
        return ()

    blockers: list[str] = []
    if surface_label != "sequential":
        blockers.append("DFlash is only supported for sequential text generation")
    if images_b64 is not None and len(images_b64) > 0:
        blockers.append("DFlash is not enabled for VLM/image surfaces yet")
    if specprefill_toggle is True:
        blockers.append("DFlash cannot be combined with SpecPrefill yet")
    if speculative_decoding_toggle is True:
        blockers.append("DFlash cannot be combined with speculative decoding yet")
    if num_draft_tokens is not None:
        blockers.append("DFlash cannot be combined with num_draft_tokens yet")
    if draft_model is not None:
        blockers.append("DFlash cannot be combined with a draft_model kwarg yet")
    if model_kit_draft_model is not None:
        blockers.append(
            "DFlash cannot be combined with an already loaded draft_model yet"
        )
    return tuple(blockers)


def _collect_prompt_cache_layers(model_kit: Any) -> tuple[Any, ...]:
    prompt_cache = getattr(getattr(model_kit, "cache_wrapper", None), "cache", None)
    if prompt_cache is None:
        prompt_cache = getattr(model_kit, "prompt_cache", None)
    if prompt_cache is None:
        return ()
    if isinstance(prompt_cache, tuple):
        return prompt_cache
    if isinstance(prompt_cache, list):
        return tuple(prompt_cache)
    try:
        return tuple(prompt_cache)
    except TypeError:
        return (prompt_cache,)


def _cache_layer_is_qwen35_sequential_arrays_cache(cache: Any) -> bool:
    """True iff ``cache`` is the proven sequential single-sequence ``ArraysCache``.

    The real mlx-lm ``ArraysCache`` loaded by Qwen3.5 / Qwen3.6
    ``ModelKit`` for sequential text generation has ``lengths`` and
    ``left_padding`` set to ``None`` with the GDN state stored in the
    ``cache`` list (``cache[0]`` is the conv window, ``cache[1]`` is the
    running gated-delta state). Any non-``None`` ``lengths`` /
    ``left_padding`` (ragged / batched variant) is NOT the proven shape
    and must remain fail-closed.
    """
    if cache is None:
        return False
    cache_list = getattr(cache, "cache", None)
    if not isinstance(cache_list, list) or len(cache_list) < 2:
        return False
    if getattr(cache, "lengths", None) is not None:
        return False
    if getattr(cache, "left_padding", None) is not None:
        return False
    return True


def _dedupe_blockers(blockers: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for blocker in blockers:
        if blocker in seen:
            continue
        seen.add(blocker)
        deduped.append(blocker)
    return tuple(deduped)


# Exact proven Qwen3.5 / Qwen3.6 27B sequential-text cache layout
# (16 KVCache full-attention layers + 48 ArraysCache GDN layers). Only
# this exact shape is allowed through the DFlash runtime validator; all
# other ragged / opaque / non-Qwen cache layouts must remain fail-closed
# unless future work proves a wider surface is rollback-safe.
DFLASH_PROVEN_QWEN35_LAYOUT = (16, 48)
DFLASH_PROVEN_QWEN35_TOTAL_LAYERS = sum(DFLASH_PROVEN_QWEN35_LAYOUT)


def _summarize_prompt_cache_layout(
    prompt_cache_layers: tuple[Any, ...],
) -> tuple[int, int, int]:
    """Return ``(kv_count, arrays_count, other_count)`` for the prompt cache."""

    kv_count = 0
    arrays_count = 0
    other_count = 0
    for cache in prompt_cache_layers:
        if cache is None:
            continue
        cache_type_name = type(cache).__name__
        if cache_type_name == "KVCache":
            kv_count += 1
        elif cache_type_name == "ArraysCache" or _cache_layer_is_qwen35_sequential_arrays_cache(
            cache
        ):
            arrays_count += 1
        else:
            other_count += 1
    return kv_count, arrays_count, other_count


def validate_dflash_runtime_compatibility(model_kit: Any) -> tuple[str, ...]:
    """Fail closed before DFlash mutates prompt caches or live history.

    Only the exact proven Qwen3.5 / Qwen3.6 sequential-text layout
    (16 ``KVCache`` layers + 48 ``ArraysCache`` GDN layers, with
    ``lengths`` and ``left_padding`` both ``None`` on the ArraysCache
    subset) is allowed through. Any other ragged, opaque, or non-Qwen
    cache shape must remain fail-closed.
    """

    blockers: list[str] = []
    if getattr(model_kit, "draft_model", None) is not None:
        blockers.append(
            "DFlash cannot be combined with an already loaded draft_model yet"
        )
    for attr_name, label in (
        ("max_kv_size", "max_kv_size"),
        ("kv_bits", "kv_bits"),
        ("kv_group_size", "kv_group_size"),
        ("quantized_kv_start", "quantized_kv_start"),
    ):
        if getattr(model_kit, attr_name, None) is not None:
            blockers.append(f"DFlash does not support {label} cache mode yet")

    prompt_cache_layers = _collect_prompt_cache_layers(model_kit)
    if not prompt_cache_layers:
        blockers.append("DFlash requires a prompt cache before runtime execution")
    else:
        kv_count = 0
        arrays_count = 0
        ragged_arrays_count = 0
        for cache in prompt_cache_layers:
            if cache is None:
                continue
            cache_type_name = type(cache).__name__
            if cache_type_name == "KVCache":
                kv_count += 1
                continue
            if cache_type_name == "RotatingKVCache" or (
                getattr(cache, "max_size", None) is not None
                and getattr(cache, "keep", None) is not None
            ):
                blockers.append(
                    "DFlash does not support bounded/rotating cache layers yet"
                )
                continue
            if cache_type_name == "BatchKVCache":
                blockers.append("DFlash does not support ragged cache layers yet")
                continue
            is_arrays_cache = cache_type_name == "ArraysCache" or isinstance(
                getattr(cache, "cache", None), list
            )
            if is_arrays_cache:
                if _cache_layer_is_qwen35_sequential_arrays_cache(cache):
                    arrays_count += 1
                else:
                    ragged_arrays_count += 1
                    blockers.append(
                        "DFlash does not support ragged ArraysCache layers yet"
                    )
                continue
            if (
                getattr(cache, "lengths", None) is not None
                or getattr(cache, "left_padding", None) is not None
            ):
                blockers.append("DFlash does not support ragged cache layers yet")
                continue
            blockers.append(
                f"DFlash does not support non-rollback-safe cache layer {cache_type_name} yet"
            )

        # Only the exact proven Qwen3.5 / Qwen3.6 sequential layout
        # (16 KVCache + 48 ArraysCache with lengths / left_padding None)
        # is allowed. Any other count, mix, or layer shape must stay
        # fail-closed so future refactors cannot silently widen the
        # DFlash runtime surface.
        proven_kv, proven_arrays = DFLASH_PROVEN_QWEN35_LAYOUT
        if (
            ragged_arrays_count > 0
            or kv_count != proven_kv
            or arrays_count != proven_arrays
        ):
            # Describe the exact gap so workers can see why their cache
            # shape is still fail-closed.
            if (
                kv_count == proven_kv
                and arrays_count != proven_arrays
                and ragged_arrays_count == 0
            ):
                blockers.append(
                    "DFlash requires exactly 16 KVCache + 48 ArraysCache "
                    f"sequential layers; got {kv_count} KVCache + "
                    f"{arrays_count} ArraysCache"
                )
            elif (
                arrays_count == proven_arrays
                and kv_count != proven_kv
                and ragged_arrays_count == 0
            ):
                blockers.append(
                    "DFlash requires exactly 16 KVCache + 48 ArraysCache "
                    f"sequential layers; got {kv_count} KVCache + "
                    f"{arrays_count} ArraysCache"
                )
            else:
                blockers.append(
                    "DFlash requires exactly 16 KVCache + 48 ArraysCache "
                    f"sequential layers; got {kv_count} KVCache + "
                    f"{arrays_count} ArraysCache"
                )

    target_model = getattr(model_kit, "model", model_kit)
    lm = (
        target_model.language_model
        if hasattr(target_model, "language_model")
        else target_model
    )
    if not hasattr(lm, "rollback_speculative_cache"):
        blockers.append(
            f"{type(lm).__name__} does not implement rollback_speculative_cache"
        )

    return _dedupe_blockers(blockers)


def build_dflash_no_go_message(
    readiness: DFlashReadinessReport,
    *,
    surface_blockers: tuple[str, ...] = (),
) -> str:
    blockers = [*surface_blockers, *readiness.blockers]
    if not blockers:
        return "DFlash boundary is wired, but no execution path exists yet"

    next_steps = (
        "Next steps: install the optional DFlash dependency, stage a local "
        "Qwen-family target/drafter pair, and keep the feature default-off "
        "until a real sequential prototype is implemented."
    )
    return "DFlash no-go: " + "; ".join(blockers) + ". " + next_steps


def build_dflash_runtime_no_go_message(blockers: tuple[str, ...]) -> str:
    if not blockers:
        return "DFlash boundary is wired, but no execution path exists yet"

    next_steps = (
        "Next steps: switch to a plain KVCache sequential path with a "
        "rollback-capable target model and keep DFlash default-off until a "
        "real sequential smoke passes."
    )
    return "DFlash no-go: " + "; ".join(blockers) + ". " + next_steps


def validate_dflash_boundary(
    *,
    options: DFlashBoundaryOptions,
    surface_label: str,
    images_b64: Optional[list[str]],
    specprefill_toggle: Optional[bool],
    speculative_decoding_toggle: Optional[bool],
    num_draft_tokens: Optional[int],
    draft_model: Any | None,
) -> None:
    """Raise if DFlash was opted in but the boundary is not ready."""

    if not options.enabled:
        return

    surface_blockers = validate_dflash_surface_compatibility(
        enabled=True,
        surface_label=surface_label,
        images_b64=images_b64,
        specprefill_toggle=specprefill_toggle,
        speculative_decoding_toggle=speculative_decoding_toggle,
        num_draft_tokens=num_draft_tokens,
        draft_model=draft_model,
    )
    readiness = probe_dflash_readiness(options)
    if surface_blockers or readiness.blockers:
        raise DFlashUnavailableError(
            build_dflash_no_go_message(
                readiness,
                surface_blockers=surface_blockers,
            )
        )
    return None
