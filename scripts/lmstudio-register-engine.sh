#!/bin/bash
set -euo pipefail

# lmstudio-register-engine.sh
#
# Register a custom MLX inference engine as a selectable backend in LM Studio.
#
# Usage:
#   ./lmstudio-register-engine.sh <engine-name> <version> [mlx-engine-source-dir]
#
# Examples:
#   ./lmstudio-register-engine.sh cheetara-mlx 1.0.0
#   ./lmstudio-register-engine.sh my-engine 2.0.0 /path/to/my/mlx_engine
#
# Step 1: Create a backend directory + manifest with your engine identity
# Step 2: Copy your mlx_engine source into all LM Studio runtime copies
# Step 3: Register in internal-engine-index.json (APPEND at end to preserve
#         meta indices that the UI dropdown depends on)
# Step 4: Register in backend preferences

ENGINE_NAME="${1:?"Usage: $0 <engine-name> <version> [mlx-engine-src-dir]"}"
ENGINE_VERSION="${2:?"Usage: $0 <engine-name> <version> [mlx-engine-src-dir]"}"
MLX_ENGINE_SRC="${3:-}"

LMSTUDIO_BACKENDS="$HOME/.lmstudio/extensions/backends"
LMSTUDIO_INTERNAL="$HOME/.lmstudio/.internal"
OFFICIAL_ID="mlx-llm-mac-arm64-apple-metal-advsimd"
BACKEND_ID="${OFFICIAL_ID}-${ENGINE_NAME}"
BACKEND_DIR="${BACKEND_ID}-${ENGINE_VERSION}"
OVERRIDE_MLX_RUNTIME_NAME="${LMSTUDIO_MLX_RUNTIME_NAME:-}"
OVERRIDE_CPYTHON_RUNTIME_NAME="${LMSTUDIO_CPYTHON_RUNTIME_NAME:-}"
SKIP_CODESIGN_CHECK="${LMSTUDIO_SKIP_RUNTIME_CODESIGN_CHECK:-}"

normalize_version() {
  local raw_version="$1"
  if [[ "$raw_version" =~ ^([0-9]{4})([0-9]{2})([0-9]{2})$ ]]; then
    echo "${BASH_REMATCH[1]}.$((10#${BASH_REMATCH[2]})).$((10#${BASH_REMATCH[3]}))"
    return
  fi
  echo "$raw_version"
}

validate_version() {
  local raw_version="$1"
  if [[ ! "$raw_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "ERROR: Version must be semver-compatible (for example 2026.6.22)." >&2
    exit 1
  fi
}

codesign_team_identifier() {
  local path="$1"
  codesign -dv --verbose=4 "$path" 2>&1 | sed -n 's/^TeamIdentifier=//p' | tail -1
}

codesign_signature_kind() {
  local path="$1"
  codesign -dv --verbose=4 "$path" 2>&1 | sed -n 's/^Signature=//p' | tail -1
}

read_official_vendor_runtime_pair() {
  local manifest_path="$LMSTUDIO_BACKENDS/${OFFICIAL_ID}-1.9.0/backend-manifest.json"
  [ -f "$manifest_path" ] || return 1
  python3 - "$manifest_path" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text())
packages = manifest.get("vendor_lib_package_names") or []
if len(packages) < 2:
    raise SystemExit(1)

for package_name in packages[:2]:
    print(package_name.split("/", 1)[-1])
PY
}

ENGINE_VERSION="$(normalize_version "$ENGINE_VERSION")"
validate_version "$ENGINE_VERSION"
BACKEND_ID="${OFFICIAL_ID}-${ENGINE_NAME}"
BACKEND_DIR="${BACKEND_ID}-${ENGINE_VERSION}"

# --- Validate ---
if [ ! -d "$LMSTUDIO_BACKENDS/${OFFICIAL_ID}-1.9.0" ]; then
  echo "ERROR: Official MLX backend not found. Install the MLX engine in LM Studio first."
  echo "  Expected: $LMSTUDIO_BACKENDS/${OFFICIAL_ID}-1.9.0"
  exit 1
fi

if [ -d "$LMSTUDIO_BACKENDS/$BACKEND_DIR" ]; then
  echo "ERROR: Backend '$BACKEND_DIR' already exists."
  echo "  Remove first: rm -rf '$LMSTUDIO_BACKENDS/$BACKEND_DIR'"
  echo "  And remove from internal-engine-index.json"
  exit 1
fi

# Resolve mlx_engine source
if [ -z "$MLX_ENGINE_SRC" ]; then
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  PROBABLE_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
  if [ -d "$PROBABLE_REPO/mlx_engine" ]; then
    MLX_ENGINE_SRC="$PROBABLE_REPO/mlx_engine"
  else
    echo "ERROR: No mlx_engine source specified. Found: $PROBABLE_REPO/mlx_engine"
    exit 1
  fi
fi

if [ ! -d "$MLX_ENGINE_SRC" ]; then
  echo "ERROR: mlx_engine source not found at: $MLX_ENGINE_SRC"
  exit 1
fi

# Determine runtime vendor path
VENDOR_DIR="$LMSTUDIO_BACKENDS/vendor/_amphibian"
RUNTIME_SELECTION_SOURCE="latest-discovered"
PREFERRED_MLX_RUNTIME_NAME=""
PREFERRED_CPYTHON_RUNTIME_NAME=""

OFFICIAL_RUNTIME_NAMES=()
while IFS= read -r runtime_name; do
  OFFICIAL_RUNTIME_NAMES+=("$runtime_name")
done < <(read_official_vendor_runtime_pair 2>/dev/null || true)

if [ "${#OFFICIAL_RUNTIME_NAMES[@]}" -ge 2 ]; then
  CANDIDATE_MLX_RUNTIME="$VENDOR_DIR/${OFFICIAL_RUNTIME_NAMES[0]}"
  CANDIDATE_CPYTHON_RUNTIME="$VENDOR_DIR/${OFFICIAL_RUNTIME_NAMES[1]}"
  if [ -d "$CANDIDATE_MLX_RUNTIME" ] && [ -d "$CANDIDATE_CPYTHON_RUNTIME" ]; then
    PREFERRED_MLX_RUNTIME_NAME="${OFFICIAL_RUNTIME_NAMES[0]}"
    PREFERRED_CPYTHON_RUNTIME_NAME="${OFFICIAL_RUNTIME_NAMES[1]}"
    RUNTIME_SELECTION_SOURCE="official-backend-manifest"
  fi
fi

if [ -n "$OVERRIDE_MLX_RUNTIME_NAME" ]; then
  PREFERRED_MLX_RUNTIME_NAME="$OVERRIDE_MLX_RUNTIME_NAME"
  RUNTIME_SELECTION_SOURCE="override"
fi
if [ -n "$OVERRIDE_CPYTHON_RUNTIME_NAME" ]; then
  PREFERRED_CPYTHON_RUNTIME_NAME="$OVERRIDE_CPYTHON_RUNTIME_NAME"
  RUNTIME_SELECTION_SOURCE="override"
fi

if [ -n "$PREFERRED_MLX_RUNTIME_NAME" ]; then
  MLX_RUNTIME="$VENDOR_DIR/$PREFERRED_MLX_RUNTIME_NAME"
else
  MLX_RUNTIME=$(ls -d "$VENDOR_DIR"/app-mlx-generate-mac*-arm64@* 2>/dev/null | sort -t@ -k2 -n | tail -1)
fi
if [ -n "$PREFERRED_CPYTHON_RUNTIME_NAME" ]; then
  CPYTHON_RUNTIME="$VENDOR_DIR/$PREFERRED_CPYTHON_RUNTIME_NAME"
else
  CPYTHON_RUNTIME=$(ls -d "$VENDOR_DIR"/cpython3.11-mac-arm64@* 2>/dev/null | sort -t@ -k2 -n | tail -1)
fi
MLX_RUNTIME_NAME=$(basename "$MLX_RUNTIME")
CPYTHON_RUNTIME_NAME=$(basename "$CPYTHON_RUNTIME")

if [ ! -d "$MLX_RUNTIME" ]; then
  echo "ERROR: MLX runtime not found: $MLX_RUNTIME"
  exit 1
fi

if [ ! -d "$CPYTHON_RUNTIME" ]; then
  echo "ERROR: CPython runtime not found: $CPYTHON_RUNTIME"
  exit 1
fi

SELECTED_MLX_CORE="$MLX_RUNTIME/lib/python3.11/site-packages/mlx/core.cpython-311-darwin.so"
LMSTUDIO_WORKER_BIN="$HOME/.lmstudio/.internal/utils/node"
if [ ! -f "$SELECTED_MLX_CORE" ]; then
  echo "ERROR: Selected MLX runtime core not found: $SELECTED_MLX_CORE"
  exit 1
fi
if [ ! -f "$LMSTUDIO_WORKER_BIN" ]; then
  echo "ERROR: LM Studio worker binary not found: $LMSTUDIO_WORKER_BIN"
  exit 1
fi

if [[ ! "$SKIP_CODESIGN_CHECK" =~ ^(1|true|yes|on)$ ]]; then
  WORKER_TEAM_ID="$(codesign_team_identifier "$LMSTUDIO_WORKER_BIN")"
  WORKER_SIGNATURE_KIND="$(codesign_signature_kind "$LMSTUDIO_WORKER_BIN")"
  SELECTED_CORE_TEAM_ID="$(codesign_team_identifier "$SELECTED_MLX_CORE")"
  SELECTED_CORE_SIGNATURE_KIND="$(codesign_signature_kind "$SELECTED_MLX_CORE")"
  if [ "$WORKER_TEAM_ID" != "$SELECTED_CORE_TEAM_ID" ]; then
    echo "ERROR: Selected MLX runtime is not signed for the active LM Studio worker." >&2
    echo "  Worker: $LMSTUDIO_WORKER_BIN" >&2
    echo "    TeamIdentifier=${WORKER_TEAM_ID:-<unset>} Signature=${WORKER_SIGNATURE_KIND:-<unknown>}" >&2
    echo "  MLX core: $SELECTED_MLX_CORE" >&2
    echo "    TeamIdentifier=${SELECTED_CORE_TEAM_ID:-<unset>} Signature=${SELECTED_CORE_SIGNATURE_KIND:-<unknown>}" >&2
    echo "  LM Studio will reject this runtime during model load." >&2
    echo "  Re-sign the selected runtime with the LM Studio Team ID or use a vendor runtime signed for this worker." >&2
    echo "  Set LMSTUDIO_SKIP_RUNTIME_CODESIGN_CHECK=1 only to bypass this preflight intentionally." >&2
    exit 1
  fi
fi

echo "Using runtime: vendor/_amphibian/$MLX_RUNTIME_NAME"
echo "Using python runtime: vendor/_amphibian/$CPYTHON_RUNTIME_NAME"
echo "Runtime selection source: $RUNTIME_SELECTION_SOURCE"

# --- Step 1: Create backend directory + patch manifest ---
echo "[1/5] Creating backend directory $BACKEND_DIR ..."
cp -R "$LMSTUDIO_BACKENDS/${OFFICIAL_ID}-1.9.0" "$LMSTUDIO_BACKENDS/$BACKEND_DIR"

MANIFEST="$LMSTUDIO_BACKENDS/$BACKEND_DIR/backend-manifest.json"
python3 -c "
import json
with open('$MANIFEST') as f:
    m = json.load(f)
m['name'] = '${BACKEND_ID}'
m['version'] = '${ENGINE_VERSION}'
m['vendor_lib_package_names'] = [
    '_amphibian/${MLX_RUNTIME_NAME}',
    '_amphibian/${CPYTHON_RUNTIME_NAME}',
]
with open('$MANIFEST', 'w') as f:
    json.dump(m, f, indent=2)
print('  name:', m['name'])
print('  version:', m['version'])
print('  vendor_lib_package_names:', m.get('vendor_lib_package_names'))
"

# --- Step 2: Generate display-data.json ---
echo "[2/5] Generating display-data.json ..."
DISPLAY="$LMSTUDIO_BACKENDS/$BACKEND_DIR/display-data.json"
python3 -c "
import json
data = [[\"en\", {
    \"langKey\": \"en\",
    \"displayName\": \"${ENGINE_NAME}\",
    \"description\": \"Custom MLX engine: ${ENGINE_NAME} v${ENGINE_VERSION}\",
    \"releaseNotes\": [
        {\"version\": \"${ENGINE_VERSION}\", \"releaseNotes\": \"Custom build from: ${MLX_ENGINE_SRC}\"}
    ]
}]]
with open('$DISPLAY', 'w') as f:
    json.dump(data, f, indent=2)
print('  displayName: ${ENGINE_NAME}')
"

# --- Step 3: Copy mlx_engine source into runtimes ---
echo "[3/5] Deploying mlx_engine source into all runtime copies ..."
if [ -n "$OVERRIDE_MLX_RUNTIME_NAME" ]; then
  RUNTIME_SITE_PACKAGES=("$MLX_RUNTIME/lib/python3.11/site-packages")
else
  RUNTIME_SITE_PACKAGES=("$VENDOR_DIR"/app-mlx-generate-mac*-arm64@*/lib/python3.11/site-packages)
fi
for sp in "${RUNTIME_SITE_PACKAGES[@]}"; do
  echo "  -> $sp/mlx_engine"
  rm -rf "$sp/mlx_engine"
  cp -R "$MLX_ENGINE_SRC" "$sp/mlx_engine"
done

# --- Step 4: Register in internal-engine-index.json (APPEND only) ---
echo "[4/5] Registering in internal-engine-index.json ..."
python3 -c "
import copy
import glob
import json
import os
import shutil

index_path = '$LMSTUDIO_INTERNAL/internal-engine-index.json'
backend_dir = '$LMSTUDIO_BACKENDS/$BACKEND_DIR'
backend_id = '$BACKEND_ID'
backend_version = '$ENGINE_VERSION'
official_id = '$OFFICIAL_ID'
backup_path = index_path + '.script-backup'

with open(index_path) as f:
    data = json.load(f)

items = data.get('json', data)
meta = data.get('meta', {})
original_count = len(items)

if original_count == 0:
    print('  ERROR: Refusing to modify an empty internal-engine-index.json')
    exit(1)

shutil.copy2(index_path, backup_path)

def find_template(candidates):
    preferred = None
    fallback = None
    for item in candidates:
        m = item.get('manifest', {})
        if m.get('name') != official_id:
            continue
        if fallback is None:
            fallback = copy.deepcopy(item)
        if m.get('version') == '1.9.0':
            preferred = copy.deepcopy(item)
            break
    return preferred or fallback

# Template: prefer official MLX 1.9.0 from the current index, then fall back to backups.
template = find_template(items)

if template is None:
    backup_candidates = sorted(glob.glob(index_path + '.bak.*'), reverse=True)
    for candidate_path in backup_candidates:
        with open(candidate_path) as candidate_file:
            candidate_data = json.load(candidate_file)
        candidate_items = candidate_data.get('json', candidate_data)
        template = find_template(candidate_items)
        if template is not None:
            print(f'  Using official MLX template from backup: {candidate_path}')
            break

if template is None:
    print('  ERROR: Could not find official MLX 1.9.0 template')
    exit(1)

# Remove only an existing entry for this exact backend name + version
items[:] = [
    item
    for item in items
    if not (
        item.get('manifest', {}).get('name') == backend_id
        and item.get('manifest', {}).get('version') == backend_version
    )
]

# Build new entry
new_entry = copy.deepcopy(template)
new_entry['manifest']['name'] = backend_id
new_entry['manifest']['version'] = backend_version
new_entry['manifest']['vendor_lib_package_names'] = [
    '_amphibian/${MLX_RUNTIME_NAME}',
    '_amphibian/${CPYTHON_RUNTIME_NAME}',
]
new_entry['libLmStudioPath'] = os.path.join(backend_dir, 'liblmstudio_bindings.node')
new_entry['launchInfo']['engineLibPath'] = os.path.join(backend_dir, 'llm_engine_mlx_amphibian.node')
new_entry['amphibianPath'] = os.path.join('$VENDOR_DIR', '${MLX_RUNTIME_NAME}')
new_entry['envVars'] = {
    'DYLD_LIBRARY_PATH': os.path.join('$VENDOR_DIR', '${CPYTHON_RUNTIME_NAME}', 'lib')
}
new_entry['displayData'] = [[\"en\", {
    \"langKey\": \"en\",
    \"displayName\": \"${ENGINE_NAME}\",
    \"description\": \"Custom MLX engine: ${ENGINE_NAME} v${ENGINE_VERSION}\",
    \"releaseNotes\": [{\"version\": \"${ENGINE_VERSION}\", \"releaseNotes\": \"Custom build\"}]
}]]

# APPEND at end (critical: preserving meta indices for UI dropdown)
items.append(new_entry)

# Rebuild meta with ONLY displayData + visibleDevicesConfig (the two keys
# the UI uses for the engine dropdown)
new_values = {}
for i in range(len(items)):
    new_values[f'{i}.displayData'] = ['map']
    new_values[f'{i}.visibleDevicesConfig'] = ['undefined']

meta['values'] = new_values

with open(index_path, 'w') as f:
    json.dump(data, f, indent=2)

if len(items) < original_count:
    print('  ERROR: Entry count shrank unexpectedly')
    shutil.copy2(backup_path, index_path)
    exit(1)

print(f'  Registered {backend_id} v{backend_version} at index {len(items)-1}')
print(f'  Meta entries: {len(new_values)} (for {len(items)} items)')
print(f'  Backup: {backup_path}')
"

# --- Step 5: Register in backend-preferences-v1.json ---
echo "[5/5] Registering in backend-preferences-v1.json ..."
python3 -c "
import json

prefs_file = '$LMSTUDIO_INTERNAL/backend-preferences-v1.json'
backend_id = '$BACKEND_ID'
backend_version = '$ENGINE_VERSION'

with open(prefs_file) as f:
    prefs = json.load(f)

# Remove old entry if exists
prefs[:] = [p for p in prefs if p.get('name') != backend_id]

# Find MLX entry for model_format
mlx_pref = next((p for p in prefs if p.get('name') == '${OFFICIAL_ID}'), None)
model_format = mlx_pref['model_format'] if mlx_pref else 'safetensors'
prefs.append({
    'model_format': model_format,
    'name': backend_id,
    'version': backend_version
})
with open(prefs_file, 'w') as f:
    json.dump(prefs, f, indent=2)
print(f'  Registered {backend_id} v{backend_version} model_format={model_format}')
"

echo ""
echo "Done. Engine '${ENGINE_NAME} v${ENGINE_VERSION}' registered."
echo ""
echo "  lms runtime select ${BACKEND_ID}   # activate"
echo "  lms runtime select ${OFFICIAL_ID}   # switch back"
echo ""
echo "To remove:"
echo "  rm -rf '$LMSTUDIO_BACKENDS/$BACKEND_DIR'"
echo "  # Then re-run this script (any name/version) to rebuild index"
