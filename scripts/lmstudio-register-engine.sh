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
MLX_RUNTIME=$(ls -d "$VENDOR_DIR"/app-mlx-generate-mac*-arm64@* 2>/dev/null | sort -t@ -k2 -n | tail -1)
CPYTHON_RUNTIME=$(ls -d "$VENDOR_DIR"/cpython3.11-mac-arm64@* 2>/dev/null | sort -t@ -k2 -n | tail -1)
MLX_RUNTIME_NAME=$(basename "$MLX_RUNTIME")
CPYTHON_RUNTIME_NAME=$(basename "$CPYTHON_RUNTIME")

echo "Using runtime: vendor/_amphibian/$MLX_RUNTIME_NAME"

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
with open('$MANIFEST', 'w') as f:
    json.dump(m, f, indent=2)
print('  name:', m['name'])
print('  version:', m['version'])
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
for sp in "$VENDOR_DIR"/app-mlx-generate-mac*-arm64@*/lib/python3.11/site-packages; do
  echo "  -> $sp/mlx_engine"
  rm -rf "$sp/mlx_engine"
  cp -R "$MLX_ENGINE_SRC" "$sp/mlx_engine"
done

# --- Step 4: Register in internal-engine-index.json (APPEND only) ---
echo "[4/5] Registering in internal-engine-index.json ..."
python3 -c "
import json, os, copy

index_path = '$LMSTUDIO_INTERNAL/internal-engine-index.json'
backend_dir = '$LMSTUDIO_BACKENDS/$BACKEND_DIR'
backend_id = '$BACKEND_ID'
backend_version = '$ENGINE_VERSION'
official_id = '$OFFICIAL_ID'

with open(index_path) as f:
    data = json.load(f)

items = data.get('json', data)
meta = data.get('meta', {})

# Template: MLX 1.9.0
template = None
for item in items:
    m = item.get('manifest', {})
    if m.get('name') == official_id and m.get('version') == '1.9.0':
        template = copy.deepcopy(item)
        break

if template is None:
    print('  ERROR: Could not find official MLX 1.9.0 template')
    exit(1)

# Remove any existing entry with this backend_id
items[:] = [item for item in items if item.get('manifest', {}).get('name') != backend_id]

# Build new entry
new_entry = copy.deepcopy(template)
new_entry['manifest']['name'] = backend_id
new_entry['manifest']['version'] = backend_version
new_entry['libLmStudioPath'] = os.path.join(backend_dir, 'liblmstudio_bindings.node')
new_entry['launchInfo']['engineLibPath'] = os.path.join(backend_dir, 'llm_engine_mlx_amphibian.node')
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

print(f'  Registered {backend_id} v{backend_version} at index {len(items)-1}')
print(f'  Meta entries: {len(new_values)} (for {len(items)} items)')
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
if mlx_pref:
    prefs.append({
        'model_format': mlx_pref['model_format'],
        'name': backend_id,
        'version': backend_version
    })
    with open(prefs_file, 'w') as f:
        json.dump(prefs, f, indent=2)
    print(f'  Registered {backend_id} v{backend_version}')
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
