#!/usr/bin/env bash
set -euo pipefail

# fetch_data.sh
#
# Purpose
# - Download historical parquet datasets from a GitHub Release (NOT git history).
# - Verify SHA256 (and optional byte size) per data/data_manifest.json.
#
# Default behavior
# - With no arguments: fetch ONLY the 3-month dataset used by run_phase3_local.py.
#
# Optional behavior
# - You may pass a dataset selector to fetch a different asset, if present in the manifest:
#     ./scripts/fetch_data.sh 3mo
#     ./scripts/fetch_data.sh 13mo
#     ./scripts/fetch_data.sh full
#
# Requirements
# - data/data_manifest.json with fields:
#   {
#     "release_tag": "data-vX",
#     "assets": [
#       { "name": "...", "destination": "...", "sha256": "...", "bytes": 0, "id": "3mo" },
#       ...
#     ]
#   }
#
# Notes
# - If "id" is not present on an asset, this script falls back to matching by common name patterns.

repo_owner="Derpbert-Psyc"
repo_name="btc_alpha"
manifest_path="data/data_manifest.json"

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "error: missing dependency: $1"; exit 1; }; }
need_cmd python3
need_cmd curl
need_cmd sha256sum

if [ ! -f "$manifest_path" ]; then
  echo "error: manifest not found at $manifest_path"
  exit 1
fi

selector="${1:-3mo}"

release_tag="$(python3 - <<'PY'
import json
m=json.load(open("data/data_manifest.json"))
print(m["release_tag"])
PY
)"

api_url="https://api.github.com/repos/${repo_owner}/${repo_name}/releases/tags/${release_tag}"

auth_header=()
if [ "${GITHUB_TOKEN:-}" != "" ]; then
  auth_header=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
fi

release_json="$(curl -fsSL "${auth_header[@]}" "$api_url")"

python3 - <<'PY' "$release_json"
import json,sys
r=json.loads(sys.argv[1])
tag=r.get("tag_name","")
assets=r.get("assets",[])
if not tag:
    raise SystemExit("error: release json missing tag_name")
if not isinstance(assets,list):
    raise SystemExit("error: release json missing assets list")
PY

mkdir -p historic_data

python3 - <<'PY' "$release_json" "$selector"
import json, os, sys, subprocess, hashlib, re

release = json.loads(sys.argv[1])
selector = sys.argv[2].strip().lower()
manifest = json.load(open("data/data_manifest.json"))

def find_url(asset_name: str):
    for a in release.get("assets", []):
        if a.get("name") == asset_name:
            return a.get("browser_download_url")
    return None

def normalize(s: str) -> str:
    return re.sub(r"\s+", "", s.strip().lower())

def infer_id_from_name(name: str) -> str:
    n = normalize(name)
    # Heuristics only; prefer explicit manifest "id".
    if "2018" in n and "2026" in n:
        return "full"
    if "2025-01-01" in n and "2026-01-31" in n:
        return "13mo"
    if "2025-10-01" in n and "2025-12-31" in n:
        return "3mo"
    return ""

token = os.environ.get("GITHUB_TOKEN", "").strip()
curl_base = ["curl", "-fL"]
if token:
    curl_base += ["-H", f"Authorization: Bearer {token}"]

assets = manifest.get("assets", [])
if not isinstance(assets, list) or not assets:
    raise SystemExit("error: manifest has no assets list")

# Select which manifest items to download.
selected = []
for item in assets:
    for k in ["name", "destination", "sha256", "bytes"]:
        if k not in item:
            raise SystemExit(f"error: manifest asset missing field: {k}")

    asset_id = normalize(str(item.get("id", "")))
    name = str(item["name"])
    inferred = infer_id_from_name(name)

    if selector in ("3mo", "13mo", "full"):
        if asset_id == selector or inferred == selector:
            selected.append(item)
    else:
        # Unknown selector: attempt exact match against name or destination
        if normalize(selector) == normalize(name) or normalize(selector) == normalize(str(item["destination"])):
            selected.append(item)

if not selected:
    known_ids = sorted({normalize(str(a.get("id",""))) for a in assets if str(a.get("id","")).strip()})
    raise SystemExit(
        "error: no assets matched selector.\n"
        f"  selector: {selector}\n"
        f"  known ids in manifest: {known_ids if known_ids else '(none; add id fields for reliability)'}\n"
        "  hint: use one of: 3mo | 13mo | full\n"
    )

if selector == "3mo" and len(selected) != 1:
    raise SystemExit(f"error: selector 3mo matched {len(selected)} assets; expected exactly 1. Add explicit id fields.")

# Download + verify each selected asset
for item in selected:
    name = str(item["name"])
    dest = str(item["destination"])
    sha_expected = str(item["sha256"]).strip().lower()
    bytes_expected = int(item["bytes"])

    if not sha_expected or sha_expected == "put_sha256_here":
        raise SystemExit(f"error: sha256 not set for {name} in manifest")

    url = find_url(name)
    if not url:
        raise SystemExit(f"error: asset not found in release: {name}")

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    tmp = dest + ".partial"
    print(f"downloading {name}")
    subprocess.check_call(curl_base + ["-o", tmp, url])

    if bytes_expected > 0:
        actual_size = os.path.getsize(tmp)
        if actual_size != bytes_expected:
            raise SystemExit(f"error: size mismatch for {name}: expected {bytes_expected}, got {actual_size}")

    h = hashlib.sha256()
    with open(tmp, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    actual_sha = h.hexdigest()

    if actual_sha != sha_expected:
        raise SystemExit(f"error: sha256 mismatch for {name}: expected {sha_expected}, got {actual_sha}")

    os.replace(tmp, dest)
    print(f"ok: {dest}")

print("done")
PY
