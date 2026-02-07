#!/usr/bin/env bash
set -euo pipefail

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

release_tag="$(python3 -c 'import json;print(json.load(open("'"$manifest_path"'"))["release_tag"])')"

auth_header=()
if [ "${GITHUB_TOKEN:-}" != "" ]; then
  auth_header=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
fi

api_url="https://api.github.com/repos/${repo_owner}/${repo_name}/releases/tags/${release_tag}"
release_json="$(curl -fsSL "${auth_header[@]}" "$api_url")"

assets_json="$(python3 -c 'import json;print(json.dumps(json.load(open("'"$manifest_path"'"))["assets"]))')"

mkdir -p historic_data data

python3 - <<'PY'
import json
m=json.load(open("data/data_manifest.json"))
for a in m["assets"]:
    for k in ["name","destination","sha256","bytes"]:
        if k not in a:
            raise SystemExit(f"manifest missing field {k} in asset entry")
PY

python3 - <<'PY' "$assets_json" "$release_json"
import json,sys,os,subprocess,hashlib

assets=json.loads(sys.argv[1])
release=json.loads(sys.argv[2])

def find_url(name):
    for a in release.get("assets",[]):
        if a.get("name")==name:
            return a.get("browser_download_url")
    return None

token=os.environ.get("GITHUB_TOKEN","").strip()
curl_base=["curl","-fL"]
if token:
    curl_base += ["-H", f"Authorization: Bearer {token}"]

for item in assets:
    name=item["name"]
    dest=item["destination"]
    sha=item["sha256"].lower()
    size=item["bytes"]

    if sha == "put_sha256_here" or not sha:
        raise SystemExit(f"error: sha256 not set for {name} in manifest")

    url=find_url(name)
    if not url:
        raise SystemExit(f"error: asset not found in release: {name}")

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    tmp=dest + ".partial"
    print(f"downloading {name}")
    subprocess.check_call(curl_base + ["-o", tmp, url])

    if size and size > 0:
        actual=os.path.getsize(tmp)
        if actual != size:
            raise SystemExit(f"error: size mismatch for {name}: expected {size}, got {actual}")

    h=hashlib.sha256()
    with open(tmp,"rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    actual_sha=h.hexdigest()
    if actual_sha != sha:
        raise SystemExit(f"error: sha256 mismatch for {name}: expected {sha}, got {actual_sha}")

    os.replace(tmp,dest)
    print(f"ok: {dest}")
PY

echo "done"
