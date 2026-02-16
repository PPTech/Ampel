#!/usr/bin/env bash
# Version: 0.9.1
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
# Author: Dr. Babak Sorkhpour with support from ChatGPT
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$ROOT_DIR/data/external_datasets_manifest.json"
OUT_DIR="$ROOT_DIR/data/local_datasets"

mkdir -p "$OUT_DIR"
python3 - "$MANIFEST" "$OUT_DIR" <<'PY'
import json
import re
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
data = json.loads(manifest.read_text(encoding='utf-8'))
rows = list(data.get('datasets', []))

def slug(name: str) -> str:
    s = re.sub(r'[^a-zA-Z0-9]+', '-', name.strip().lower()).strip('-')
    return s or 'dataset'

for ds in rows:
    ds_dir = out_dir / slug(str(ds.get('name', 'dataset')))
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / 'SOURCE.json').write_text(json.dumps(ds, ensure_ascii=False, indent=2), encoding='utf-8')
    (ds_dir / 'README.txt').write_text(
        'This folder stores local dataset metadata stubs only.\n'
        'Download and usage must comply with upstream license and terms.\n',
        encoding='utf-8',
    )

index = {
    'version': '0.9.2',
    'mode': 'local_metadata_stub',
    'dataset_count': len(rows),
    'source_manifest': str(manifest),
    'privacy_note': 'No raw GPS/user data are uploaded by this script.',
}
(out_dir / 'INDEX.json').write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')

steps=['# Dataset download steps (license-aware)', '']
for i,ds in enumerate(rows, start=1):
    name=str(ds.get('name','dataset'))
    url=str(ds.get('url',''))
    license_name=str(ds.get('license','unknown'))
    slug_name=slug(name)
    steps.append(f"{i}. {name}")
    steps.append(f"   - License: {license_name}")
    steps.append(f"   - Source: {url}")
    steps.append(f"   - Local folder: data/local_datasets/{slug_name}")
    steps.append("   - Suggested command: manual download from source portal, then place files under local folder respecting license terms.")
    steps.append('')
(out_dir / 'DOWNLOAD_STEPS.md').write_text('\n'.join(steps), encoding='utf-8')
print(json.dumps(index, ensure_ascii=False))
PY

echo "[import] local dataset metadata stubs prepared at: $OUT_DIR"
