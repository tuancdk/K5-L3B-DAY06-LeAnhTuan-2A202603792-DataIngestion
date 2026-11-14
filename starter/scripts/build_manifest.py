"""Recompute per-page stats in data/manifest.json and validate the corpus.

Static fields (source, license, exclusions, sections) are edited by hand; this script
only refreshes 'pages', 'counted_content_pages', 'total_chars' and checks sha256.
Run from starter/:  python scripts/build_manifest.py
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingestion import DEFAULT_MANIFEST, REPO_ROOT, load_corpus  # noqa: E402

MIN_CONTENT_PAGES = 30


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    pages, errors = load_corpus(DEFAULT_MANIFEST)

    for src in manifest["sources"]:
        local = REPO_ROOT / src["local_path"]
        if local.exists():
            digest = hashlib.sha256(local.read_bytes()).hexdigest()
            if digest != src["snapshot"]["sha256"]:
                errors.append({"document_id": src["source_id"],
                               "error": f"sha256 mismatch: {digest}"})
        else:
            print(f"[warn] {src['local_path']} not present locally; sha256 not checked")

        own = [p for p in pages if p.document_id == src["source_id"]]
        src["pages"] = [{
            "page": p.page,
            "page_id": f"{p.document_id}:p{p.page:03d}",
            "section": p.metadata.get("section"),
            "chars": len(p.text),
            "ocr_mean_confidence": p.metadata.get("ocr_mean_confidence"),
        } for p in own]
        src["counted_content_pages"] = len(own)
        src["total_chars"] = sum(len(p.text) for p in own)
        print(f"{src['source_id']}: {len(own)} content pages, {src['total_chars']} chars "
              f"(min {min(len(p.text) for p in own)}, max {max(len(p.text) for p in own)})")

    DEFAULT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    total = sum(s["counted_content_pages"] for s in manifest["sources"])
    print(f"Total content pages: {total} (required >= {MIN_CONTENT_PAGES})")
    for err in errors:
        print(f"[error] {err}")
    if errors or total < MIN_CONTENT_PAGES:
        sys.exit(1)


if __name__ == "__main__":
    main()
