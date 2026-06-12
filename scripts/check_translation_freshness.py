#!/usr/bin/env python3
"""
Translation Freshness Checker

Usage:
    python3 scripts/check_translation_freshness.py --check         # report stale zh.md translations
    python3 scripts/check_translation_freshness.py --update        # rebuild translation_manifest.json
    python3 scripts/check_translation_freshness.py --check --json  # machine-readable output

The manifest maps each en.md to its SHA256 hash.
If en.md content changes, the hash diverges and the script flags zh.md as stale.

Add to CI:
    python3 scripts/check_translation_freshness.py --check
    if exit code != 0: a translation is out of date
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "translation_manifest.json"
MANIFEST_VERSION = 1


def sha256_of(path: Path) -> str:
    """Return lowercase hex SHA256 of the file content."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def discover_pairs() -> list[tuple[Path, Path]]:
    """Return [(en.md, zh.md), ...] for every lesson with both files."""
    pairs = []
    for zh in sorted(REPO_ROOT.glob("phases/*/*/docs/zh.md")):
        en = zh.with_name("en.md")
        if en.exists():
            pairs.append((en, zh))
    return pairs


def build_manifest() -> dict:
    """Build a manifest dict from current disk state."""
    pairs = discover_pairs()
    translations = {}
    for en, zh in pairs:
        rel_en = en.relative_to(REPO_ROOT).as_posix()
        rel_zh = zh.relative_to(REPO_ROOT).as_posix()
        translations[rel_en] = {
            "sha256": sha256_of(en),
            "zh_path": rel_zh,
        }
    return {
        "version": MANIFEST_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "translations": translations,
    }


def load_manifest() -> dict | None:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return None


def cmd_check(args):
    manifest = load_manifest()
    if manifest is None:
        print(
            "translation_manifest.json not found. Run --update first.",
            file=sys.stderr,
        )
        sys.exit(1)

    pairs = discover_pairs()
    results = []
    stale_count = 0
    missing_count = 0
    fresh_count = 0

    for en, zh in pairs:
        rel_en = en.relative_to(REPO_ROOT).as_posix()
        rel_zh = zh.relative_to(REPO_ROOT).as_posix()

        if rel_en not in manifest["translations"]:
            results.append(
                {
                    "en": rel_en,
                    "zh": rel_zh,
                    "status": "missing",
                    "detail": "not in manifest",
                }
            )
            missing_count += 1
            continue

        current_hash = sha256_of(en)
        stored_hash = manifest["translations"][rel_en]["sha256"]

        if current_hash == stored_hash:
            results.append(
                {"en": rel_en, "zh": rel_zh, "status": "fresh", "detail": ""}
            )
            fresh_count += 1
        else:
            results.append(
                {
                    "en": rel_en,
                    "zh": rel_zh,
                    "status": "stale",
                    "detail": f"hash changed: {stored_hash} -> {current_hash}",
                }
            )
            stale_count += 1

    # Check for orphaned manifest entries
    orphan_count = 0
    current_ens = {en.relative_to(REPO_ROOT).as_posix() for en, _ in pairs}
    for rel_en in manifest["translations"]:
        if rel_en not in current_ens:
            orphan_count += 1

    if args.json:
        print(
            json.dumps(
                {
                    "results": results,
                    "summary": {
                        "total": len(results),
                        "fresh": fresh_count,
                        "stale": stale_count,
                        "missing": missing_count,
                        "orphaned": orphan_count,
                    },
                },
                indent=2,
            )
        )
    else:
        print(f"Translation freshness check")
        print(f"{'=' * 60}")
        print(f"Total: {len(results)}  Fresh: {fresh_count}  "
              f"Stale: {stale_count}  Missing: {missing_count}  "
              f"Orphaned: {orphan_count}")
        print()

        if stale_count:
            print("--- STALE translations (en.md hash changed) ---")
            for r in results:
                if r["status"] == "stale":
                    print(f"  {r['en']}")
                    print(f"    zh: {r['zh']}")
                    print(f"    {r['detail']}")
            print()

        if missing_count:
            print("--- MISSING from manifest ---")
            for r in results:
                if r["status"] == "missing":
                    print(f"  {r['en']}")
            print()

        if orphan_count:
            print(f"(Note: {orphan_count} manifest entries have no matching en.md "
                  f"on disk and can be cleaned up.)")
            print()

        if stale_count == 0 and missing_count == 0:
            print("All translations are fresh.")
        else:
            print(f"ACTION REQUIRED: {stale_count} stale + {missing_count} missing")

    return stale_count + missing_count


def cmd_update(args):
    manifest = build_manifest()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    total = len(manifest["translations"])
    print(f"translation_manifest.json updated: {total} translations registered.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Check or update zh.md translation freshness via en.md content hash."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--check", action="store_true", help="Compare manifest hashes against current en.md"
    )
    group.add_argument(
        "--update", action="store_true", help="Rebuild translation_manifest.json"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output JSON (valid with --check)"
    )
    args = parser.parse_args()

    if args.check:
        rc = cmd_check(args)
    elif args.update:
        rc = cmd_update(args)
    else:
        rc = 1

    sys.exit(rc)


if __name__ == "__main__":
    main()
