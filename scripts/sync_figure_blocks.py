#!/usr/bin/env python3
"""
Add missing ```figure blocks from en.md into zh.md translations.

Usage:
    python3 scripts/sync_figure_blocks.py           # apply all missing figure blocks
    python3 scripts/sync_figure_blocks.py --dry-run  # preview without writing
    python3 scripts/sync_figure_blocks.py --json     # machine-readable report

Strategy:
    Uses heading structure matching rather than text matching.
    Both en.md and zh.md share the same document skeleton (heading levels,
    ordering, hierarchy). The script finds figure blocks in en.md by their
    heading path + paragraph offset, then inserts at the same structural
    position in zh.md.
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_heading_tree(lines: list[str]) -> list[dict]:
    """Parse headings into a flat list with {level, text, line_idx, section_end}."""
    headings = []
    heading_pat = re.compile(r"^(#{1,6})\s+(.+)$")
    for i, line in enumerate(lines):
        m = heading_pat.match(line)
        if m:
            headings.append(
                {"level": len(m.group(1)), "text": m.group(2), "line_idx": i}
            )
    # set section_end for each heading
    for idx, h in enumerate(headings):
        if idx + 1 < len(headings):
            h["section_end"] = headings[idx + 1]["line_idx"]
        else:
            h["section_end"] = len(lines)
    return headings


def find_heading_path(
    line_idx: int, headings: list[dict]
) -> list[dict]:
    """Return the heading hierarchy enclosing a given line."""
    path = []
    for h in headings:
        if h["line_idx"] < line_idx:
            # Remove any headings at deeper or equal level
            while path and path[-1]["level"] >= h["level"]:
                path.pop()
            path.append(h)
        elif h["line_idx"] >= line_idx:
            break
    return path


def heading_section_match(
    en_headings: list[dict],
    zh_headings: list[dict],
) -> dict[int, int]:
    """Match en.md heading indices to zh.md heading indices by structure.

    Uses heading level + relative position within parent to match,
    even when heading text is in different languages.

    Returns dict mapping en_heading_index -> zh_heading_index.
    """
    en_by_level: dict[int, list[int]] = {}
    for idx, h in enumerate(en_headings):
        en_by_level.setdefault(h["level"], []).append(idx)

    zh_by_level: dict[int, list[int]] = {}
    for idx, h in enumerate(zh_headings):
        zh_by_level.setdefault(h["level"], []).append(idx)

    # Match by level and ordinal position within level
    matching = {}
    for level, en_indices in en_by_level.items():
        zh_indices = zh_by_level.get(level, [])
        for pos, en_idx in enumerate(en_indices):
            if pos < len(zh_indices):
                matching[en_idx] = zh_indices[pos]

    return matching


def compute_para_offset(
    line_idx: int, section_start: int, section_end: int, lines: list[str]
) -> int:
    """Count non-empty paragraphs between section_start and the figure block."""
    offset = 0
    for i in range(section_start, min(line_idx, section_end)):
        line = lines[i].strip()
        if line and not line.startswith("```"):
            offset += 1
    return offset


def find_insertion_point(
    zh_lines: list[str],
    en_headings: list[dict],
    zh_headings: list[dict],
    heading_match: dict[int, int],
    block: dict,
) -> int | None:
    """Find where to insert the figure block in zh.md using structural matching.

    1. Find the en.md heading that contains the figure block
    2. Find the matching zh.md heading (via heading_match)
    3. Within that zh.md section, use paragraph offset to pinpoint insertion
    """
    en_h_path = find_heading_path(block["start_line"], en_headings)
    if not en_h_path:
        return None

    innermost = en_h_path[-1]
    en_h_idx = next(
        (i for i, h in enumerate(en_headings) if h["line_idx"] == innermost["line_idx"]),
        None,
    )
    if en_h_idx is None or en_h_idx not in heading_match:
        return None

    zh_h_idx = heading_match[en_h_idx]
    zh_h = zh_headings[zh_h_idx]
    section_start = zh_h["line_idx"]
    section_end = zh_h["section_end"]

    para_offset = compute_para_offset(
        block["start_line"],
        innermost["line_idx"],
        innermost["section_end"],
        block["_all_lines"],
    )

    # Walk through zh.md section to same paragraph offset
    para_count = 0
    insert_at = section_start + 1
    for i in range(section_start + 1, min(section_end, len(zh_lines))):
        line = zh_lines[i].strip()
        if line and not line.startswith("```"):
            para_count += 1
            if para_count > para_offset:
                insert_at = i
                break
        insert_at = i + 1

    # clamp
    if insert_at > section_end:
        insert_at = section_end

    return insert_at


def parse_figure_blocks(text: str, source_lines: list[str]) -> list[dict]:
    """Extract ```figure blocks from text."""
    lines = text.split("\n")
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```figure"):
            name = lines[i + 1].strip() if i + 1 < len(lines) else ""
            end_idx = i
            for j in range(i + 1, min(i + 10, len(lines))):
                if lines[j].strip() == "```":
                    end_idx = j
                    break
            block_lines = lines[i : end_idx + 1]
            blocks.append(
                {
                    "name": name,
                    "start_line": i,
                    "block_text": "\n".join(block_lines),
                    "_all_lines": source_lines,
                }
            )
            i = end_idx + 1
            continue
        i += 1
    return blocks


def sync_lesson(en_path: Path, zh_path: Path, dry_run: bool = False) -> dict | None:
    """Sync figure blocks from en.md to zh.md. Returns change report or None."""
    en_text = en_path.read_text()
    zh_text = zh_path.read_text()

    if "```figure" in zh_text:
        return None

    en_lines = en_text.split("\n")
    zh_lines = zh_text.split("\n")

    en_headings = parse_heading_tree(en_lines)
    zh_headings = parse_heading_tree(zh_lines)
    heading_match = heading_section_match(en_headings, zh_headings)

    if not heading_match:
        return None

    en_blocks = parse_figure_blocks(en_text, en_lines)
    if not en_blocks:
        return None

    changes = []
    for block in en_blocks:
        insert_at = find_insertion_point(
            zh_lines, en_headings, zh_headings, heading_match, block
        )
        if insert_at is not None:
            changes.append(
                {
                    "figure": block["name"],
                    "insert_at": insert_at,
                    "block_text": block["block_text"],
                }
            )

    if not changes:
        return None

    if dry_run:
        return {
            "en": str(en_path.relative_to(REPO_ROOT)),
            "zh": str(zh_path.relative_to(REPO_ROOT)),
            "changes": [
                {"figure": c["figure"], "insert_at": c["insert_at"]}
                for c in changes
            ],
        }

    changes.sort(key=lambda c: c["insert_at"], reverse=True)
    for change in changes:
        zh_lines.insert(change["insert_at"], "")
        zh_lines.insert(change["insert_at"], change["block_text"])

    zh_path.write_text("\n".join(zh_lines))

    return {
        "en": str(en_path.relative_to(REPO_ROOT)),
        "zh": str(zh_path.relative_to(REPO_ROOT)),
        "changes": [{"figure": c["figure"], "insert_at": c["insert_at"]} for c in changes],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sync ```figure blocks from en.md into zh.md translations."
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    args = parser.parse_args()

    results = []
    total_figures_added = 0
    lessons_modified = 0

    for zh_path in sorted(REPO_ROOT.glob("phases/*/*/docs/zh.md")):
        en_path = zh_path.with_name("en.md")
        if not en_path.exists():
            continue
        result = sync_lesson(en_path, zh_path, dry_run=args.dry_run)
        if result:
            results.append(result)
            lessons_modified += 1
            total_figures_added += len(result["changes"])

    if args.json:
        print(
            json.dumps(
                {
                    "dry_run": args.dry_run,
                    "lessons_modified": lessons_modified,
                    "figures_added": total_figures_added,
                    "results": results,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        action = "Would add" if args.dry_run else "Added"
        print(f"Figure block sync ({'dry-run' if args.dry_run else 'live'})")
        print(f"{'=' * 60}")
        print(f"Lessons modified: {lessons_modified}")
        print(f"Figures {action.lower()}: {total_figures_added}")
        print()
        if results:
            for r in results:
                figures = ", ".join(c["figure"] for c in r["changes"])
                print(f"  {r['en']}")
                print(f"    {action}: {figures}")
            print()
        if args.dry_run:
            print("Run without --dry-run to apply these changes.")


if __name__ == "__main__":
    main()
