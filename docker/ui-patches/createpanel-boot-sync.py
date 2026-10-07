#!/usr/bin/env python3
"""Fix lmModel useState + boot-sync Gradio to localStorage.

Must run LAST among CreatePanel patches. Aggressively rewrites any
lmModel useState form (including the broken double-arrow variant).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

MARKER = "[XPU-BOOT-SYNC]"

GOOD_LM_STATE = (
    "const [lmModel, setLmModel] = useState(() => "
    "(typeof localStorage !== 'undefined' && localStorage.getItem('ace-lmModel')) "
    "|| 'acestep-5Hz-lm-1.7B');"
)

BOOT_EFFECT = ''


def repair_lm_model_state(text: str) -> str:
    """Replace from 'const [lmModel, setLmModel]' through end of that statement."""
    start = text.find("const [lmModel, setLmModel]")
    if start < 0:
        print("WARN: lmModel state not found")
        return text

    # Walk from start until we have balanced braces/parens ending with ;
    i = start
    n = len(text)
    depth_paren = 0
    depth_brace = 0
    seen_open = False
    while i < n:
        ch = text[i]
        if ch == "(":
            depth_paren += 1
            seen_open = True
        elif ch == ")":
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
            seen_open = True
        elif ch == "}":
            depth_brace -= 1
        elif ch == ";" and seen_open and depth_paren <= 0 and depth_brace <= 0:
            end = i + 1
            old = text[start:end]
            text = text[:start] + GOOD_LM_STATE + text[end:]
            print("Repaired lmModel useState block:")
            print("  OLD:", repr(old[:120]) + ("..." if len(old) > 120 else ""))
            print("  NEW:", GOOD_LM_STATE)
            return text
        i += 1

    # Fallback: line-based nuke through next standalone });
    lines = text[start:].splitlines(keepends=True)
    consume = 0
    buf = ""
    for line in lines:
        buf += line
        consume += len(line)
        if "});" in line or (line.strip().endswith(";") and "useState" not in line and consume > 40):
            break
    text = text[:start] + GOOD_LM_STATE + "\n" + text[start + consume :]
    print("Repaired lmModel useState (line fallback)")
    return text


def main() -> None:
    hits = [p for p in Path(".").rglob("CreatePanel.tsx") if "node_modules" not in str(p)]
    if not hits:
        print("CreatePanel.tsx not found", file=sys.stderr)
        sys.exit(1)
    path = hits[0]
    text = path.read_text()

    text = repair_lm_model_state(text)

    # Verify parse-ish: no double '=> {' after 1.7B
    if "1.7B') =>" in text or "1.7B') => {" in text:
        print("ERROR: broken pattern still present after repair", file=sys.stderr)
        sys.exit(1)

#    if MARKER not in text:
#        m = re.search(r"\n  useEffect\(", text)
#        if m:
#            text = text[: m.start()] + "\n" + BOOT_EFFECT + text[m.start() :]
#        else:
#            text = text + "\n" + BOOT_EFFECT
#        print(f"OK boot-sync effect in {path}")
#    else:
#        print("boot-sync effect already present")

    path.write_text(text)
    print("createpanel-boot-sync complete")


if __name__ == "__main__":
    main()
