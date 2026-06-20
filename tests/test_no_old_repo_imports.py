from __future__ import annotations

from pathlib import Path


def test_no_old_repo_imports_or_path_hacks() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden = [
        "time" + "-series" + "-simulator",
        "digital" + "-twin" + "-engine",
        "flux" + "-attention" + "-engine",
        "plant" + "-scenario" + "-compiler",
        "sys" + ".path.append",
        "sys" + ".path.insert",
        "PYTHON" + "PATH",
    ]
    findings = []
    for base in [root / "src", root / "scripts", root / "tests"]:
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    findings.append((path.relative_to(root), token))
    assert findings == []

