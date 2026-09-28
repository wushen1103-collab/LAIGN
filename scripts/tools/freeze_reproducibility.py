#!/usr/bin/env python3
import datetime as dt
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


COMMANDS = [
    ("git_commit", "git rev-parse --short HEAD"),
    ("hostname", "hostname"),
    ("kernel", "uname -a"),
    ("python3", "python3 --version"),
    ("gcc", "gcc --version | head -n 1"),
    ("docker", "docker --version"),
    ("nvidia_smi_query", "nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv"),
    ("nvidia_smi_full", "nvidia-smi"),
    ("conda", "conda --version"),
    ("torch", "python - <<'PY'\nimport torch\nprint(torch.__version__)\nprint(torch.version.cuda)\nPY"),
    ("rdkit", "python - <<'PY'\nimport rdkit\nprint(rdkit.__version__)\nPY"),
    ("plip", "python - <<'PY'\nimport plip\nprint(getattr(plip, '__version__', 'unknown'))\nPY"),
    ("mmseqs", "mmseqs version"),
    ("foldseek", "foldseek version"),
    ("vina", "vina --version"),
    ("openbabel", "obabel -V"),
]


def run(command: str) -> str:
    try:
        return subprocess.check_output(
            command,
            shell=True,
            text=True,
            stderr=subprocess.STDOUT,
            timeout=60,
        ).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def main() -> None:
    lines = [
        "# Reproducibility",
        "",
        f"updated_utc: {dt.datetime.utcnow().isoformat()}Z",
        f"repo: {ROOT}",
        "",
        "## Version Snapshot",
        "",
    ]
    for name, command in COMMANDS:
        lines.append(f"### {name}")
        lines.append("")
        lines.append("```text")
        lines.append(run(command))
        lines.append("```")
        lines.append("")

    lines.append("## Important Files")
    lines.append("")
    for rel in ["environment.yml", "requirements.txt", "Dockerfile", "configs/datasets.json"]:
        path = ROOT / rel
        if path.exists():
            lines.append(f"- {rel}: {path.stat().st_size} bytes")
    lines.append("")

    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "reproducibility.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
