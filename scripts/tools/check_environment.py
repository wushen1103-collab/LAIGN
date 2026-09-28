#!/usr/bin/env python3
import importlib
import json
import shutil
import subprocess
import sys


def run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def module_version(name):
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": True,
        "version": getattr(mod, "__version__", "unknown"),
    }


def main():
    info = {"python": sys.version}
    torch_info = module_version("torch")
    if torch_info["available"]:
        import torch

        torch_info.update(
            {
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": getattr(torch.version, "cuda", None),
                "device_count": torch.cuda.device_count(),
            }
        )
    info["torch"] = torch_info
    for name in ["rdkit", "plip", "esm", "torch_geometric", "numpy", "pandas", "sklearn"]:
        info[name] = module_version(name)
    info["tools"] = {
        "plip": run([sys.executable, "-m", "plip.plipcmd", "-h"])[:300],
        "mmseqs": run(["mmseqs", "version"]) if shutil.which("mmseqs") else "UNAVAILABLE",
        "foldseek": run(["foldseek", "version"]) if shutil.which("foldseek") else "UNAVAILABLE",
        "vina": run(["vina", "--version"]) if shutil.which("vina") else "UNAVAILABLE",
    }
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
