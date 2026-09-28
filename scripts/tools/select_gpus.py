#!/usr/bin/env python3
import csv
import subprocess
import sys


def main() -> int:
    mem_threshold = int(sys.argv[1]) if len(sys.argv) > 1 else 18000
    util_threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True)
    except Exception:
        return 1

    selected = []
    for row in csv.reader(output.splitlines()):
        if len(row) != 3:
            continue
        idx = row[0].strip()
        mem_free = int(row[1].strip())
        util = int(row[2].strip())
        if mem_free >= mem_threshold and util <= util_threshold:
            selected.append(idx)

    print(",".join(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
