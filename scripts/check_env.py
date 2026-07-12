#!/usr/bin/env python
"""Environment sanity check (TASKS.md T-02).

Prints torch version and accelerator availability, and confirms the selected
device. Exits 0 on a clean import + device probe.

    python scripts/check_env.py
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    import torch  # noqa: F401  (import is part of the check)

    from falsetto import __version__
    from falsetto.utils.device import device_report, select_device

    report = device_report()
    report["falsetto"] = __version__
    print(json.dumps(report, indent=2))

    device = select_device("auto")
    print(f"\nSelected device: {device}")
    # Tiny op to confirm the device actually works, not just that it's reported.
    x = torch.randn(4, 4, device=device)
    y = (x @ x.T).sum().item()
    print(f"Matmul smoke test on {device}: sum={y:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
