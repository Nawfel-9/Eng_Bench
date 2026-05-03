"""Generate download commands for all GGUF models.

This script prints (or writes) ``hf download`` commands
for every model × quantisation combination defined in the registry.
It does **not** execute any downloads itself.

Usage
-----
    # Print all commands to stdout
    python scripts/download_models.py

    # Write a master PowerShell script
    python scripts/download_models.py --output scripts/download_all_models.ps1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this file directly or as a module.
try:
    from models.registry import BENCHMARK_ORDER, MODEL_REGISTRY, MODELS_ROOT, QUANT_OPTIONS
except ModuleNotFoundError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from models.registry import BENCHMARK_ORDER, MODEL_REGISTRY, MODELS_ROOT, QUANT_OPTIONS


def generate_commands() -> list[str]:
    """Return a list of ``hf download`` shell commands."""
    commands: list[str] = []

    for model_key in BENCHMARK_ORDER:
        spec = MODEL_REGISTRY[model_key]
        hf_repo = spec["hf_repo"]
        local_dir = MODELS_ROOT / spec["model_dir"]

        commands.append(f"# -- {spec['display_name']} ({model_key}) --")
        commands.append(f"# HuggingFace repo: https://huggingface.co/{hf_repo}")

        # Multimodal projector (usually f16 or bf16, download once)
        mmproj = spec["mmproj_file"]
        commands.append(
            f'hf download {hf_repo} {mmproj}'
            f' --local-dir "{local_dir}"'
        )

        # One GGUF per quantisation level
        for quant in spec.get("available_quants", QUANT_OPTIONS):
            gguf = spec["gguf_file"].format(quant=quant)
            commands.append(
                f'hf download {hf_repo} {gguf}'
                f' --local-dir "{local_dir}"'
            )

        commands.append("")  # blank separator

    return commands


def write_powershell_script(path: Path, commands: list[str]) -> None:
    """Write *commands* as a PowerShell ``.ps1`` script."""
    lines = [
        "# ============================================================",
        "# EEE_Bench - Download all GGUF models from HuggingFace",
        "# ============================================================",
        "#",
        "# Prerequisites:",
        "#   pip install huggingface-hub",
        "#",
        "# Usage:",
        "#   .\\scripts\\download_all_models.ps1",
        "#",
        "# To download only a specific quantisation, comment out the",
        "# lines you do not need.",
        "# ============================================================",
        "",
        "# Ensure the target directory exists",
        f'New-Item -ItemType Directory -Force -Path "{MODELS_ROOT}" | Out-Null',
        "",
    ]
    lines.extend(commands)

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote PowerShell download script -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate hf download commands for GGUF models."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write a .ps1 script instead of printing to stdout.",
    )
    args = parser.parse_args()

    commands = generate_commands()

    if args.output:
        write_powershell_script(Path(args.output), commands)
    else:
        print("\n".join(commands))


if __name__ == "__main__":
    main()
