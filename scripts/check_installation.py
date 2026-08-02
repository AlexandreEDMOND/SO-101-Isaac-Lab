#!/usr/bin/env python3
"""Report whether the pinned SO-101 Isaac Lab development stack is available.

The script has no Isaac Lab dependency itself: failed imports are reported with
actionable next steps instead of an uncaught import error.
"""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass
from types import ModuleType

TARGET_PYTHON = (3, 11)


@dataclass(frozen=True)
class CheckResult:
    """Result of one diagnostic check."""

    name: str
    ok: bool
    detail: str
    next_step: str | None = None


def short_exception(error: BaseException) -> str:
    """Keep optional-dependency errors readable on one line."""

    message = " ".join(str(error).split())
    return f"{type(error).__name__}: {message}"[:300]


def import_module(module_name: str, display_name: str, next_step: str) -> CheckResult:
    """Import a module while preserving a concise remediation message."""

    try:
        module: ModuleType = importlib.import_module(module_name)
    except Exception as error:  # Optional native dependencies may raise varied errors.
        return CheckResult(display_name, False, short_exception(error), next_step)

    version = getattr(module, "__version__", "version inconnue")
    return CheckResult(display_name, True, f"import réussi ({version})")


def check_python() -> CheckResult:
    """Check the Python ABI required by the pinned Isaac Sim 5.1 route."""

    current = sys.version_info[:2]
    detail = f"{platform.python_version()} ({platform.python_implementation()})"
    if current == TARGET_PYTHON:
        return CheckResult("Python", True, detail)
    return CheckResult(
        "Python",
        False,
        f"{detail}; la pile SO-101 retenue requiert Python {TARGET_PYTHON[0]}.{TARGET_PYTHON[1]}",
        "Créer un environnement Python 3.11 avant d'installer Isaac Sim 5.1 et Isaac Lab 2.3.x.",
    )


def check_torch() -> tuple[CheckResult, CheckResult]:
    """Check PyTorch and the CUDA runtime visible from it."""

    try:
        torch = importlib.import_module("torch")
    except Exception as error:
        message = short_exception(error)
        return (
            CheckResult(
                "PyTorch",
                False,
                message,
                "Installer la roue PyTorch CUDA compatible avec la matrice de versions retenue.",
            ),
            CheckResult(
                "CUDA",
                False,
                "PyTorch indisponible",
                "Installer puis re-lancer le diagnostic.",
            ),
        )

    torch_version = getattr(torch, "__version__", "version inconnue")
    torch_result = CheckResult("PyTorch", True, torch_version)
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as error:
        return torch_result, CheckResult(
            "CUDA", False, short_exception(error), "Vérifier le pilote NVIDIA."
        )

    if not cuda_available:
        return (
            torch_result,
            CheckResult(
                "CUDA",
                False,
                "torch.cuda.is_available() retourne False",
                "Vérifier le pilote NVIDIA, le GPU visible dans le conteneur et la roue PyTorch "
                "CUDA.",
            ),
        )

    try:
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        detail = f"{gpu_name} (compute capability {capability[0]}.{capability[1]})"
    except Exception as error:
        detail = f"CUDA disponible, mais GPU non lisible : {short_exception(error)}"
    return torch_result, CheckResult("CUDA", True, detail)


def print_result(result: CheckResult) -> None:
    """Print one result and its specific recovery action."""

    status = "OK" if result.ok else "ECHEC"
    print(f"[{status}] {result.name}: {result.detail}")
    if result.next_step:
        print(f"        Prochaine action : {result.next_step}")


def main() -> int:
    """Run all checks and return a shell-friendly status code."""

    print("Diagnostic SO-101 Isaac Lab")
    print("Profil vérifié : Python 3.11 + Isaac Sim 5.1.0 + Isaac Lab 2.3.x.")
    print()

    results = [check_python()]
    torch_result, cuda_result = check_torch()
    results.extend([torch_result, cuda_result])
    results.append(
        import_module(
            "isaacsim",
            "Isaac Sim",
            "Installer Isaac Sim 5.1.0 depuis l'index NVIDIA, puis suivre "
            "docs/technical_research.md.",
        )
    )
    results.append(
        import_module(
            "isaaclab",
            "Isaac Lab",
            "Installer Isaac Lab v2.3.x correspondant à Isaac Sim 5.1.0.",
        )
    )

    for result in results:
        print_result(result)

    failures = [result.name for result in results if not result.ok]
    if failures:
        print(f"\nPile incomplète : {', '.join(failures)}.")
        return 1

    print("\nPile de base disponible. Lancer ensuite la validation de la phase 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
