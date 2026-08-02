#!/usr/bin/env python3
"""Locate a downloaded LeIsaac SO-101 follower asset without loading Isaac Sim.

This is a preflight check only: it does not claim that the USD has valid physics
or actuators. Those checks belong to roadmap phase 1.
"""

from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path

ASSET_NAME = "so101_follower.usd"


def find_assets(root: Path) -> list[Path]:
    """Return all matching assets under an explicitly supplied local root."""

    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob(ASSET_NAME) if path.is_file())


def parse_args() -> argparse.Namespace:
    """Parse the optional directory containing an inspected LeIsaac checkout."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root",
        type=Path,
        help=(
            "Répertoire LeIsaac/assets à rechercher. Aucune recherche large n'est effectuée "
            "sans cette option."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Print a deliberate asset-inspection result."""

    args = parse_args()
    if args.asset_root is None:
        try:
            version = metadata.version("leisaac")
        except metadata.PackageNotFoundError:
            print("[ECHEC] LeIsaac n'est pas installé et aucun --asset-root n'a été fourni.")
            print(
                "        Installer/inspecter un release LeIsaac, puis passer son répertoire assets."
            )
            return 1
        print(f"[INFO] LeIsaac {version} est installé.")
        print("[INFO] Fournir --asset-root pour vérifier explicitement l'asset téléchargé.")
        return 0

    matches = find_assets(args.asset_root)
    if not matches:
        print(f"[ECHEC] {ASSET_NAME} introuvable sous {args.asset_root}")
        print("        Vérifier le release LeIsaac, le téléchargement et la licence de l'asset.")
        return 1

    for asset in matches:
        print(f"[OK] Asset trouvé : {asset}")
    print("[INFO] Étape suivante : charger l'asset dans Isaac Lab et vérifier ses articulations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
