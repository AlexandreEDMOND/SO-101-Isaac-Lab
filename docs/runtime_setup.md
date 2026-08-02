# Runtime PickOrange — pile actuelle

Cette procédure prépare volontairement un environnement Isaac séparé du petit
environnement de développement de ce dépôt. `uv sync --extra dev` installe
seulement Ruff, pytest et PyYAML : il **n’installe pas** Isaac Sim.

La combinaison figée est Python 3.11, Isaac Sim 5.1.0, Isaac Lab v2.3.0,
LeIsaac v0.4.0 (`1651c321e9b0c1bb54233211fc7b3cd70d8373d5`) et LeRobot 0.4.1.
Elle est publiée par la [matrice LeIsaac](https://lightwheelai.github.io/leisaac/docs/getting_started/installation/)
et Isaac Lab 2.3 est explicitement construit sur Isaac Sim 5.1 dans ses
[notes de version](https://isaac-sim.github.io/IsaacLab/main/source/refs/release_notes.html).

## Installation sur la RTX 3090

Sur Ubuntu avec un pilote NVIDIA compatible, depuis le répertoire parent du
dépôt :

```bash
uv venv --python 3.11 .venv-pick-orange
source .venv-pick-orange/bin/activate
uv pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
git clone --branch v2.3.0 https://github.com/isaac-sim/IsaacLab.git third_party/IsaacLab
cd third_party/IsaacLab && ./isaaclab.sh --install && cd ../..
git clone --branch v0.4.0 --recursive https://github.com/LightwheelAI/leisaac.git third_party/leisaac
uv pip install -e "third_party/leisaac/source/leisaac[lerobot]"
uv pip install lerobot==0.4.1 numpy==1.26.0
uv pip install "transformers>=4.53,<5"
uv pip install -e ".[dev]"
```

Ces commandes reprennent les paquets et versions prescrits par les guides
officiels : l’[installation pip Isaac Lab 2.3](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html)
recommande aussi explicitement `uv venv --python 3.11` et l’installation
`isaacsim[all,extscache]==5.1.0`. Ne clonez pas les répertoires `third_party/`
dans Git de ce projet ; ils restent locaux.

Avant d’installer, vérifiez `ldd --version` (le paquet pip Isaac Sim requiert
GLIBC 2.35+) et acceptez l’EULA NVIDIA lors du premier lancement. Téléchargez
ensuite les assets PickOrange/`so101_follower.usd` depuis la release LeIsaac
dans `third_party/leisaac/assets/`, conformément à leur
[guide d’assets](https://lightwheelai.github.io/leisaac/docs/getting_started/installation/).

Avant toute commande LeIsaac, pointez explicitement ses assets locaux :

```bash
export LEISAAC_ASSETS_ROOT="$PWD/third_party/leisaac/assets"
export OMNI_KIT_ACCEPT_EULA=YES
```

## Vérification et gel de l’environnement

Dans l’environnement activé, à la racine de ce dépôt :

```bash
uv run python scripts/check_installation.py
uv run python scripts/inspect_pick_orange_environment.py --headless \
  --frozen-config configs/pick_orange_current_stack.yaml
uv run python scripts/capture_environment_observations.py --headless
```

Le second script écrit `outputs/compatibility/environment_manifest.json`, avec
le `environment_fingerprint`. Conservez ce manifeste avec chaque campagne ; il
est ignoré par Git. Si une valeur extraite diffère du contrat YAML, interrompez
la collecte et mettez d’abord à jour le contrat avec une preuve source.

## État runtime mesuré (2026-08-02)

L'installation Python est présente et `scripts/check_installation.py` a validé
PyTorch 2.7.0+cu128, CUDA, Isaac Sim 5.1.0, Isaac Lab (tag source `v2.3.0`,
distribution `0.47.2`), LeIsaac 0.4.0 et LeRobot 0.4.1 sur une RTX 3090.

Le pilote `580.173.02` est actif. Avec `LEISAAC_ASSETS_ROOT` configuré,
`LeIsaac-SO101-PickOrange-v0` charge réellement la scène et les deux caméras
RGB 640×480, à 60 Hz de contrôle. Le manifeste runtime est écrit dans
`outputs/compatibility/environment_manifest.json`. Les avertissements PhysX
émis par l'asset de cuisine sont conservés dans le log Isaac Sim ; ils n'ont pas
empêché le chargement ni le rendu RTX.
