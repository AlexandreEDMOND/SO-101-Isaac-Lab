# SO-101 Isaac Lab

Projet open source pour étudier l'apprentissage par imitation d'un bras SO-101 simulé dans Isaac Lab. La cible finale est le tri de briques rouges, vertes et bleues dans leurs bacs respectifs avec une politique ACT entraînée à partir de démonstrations.

> **État : Isaac Sim 5.1, Isaac Lab et le rendu RTX PickOrange sont validés sur la RTX 3090 avec le pilote NVIDIA 580.173.02.** La baseline active utilise le dataset humain public PickOrange ; ni téléopération ni collecte manuelle ne sont requises.

## Pipeline visé

```text
Isaac Lab scene → téléopération → LeRobotDataset → entraînement ACT → évaluation → analyse
```

## Objectifs

- Construire une scène Isaac Lab reproductible pour le SO-101 et des objets rigides.
- Collecter des démonstrations simulées compatibles avec LeRobot.
- Entraîner puis évaluer ACT sur des positions de briques inédites.
- Publier un projet lisible, sans assets, datasets ni checkpoints propriétaires ou volumineux.

## Périmètre de la première version

La première version couvre une seule brique et une seule cible de dépôt. Le tri multi-briques, les instructions en langage naturel, le PPO, les VLA, les world models, le sim-to-real avancé, les objets déformables et l'entraînement d'un modèle de vision depuis zéro restent hors périmètre.

## Prérequis prévus

La configuration de référence est Ubuntu Linux, une NVIDIA RTX 3090 (24 Go VRAM), au moins 32 Go de RAM et suffisamment d'espace SSD pour Isaac Sim et les jeux de données locaux. La RTX 3090 satisfait le seuil de VRAM de 16 Go d'Isaac Sim, mais elle n'est pas le GPU de référence actuel de NVIDIA : valider la machine avec le Compatibility Checker avant installation.

La combinaison retenue pour l'exploration SO-101 est actuellement Python 3.11, Isaac Sim 5.1.0, Isaac Lab 2.3.x et LeIsaac 0.4.x. Les raisons, limites et sources sont détaillées dans la [recherche technique](docs/technical_research.md).

## Quick start : diagnostic uniquement

Ce dépôt ne télécharge ni Isaac Sim ni aucun asset automatiquement.

Le dépôt utilise [uv](https://docs.astral.sh/uv/) pour l'environnement et les
outils Python légers ; `.python-version` fixe Python 3.11 et `uv.lock` verrouille
les dépendances de développement. La stack Isaac/LeRobot reste à installer
explicitement dans l'environnement Python 3.11 retenu.

```bash
uv sync --extra dev
uv run python scripts/check_installation.py
```

Le diagnostic est volontairement exécutable avant installation complète ; il explique les dépendances manquantes et retourne un code non nul tant que la stack cible n'est pas prête.

## Existing-task baseline

Avant de construire la tâche de tri, le dépôt prépare une baseline sur
`LightwheelAI/leisaac-pick-orange`, l'environnement existant
`LeIsaac-SO101-PickOrange-v0` et ACT. Elle permet de valider le format SO-101,
les artefacts d'entraînement et le chemin d'évaluation sans confondre ces
risques avec notre future scène.

Le dataset publié est actuellement en LeRobotDataset v2.1, alors que le pin
LeRobot 0.4.1 de ce projet attend v3. Il doit donc être converti avant la
visualisation de frames ou ACT. L'inspection suivante documente le schéma et ce
blocage sans télécharger le dataset complet :

```bash
uv run python scripts/inspect_dataset.py --repo-id LightwheelAI/leisaac-pick-orange
```

Après conversion explicite en v3 dans un dépôt sous votre contrôle :

```bash
uv run python scripts/inspect_dataset.py --repo-id VOTRE_ORGANISATION/leisaac-pick-orange-v3
uv run python scripts/train_act_pick_orange.py --mode smoke \
  --dataset-repo-id VOTRE_ORGANISATION/leisaac-pick-orange-v3 --run
uv run python scripts/analyze_training.py --training-dir outputs/training/act_pick_orange_smoke \
  --log-file outputs/training_logs/act_pick_orange_smoke_YYYYMMDDTHHMMSSZ.log
```

L'entraînement RTX 3090 est préparé avec `--mode full`; il n'est jamais lancé
automatiquement. Le chargement direct d'un checkpoint ACT 0.4.1 dans LeIsaac
n'est pas encore validé, donc aucune évaluation simulée n'est revendiquée.
Consulter la [note de baseline PickOrange](docs/pick_orange_baseline.md) pour
la conversion, la reprise, l'analyse et le blocage d'évaluation précis.
L'[audit de compatibilité](docs/pick_orange_compatibility.md) est un préalable
à toute interprétation d'une évaluation simulée : le dataset public ne conserve
ni son commit LeIsaac exact ni ses états initiaux de scène.

## Baseline PickOrange avec dataset public

La baseline active part du dataset humain
[`LightwheelAI/leisaac-pick-orange`](https://huggingface.co/datasets/LightwheelAI/leisaac-pick-orange).
La copie doit être inspectée puis convertie de v2.1 vers LeRobotDataset v3 ;
ensuite seulement, nous reproduirons son environnement, rejouerons quelques
actions et entraînerons ACT. La [roadmap](docs/roadmap.md) donne les sept étapes
et la [note de baseline](docs/pick_orange_baseline.md) contient les commandes.

Les scripts de téléopération et de collecte locale sont conservés pour une
future tâche personnalisée, mais ne font pas partie du chemin actif.

## Documentation

- [Roadmap](docs/roadmap.md) : périmètre et critères d'acceptation par phase.
- [Recherche technique](docs/technical_research.md) : choix de versions, compatibilités et sources vérifiées.
- [Baseline PickOrange](docs/pick_orange_baseline.md) : compatibilité v2.1/v3, ACT et évaluation LeIsaac.
- [Audit PickOrange](docs/pick_orange_compatibility.md) : provenance, mapping et garde-fous d'évaluation.
- [Runtime PickOrange](docs/runtime_setup.md) : installation Isaac explicite et validation de la pile actuelle.
- [Collecte PickOrange](docs/pick_orange_collection_protocol.md) : protocole v3, validation et visualisation.

## Développement

Les outils locaux de qualité sont définis dans `pyproject.toml` : Ruff (lint/format), pytest et annotations de type. Les datasets, vidéos, caches Isaac Sim, checkpoints et modèles volumineux sont ignorés par Git.

Ce projet est encore en développement : les commandes de simulation, d'enregistrement et d'entraînement seront ajoutées et documentées seulement après validation de chaque phase de la roadmap.
