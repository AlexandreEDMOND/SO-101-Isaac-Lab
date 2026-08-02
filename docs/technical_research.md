# Recherche technique

Dernière vérification : 2 août 2026. Cette note sépare les compatibilités documentées des hypothèses à valider localement. Aucun asset ni dépendance lourde n'est téléchargé par ce dépôt.

## Décision de versions

| Composant | Version retenue pour le prototype SO-101 | Motif |
| --- | --- | --- |
| Python | 3.11 | La matrice publiée par LeIsaac associe Isaac Sim 5.1 à Python 3.11. |
| Isaac Sim | 5.1.0 | Dernière version explicitement couverte par la matrice LeIsaac examinée. |
| Isaac Lab | v2.3.0 (ou correctif 2.3.x validé ensemble) | Version associée à Isaac Sim 5.1 par LeIsaac et compatible selon la matrice Isaac Lab. |
| PyTorch / CUDA | 2.7.0 / CUDA 12.8 | Versions de la procédure LeIsaac pour ce couple. |
| LeIsaac | v0.4.x, commit/tag épinglé avant usage | Fournit le SO101 Follower, les parcours de téléopération et l'enregistreur LeRobot. |
| LeRobot | 0.4.1 pour le premier essai intégré | Version utilisée dans le guide EnvHub LeIsaac ; `LeRobotDataset` v3 est annoncé à partir de 0.4.0. |

NVIDIA recommande désormais Isaac Sim 6.0.1 avec Isaac Lab 3.0 beta (Python 3.12) pour une nouvelle installation générale. Ce n'est **pas** la pile choisie ici : la documentation officielle de LeIsaac consultée ne publie une matrice SO-101 que pour Isaac Sim 4.5, 5.0 et 5.1, avec Isaac Lab jusqu'à v2.3.0. Passer à 6.x/3.x avant une validation LeIsaac serait un risque de migration, pas une mise à jour transparente.

## Sources primaires et état de maturité

- [Matrice de dépendances Isaac Lab](https://github.com/isaac-sim/IsaacLab#isaac-sim-version-dependency) : `v2.3.x` accepte Isaac Sim 4.5, 5.0 et 5.1 ; la branche 3.0 vise Isaac Sim 6.0.x.
- [Installation Isaac Lab 3.0 beta](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/setup/installation/index.html) : Isaac Sim 6.x requiert Python 3.12, 16 Go de VRAM au minimum et NVIDIA recommande l'installation pip/uv pour la pile récente.
- [Exigences Isaac Sim 5.1](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html) : Ubuntu 22.04/24.04, 32 Go de RAM et 16 Go de VRAM sont les minimums indiqués ; la vérification de compatibilité NVIDIA doit être exécutée sur la machine cible.
- [Générateur de projet externe Isaac Lab](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/overview/own-project/template.html) : le générateur `isaaclab.sh --new` et un projet **External** sont la méthode recommandée. L'ancien dépôt `IsaacLabExtensionTemplate` est archivé ; il ne doit pas servir de base neuve.
- [Installation et matrice LeIsaac](https://lightwheelai.github.io/leisaac/docs/getting_started/installation/) : confirme le couple 5.1.0 / v2.3.0 / Python 3.11 et documente l'asset `so101_follower.usd`.
- [LeIsaac × LeRobot EnvHub](https://huggingface.co/docs/lerobot/main/envhub_leisaac) : intégration officiellement documentée par LeRobot, avec des tâches SO-101 et une installation de référence LeRobot 0.4.1.
- [Format LeRobotDataset v3](https://huggingface.co/docs/lerobot/main/lerobot-dataset-v3) : Parquet pour les signaux temporels, MP4 pour les caméras, métadonnées relationnelles ; v3 est disponible dans `lerobot >= 0.4.0`.
- [ACT dans LeRobot](https://huggingface.co/docs/lerobot/v0.4.3/act) : ACT fait partie de l'installation de base LeRobot et consomme images RGB et positions articulaires pour produire des séquences d'actions.
- [Portage LeRobotDataset v2.1 vers v3](https://huggingface.co/docs/lerobot/porting_datasets_v3) : LeRobot publie un convertisseur v2.1→v3. La vérification du code source du tag 0.4.1 confirme qu'il refuse de charger v2.1 directement.
- [Périphériques Isaac Lab](https://isaac-sim.github.io/IsaacLab/v2.0.1/source/api/lab/isaaclab.devices.html) : `Se3Keyboard` et `Se3Gamepad` produisent une pose delta 6D plus une commande de pince.

## Architecture retenue

Le dépôt reste un paquet Python léger et indépendant du simulateur tant que la phase 0 n'est pas validée. La commande officielle à utiliser après installation d'Isaac Lab pour créer une extension est `isaaclab.sh --new`, type **External**. À ce stade, la structure volontairement réduite est :

```text
configs/                         # paramètres reproductibles, ajoutés avec la scène
docs/                            # décisions et roadmap
scripts/check_installation.py    # diagnostic sans dépendance Isaac
src/so101_sorting/               # paquet sans import Isaac Sim
├── __init__.py
└── assets/README.md             # politique d'assets
tests/                           # tests sans simulateur
```

Les modules `environments`, `teleoperation`, `datasets`, `policies` et `evaluation` seront ajoutés dans leurs phases respectives, avec le template externe généré comme référence de convention. Les ajouter maintenant ne créerait que des coquilles sans API validée.

## SO-101 : stratégie d'asset et de contrôle

Le modèle retenu est le **SO101 Follower** de LeIsaac, et non un URDF/USD reconstruit dans ce dépôt. LeIsaac publie l'asset attendu sous `assets/robots/so101_follower.usd` et des tâches SO-101 déjà exposées dans LeRobot EnvHub. Avant de l'utiliser :

1. épingler le tag ou commit LeIsaac inspecté ;
2. télécharger l'asset explicitement, hors Git et dans un répertoire ignoré ;
3. vérifier la licence, la chaîne de dépendances USD et les noms d'articulations ;
4. charger l'asset dans une scène vide, vérifier les limites, la pince et la stabilité ;
5. seulement ensuite créer la scène table/cube du projet.

Le premier moyen de téléopération sera le clavier (`Se3Keyboard`) ou une manette (`Se3Gamepad`) d'Isaac Lab, avec contrôle d'effecteur via IK à intégrer et valider. Le clavier a des mappages documentés (W/S, A/D, Q/E pour la translation, touches dédiées pour rotation et `K` pour la pince). Une fois la scène stable, le bras leader SO-101 de LeRobot/LeIsaac pourra servir à collecter des démonstrations articulaires. LeIsaac documente aussi `keyboard`, `gamepad` et `so101leader` comme options de son outil de téléopération.

## Données et intégration LeRobot

La cible est un `LeRobotDataset` v3 local, sans publication automatique. Par épisode réussi, l'enregistreur devra produire des observations synchronisées (positions articulaires, caméra RGB et éventuels états de tâche), les actions appliquées, l'index d'épisode et la description de tâche. Le format v3 stocke les vecteurs dans `data/*.parquet`, les flux vidéo dans `videos/` et les métadonnées dans `meta/` ; l'API `LeRobotDataset` doit créer ces fichiers plutôt qu'un export artisanal.

Deux voies existent et sont confirmées :

1. **Voie initiale préférée :** utiliser l'enregistreur LeRobot direct de LeIsaac, qui tamponne les frames et n'écrit que les épisodes réussis. Elle réduit le code de conversion à maintenir, mais peut légèrement ralentir la téléopération.
2. **Repli :** enregistrer le HDF5 de téléopération LeIsaac, relire puis convertir avec son outil vers LeRobotDataset. Cette voie isole mieux la collecte de la sérialisation, au prix d'une étape supplémentaire.

EnvHub est une intégration confirmée par la documentation LeRobot, mais il n'est pas requis pour le premier prototype. Si le projet est publié via EnvHub plus tard, il devra fournir un `make_env(...)` Gym vectorisé et une révision épinglée ; `trust_remote_code=True` ne sera jamais activé pour un dépôt non audité.

La baseline existante PickOrange est documentée séparément dans
[`pick_orange_baseline.md`](pick_orange_baseline.md). Son dataset publié est en
v2.1 et impose une conversion officielle vers v3 avant usage avec notre pin
LeRobot 0.4.1. L'évaluation via le client de service LeIsaac reste à valider :
le code LeIsaac 0.4.0 examiné cible le protocole LeRobot 0.3.3.

L'audit de compatibilité de cette baseline est consigné dans
[`pick_orange_compatibility.md`](pick_orange_compatibility.md). Il établit le
schéma dataset, mais pas la provenance exacte de la scène : le commit historique
et les versions Isaac ne sont pas enregistrés dans le dataset.

## Baseline PickOrange de la pile actuelle

La référence d'évaluation devient la configuration locale
[`pick_orange_current_stack.yaml`](../configs/pick_orange_current_stack.yaml),
pas le dataset public historique. Elle fige LeIsaac v0.4.0 au commit
`1651c321e9b0c1bb54233211fc7b3cd70d8373d5`, le SO101 Follower, les deux
caméras `front`/`wrist`, l'action absolue six articulations et les fréquences
source-dérivées : physique/contrôle/rendu 60 Hz et caméra/enregistrement 30 Hz.

Le contrat est hashé de manière canonique (`sha256`) par
`so101_sorting.current_stack.environment_fingerprint`. Le manifeste runtime,
le sidecar de dataset et le manifeste d'entraînement doivent contenir ce hash.
Un changement de caméra, fréquence, ordre d'articulation, randomisation ou
version change le hash et bloque l'évaluation normale. La procédure exacte est
dans [`runtime_setup.md`](runtime_setup.md).

Le recorder direct ajouté dans LeIsaac v0.4.0 est confirmé par ses notes de
release et par la documentation de téléopération ; il permet d'écrire le format
LeRobot v3 sans conversion HDF5 supplémentaire. En revanche, le client de
policy de LeIsaac 0.4.0 inspecté cible encore le protocole de service LeRobot
0.3.3. Un adaptateur ACT 0.4.1 direct n'est donc pas revendiqué avant un test
sur la pile installée.

## Incertitudes et risques

| Sujet | Risque | Réduction prévue |
| --- | --- | --- |
| Isaac Lab 3 / Isaac Sim 6 | API et ABI Python incompatibles avec le chemin LeIsaac validé. | Rester sur 5.1 / 2.3.x jusqu'à un test de migration isolé. |
| RTX 3090 | 24 Go de VRAM sont suffisants sur le papier, mais le GPU n'est pas le modèle de référence actuel NVIDIA. | Compatibility Checker, pilote production récent et essai d'une scène/caméra minimale. |
| Asset SO-101 | Licence, chemins USD, gains d'actionneur et géométrie de collision doivent être contrôlés. | Pas de copie dans Git ; inspection manuelle du release et test de phase 1. |
| Téléopération clavier/manette | L'IK, les limites d'articulations et le taux interactif restent à adapter au SO-101. | Commencer à un seul environnement, sans enregistrement ni caméra lourde. |
| LeRobot v3 | L'enregistreur direct et la version LeRobot doivent être testés ensemble. | Valider écriture, relecture et schéma d'un épisode avant ACT. |
| Dépendances Python | LeIsaac documente `numpy==1.26.0` avec son option LeRobot. | Environnement dédié, versions verrouillées après phase 0, pas d'installation implicite via `pyproject.toml`. |

## Décisions explicites

- Pas d'installation, d'asset, de modèle ou de dataset téléchargé automatiquement.
- Pas de faux modèle SO-101 : une absence d'asset est un échec explicite de la phase 1.
- Pas d'environnement de tri ni d'entraînement dans cette première étape.
- Aucun checkpoint, dataset, vidéo, cache Isaac Sim ou asset à licence non vérifiée ne sera versionné.
