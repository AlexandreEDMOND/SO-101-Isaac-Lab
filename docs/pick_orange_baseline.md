# Baseline existante : SO-101 PickOrange

Dernière vérification : 2 août 2026. Cette baseline précède volontairement la tâche de tri personnalisée : elle valide d'abord un schéma de données SO-101, ACT et une chaîne d'artefacts reproductible sur une tâche existante.

## Résultat de compatibilité

Le dataset [`LightwheelAI/leisaac-pick-orange`](https://huggingface.co/datasets/LightwheelAI/leisaac-pick-orange) est publiquement disponible (60 épisodes, 36 293 frames, 30 Hz), mais son `meta/info.json` déclare **`codebase_version: v2.1`**. Le code source de LeRobot **0.4.1** ne le charge pas directement : `LeRobotDataset` accepte son format v3 et lève explicitement une erreur de compatibilité pour v2.1 avec la commande de conversion v3 indiquée ci-dessous. Ce n'est donc pas un dataset entraînable directement avec la version du projet.

| Élément publié | Constat vérifié | Décision |
| --- | --- | --- |
| `observation.state` | `float32[6]`, ordre SO-101 publié | entrée proprioception ACT après conversion v3 |
| `action` | `float32[6]`, mêmes six noms d'articulation | sortie ACT à conserver dans le même ordre |
| Caméra frontale | `observation.images.front`, vidéo 480×640, AV1/yuv420p | entrée visuelle requise |
| Caméra poignet | `observation.images.wrist`, vidéo 480×640, AV1/yuv420p | entrée visuelle requise |
| `task_index` | présent (`int64`) | conserver lors de la conversion |

Les six noms sont `shoulder_pan.pos`, `shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos`, `gripper.pos`. Dans LeIsaac 0.4.0, le chemin SO-101 leader construit une action `JointPositionActionCfg` de six positions articulaires dans cet ordre (cinq articulations et pince) : les actions sont donc bien des **commandes de position articulaires**, pas des vitesses. Les échantillons v2 publiés ont des valeurs qui ressemblent à des degrés moteur absolus ; `info.json` ne publie toutefois pas l'unité ni la calibration. Il faut laisser le convertisseur préserver les valeurs et valider l'unité, l'ordre et les limites sur le dataset v3 résultant avant une rollout.

## Conversion obligatoire

Dans l'environnement Python 3.11 avec **LeRobot 0.4.1**, télécharger le dataset dans un répertoire ignoré puis exécuter l'utilitaire fourni par LeRobot, sans publication automatique :

```bash
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
  --repo-id=LightwheelAI/leisaac-pick-orange \
  --root=/chemin/ignore/vers/cache-lerobot \
  --push-to-hub=false
```

`--root` est le dossier parent de cache : l'outil y télécharge
`LightwheelAI/leisaac-pick-orange`, garde l'original sous le suffixe `_old` et
remplace le dossier du dataset par le v3. Employer donc un cache neuf et ignoré
par Git ; ne jamais viser une copie irremplaçable. Pour l'entraînement, publier
explicitement cette conversion dans un dépôt Hub privé/public sous votre
contrôle, ou employer son dossier local v3 avec `--dataset-root`. Ne pas
écraser le dataset source ni versionner la copie locale.

La conversion de données LeIsaac HDF5 vers v3 documentée actuellement par LeIsaac demande `lerobot==0.4.2`; elle ne rend pas automatiquement le dataset PickOrange v2.1 compatible avec notre pin 0.4.1. Pour ce dataset déjà au format LeRobot v2.1, le convertisseur officiel 0.4.1 ci-dessus est la voie retenue.

### Exécution locale vérifiée

Le 2 août 2026, la révision Hugging Face
`fa6e0625d814352b8e6ee1c6d2482194e4da8ed3` a été téléchargée localement
(environ 0,65 Go avant conversion) et convertie avec LeRobot 0.4.1, sans
publication. Le convertisseur a conservé le v2.1 dans
`datasets/lerobot_cache/LightwheelAI/leisaac-pick-orange_old` et a placé le v3
dans `datasets/lerobot_cache/LightwheelAI/leisaac-pick-orange`. Le cache fait
environ 1,4 Go pour conserver les deux versions. Ces chemins sont ignorés par
Git ; les emplacements peuvent donc être adaptés sans changer le dépôt.

## Inspection et visualisation

Avant conversion, l'inspection télécharge uniquement `meta/info.json` et `meta/episodes.jsonl`, écrit un rapport et s'arrête avec le code 2, ce qui documente correctement le blocage :

```bash
python scripts/inspect_dataset.py --repo-id LightwheelAI/leisaac-pick-orange
```

Après conversion et publication en v3, l'API officielle `LeRobotDataset` est utilisée pour les frames, les statistiques numériques et une planche de la caméra frontale/poignet :

```bash
python scripts/inspect_dataset.py \
  --repo-id VOTRE_ORGANISATION/leisaac-pick-orange-v3 \
  --episode-index 0
```

Ou, sans publication Hub, après la conversion ci-dessus :

```bash
python scripts/inspect_dataset.py \
  --repo-id LightwheelAI/leisaac-pick-orange \
  --root /chemin/ignore/vers/cache-lerobot/LightwheelAI/leisaac-pick-orange
```

Les rapports JSON et la planche PNG sont écrits dans `outputs/dataset_inspection/`. AV1 est le principal risque de décodage : si la planche échoue, vérifier la prise en charge AV1 de FFmpeg/PyAV dans l'environnement LeRobot, conserver `--video-backend pyav` et ne pas conclure que les images sont invalides sur la seule base de cet échec local.

## ACT reproductible

Les champs YAML ont été vérifiés dans `TrainPipelineConfig` et `ACTConfig` de LeRobot 0.4.1. Les deux caméras et `observation.state` sont déduits des métadonnées v3 par LeRobot : aucune clé d'entrée inventée n'est passée dans les YAML. Les configurations utilisent `pretrained_backbone_weights: null` pour ne pas déclencher un téléchargement de poids implicite.

Avant tout entraînement avec intention d'évaluer en simulation, consulter
[`pick_orange_compatibility.md`](pick_orange_compatibility.md). Le lanceur
accepte `--require-compatibility-report` et refuse alors un rapport contenant
un `FAIL` bloquant. Cette option ne transforme pas les `UNKNOWN` historiques en
preuves de compatibilité.

Smoke test (20 étapes, sauvegardes aux étapes 10 et 20) :

```bash
python scripts/train_act_pick_orange.py \
  --mode smoke \
  --dataset-repo-id VOTRE_ORGANISATION/leisaac-pick-orange-v3 \
  --run
```

Pour la copie v3 locale, ajouter
`--dataset-root /chemin/ignore/vers/cache-lerobot/LightwheelAI/leisaac-pick-orange`.

Reprise du smoke test depuis `checkpoints/last` :

```bash
python scripts/train_act_pick_orange.py \
  --mode smoke \
  --dataset-repo-id VOTRE_ORGANISATION/leisaac-pick-orange-v3 \
  --resume --run
```

Le lanceur n'exécute rien sans `--run`, vérifie CUDA, le GPU, la VRAM, `lerobot-train` et la version exacte 0.4.1. LeRobot enregistre la configuration effective dans chaque checkpoint ; le lanceur capture également son stdout localement. Weights & Biases reste opt-in avec `--wandb`.

Entraînement de référence RTX 3090 (100 000 étapes, batch 8, 4 workers) :

```bash
python scripts/train_act_pick_orange.py \
  --mode full \
  --dataset-repo-id VOTRE_ORGANISATION/leisaac-pick-orange-v3 \
  --run
```

Analyser le log produit :

```bash
python scripts/analyze_training.py \
  --training-dir outputs/training/act_pick_orange \
  --log-file outputs/training_logs/act_pick_orange_full_YYYYMMDDTHHMMSSZ.log
```

L'analyse produit `training_loss.svg`, `summary.json` et `summary.md` dans `outputs/training_analysis/`. LeRobot 0.4.1 écrit les losses de train dans son stdout mais n'écrit pas de loss de validation dans cette configuration (pas d'environnement d'évaluation configuré) : le script ne désigne donc pas un « meilleur » checkpoint fictif et sélectionne le dernier en l'indiquant.

## Évaluation dans LeIsaac : blocage confirmé

La tâche [`LeIsaac-SO101-PickOrange-v0`](https://huggingface.co/docs/lerobot/main/envhub_leisaac) existe et est documentée par LeRobot/LeIsaac. Le script d'inférence LeIsaac 0.4.0 expose bien l'option `lerobot-act` avec un chemin de checkpoint, mais son `LeRobotServicePolicyClient` indique dans son code source qu'il cible **LeRobot v0.3.3** et communique avec un serveur RPC. La compatibilité de protocole et de pré/post-traitement avec un checkpoint ACT 0.4.1 n'est pas démontrée.

Par conséquent, aucun `scripts/evaluate_pick_orange.py` n'est créé : appeler ce client avec un checkpoint 0.4.1 serait une fausse implémentation. Le prochain spike technique doit, dans l'environnement Isaac Sim 5.1 / Isaac Lab 2.3, lancer le serveur d'inférence LeRobot 0.4.1, comparer précisément les features (`front`, `wrist`, état 6D), les unités/ordre d'action et le protocole RPC, puis faire une rollout unique avant d'écrire un adaptateur. Les vidéos, succès et taux de succès resteront indisponibles tant que ce test n'aura pas réussi.

## Sources

- [Dataset PickOrange et ses fichiers de métadonnées](https://huggingface.co/datasets/LightwheelAI/leisaac-pick-orange)
- [Portage officiel LeRobot vers v3](https://huggingface.co/docs/lerobot/porting_datasets_v3)
- [ACT LeRobot 0.4.x](https://huggingface.co/docs/lerobot/v0.4.3/act)
- [Support des politiques LeIsaac](https://lightwheelai.github.io/leisaac/docs/getting_started/policy_support/)
- [Support EnvHub LeIsaac](https://lightwheelai.github.io/leisaac/docs/features/envhub_support/)
