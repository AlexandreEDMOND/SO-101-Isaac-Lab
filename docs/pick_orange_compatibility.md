# Audit de compatibilité PickOrange

Dernière vérification : 2 août 2026. Le verdict est **inconnu — ne pas lancer
ACT avec intention d'évaluer dans LeIsaac tant que le manifeste runtime et le
test d'action-replay ne sont pas passés.** Les fichiers publiés permettent de
valider le schéma d'entraînement, mais pas de prouver l'identité de la scène et
de la dynamique d'évaluation.

## Provenance et niveau de preuve

| Information | Valeur | Niveau |
| --- | --- | --- |
| Publication dataset | 30 juillet 2025, `fa6e0625d814352b8e6ee1c6d2482194e4da8ed3` | confirmed |
| Dataset | LeRobotDataset v2.1, 60 épisodes, 36 293 frames | confirmed |
| Tâche | `Grab orange and place into plate` | confirmed |
| Robot | `so101_follower` | confirmed |
| LeIsaac historique plausible | `5ff548dcc6b67ec5170cd539f01fb84d6bace0c0` (28 juillet 2025) | inferred |
| LeRobot historique plausible | `26cb4614c961e6da04e4b83b6178331f4150650d` | inferred |
| Stack historique plausible | Python 3.10, PyTorch 2.5.1/CUDA 11.8, Isaac Sim 4.5.0, Isaac Lab v2.1.0 | inferred |
| Commit exact de collecte | absent du dataset | unknown |

Le commit LeIsaac proposé est seulement le dernier commit public avant la mise
en ligne. Son README documente aussi la stack historique plausible et la
commande SO101 leader de collecte. Il contient précisément la tâche PickOrange,
le téléopérateur HDF5, le replay HDF5 et le convertisseur qui reproduisent le
schéma publié, mais il ne constitue pas une preuve du commit utilisé. C'est un
risque majeur de reproductibilité.

## Mapping entraînement → évaluation

| Entrée/sortie ACT | Dataset | Environnement historique plausible | État |
| --- | --- | --- | --- |
| État | `observation.state`, `float32[6]`, suffixes `.pos` | six joints follower, radians avant conversion | unité dataset inférée : coordonnées moteur LeRobot, pas radians |
| Action | `action`, `float32[6]`, même ordre | `JointPositionActionCfg`, cible absolue en radians pour leader | adaptateur inverse obligatoire vers radians |
| Front | `observation.images.front`, RGB HWC 480×640, AV1 | capteur `front`, base robot, même résolution | géométrie plausible ; randomisation historique ±5 cm/±5° |
| Poignet | `observation.images.wrist`, RGB HWC 480×640, AV1 | capteur `wrist`, pince, même résolution | géométrie plausible, pas de randomisation déclarée |

Les conversions historiques sont explicites : les actions/états Isaac Lab sont
d'abord convertis radians → degrés USD → coordonnées moteur `[-100,100]` (pince
`[0,100]`). LeRobot 0.4.1 attendra ensuite les mêmes valeurs pendant
l'entraînement et normalisera ses features de politique. En évaluation, une
sortie ACT doit faire le chemin inverse fourni par LeIsaac
`convert_lerobot_action_to_leisaac`; passer les six valeurs directement à
`env.step()` serait une incompatibilité bloquante.

### Replay runtime exécuté

Le 2 août 2026, l'épisode v3 `0` (774 frames, 25,8 s) a été rejoué sur la pile
actuelle avec une nouvelle seed `42`. Chaque action dataset à 30 Hz a été
convertie avec `convert_lerobot_action_to_leisaac`, puis maintenue deux pas dans
l'environnement à 60 Hz. La limite d'épisode de l'environnement a été étendue
uniquement pour ce replay de 25,0 s à 25,817 s afin d'éviter un reset pendant
les dernières frames ; cette adaptation est enregistrée dans le rapport.

Le résultat est une RMSE articulaire globale de `0,0438` radian (environ 2,5°),
sans reset durant la trajectoire. Cela confirme le chemin de conversion et le
maintien temporel des actions pour cet épisode ; ce n'est **pas** une preuve de
succès de tâche, puisque les poses initiales historiques des oranges et de
l'assiette ne sont pas publiées. Les artefacts locaux, hors Git, sont
`outputs/compatibility/replay/episode_000_action_replay.json` et
`outputs/compatibility/replay/episode_000_trajectory_comparison.png`.

Le prétraitement image de LeRobot 0.4.1 pour une observation Gym RGB est : HWC
`uint8` → BCHW `float32` → division par 255. Les caméras du dataset sont encodées
en AV1/yuv420p : une comparaison visuelle doit utiliser les frames décodées,
pas les octets vidéo. Les statistiques de dataset montrent des pixels ramenés à
[0,1] pour l'analyse, mais ne définissent pas à elles seules le preprocessing
d'inférence.

## Fréquences : résultat strict

La fréquence déclarée par le dataset est **30 Hz** et les timestamps publiés
sont bien `n / 30`. Les deux capteurs historiques demandent `update_period=1/30`.
Toutefois, le téléopérateur historique utilise `--step_hz=60` par défaut et
enregistre chaque `env.step`; son convertisseur parcourt toutes les frames HDF5
(sauf les cinq premières) sans sous-échantillonnage, puis fixe `fps=30` dans le
dataset LeRobot.

Il y a donc trois possibilités non départagées par le dataset : collecte lancée
avec `--step_hz 30`, collecte à 60 Hz mais timestamps déclarés à 30 Hz, ou boucle
effective différente à cause du rendu. Le `sim.dt`/rendu historiques et la
commande de collecte sont inconnus. Il est interdit de conclure que la physique,
le contrôle et le dataset étaient tous à 30 Hz.

Décision : l'évaluation doit d'abord inspecter `sim.dt`, `decimation` et la
fréquence de contrôle runtime. Elle ne peut utiliser 30 Hz qu'après une décision
explicite de maintien, répétition ou resampling des actions, validée par
action-replay. Aucun resampling implicite n'est fourni par ce dépôt.

## Diff historique plausible → LeIsaac v0.4.0 actuel

- Robot SO101 Follower, action leader absolue et deux configurations de caméra
  sont textuellement inchangés dans les fichiers comparés.
- La scène PickOrange a été refactorisée dans un template. Les positions des
  objets et de l'assiette passent de ±5 cm à ±3 cm ; la caméra frontale de
  ±5°/±5 cm à ±2,5°/±2,5 cm.
- La durée d'épisode du template actuel vaut 25 s ; l'ancien cfg valait 8 s,
  mais le script de téléopération désactivait timeout et succès automatiques.
- Le v0.4.0 ajoute un enregistreur LeRobot direct et un convertisseur v3, mais
  ces ajouts sont postérieurs au dataset v2.1 et ne prouvent rien sur sa collecte.

Ces changements empêchent de qualifier v0.4.0 de reproduction exacte. La
stratégie recommandée est donc **3 : ne pas utiliser ce dataset pour revendiquer
une évaluation simulée tant qu'un commit historique reproductible n'a pas été
validé ; collecter ensuite un dataset neuf avec manifest runtime si nécessaire.**
Il reste acceptable pour valider offline la lecture v3, ACT et les outils.

## Outils de l'audit

```bash
# Métadonnées légères uniquement : ne télécharge pas les vidéos.
uv run --with huggingface-hub python scripts/build_pick_orange_dataset_manifest.py

# Dans la pile Isaac/LeIsaac installée, sans politique.
uv run python scripts/inspect_pick_orange_environment.py --headless
uv run python scripts/check_dataset_environment_compatibility.py

# Captures multi-seeds et planche. Ajouter un repo v3 pour une frame dataset.
uv run python scripts/capture_environment_observations.py --headless
```

Le script de replay ne restaure jamais une scène exacte : ni seed, ni poses des
oranges/assiette, ni état PhysX ne sont publiés. Il force l'utilisateur à le
reconnaître et produit seulement un RMSE des joints pour une action replay :

```bash
uv run python scripts/replay_lerobot_episode.py --headless \
  --dataset-repo-id VOTRE_ORG/pick-orange-v3 --episode-index 0 \
  --action-rate-hz 30 --allow-nondeterministic-action-replay
```

Répéter ensuite sur un épisode court, médian et long choisis à partir de
`meta/episodes.jsonl`. Les vidéos et comparaison côte à côte sont volontairement
différés : sans scène initiale correspondante, elles n'établiraient pas un replay
déterministe.

Le LeIsaac historique plausible fournit bien un `replay.py` pour le HDF5 local,
ce qui aurait permis un replay de collecte plus fidèle. Ce HDF5 n'est pas publié
dans le dataset Hub : cette API ne peut donc pas être utilisée pour les 60
épisodes publiés.

## Politique de compatibilité

**Pour l'entraînement**, il faut décoder les deux vidéos, conserver les clés et
l'ordre des vecteurs, et appliquer la conversion v2.1→v3 sans modifier les
valeurs.

**Pour l'évaluation**, il faut en plus le même SO101 Follower, le mapping
explicite des sorties ACT coordonnées moteur → radians, les deux caméras avec
la géométrie/preprocessing validés, et une décision temporelle explicite.

**Pour une reproductibilité exacte**, il faudrait en plus le commit LeIsaac,
les versions Isaac, les USD hashés, les paramètres PhysX, les randomisations,
la seed et les états initiaux par épisode. Le dataset publié ne contient pas ces
éléments. Un entraînement peut donc techniquement aboutir tout en laissant son
évaluation simulée invalide.
