# Protocole de collecte PickOrange — pile actuelle

Ce protocole vise une baseline déterministe et petite. Il ne concerne pas la
future tâche de tri de briques.

## Avant de collecter

1. Installer la pile de [`runtime_setup.md`](runtime_setup.md), télécharger les
   assets LeIsaac hors Git, puis générer le manifeste de l’environnement.
2. Vérifier que son `environment_fingerprint` est celui calculé depuis
   `configs/pick_orange_current_stack.yaml`.
3. Garder les randomisations listées comme désactivées dans ce contrat. Ne pas
   modifier caméra, action, fréquence ou asset en cours de collecte.

## Téléopération et acceptation

Lancer d’abord la téléopération sans écriture :

```bash
uv run python scripts/teleoperate_pick_orange.py --teleop-device keyboard
```

`keyboard` et `gamepad` sont des périphériques officiellement déclarés par
LeIsaac. La fenêtre Isaac affiche le mapping exact fourni par
`SO101Keyboard`/`SO101Gamepad`. `R` réinitialise et rejette l’essai ; `N`
accepte un épisode réussi. Le guide LeIsaac indique également `b` pour démarrer
la téléopération, puis `r`/`n` pour réinitialiser échec/succès : suivez la
légende affichée par la version installée plutôt que de supposer une touche.

Une démonstration acceptable prend les trois oranges, les place dans l’assiette
et termine dans une posture repos. Ne validez pas une trajectoire avec collision
évidente, caméra masquée, orange tombée ou reset accidentel.

## Enregistrer

Le clavier et la manette LeIsaac émettent une commande IK relative 8D. Le
collecteur ne l’écrit jamais comme action ACT : après `env.step`, il lit la
sortie réellement résolue par Isaac Lab dans `robot.data.joint_pos_target`, la
réordonne en six cibles absolues (radians) et l’écrit comme `action`. L’état
est `robot.data.joint_pos` juste avant cette transition. Le collecteur utilise
donc uniquement le SO-101 Follower simulé, échantillonne explicitement une
action sur deux à partir de la boucle contrôle 60 Hz, et exporte seulement les
épisodes acceptés. Il produit un LeRobotDataset v3 et son
`meta/current_stack_manifest.json` (fingerprint, contrat gelé, profil). Les
vecteurs SO-101 sont encodés dans la représentation moteur LeRobot, conformément
à l’adaptateur LeIsaac ; ils ne sont pas des radians bruts.

Smoke — cinq réussites :

```bash
uv run python scripts/record_pick_orange_dataset.py --max-successes 1 \
  --repo-id VOTRE_ORGANISATION/pick-orange-current-stack-smoke-v3
```

Baseline — 30 à 50 réussites (le profil fixe 40) :

```bash
uv run python scripts/record_pick_orange_dataset.py --max-successes 40 \
  --repo-id VOTRE_ORGANISATION/pick-orange-current-stack-v3
```

Changez légèrement les poses initiales prévues par les seeds 0–4 et les
trajectoires de préhension, mais pas la configuration de caméra ou les unités.
Pour une première baseline, n’enregistrez que les réussites ; les échecs sont
réinitialisés, non exportés.

## Vérifier et visualiser

```bash
uv run python scripts/validate_recorded_dataset.py \
  --repo-id VOTRE_ORGANISATION/pick-orange-current-stack-smoke-v3
uv run python scripts/visualize_recorded_episode.py \
  --repo-id VOTRE_ORGANISATION/pick-orange-current-stack-smoke-v3 --episode-index 0
```

Le validateur bloque l’entraînement si le format n’est pas v3, si les deux
vidéos sont absentes, si les vecteurs ne sont pas `[T, 6]` finis, si le FPS
n’est pas 30 ou si le fingerprint diverge. La visualisation exporte des images
RGB front/poignet et les traces état/action ; comparez-les plus tard aux images
de l’évaluation. Les datasets, vidéos et sorties restent ignorés par Git.

## Métadonnées

Le recorder officiel assure les champs de trame `observation.state`, `action`,
`timestamp`, `episode_index` et les deux flux caméra. La version, le commit,
la seed et le fingerprint sont conservés au niveau de campagne dans le sidecar
projet. Le recorder v0.4.0 inspecté n’expose pas encore un champ libre par
épisode pour enrichir le parquet : cette limite est documentée, non contournée
par une écriture artisanale incompatible.
