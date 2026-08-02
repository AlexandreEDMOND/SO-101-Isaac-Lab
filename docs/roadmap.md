# Roadmap

Cette roadmap est séquentielle : une phase ne démarre que lorsque ses critères d'acceptation sont satisfaits. Les éléments hors périmètre sont aussi des garde-fous contre une extension prématurée du projet.

## Priorité actuelle — Dataset public PickOrange en 7 étapes

Cette baseline utilise [`LightwheelAI/leisaac-pick-orange`](https://huggingface.co/datasets/LightwheelAI/leisaac-pick-orange). Elle ne requiert ni SO-101 Leader physique ni téléopération. Le dataset brut reste hors Git et toute conversion travaille sur une copie.

1. [x] **Télécharger et figer le dataset brut.** Révision `fa6e0625d814352b8e6ee1c6d2482194e4da8ed3` téléchargée hors Git. Hors périmètre : modifier la source.
2. [x] **Inspecter schéma et épisodes.** Deux caméras, état/action 6D, 30 Hz, codec, NaN et bornes ont été vérifiés ; l'épisode 0 a été visualisé. Les unités restent explicitement inconnues.
3. [x] **Convertir une copie vers LeRobotDataset v3.** Le convertisseur officiel v2.1→v3 de LeRobot 0.4.1 a produit une copie locale, qui est chargée et inspectée avec l'API LeRobot. Hors périmètre : réécrire manuellement les actions ou publier automatiquement sur Hugging Face.
4. [x] **Reproduire l'environnement historique au plus près.** La pile actuelle a été inspectée en runtime : SO-101, ordre articulaire, deux caméras, résolutions et fréquence sont confirmés ; le rapport ne contient aucun échec bloquant. Le commit et l'état initial historiques restent inconnus : ce n'est pas une reproduction exacte.
5. [ ] **Valider par replay et comparaison visuelle.** Rejouer des actions converties dans Isaac Sim et comparer états, fréquence et images. Le replay reste non déterministe si l'état initial de scène manque. Hors périmètre : contourner un mismatch par reshape ou normalisation arbitraire.
6. [ ] **Entraîner ACT sur le dataset validé.** Faire d'abord un smoke test (forward, backward, checkpoint, reprise), puis un entraînement RTX 3090 reproductible avec losses locales. Hors périmètre : entraînement long automatique ou autres politiques.
7. [ ] **Évaluer ACT dans la simulation correspondante.** Exécuter plusieurs épisodes avec le même schéma d'observation/action, produire vidéos, trajectoires, succès et causes d'échec. Hors périmètre : revendiquer une évaluation exacte tant que scène/commit/états initiaux historiques restent inconnus.

Les scripts de téléopération restent disponibles pour une future tâche personnalisée, mais ne font pas partie de cette baseline active.

## Phase 0 — Installation et validation de la stack

**Résultat attendu.** Un environnement Python 3.11 reproductible exécute Isaac Sim 5.1.0, Isaac Lab 2.3.x et PyTorch CUDA sur la machine cible.

**Critères d'acceptation.** `scripts/check_installation.py` réussit ; la version GPU/pilote est consignée ; un exemple Isaac Lab officiel démarre sans erreur d'import.

**Dépendances.** Ubuntu, pilote NVIDIA compatible, RTX 3090 ou GPU équivalent, espace disque et accès à l'index NVIDIA.

**Hors périmètre.** Scène métier, asset SO-101, téléopération et entraînement.

## Phase 0B — Baseline PickOrange existante

**Résultat attendu.** Le dataset SO-101 PickOrange existant est converti en
LeRobotDataset v3, inspecté par l'API LeRobot, puis utilisé pour un smoke test
ACT reproductible avec checkpoint et analyse locale.

**Critères d'acceptation.** Le schéma deux caméras/état/actions est validé ;
un épisode est visualisé ; un entraînement de 20 étapes effectue forward et
backward, sauvegarde puis reprend un checkpoint ; les losses sont analysées.
Le chemin exact vers une rollout LeIsaac est validé, ou son incompatibilité est
documentée avec une prochaine action vérifiable.

**Dépendances.** Phase 0, LeRobot 0.4.1, conversion officielle v2.1→v3,
FFmpeg/PyAV avec AV1 et RTX 3090 pour l'entraînement réel.

**Hors périmètre.** Modification de PickOrange, création de la tâche de tri,
entraînement long automatique, publication automatique de dataset/checkpoint et
adaptateur d'évaluation non validé.

## Phase 0C — Baseline PickOrange sur la pile actuelle

**Résultat attendu.** La même configuration LeIsaac 0.4.0/Isaac Lab 2.3 est
inspectée, figée et fingerprintée ; cinq démonstrations SO-101 acceptées sont
écrites directement en LeRobotDataset v3 puis utilisées pour un smoke ACT.

**Critères d'acceptation.** Le manifeste runtime correspond au contrat ; les
deux flux caméra et les vecteurs `[T,6]` passent la validation ; le fingerprint
est présent dans dataset/entraînement/évaluation ; le smoke ACT sauvegarde et
relit un checkpoint ; une évaluation ne démarre pas avec un fingerprint
différent. La rollout ACT elle-même n'est considérée validée qu'après un
adaptateur LeIsaac 0.4.0 ↔ LeRobot 0.4.1 testé.

**Dépendances.** Phase 0, assets LeIsaac téléchargés explicitement, téléopération
officielle clavier ou manette, LeRobot 0.4.1 et RTX 3090.

**Hors périmètre.** Dataset historique comme référence simulée, entraînement
long automatique, modification du code LeIsaac, randomisation agressive,
création de la tâche de tri ou métrique d'évaluation non vérifiée.

## Phase 1 — Charger et articuler le SO-101

**Résultat attendu.** Le SO101 Follower issu d'une version LeIsaac épinglée est chargé dans une scène vide et ses articulations/pince répondent à des commandes sûres.

**Critères d'acceptation.** Asset et licence revus ; noms et limites des articulations consignés ; posture initiale stable ; ouverture/fermeture de pince et mouvement limité observables.

**Dépendances.** Phase 0, asset LeIsaac téléchargé explicitement, configuration d'actionneurs validée.

**Hors périmètre.** Import/reconstruction d'un faux URDF, table, objets ou caméra d'apprentissage.

## Phase 2 — Scène minimale avec table et cube

**Résultat attendu.** Une scène PhysX contient le bras fixé, une table et un cube rigide avec collisions stables.

**Critères d'acceptation.** Reset déterministe ; cube ne traverse ni table ni pince au repos ; caméra de débogage et pas de simulation vérifiés.

**Dépendances.** Phase 1, primitives ou assets à licence compatible, configuration PhysX.

**Hors périmètre.** Bacs colorés, randomisation, succès de prise et enregistrement de données.

## Phase 3 — Téléopération

**Résultat attendu.** L'opérateur pilote l'effecteur du SO-101 avec clavier, puis éventuellement manette, avec une pince commandable.

**Critères d'acceptation.** Les commandes sont documentées ; l'IK respecte les limites utiles ; un opérateur peut approcher, saisir et poser le cube de façon répétable.

**Dépendances.** Phase 2, `Se3Keyboard`/`Se3Gamepad`, contrôleur IK adapté au bras.

**Hors périmètre.** Bras leader physique, collecte de dataset, multi-environnements et politiques apprises.

## Phase 4 — Enregistrer et relire des démonstrations

**Résultat attendu.** Des épisodes réussis sont enregistrés localement puis relus comme `LeRobotDataset` v3.

**Critères d'acceptation.** Observations, actions, RGB, FPS et frontières d'épisodes sont synchronisés ; un épisode se relit et se visualise ; échecs exclus ou annotés selon la règle documentée.

**Dépendances.** Phase 3, LeRobot compatible, enregistreur LeIsaac direct ou conversion HDF5 validée.

**Hors périmètre.** Publication Hub automatique, entraînement ACT, augmentation synthétique et collecte massive.

## Phase 5 — Tâche de tri avec une seule brique

**Résultat attendu.** Une brique colorée doit être placée dans son unique bac correspondant dans une configuration contrôlée.

**Critères d'acceptation.** Réinitialisation, définition géométrique de succès, timeout et jeu de démonstrations de référence sont tous testés ; un épisode peut être rejoué.

**Dépendances.** Phase 4, modèle de bac à licence compatible, métrique de placement fiable.

**Hors périmètre.** Trois couleurs actives simultanément, langage naturel et curriculum complexe.

## Phase 6 — Entraîner ACT

**Résultat attendu.** Une politique ACT LeRobot est entraînée sur le dataset de la phase 5 avec une configuration versionnée.

**Critères d'acceptation.** Split train/validation explicite ; artefacts hors Git ; courbes et checkpoint reproductibles ; exécution d'inférence simulée sans erreur de schéma.

**Dépendances.** Phase 5, `LeRobotDataset` relisible, GPU CUDA et budget de démonstrations suffisant.

**Hors périmètre.** PPO, VLA, world models, entraînement de vision depuis zéro et comparaison exhaustive de politiques.

## Phase 7 — Évaluer sur des positions inédites

**Résultat attendu.** ACT est mesuré sur des positions de brique/bac non vues pendant l'entraînement.

**Critères d'acceptation.** Seeds et distributions de test séparées ; taux de succès, catégories d'échecs, vidéos d'exemples et analyse sont produits.

**Dépendances.** Phase 6, protocole de reset stable et export de politique.

**Hors périmètre.** Sim-to-real avancé, benchmark externe et revendication de généralisation hors distribution.

## Phase 8 — Étendre à plusieurs briques

**Résultat attendu.** La scène gère plusieurs briques rouges, vertes et bleues et leurs trois bacs, avec une règle de tri mesurable.

**Critères d'acceptation.** Spawn sans collisions initiales ; succès par brique et succès global définis ; dataset et protocole d'évaluation mis à jour sans fuite entre entraînement et test.

**Dépendances.** Phase 7, stratégie de données multi-objets et budget de calcul/stockage.

**Hors périmètre.** Instructions multi-langage, objets déformables, VLA, world models et transfert réel non validé.

## Hors périmètre global actuel

- PPO ;
- VLA ;
- world models ;
- sim-to-real avancé ;
- objets déformables ;
- entraînement depuis zéro d'un modèle de vision ;
- tâche multi-instructions en langage naturel.
