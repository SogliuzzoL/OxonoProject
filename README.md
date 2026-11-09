# Oxono

Agents pour **Oxono**, projet du cours d'intelligence artificielle (UCLouvain, groupe 114). Le rapport est dans [OxonoReport](https://github.com/SogliuzzoL/OxonoReport).

## Agents

| Fichier | Principe |
|---|---|
| `random_agent.py` | Coups aléatoires, agent de référence |
| `minmax_agent.py` | Minimax avec élagage, quiescence, heuristique et table de Zobrist |
| `mcts_agent.py` | Monte Carlo Tree Search avec PUCT |
| `zero_agent.py` | AlphaZero : réseau `OxonoNet` + MCTS |
| `athenan_agent.py` | Agent entraîné du groupe : réseau `R2AthenanNet` + arbre `AthenanTree` |

## Structure

- `oxono.py`, `agent.py` — moteur du jeu et classe de base des agents ;
- `manager.py`, `visual_manager.py`, `replayer.py` — exécution de parties, interface Pygame et rejeu ;
- `train/` — entraînement et MCTS en Cython (`mcts_train.pyx`, `oxono_train.pyx`) ;
- `swiss_tournament.py`, `run_stats.py`, `stats_plot.py` — tournois, statistiques et figures ;
- `files_99942500_26482510/` — paquet d'agent au format attendu par la plateforme du cours ; le nom du dossier est généré par la plateforme et les imports s'y réfèrent tels quels.

## Lancer

```bash
uv sync
uv run setup.py build_ext --inplace          # compilation Cython

# tournoi entre deux agents
uv run manager.py -n 10 -p0 random_agent.py -p1 zero_agent.py

# jouer contre un agent
uv run visual_manager.py -p1 zero_agent.py
```

## Entraînement

```bash
uv run run_train.py        # AlphaZero (OxonoNet)
uv run athenan_train.py    # variante Athenan (R2AthenanNet)
```

Les poids (`*.pth`) et les sorties de tournoi (`*.zip`, `*.png`) ne sont pas versionnés : il faut entraîner avant d'utiliser `zero_agent.py` ou `athenan_agent.py`.

## Limites

- `zero_agent.py` et `athenan_agent.py` ont besoin des poids entraînés, absents du dépôt.
- Pas de tests automatisés ; la validation se fait par tournois (`swiss_tournament.py`).
- Projet rendu dans un cadre de cours, co-écrit avec Friedrich Röben.
