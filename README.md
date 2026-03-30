# OxonoProject

## Install dependencies

```bash
uv sync
```

## Run the project

* For training:
```bash
uv run setup.py build_ext --inplace # Compile the Cython code
uv run train.py
```

* For evaluation:
```bash
uv run manager.py -n 10 -p0 random_agent.py -p1 zero_agent.py
```

* To play against the agent:
```bash
uv run visual_manager.py -p1 zero_agent.py
```

## Project structure

* `agent.py`, `main.py`, `manager.py`, `oxono.py`, `random_agent.py`, `replayer.py`, `visual_manager.py`: Default Python files for the project.
* `mcts.py`: Implementation of the Monte Carlo Tree Search algorithm with PUCT and Zobrist table.
* `mcts_train.pyx`: Cython implementation of the MCTS algorithm for training.
* `network_agent.py`: Implementation of a neural network-based agent.
* `oxono_train.pyx`: Cython implementation of the training.
* `oxononet.py`: Implementation of the neural network architecture for the agent.
* `setup.py`: Cython setup file for compiling the Cython code.
* `train.py`: Script for training the neural network agent.
* `zero_agent.py`: Implementation of AlphaZero agent.
