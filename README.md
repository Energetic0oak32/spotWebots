# Spot Webots PPO

Reinforcement learning environment for training a Spot-like quadruped robot in Webots using Gymnasium and Stable-Baselines3 PPO.

The current documented workflow focuses on **parallel PPO training across multiple independent Webots simulations**. A legacy single-instance runner remains under `controllers/main/main.py`, but the parallel training infrastructure and Spot Training Manager are the primary development path.

## Overview

The environment gives the PPO policy continuous control over the robot's 12 actuated joints.

Current environment configuration:

- **Algorithm:** PPO
- **Policy:** MLP
- **Action space:** continuous, 12 dimensions
- **Observation space:** 48 dimensions
- **Control interval:** 32 ms
- **Maximum episode length:** 1000 control steps
- **Default learning rate:** `5e-5`

The current observation includes:

- 12 normalized joint positions;
- 12 normalized joint velocities;
- 6 orientation features;
- 3 normalized angular velocities;
- 3 normalized local linear velocities;
- 12 previous-action values.

## Parallel training

Parallel training uses multiple independent Webots processes controlled by a single PPO policy.

```text
                         PPO
                          |
                    WebotsVecEnv
                    /     |     \
                   /      |      \
             Worker 0  Worker 1  Worker N
                |         |         |
             Webots     Webots     Webots
                |         |         |
              Spot      Spot      Spot
```

Each Webots instance runs its own `SpotEnv`.

`WebotsVecEnv` sends actions to all environments before waiting for their responses, allowing the independent simulations to advance concurrently.

For `N` environments, the effective PPO rollout size is:

```text
rollout_size = N * n_steps
```

For example:

```text
2 environments × 1024 n_steps = 2048 samples
```

## Project structure

```text
spotWebots/
|
+-- app/
|   +-- config.py
|   +-- gui.py
|   `-- launcher.py
|
+-- controllers/
|   +-- debugging/
|   |   `-- debug.py
|   |
|   `-- main/
|       +-- main.py
|       +-- motors.py
|       +-- parallel_worker.py
|       +-- reward.py
|       +-- sensors.py
|       `-- spot_env.py
|
+-- training/
|   +-- train_parallel.py
|   `-- webots_vec_env.py
|
+-- docs/
|   `-- REPRODUCING.md
|
+-- models/                # runtime, ignored by Git
|
+-- logs/
|   `-- tensorboard/       # runtime, ignored by Git
|
+-- worlds/
|   +-- static_spot_ppo.wbt
|   `-- static_spot_parallel.wbt
|
+-- requirements.txt
`-- README.md
```

### `controllers/main`

Contains the robot environment and code that directly interacts with Webots:

- motor control;
- sensors;
- reward;
- Gymnasium environment;
- external parallel worker.

### `training`

Contains the Stable-Baselines3 training infrastructure and the custom `WebotsVecEnv` used to communicate with multiple Webots instances.

### `app`

Contains the PySide6 **Spot Training Manager**.

The manager currently supports:

- Webots installation selection;
- world selection;
- Python environment selection;
- controller directory selection;
- configurable number of Webots instances;
- configurable base port;
- PPO timesteps;
- learning rate;
- `n_steps`;
- rollout-size display;
- Webots launch/stop;
- PPO launch;
- live trainer output;
- graceful training stop.

### Runtime directories

`models/` and `logs/` are intentionally ignored by Git because they contain generated artifacts.

The Spot Training Manager creates the required runtime directories automatically before training.

## Installation

Clone the current parallel-training branch:

```powershell
git clone --branch parallel-webots https://github.com/Energetic0oak32/spotWebots.git
cd spotWebots
```

Create a Python virtual environment:

```powershell
python -m venv env
```

Activate it:

```powershell
.\env\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Webots must be installed separately.

The Webots Python `controller` module is provided by Webots itself and should **not** be installed from PyPI.

## Spot Training Manager

Start the manager from the repository root:

```powershell
python .\app\gui.py
```

A typical workflow is:

```text
1. Verify configuration
2. Start Webots
3. Start training
4. Stop training when desired
5. Stop Webots when finished
```

The launcher starts Webots in `fast` mode.

Stopping training through the manager is graceful: the trainer receives a stop signal, exits `model.learn()`, saves the model, closes the external workers, and leaves the Webots instances open.

## Reproducing experiments

Detailed setup, experiment parameters, troubleshooting, and reproducibility guidance are available in:

[docs/REPRODUCING.md](docs/REPRODUCING.md)

## Development status

The project is under active development.

Current work focuses on:

- reliable parallel PPO training;
- quadruped locomotion;
- reward design;
- action smoothness;
- reproducible experiments;
- training and simulation management.

Planned improvements include:

- automatic checkpoints;
- experiment metadata;
- model selection and compatibility checks;
- integrated TensorBoard controls;
- automated evaluation;
- additional locomotion and recovery tasks.
