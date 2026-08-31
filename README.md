# Spot Webots PPO

Reinforcement learning environment for training a Spot-like quadruped robot in Webots using Gymnasium and Stable-Baselines3 PPO.

The project currently supports both single-instance experiments and parallel training across multiple independent Webots simulations.

## Overview

The environment gives the PPO policy continuous control over the robot's 12 actuated joints.

Current environment configuration:

* **Algorithm:** PPO
* **Policy:** MLP
* **Action space:** continuous, 12 dimensions
* **Observation space:** 48 dimensions
* **Control interval:** 32 ms
* **Maximum episode length:** 1000 control steps
* **Default learning rate:** `5e-5`

The observation currently includes normalized joint positions, joint velocities, robot orientation, angular velocity, local linear velocity, and the previous action.

## Parallel training

Parallel training uses multiple independent Webots processes controlled by a single PPO model.

The architecture is:

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

Each Webots instance runs its own `SpotEnv`. Actions are sent to all environments before the trainer waits for their results, allowing simulation steps to execute concurrently.

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
+-- models/
|
+-- logs/
|   `-- tensorboard/
|
+-- worlds/
|   +-- static_spot_ppo.wbt
|   `-- static_spot_parallel.wbt
|
+-- requirements.txt
`-- README.md
```

### `controllers/main`

Contains the robot environment and the code that directly interacts with Webots.

### `training`

Contains the Stable-Baselines3 training infrastructure and the custom vectorized environment used to communicate with multiple Webots instances.

### `app`

Contains the Spot Training Manager desktop interface.

The manager can configure and launch Webots instances, choose the number of parallel environments, configure PPO parameters, start training, and display training output.

### `models`

Local directory for trained PPO models.

Model files are intentionally ignored by Git.

### `logs`

Contains generated training logs, including TensorBoard runs.

Logs are intentionally ignored by Git.

## Installation

Clone the repository:

```powershell
git clone https://github.com/Energetic0oak32/spotWebots.git
cd spotWebots
```

Create a Python environment:

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

Webots must also be installed separately.

The Python `controller` module used by the project is provided by Webots and should not be installed from PyPI.

## Spot Training Manager

Start the manager from the repository root:

```powershell
python .\app\gui.py
```

The interface allows configuration of:

* Webots installation path
* World file
* Python environment
* Controller directory
* Number of Webots instances
* Initial Webots port
* Total PPO timesteps
* Learning rate
* `n_steps`

The manager also displays the effective rollout size before training.

## Reproducing experiments

Detailed setup and reproduction instructions are available in:

```text
docs/REPRODUCING.md
```

## Status

The project is under active development.

Current work focuses on:

* reliable parallel PPO training;
* improving quadruped locomotion;
* reward design;
* action smoothness;
* reproducible training experiments;
* training and simulation management tools.

Future work may include checkpoint management, experiment metadata, model compatibility checks, evaluation tools, and additional locomotion tasks.
