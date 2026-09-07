# Webots RL Interface

A reinforcement-learning interface for training and evaluating robots in Webots with Gymnasium and Stable-Baselines3.

The current MVP focuses on a Spot-like quadruped trained with PPO. It supports parallel Webots instances, GUI-based configuration, model inspection, compatibility checks, deterministic testing, experiment metadata, and controlled failure handling.

> **Current status:** Spot + PPO MVP validated.

## Current MVP

The current implementation supports:

- Spot-like quadruped environment in Webots
- Stable-Baselines3 PPO
- continuous control of 12 actuated joints
- 48-dimensional observation space
- parallel training with multiple Webots instances
- configurable PPO hyperparameters
- fixed or randomized seeds
- loading and continuing an existing model
- creating a new PPO model from scratch
- separate input and output model paths
- model inspection
- environment/model compatibility checks
- deterministic evaluation
- graceful training stop with model saving
- live trainer/test output in the GUI
- TensorBoard logging
- per-run configuration snapshots
- per-run logs and metadata
- worker/Webots disconnect detection
- controlled failure handling when a simulation instance is closed

The GUI currently exposes **Spot + PPO** only. Support for additional robots and algorithms is planned after this MVP.

## Spot environment

Current environment configuration:

- **Algorithm:** PPO
- **Policy:** MLP
- **Action space:** continuous, 12 dimensions
- **Observation space:** 48 dimensions
- **Control interval:** 32 ms
- **Maximum episode length:** 1000 control steps
- **Default learning rate:** `5e-5`

The observation currently includes:

- 12 normalized joint positions
- 12 normalized joint velocities
- 6 orientation features
- 3 normalized angular velocities
- 3 normalized local linear velocities
- 12 previous-action values

The reward and locomotion logic remain robot-specific and live under `controllers/main/`.

## Parallel training

Parallel training uses multiple independent Webots simulations controlled by one PPO model.

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

For `N` environments, the effective PPO rollout size is:

```text
rollout_size = N * n_steps
```

Example:

```text
2 environments × 512 n_steps = 1024 samples
```

## Project structure

```text
spotWebots/
|
+-- app/
|   +-- config.py
|   +-- gui.py
|   +-- launcher.py
|   +-- model_inspector.py
|   `-- run_manager.py
|
+-- controllers/
|   +-- debugging/
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
|   +-- check_compatibility.py
|   +-- test_model.py
|   +-- train_parallel.py
|   `-- webots_vec_env.py
|
+-- docs/
|   `-- REPRODUCING.md
|
+-- models/                 # runtime, ignored by Git
+-- logs/
|   `-- tensorboard/        # runtime, ignored by Git
+-- runs/                   # runtime, ignored by Git
|
+-- worlds/
|   +-- static_spot_ppo.wbt
|   `-- static_spot_parallel.wbt
|
+-- requirements.txt
`-- README.md
```

## Installation

Clone the current development branch:

```powershell
git clone --branch webots-rl-interface https://github.com/Energetic0oak32/spotWebots.git
cd spotWebots
```

Create and activate a Python virtual environment:

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Webots must be installed separately.

The Webots Python `controller` module is provided by Webots itself and should not be installed from PyPI.

## Running the interface

Start the GUI from the repository root:

```powershell
python .\app\gui.py
```

Typical workflow:

```text
1. Select robot and algorithm
2. Select the input model, or choose a new model path
3. Select the output model path
4. Configure Webots, world, environment and controller paths
5. Choose instance count and base port
6. Configure PPO hyperparameters
7. Choose a fixed or randomized seed
8. Verify configuration
9. Start Webots
10. Verify model/environment compatibility
11. Start training or deterministic testing
12. Stop the operation gracefully when desired
13. Inspect the generated run directory
```

## Models

Training distinguishes between the model used as input and the model written at the end.

If the input model exists, it is loaded and training continues from its current state. If it does not exist, a new PPO model is created.

The final model is saved to the configured output path, allowing the original baseline to be preserved.

Example:

```text
Input:
models/ppo_spot.zip

Output:
models/ppo_spot_experiment.zip
```

## Deterministic testing

The GUI can launch deterministic evaluation with:

```text
deterministic=True
```

The number of episodes and seed are configurable.

Testing runs in real-time simulation mode for visual inspection.

## Seeds

The interface supports:

- **fixed seed:** the configured value is reused
- **random seed:** a concrete seed is generated when the operation begins

Even in random mode, the concrete generated seed is stored in run metadata so the experiment can be reproduced later.

## Experiment runs

Training and testing create a dedicated directory under:

```text
runs/
```

Example:

```text
runs/
`-- 2026-09-07_04-01-55_spot_ppo_training/
    +-- config.json
    +-- run.json
    `-- training.log
```

A testing run uses `test.log` instead.

`run.json` stores metadata including:

- run ID
- operation type
- status
- start and finish timestamps
- robot
- algorithm
- concrete seed
- random-seed flag
- ports
- number of instances
- input model path
- output model path
- whether model files existed at startup
- world path
- Git branch
- Git commit
- requested timesteps or test parameters
- exit code
- user-stop flag
- transport failure
- Webots-closed flag

Typical final statuses are:

```text
completed
stopped
failed
```

TensorBoard data is written under:

```text
logs/tensorboard/
```

## Graceful stop and failure handling

Stopping training through the GUI is graceful: the trainer receives a stop request, exits `model.learn()`, saves the output model, closes the external workers, and finishes the run as `stopped`.

If a Webots instance is closed during parallel training, the worker disconnect is detected, the remaining session is released, and the run is recorded as failed.

Example:

```json
{
  "status": "failed",
  "stopped_by_user": false,
  "transport_failure": "WORKER_DISCONNECTED",
  "webots_closed": true
}
```

## Validated MVP scenarios

The Spot/PPO MVP has been manually tested with:

- existing PPO model loading
- new PPO model creation
- continued training
- changed PPO hyperparameters
- one Webots instance
- two parallel Webots instances
- fixed seed
- random seed
- deterministic evaluation
- model/environment compatibility checking
- separate input/output model files
- experiment run creation
- graceful training stop
- worker disconnect during training
- incompatible model handling
- invalid PPO configuration handling
- operations blocked while Webots is not running
- GUI resizing and scrolling

## Known limitations

### Spot + PPO only

The GUI currently exposes only the Spot environment and PPO. The next major development stage is to generalize robot and algorithm support.

### No automatic checkpoints

Training currently saves the output model when training finishes or is stopped gracefully.

There are no automatic intermediate checkpoints yet. A crash, forced termination, or power loss may therefore lose progress since the previous saved model.

Automatic checkpoints are planned as a lower-priority improvement.

### Hyperparameter validation

Basic configuration is checked before launch, but not every invalid Stable-Baselines3 hyperparameter combination is rejected directly by the GUI.

Some invalid combinations are rejected by Stable-Baselines3 when the trainer starts. These failures are logged and the run is marked as failed.

### Compatibility and testing

The compatibility checker detects incompatible model/environment spaces.

The backend also validates compatibility when loading a model for testing or training, providing a second failure boundary.

## Next development stage

With the Spot/PPO MVP complete, the next major goal is to generalize the interface.

Planned direction:

1. separate robot-specific environment definitions from the generic training interface
2. add a second robot/environment
3. add a second algorithm
4. validate algorithm/action-space compatibility
5. use that second combination to remove remaining Spot/PPO assumptions from the application

The intended next validation target is a mobile robot such as the Pioneer with a discrete action space and DQN.

Additional future improvements include:

- automatic checkpoints
- richer run comparison tools
- integrated TensorBoard controls
- more evaluation metrics
- additional locomotion and recovery tasks

## Reproducing experiments

Detailed setup and reproducibility guidance are available in:

```text
docs/REPRODUCING.md
```

## About

This repository started as a reinforcement-learning project for training a Spot-like quadruped in Webots with PPO.

The `webots-rl-interface` branch expands that work into an experiment interface intended to support multiple Webots robots and reinforcement-learning algorithms while keeping training, testing, configuration, and experiment tracking in one workflow.
