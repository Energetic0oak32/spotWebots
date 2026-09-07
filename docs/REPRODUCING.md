# Reproducing Spot PPO Experiments

This document describes how to reproduce the current Spot reinforcement-learning experiments using Webots, Gymnasium, Stable-Baselines3, and the Webots RL Interface.

The documented workflow targets the `webots-rl-interface` branch and the validated Spot + PPO MVP.

---

## 1. System requirements

The current development workflow uses Windows and PowerShell.

Required software:

- Git
- Python
- Webots

Python dependencies are listed in:

```text
requirements.txt
```

The main dependencies include:

- NumPy
- Gymnasium
- Stable-Baselines3
- TensorBoard
- PySide6

The Webots Python `controller` module is provided by Webots itself and should not be installed separately from PyPI.

---

## 2. Clone the correct branch

To reproduce the current interface:

```powershell
git clone --branch webots-rl-interface https://github.com/Energetic0oak32/spotWebots.git
cd spotWebots
```

If the repository has already been cloned:

```powershell
git fetch origin
git switch webots-rl-interface
git pull
```

For a specific published experiment, prefer checking out the exact commit associated with that run:

```powershell
git checkout <commit-hash>
```

The interface records the active Git branch and commit in each run's metadata.

---

## 3. Create the Python environment

Create the virtual environment:

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

Verify the selected Python executable:

```powershell
(Get-Command python).Source
```

It should point to something similar to:

```text
...\spotWebots\env\Scripts\python.exe
```

The GUI can also be configured to use an existing compatible virtual environment.

---

## 4. Install and locate Webots

Webots must be installed independently.

A typical Windows installation is:

```text
C:\Program Files\Webots
```

The project uses:

```text
webots.exe
webots-controller.exe
```

A typical external-controller executable is:

```text
C:\Program Files\Webots\msys64\mingw64\bin\webots-controller.exe
```

The GUI verifies the main configured paths before operations are launched.

---

## 5. Runtime directories

The following directories contain generated artifacts and are intentionally ignored by Git:

```text
models/
logs/
runs/
```

TensorBoard data is stored under:

```text
logs/tensorboard/
```

Training and testing runs are stored under:

```text
runs/
```

These directories are created as needed.

On a fresh clone, ignored empty directories may not exist until the application creates them.

---

## 6. Current architecture

The current Spot + PPO pipeline is:

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

Each Webots instance runs an independent Spot environment.

The external worker is implemented in:

```text
controllers/main/parallel_worker.py
```

The vectorized trainer interface is implemented in:

```text
training/webots_vec_env.py
```

The GUI orchestrates:

- Webots processes;
- worker/trainer processes;
- model inspection;
- compatibility checking;
- training;
- deterministic testing;
- run metadata and logs.

---

## 7. Webots ports

Each Webots instance receives its own external-controller port.

Ports are generated as:

```text
port = base_port + instance_index
```

Example:

```text
base_port = 1234
instances = 4
```

produces:

```text
1234
1235
1236
1237
```

The trainer and external workers also use temporary localhost IPC ports internally.

Those IPC ports are independent from the Webots instance ports.

---

## 8. Webots execution mode

Training Webots instances are launched in fast simulation mode.

The simulation therefore runs as quickly as the machine can execute it rather than being limited to real time.

Deterministic visual evaluation uses real-time simulation mode.

Training and testing are separate operations in the GUI.

---

## 9. `runtime.ini`

The current parallel workflow uses external controllers and does not require a controller-local `runtime.ini`.

If a local `runtime.ini` causes external-controller errors, it may be disabled, for example:

```powershell
Rename-Item runtime.ini runtime.ini.disabled
```

Run that command only from the directory where the file actually exists.

The trainer is launched with the Python interpreter selected by the configured virtual environment, so it does not depend on PowerShell activation after the GUI has started.

---

## 10. Current environment definition

### Action space

The current action space is:

```text
Box(-1, 1, shape=(12,))
```

Each action dimension controls one actuated joint.

Normalized actions are mapped to the mechanical range of their corresponding joints.

### Observation space

The current observation space is:

```text
Box(..., shape=(48,))
```

Current composition:

```text
12 normalized joint positions
12 normalized joint velocities
 6 orientation features
 3 normalized angular velocities
 3 normalized local linear velocities
12 previous-action values
---------------------------------
48 values
```

Orientation uses sine/cosine features for roll, pitch, and yaw.

The previous action is included explicitly because the reward contains an action-rate penalty.

---

## 11. Current control timing

The environment currently uses:

```text
control_step = 32 ms
```

A new PPO action is therefore applied every 32 ms of simulated time.

The Webots physics basic timestep is still obtained from:

```python
robot.getBasicTimeStep()
```

The physics timestep and the reinforcement-learning control interval are conceptually separate.

---

## 12. Episode termination

Current maximum episode duration:

```text
1000 control steps
```

Episodes may terminate earlier if the robot is classified as fallen.

Fall detection uses robot orientation and body-height conditions.

Because environment and reward logic can evolve, reproducible experiments should always be associated with the exact Git commit that produced them.

---

## 13. Current reward

The current reward is implemented in:

```text
controllers/main/reward.py
```

It combines:

Positive terms:

- alive reward;
- forward local velocity.

Scaling factors:

- upright/stability factor;
- body-height factor.

Penalties include:

- vertical velocity;
- change in vertical velocity;
- angular velocity;
- action-rate change.

A larger negative reward is applied when the robot falls.

Reward design is robot-specific.

---

## 14. Start the Webots RL Interface

From the repository root:

```powershell
python .\app\gui.py
```

The current GUI allows configuration of:

- robot;
- algorithm;
- input model;
- output model;
- Webots installation;
- world file;
- Python environment;
- controller directory;
- number of Webots instances;
- base port;
- total timesteps;
- learning rate;
- `n_steps`;
- batch size;
- PPO epochs;
- gamma;
- fixed or random seed;
- number of deterministic test episodes.

The current MVP exposes:

```text
Robot: Spot
Algorithm: PPO
```

---

## 15. Recommended GUI workflow

A normal training workflow is:

```text
1. Open the Webots RL Interface
2. Select the input model, or choose a path for a new model
3. Select the output model path
4. Verify paths and configuration
5. Configure PPO hyperparameters
6. Choose a fixed or random seed
7. Start Webots
8. Verify model/environment compatibility
9. Start training
10. Observe trainer output
11. Stop training gracefully when desired
12. Stop Webots when finished
13. Inspect the generated run directory
```

For deterministic testing:

```text
1. Load a trained model
2. Start Webots
3. Verify compatibility
4. Configure the test seed and number of episodes
5. Start deterministic testing
6. Observe the robot in real time
7. Inspect the generated testing run
```

---

## 16. Input and output models

Training distinguishes between the model used as the starting point and the model written at the end.

Example:

```text
Input:
models/ppo_spot.zip

Output:
models/ppo_spot_experiment.zip
```

If the input model exists:

```text
PPO.load(...)
```

is used and training continues from that model.

If the input model does not exist:

```text
PPO(...)
```

creates a new model.

The resulting model is saved only to the configured output path.

This separation allows a baseline model to remain unchanged while experiments are written to new files.

---

## 17. Model inspection

When an existing model is loaded in the GUI, the interface can inspect information such as:

- algorithm;
- observation space;
- action space;
- PPO hyperparameters.

For the current Spot model, the expected spaces are:

```text
Observation: 48 values
Action:      12 continuous values
```

Model inspection alone does not prove semantic compatibility; it only exposes the saved model structure and parameters.

---

## 18. Compatibility checking

The GUI includes a model/environment compatibility check.

The checker:

1. connects to the selected Webots environment;
2. reads the environment observation and action spaces;
3. loads the selected model against that environment;
4. reports compatible or incompatible.

A typical incompatible result is:

```text
INCOMPATIBLE|Observation spaces do not match: ...
```

A missing input model is treated as a new-model case rather than as an incompatibility.

For major changes to:

- observation shape;
- observation meaning;
- normalization;
- action mapping;

a new model should normally be trained.

---

## 19. Graceful training stop

The GUI and trainer communicate through:

```text
training/.stop_training
```

When **Stop training** is pressed:

```text
GUI creates .stop_training
        |
        v
StopTrainingCallback detects it
        |
        v
model.learn() returns normally
        |
        v
model.save(output_model)
        |
        v
WebotsVecEnv closes workers
        |
        v
run status = stopped
```

The stop file is removed after the trainer finishes.

The GUI also clears stale stop files before starting new operations.

A graceful stop preserves the output model.

---

## 20. Failure handling

The interface monitors Webots and worker health.

If a Webots instance is closed while training, the corresponding worker connection fails.

A typical failure is:

```text
WORKER_DISCONNECTED
```

The training session is then released instead of continuing with a partial environment set.

The run is recorded with failure metadata, for example:

```json
{
  "status": "failed",
  "stopped_by_user": false,
  "transport_failure": "WORKER_DISCONNECTED",
  "webots_closed": true
}
```

Invalid Stable-Baselines3 configurations are also captured as failed runs when the trainer process exits with an error.

---

## 21. PPO rollout size

In Stable-Baselines3 PPO, `n_steps` is collected per environment.

Therefore:

```text
effective_rollout = n_envs × n_steps
```

Examples:

```text
1 environment  × 2048 = 2048 samples
2 environments × 1024 = 2048 samples
4 environments ×  512 = 2048 samples
8 environments ×  256 = 2048 samples
```

Increasing the number of environments without reducing `n_steps` increases the total rollout size.

For example:

```text
2 environments × 2048 = 4096 samples
```

This is valid, but is not methodologically identical to a 2048-sample rollout.

The GUI displays the effective rollout.

---

## 22. Short smoke test

Before a long training run, use a short validation experiment.

Suggested configuration:

```text
Instances:       2
Base port:       1234
Timesteps:       2048
Learning rate:   5e-5
n_steps:         512
Batch size:      64
N epochs:        10
Gamma:           0.99
Seed:            42
```

Effective rollout:

```text
2 × 512 = 1024 samples
```

Expected behavior:

```text
1. Two Webots instances open.
2. Both external workers connect.
3. WebotsVecEnv reports two environments.
4. PPO begins collecting rollouts.
5. Training output appears in the GUI.
6. The model is saved to the selected output path.
7. A run directory is created.
8. config.json, run.json, and training.log are written.
```

---

## 23. Deterministic evaluation

The current deterministic tester uses:

```text
deterministic=True
```

The GUI allows configuration of:

```text
test episodes
seed
```

Testing runs in real-time simulation mode.

A fixed base seed makes the evaluation reproducible.

The test operation writes its own run directory and log.

---

## 24. Seeds

The interface supports:

```text
fixed seed
random seed
```

With a fixed seed, the configured value is reused.

With random seed enabled, the GUI resolves a concrete seed when the operation begins.

That concrete seed is stored in `run.json` and `config.json`.

This allows the generated seed to be recovered later.

---

## 25. Run directories

Every training or testing operation creates a directory under:

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

A testing run contains:

```text
test.log
```

instead of `training.log`.

---

## 26. `config.json`

`config.json` is a snapshot of the effective GUI configuration at operation start.

It records the settings used to launch that specific training or testing run.

When random seed mode is enabled, the resolved concrete seed is stored in the snapshot.

---

## 27. `run.json`

`run.json` contains experiment metadata.

Current metadata includes:

```text
run ID
operation
status
start time
finish time
robot
algorithm
seed
random-seed flag
ports
number of instances
input model path
output model path
input model existed at start
output model existed at start
world path
Git branch
Git commit
requested timesteps or test settings
exit code
stopped-by-user flag
transport failure
Webots-closed flag
```

Typical final statuses are:

```text
completed
stopped
failed
```

Example successful stop:

```json
{
  "status": "stopped",
  "exit_code": 0,
  "stopped_by_user": true,
  "transport_failure": null,
  "webots_closed": false
}
```

---

## 28. TensorBoard

Training logs are written under:

```text
logs/tensorboard/
```

Start TensorBoard from the repository root:

```powershell
tensorboard --logdir .\logs\tensorboard
```

Useful metrics include:

- episode reward;
- episode length;
- approximate KL divergence;
- clip fraction;
- learning rate;
- policy loss;
- value loss;
- explained variance.

---

## 29. Experimental reproducibility

For every relevant experiment, record or preserve at least:

```text
Run identifier
Git commit
World file
Robot
Algorithm
Number of environments
Ports
Total timesteps
n_steps
Effective rollout size
Batch size
N epochs
Gamma
Learning rate
Control step
Observation-space version
Reward configuration
Input model
Output model
Concrete seed
```

Much of this is now recorded automatically under `runs/`.

For full reproducibility, still prefer the exact Git commit over only the branch name.

---

## 30. Example experiment

```text
Experiment:
spot_parallel_baseline

Git commit:
<commit hash>

World:
static_spot_parallel.wbt

Algorithm:
PPO / MlpPolicy

Instances:
2

Ports:
1234, 1235

n_steps:
512

Effective rollout:
1024

Batch size:
64

N epochs:
10

Gamma:
0.99

Learning rate:
5e-5

Control step:
32 ms

Observation space:
48

Action space:
12

Total timesteps:
500000

Input model:
None

Output model:
models/spot_parallel_baseline.zip

Seed:
42
```

---

## 31. Common issues

### External controller cannot find a robot

Check that:

- the correct world was opened;
- the selected Webots port matches the worker;
- the world contains the expected external-controller configuration;
- the simulation is running.

### `webots-controller.exe` not found

Check the Webots installation configured in the GUI.

Typical installation:

```text
C:\Program Files\Webots
```

### Python environment not found

Verify the configured virtual-environment path.

From an activated PowerShell environment:

```powershell
(Get-Command python).Source
```

### Training immediately stops

Check for a stale:

```text
training/.stop_training
```

The GUI normally removes stale stop files before starting a new training session.

### Existing PPO model fails to load

Check:

- action-space shape;
- observation-space shape;
- observation semantics;
- normalization;
- action mapping;
- timing.

Use the GUI compatibility checker before training or testing.

### Invalid PPO hyperparameters

Not every invalid Stable-Baselines3 hyperparameter combination is rejected by the GUI before launch.

The backend may reject invalid values when creating or loading PPO.

In that case:

```text
status = failed
```

is recorded for the run, and no valid output model is saved if model creation fails.

### Worker disconnect

If one Webots instance is closed during training, the session should fail instead of continuing with fewer environments.

Inspect:

```text
transport_failure
webots_closed
```

in `run.json`.

### Text encoding in Windows PowerShell

Run logs are written in UTF-8.

Older Windows PowerShell versions may display UTF-8 text incorrectly when using `Get-Content` without an explicit encoding.

Use:

```powershell
Get-Content .\path\to\training.log -Encoding UTF8
```

This affects terminal display, not the log file itself.

### Rollout changed after adding environments

Remember:

```text
rollout = n_envs × n_steps
```

Adjust `n_steps` if maintaining a constant total rollout is part of the experimental design.

---

## 32. Legacy single-instance runner

`controllers/main/main.py` remains in the repository as the older single-instance runner.

The actively developed workflow is the parallel infrastructure under:

```text
training/
```

and the GUI under:

```text
app/
```

Do not assume the legacy runner shares every current model-path, logging, seed, or run-management convention unless it has been updated explicitly.

---

## 33. Known limitations

### Spot + PPO only

The current GUI exposes only Spot and PPO.

The architecture will be generalized after the validated MVP.

### No automatic checkpoints

The trainer currently saves the model when training completes or when it is stopped gracefully.

Automatic intermediate checkpoints are not yet implemented.

A crash, hard process kill, or power loss may therefore lose progress since the previous saved model.

### Hyperparameter validation

The GUI performs basic pre-flight validation, but some invalid PPO parameter combinations are still rejected only by Stable-Baselines3.

### Model compatibility

The compatibility checker validates saved model/environment spaces.

Matching shapes alone do not guarantee semantic compatibility after major changes to observations, actions, normalization, reward, or timing.

---

## 34. Validated MVP scenarios

The current Spot + PPO MVP has been manually validated for:

```text
Existing model loading
New PPO model creation
Continued training
Changed PPO hyperparameters
One Webots instance
Two parallel Webots instances
Fixed seed
Random seed
Deterministic testing
Compatibility checking
Separate input and output model paths
Run metadata and logs
Graceful training stop
Worker/Webots disconnect handling
Incompatible-model failure handling
Invalid-PPO-configuration failure handling
Operations blocked without running Webots
GUI resizing and scrolling
```

---

## 35. Next development stage

The Spot + PPO MVP is considered complete.

The next major goal is generalization.

The intended direction is:

```text
1. Isolate remaining Spot-specific assumptions
2. Isolate remaining PPO-specific assumptions
3. Add a second robot/environment
4. Add a second algorithm
5. Validate action-space/algorithm compatibility
6. Use the new combination to verify the generic interface
```

The intended second validation target is a Pioneer-style mobile robot with a discrete action space and DQN.

Additional future improvements include:

- automatic checkpoints;
- richer run comparison;
- integrated TensorBoard controls;
- additional evaluation metrics;
- additional locomotion tasks;
- standing/recovery behaviors.

---

## 36. Reproducibility rule

For any result that matters, keep:

```text
run directory
model output
Git commit
world file
configuration snapshot
seed
training/test log
```

A branch name alone is not enough to reproduce a historical experiment after the code evolves.
