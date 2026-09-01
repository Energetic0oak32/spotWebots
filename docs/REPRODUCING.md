# Reproducing Spot PPO Experiments

This document describes how to reproduce the current Spot reinforcement-learning experiments using Webots, Gymnasium, and Stable-Baselines3.

The documented workflow targets the `parallel-webots` branch and focuses on the current parallel PPO training architecture.

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

To reproduce the current parallel-training version directly:

```powershell
git clone --branch parallel-webots https://github.com/Energetic0oak32/spotWebots.git
cd spotWebots
```

Alternatively, if the repository has already been cloned:

```powershell
git fetch origin
git switch parallel-webots
git pull
```

For a specific published experiment, prefer checking out the exact commit associated with that experiment:

```powershell
git checkout <commit-hash>
```

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

A typical external-controller path is:

```text
C:\Program Files\Webots\msys64\mingw64\bin\webots-controller.exe
```

The Spot Training Manager verifies these paths during its pre-flight check.

---

## 5. Runtime directories

The following directories contain generated artifacts and are intentionally ignored by Git:

```text
models/
logs/tensorboard/
```

When training is launched through the Spot Training Manager, these directories are created automatically if they do not exist.

This matters on a fresh clone because empty ignored directories are not stored by Git.

---

## 6. External-controller architecture

Parallel training uses one Webots process and one external worker per environment.

Example for one environment:

```text
Webots
port 1234
   |
webots-controller
   |
parallel_worker.py
   |
SpotEnv
```

For multiple environments, ports are generated as:

```text
port = base_port + instance_index
```

For example:

```text
base_port = 1234
instances = 4
```

results in:

```text
1234
1235
1236
1237
```

The trainer and workers also use temporary localhost IPC ports internally. These are independent from the Webots instance ports.

---

## 7. Webots execution mode

The manager launches Webots with:

```text
--mode=fast
```

Therefore the simulation should start running automatically instead of opening paused.

The Webots windows remain open when training is stopped through the manager.

---

## 8. `runtime.ini`

The current parallel workflow uses external controllers and does not require a controller-local `runtime.ini`.

If a local `runtime.ini` causes external-controller errors, it may be disabled, for example:

```powershell
Rename-Item runtime.ini runtime.ini.disabled
```

Run that command only from the directory where the file actually exists.

The manager launches training with the Python interpreter selected from the configured virtual environment, so it does not rely on PowerShell environment activation for the trainer process itself.

---

## 9. Current environment definition

### Action space

The current action space is:

```text
Box(-1, 1, shape=(12,))
```

Each action dimension controls one actuated joint.

The normalized action is mapped over the mechanical range of its joint.

### Observation space

The current observation space is:

```text
Box(-1, 1, shape=(48,))
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

Including the previous action makes the previous command explicit to the policy while the reward contains an action-rate penalty.

---

## 10. Current control timing

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

## 11. Episode termination

Current maximum episode duration:

```text
1000 control steps
```

Episodes may terminate earlier if the robot is classified as fallen.

Fall detection currently uses:

- orientation;
- minimum body height;
- persistence over multiple control steps.

The persistence requirement helps avoid terminating an episode because of a single transient sample.

---

## 12. Current reward

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

Penalties:

- vertical velocity;
- change in vertical velocity;
- angular velocity;
- action-rate change.

A larger negative reward is applied when the robot falls.

Because reward design is still evolving, a result should always be associated with the exact Git commit that produced it.

---

## 13. Start the Spot Training Manager

From the repository root:

```powershell
python .\app\gui.py
```

The manager allows configuration of:

- Webots installation;
- world file;
- Python environment;
- controller directory;
- number of environments;
- base port;
- total timesteps;
- learning rate;
- `n_steps`.

Use **Verify configuration** before launching Webots.

---

## 14. Recommended GUI workflow

```text
1. Open Spot Training Manager
2. Select/verify paths
3. Configure PPO parameters
4. Verify configuration
5. Start Webots
6. Start training
7. Observe trainer output
8. Stop training when desired
9. Stop Webots when finished
```

Stopping training does not close the Webots instances.

---

## 15. Graceful training stop

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
model.save(...)
        |
        v
WebotsVecEnv closes workers
        |
        v
Webots instances remain open
```

The stop file is removed after the trainer finishes.

The GUI also clears a stale stop file before starting a new training session.

If the entire manager window is closed while training is active, the GUI first attempts the same graceful stop and only kills the trainer if it does not exit within a short timeout.

---

## 16. PPO rollout size

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

Increasing the number of environments without changing `n_steps` also increases the rollout size.

For example:

```text
2 environments × 2048 = 4096 samples
```

This is valid, but it is not methodologically identical to a 2048-sample rollout.

---

## 17. Short smoke test

Before a long training run, use a short validation experiment.

Suggested configuration:

```text
Instances:       2
Base port:       1234
Timesteps:       10000
Learning rate:   5e-5
n_steps:         1024
```

Effective rollout:

```text
2 × 1024 = 2048 samples
```

Expected behavior:

```text
1. Two Webots instances open.
2. Both simulations start in fast mode.
3. Two external workers connect.
4. WebotsVecEnv reports two environments.
5. PPO begins collecting rollouts.
6. Training output appears in the manager.
7. Stopping training saves the model.
8. Workers disconnect.
9. Webots windows remain open.
```

If all of these occur, the current parallel pipeline is operational.

---

## 18. Models

The default parallel trainer uses:

```text
models/ppo_spot.zip
```

Stable-Baselines3 saves models as `.zip` files.

Model artifacts are excluded from Git.

When continuing training, the saved policy must be compatible with the current environment.

At minimum, check:

- action-space shape;
- observation-space shape.

A matching shape is not sufficient if the semantic meaning or normalization of observation components changed.

For that reason, major observation changes should normally start a new model.

---

## 19. TensorBoard

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

## 20. Experimental reproducibility

For every relevant experiment, record at least:

```text
Experiment identifier
Git commit
World file
Number of environments
Base port
Total timesteps
n_steps
Effective rollout size
Learning rate
Control step
Observation-space version
Reward configuration
Starting model
Random seed, when fixed
```

Example:

```text
Experiment: P01_parallel_baseline

Git commit:
<commit hash>

World:
static_spot_parallel.wbt

Algorithm:
PPO / MlpPolicy

Instances:
2

Base port:
1234

n_steps:
1024

Effective rollout:
2048

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

Starting model:
None
```

For a reproducible result, prefer recording the exact commit rather than only the branch name.

---

## 21. Common issues

### External controller cannot find a robot

Check that:

- the correct world was opened;
- the selected Webots port matches the worker;
- the world contains the expected external-controller configuration;
- the simulation is running.

### `webots-controller.exe` not found

Check the Webots installation path configured in the manager.

Typical installation:

```text
C:\Program Files\Webots
```

### Python environment not found

Verify the venv path in the manager.

From an activated PowerShell environment:

```powershell
(Get-Command python).Source
```

### Training immediately stops

Check for a stale:

```text
training/.stop_training
```

The corrected manager removes this automatically before starting training.

### Existing PPO model fails to load

Check whether the saved model was trained with the same action and observation spaces.

Also consider semantic changes to:

- observations;
- normalization;
- action mapping;
- reward;
- timing.

### Rollout changed after adding environments

Remember:

```text
rollout = n_envs × n_steps
```

Adjust `n_steps` if maintaining a constant total rollout is part of the experiment design.

---

## 22. Legacy single-instance runner

`controllers/main/main.py` remains in the repository as the older single-instance runner.

The current documented and actively developed workflow is the parallel infrastructure under:

```text
training/
```

and the manager under:

```text
app/
```

Do not assume the legacy runner shares all current model-path and management conventions unless it has been updated explicitly.

---

## 23. Development status

The infrastructure is still evolving.

Planned or active work includes:

- automatic checkpoints;
- experiment metadata;
- model selection;
- model compatibility checks;
- integrated TensorBoard controls;
- deterministic evaluation tooling;
- additional locomotion tasks;
- recovery/standing behaviors.

For reproducible results, always associate the experiment with a specific Git commit.
