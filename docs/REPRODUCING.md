# Reproducing Spot PPO Experiments

This document describes how to reproduce the Spot reinforcement learning experiments using Webots, Gymnasium, and Stable-Baselines3.

The instructions focus primarily on the current parallel PPO training architecture.

---

# 1. System requirements

The current development environment uses Windows and PowerShell.

Required software:

* Python
* Webots
* Git

Python dependencies are defined in:

```text
requirements.txt
```

The main dependencies include:

* NumPy
* Gymnasium
* Stable-Baselines3
* TensorBoard
* PySide6

The Webots Python `controller` module is provided by Webots itself and should not be installed separately from PyPI.

---

# 2. Clone the repository

```powershell
git clone https://github.com/Energetic0oak32/spotWebots.git
cd spotWebots
```

---

# 3. Create the Python environment

Create a virtual environment:

```powershell
python -m venv env
```

Activate it:

```powershell
.\env\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Verify the Python executable being used:

```powershell
(Get-Command python).Source
```

The output should point to the project's virtual environment, for example:

```text
...\spotWebots\env\Scripts\python.exe
```

---

# 4. Webots installation

Webots must be installed independently.

A typical Windows installation is located at:

```text
C:\Program Files\Webots
```

The project uses both:

```text
webots.exe
```

and:

```text
webots-controller.exe
```

The external controller executable is normally located under:

```text
C:\Program Files\Webots\msys64\mingw64\bin\webots-controller.exe
```

The Spot Training Manager verifies these paths before launching training.

---

# 5. External controller configuration

Parallel training uses Webots external controllers.

Each Webots instance listens on a different port, while a corresponding `webots-controller` process connects the simulation to a Python `parallel_worker.py`.

Example:

```text
Webots instance 0
port 1234
      |
webots-controller
      |
parallel_worker.py
      |
SpotEnv
```

A second instance may use:

```text
1235
```

A third:

```text
1236
```

and so on.

The ports are generated from:

```text
port = base_port + instance_index
```

With:

```text
base_port = 1234
instances = 4
```

the resulting ports are:

```text
1234
1235
1236
1237
```

## `runtime.ini`

For the external-controller workflow used by parallel training, a controller-local `runtime.ini` is not required.

If a local `runtime.ini` causes Webots external-controller errors, it can be temporarily disabled, for example:

```powershell
Rename-Item runtime.ini runtime.ini.disabled
```

Only do this from the directory where that file actually exists.

---

# 6. Environment definition

The current `SpotEnv` uses a continuous action space:

```text
Box(-1, 1, shape=(12,))
```

Each action controls one of the robot's 12 actuated joints.

The current observation space is:

```text
Box(-1, 1, shape=(48,))
```

The observation contains:

```text
12 normalized joint positions
12 normalized joint velocities
 6 orientation features
 3 normalized angular velocities
 3 normalized local linear velocities
12 previous action values
---------------------------------
48 values
```

Orientation is represented using sine and cosine features for roll, pitch, and yaw.

The previous action is included in the observation so that the policy has information about the command issued during the previous control step.

---

# 7. Control frequency

The environment currently uses:

```text
control_step = 32 ms
```

A new PPO action is therefore applied every 32 milliseconds of simulated time.

The Webots physics timestep is obtained directly from the world using:

```python
robot.getBasicTimeStep()
```

while the reinforcement-learning control interval is defined independently by `control_step`.

---

# 8. Episode termination

The current maximum episode length is:

```text
1000 control steps
```

Episodes may terminate earlier if the robot is considered fallen.

Fall detection currently considers robot orientation and body height.

Several consecutive invalid steps are required before declaring a fall, reducing sensitivity to short transient movements.

---

# 9. Reward function

The current locomotion reward combines several components.

Positive components include:

* alive reward;
* forward local velocity.

The forward reward is scaled by:

* robot stability;
* body-height factor.

Penalties currently include:

* vertical velocity;
* sudden changes in vertical velocity;
* angular velocity;
* large changes between consecutive actions.

A larger penalty is applied when the robot falls.

The reward function is implemented in:

```text
controllers/main/reward.py
```

Because reward design is actively being developed, experiments should record the Git commit used during training.

---

# 10. Running the Spot Training Manager

From the repository root:

```powershell
python .\app\gui.py
```

The interface allows configuration of:

```text
Webots installation
World file
Python environment
Controller directory
Number of instances
Base port
Total timesteps
Learning rate
n_steps
```

Before running an experiment, use:

```text
Verify configuration
```

The manager checks the required executables, world, Python environment, controller files, and training files.

---

# 11. Starting parallel training

A normal GUI workflow is:

```text
1. Open Spot Training Manager

2. Configure the experiment

3. Verify configuration

4. Start Webots

5. Start training
```

For two environments and base port `1234`, the manager launches:

```text
Webots 1234
Webots 1235
```

The trainer then creates one external controller worker for each Webots process.

---

# 12. PPO rollout size

In Stable-Baselines3 PPO, `n_steps` is collected by each environment.

Therefore:

```text
effective_rollout = number_of_environments * n_steps
```

Examples:

```text
1 environment  × 2048 = 2048 samples
2 environments × 1024 = 2048 samples
4 environments ×  512 = 2048 samples
8 environments ×  256 = 2048 samples
```

This is important when comparing experiments.

Increasing the number of environments without changing `n_steps` also increases the total amount of experience used in each PPO rollout.

For example:

```text
2 environments × 2048 = 4096 samples
```

This is not necessarily wrong, but it changes the training configuration.

---

# 13. Recommended baseline reproduction

A useful parallel baseline that preserves a 2048-sample rollout is:

```text
Algorithm:           PPO
Policy:              MlpPolicy

Instances:           2
Base port:           1234

n_steps:             1024
Effective rollout:   2048

Learning rate:       5e-5
Control step:        32 ms

Observation space:   48
Action space:        12
```

For a short validation run, use:

```text
Total timesteps: 10,000
```

This run is intended only to verify that the complete training pipeline works.

For real training, use a larger timestep budget appropriate to the experiment.

---

# 14. Parallel training architecture

The complete communication path is:

```text
Spot Training Manager
        |
        | launches
        v
   Webots processes
        |
        | ports 1234+
        v
webots-controller processes
        |
        v
 parallel_worker.py
        |
        v
     SpotEnv
        ^
        |
 multiprocessing.connection
        |
        v
   WebotsVecEnv
        |
        v
       PPO
```

`WebotsVecEnv.step_async()` sends actions to all environments before waiting for their responses.

Conceptually, with two environments:

```text
actions shape:
(2, 12)

observations shape:
(2, 48)
```

Both Webots simulations can therefore advance independently while sharing the same PPO policy.

---

# 15. Models

Trained models are stored locally under:

```text
models/
```

Stable-Baselines3 models are stored as `.zip` files.

Model files are intentionally excluded from Git because they are generated training artifacts and may become large.

When continuing training from an existing model, the environment's observation and action spaces must be compatible with the saved policy.

Changes to the semantic meaning of observations may also make an old model unsuitable even when the numerical observation dimensions remain identical.

---

# 16. TensorBoard

Training logs are stored under:

```text
logs/tensorboard/
```

To inspect them:

```powershell
tensorboard --logdir .\logs\tensorboard
```

TensorBoard can be used to inspect metrics including:

```text
episode reward
episode length
learning rate
approximate KL divergence
clip fraction
policy loss
value loss
explained variance
```

---

# 17. Experimental reproducibility

For every relevant training experiment, record at least:

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
Model starting point
Random seed, when fixed
```

A minimal experiment record might look like:

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

Recording the Git commit is especially important because `SpotEnv`, observations, reward terms, normalization, and termination criteria may change during development.

---

# 18. Smoke test

Before starting a long experiment, run a short training session.

Suggested configuration:

```text
Instances:       2
Base port:       1234
Timesteps:       10000
Learning rate:   5e-5
n_steps:         1024
```

Expected behavior:

```text
Both Webots instances open.
Both workers connect successfully.
WebotsVecEnv reports two environments.
PPO begins collecting rollouts.
Training metrics appear in the manager log.
The model is saved after training completes.
```

If these conditions are met, the parallel training pipeline is operational.

---

# 19. Common issues

## External controller cannot find a robot

Check that:

* the correct world is open;
* the world is running;
* the external controller configuration is correct;
* the selected Webots port matches the worker port.

## `webots-controller.exe` not found

Check the configured Webots installation directory.

A typical path is:

```text
C:\Program Files\Webots
```

## Python environment not found

Verify:

```powershell
(Get-Command python).Source
```

and confirm that the manager points to the intended project environment.

## Different rollout size after increasing environments

Remember:

```text
rollout = n_envs × n_steps
```

Change `n_steps` if maintaining the same total rollout size is important for the experiment.

## Existing PPO model fails to load

Check whether the saved model was trained with the same observation and action spaces.

Old models may become incompatible after changes to `SpotEnv`.

---

# 20. Development status

The training infrastructure is still evolving.

Areas planned or under development include:

* automatic checkpoints;
* experiment metadata;
* model selection and compatibility checking;
* improved training stop/resume behavior;
* integrated TensorBoard controls;
* automated evaluation;
* additional locomotion tasks;
* transfer between locomotion and recovery behaviors.

For reproducible results, always associate an experiment with a specific Git commit.
