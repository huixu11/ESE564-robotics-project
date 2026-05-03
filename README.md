# Pi 0.7-Inspired Basket Sorting

This folder contains a runnable scaffold for the project proposal:

- language task parsing,
- randomized basket-sorting environment,
- scripted FSM expert,
- differential IK / continuity-aware controller,
- demonstration collection,
- lightweight behavior cloning baseline,
- evaluation and GIF generation.

The current repository does not include the private class Panda MuJoCo model, true YCB meshes, or homework IK wrapper. For that reason, the default environment is a deterministic kinematic fallback, and `assets/mujoco/scene.xml` is a local Panda/YCB-named MuJoCo stand-in for setup and pipeline testing. The MuJoCo integration point is isolated in `src/basket_sorting/envs/mujoco_env.py`.

## Quick Start

Run the scripted demo:

```powershell
python scripts/run_demo.py --episodes 3 --save-video videos/demo.gif
```

Collect demonstrations:

```powershell
python scripts/collect_demos.py --episodes 20 --out data/demos
```

Use `--no-images` for larger smoke datasets when you only need state/action arrays.

Train the lightweight state + subtask-feature BC policy:

```powershell
python scripts/train_bc.py --data data/demos --out models/state_linear_bc.npz
```

Evaluate FSM:

```powershell
python scripts/evaluate.py --policy fsm --episodes 25 --out runs/fsm_eval.json
```

Evaluate the trained BC policy:

```powershell
python scripts/evaluate.py --policy linear_bc --model models/state_linear_bc.npz --episodes 25 --out runs/bc_eval.json
```

The NumPy BC policy is a smoke-test baseline, not the final CLIP model. If its success rate is below 60%, use the FSM executor for final evaluation, matching the proposal fallback.

Run tests:

```powershell
python -m unittest discover -s tests
```

## MuJoCo Integration

The repository includes a local MuJoCo stand-in at `assets/mujoco/scene.xml`. To use the private class assets instead:

1. Install MuJoCo: `python -m pip install mujoco`.
2. Copy the class scene to `assets/mujoco/scene.xml`, or edit `configs/mujoco_template.yaml`.
3. Make sure the joint/site/body/actuator names in `configs/mujoco_template.yaml` match the XML.
4. Validate the setup:

```powershell
python scripts/check_mujoco_setup.py --config configs/mujoco_template.yaml
```

5. Run the FSM with the MuJoCo config:

```powershell
python scripts/run_demo.py --config configs/mujoco_template.yaml --episodes 3 --save-video videos/mujoco_demo.gif
```

The FSM and data collection scripts use the same `(dx, dy, dz, gripper)` action interface for both environments.

## Class Panda Assets

Homework assets from `hw/hw2 (1).zip` were extracted into `assets/class_panda/`.
The scene at `assets/class_panda/basket_scene.mjcf` uses the homework Panda
MJCF/meshes and official YCB Google 16k visual meshes for:

- `003_cracker_box`
- `006_mustard_bottle`

### Object and Basket Color Convention

In the class Panda scene, the visual colors are:

- red object: YCB cracker box
- yellow object: YCB mustard bottle
- blue basket: left basket
- green basket: right basket

The task is still language-conditioned: the robot follows the instruction, not a
hard-coded color rule. For the clearest final demo convention, use:

- red/cracker box -> blue/left basket
- yellow/mustard bottle -> green/right basket

Example demo commands:

```powershell
.\.venv\Scripts\python.exe scripts\run_demo.py --config configs\class_panda.yaml --instruction "place the cracker box in the left basket" --save-video videos\red_cracker_to_blue_basket.gif
.\.venv\Scripts\python.exe scripts\run_demo.py --config configs\class_panda.yaml --instruction "place the mustard bottle in the right basket" --save-video videos\yellow_mustard_to_green_basket.gif
```

Random evaluation episodes can still choose any valid object/basket instruction,
including red to green or yellow to blue, because that tests whether the policy
uses the language command rather than memorizing a fixed color mapping.

Run the class-asset scene:

```powershell
.\.venv\Scripts\python.exe scripts\check_mujoco_setup.py --config configs\class_panda.yaml
.\.venv\Scripts\python.exe scripts\run_demo.py --config configs\class_panda.yaml --episodes 3 --save-video videos\class_panda_demo.gif
```

Collect demonstrations and train the smoke BC baseline:

```powershell
.\.venv\Scripts\python.exe scripts\collect_demos.py --config configs\class_panda.yaml --episodes 100 --out data\class_panda_demos --no-images
.\.venv\Scripts\python.exe scripts\train_bc.py --data data\class_panda_demos --out models\class_panda_linear_bc.npz
```

Current measured status on the class scene:

- FSM: `20/20` evaluation successes, average `88.15` steps.
- NumPy linear BC smoke baseline: `0/20` evaluation successes.

Use FSM as the final executor unless the BC policy is upgraded enough to pass the proposal's 60% threshold.
