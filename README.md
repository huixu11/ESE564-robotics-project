# Pi 0.7-Inspired Basket Sorting

This folder contains a runnable scaffold for the project proposal:

- language task parsing,
- randomized basket-sorting environment,
- scripted pick-and-place and pushing FSM experts,
- color-based camera perception for the final MuJoCo path,
- differential IK / continuity-aware controller,
- demonstration collection,
- lightweight behavior cloning baseline,
- evaluation and GIF generation.

The default environment is a deterministic kinematic fallback for quick tests. The final class Panda path uses assets in `assets/class_panda/`, color perception from rendered RGB frames, and `push_fsm` pushing in MuJoCo.

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

Run an explicit language command with the final pushing executor:

```powershell
python scripts/evaluate.py --config configs/class_panda.yaml --policy push_fsm --instruction "place the mustard bottle in the right basket" --episodes 3 --save-video videos/language_command_demo.gif
```

Saved GIFs include a top caption such as `Language: place the mustard bottle in
the right basket`, and the JSON output records the same instruction for each
episode. This makes it clear which language command the robot is following in
the video.

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

The same language command can also be passed to the main evaluation script:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --config configs\class_panda.yaml --policy push_fsm --instruction "place the cracker box in the left basket" --episodes 3 --out runs\push_language_red_to_blue.json --save-video videos\push_language_red_to_blue.gif --video-episodes 3
.\.venv\Scripts\python.exe scripts\evaluate.py --config configs\class_panda.yaml --policy push_fsm --instruction "place the mustard bottle in the right basket" --episodes 3 --out runs\push_language_yellow_to_green.json --save-video videos\push_language_yellow_to_green.gif --video-episodes 3
```

The saved GIFs show the command as an on-frame language caption, and the
corresponding JSON files store the instruction strings under each episode's
`instruction` field.

Random evaluation episodes can still choose any valid object/basket instruction,
including red to green or yellow to blue, because that tests whether the policy
uses the language command rather than memorizing a fixed color mapping.

Run the class-asset scene:

```powershell
.\.venv\Scripts\python.exe scripts\check_mujoco_setup.py --config configs\class_panda.yaml
.\.venv\Scripts\python.exe scripts\evaluate.py --config configs\class_panda.yaml --policy push_fsm --episodes 5 --out runs\class_panda_push_eval_5.json --save-video videos\class_panda_push_eval_5.gif --video-episodes 5
```

Collect demonstrations and train the smoke BC baseline:

```powershell
.\.venv\Scripts\python.exe scripts\collect_demos.py --config configs\class_panda.yaml --episodes 100 --out data\class_panda_demos --no-images
.\.venv\Scripts\python.exe scripts\train_bc.py --data data\class_panda_demos --out models\class_panda_linear_bc.npz
```

Current measured status on the class scene:

- Final `push_fsm`: `5/5` required video successes, average `41.60` steps.
- Final `push_fsm`: `20/20` randomized smoke-evaluation successes, average `38.75` steps.
- Experimental contact-only `push_fsm` in `configs/class_panda_contact.yaml`: `18/20` randomized gate successes, success rate `0.900`, average `385.25` steps.
- Experimental contact-rich TAMP grasp path in `configs/class_panda_grasp.yaml` with `--policy tamp_grasp`: `5/5` randomized successes, average `242.60` steps.
- Legacy pick-and-place FSM: `300/300` successes, but it uses a manual attachment rule and is not the final assignment-compliance path.
- NumPy linear BC smoke baseline: `0/20` evaluation successes.

Use `push_fsm` as the final executor for the report and submission video. It
uses camera color perception and MuJoCo-integrated object motion. The stable
submission config still uses a planar force/velocity push proxy during the push
phase; an experimental contact-pusher implementation is present in the scene and
environment code. The contact-only path now passes the larger `18/20`
randomized gate, but it remains separate from the default so the stable proxy
baseline is still available for comparison.

Run the experimental contact-rich path:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --config configs\class_panda_contact.yaml --policy push_fsm --episodes 5 --out runs\class_panda_contact_eval_5.json
.\.venv\Scripts\python.exe scripts\evaluate.py --config configs\class_panda_contact.yaml --policy push_fsm --episodes 20 --out runs\class_panda_contact_eval_20.json
```

This config disables the planar push proxy, uses an overhead color-marker
perception camera for object localization, and moves objects only through
MuJoCo contact with the dedicated pusher body.

The contact-rich five-task submission video is committed at:

```text
videos/class_panda_contact_eval_5.mp4
```

Run the perception sanity metric:

```powershell
.\.venv\Scripts\python.exe scripts\perception_sanity.py --config configs\class_panda.yaml --samples 100 --out runs\class_panda_perception_sanity_100.json
.\.venv\Scripts\python.exe scripts\perception_sanity.py --config configs\class_panda_contact.yaml --samples 100 --out runs\class_panda_contact_perception_sanity_100.json
```

This compares color-segmentation tabletop estimates against MuJoCo object poses
after randomized resets.

Run the experimental TAMP grasp path:

```powershell
.\.venv\Scripts\python.exe scripts\check_mujoco_setup.py --config configs\class_panda_grasp.yaml
.\.venv\Scripts\python.exe scripts\evaluate.py --config configs\class_panda_grasp.yaml --policy tamp_grasp --episodes 5 --out runs\class_panda_grasp_collision_eval_5.json --save-video videos\class_panda_grasp_collision_eval_5.gif --video-episodes 5
```

This path parses the same language commands into a symbolic
`place(object, basket)` goal, samples object-specific grasp candidates, checks
workspace/gripper feasibility, then executes
`pre_grasp -> grasp -> close -> lift -> transfer -> place -> release -> retreat`.
It is useful evidence for a grasp/TAMP direction. The grasp config enables a
dedicated MuJoCo contact gripper: a dynamic tool body welded to a mocap target,
two side pads, a top adhesive pad, and a MuJoCo adhesion actuator that turns on
while the gripper is closed. This path does not use the manual attachment
fallback, but it is still a simplified adhesive gripper rather than fully tuned
Franka finger contact. The grasp config also enables Panda link collision geoms
against the YCB objects, following the planning-scene idea used by MoveIt/TAMP:
only the intended gripper-pad contacts are allowed, while arm links can no
longer pass through objects silently. The visible Panda hand is offset above the
contact gripper target so the hand does not drive through the cracker box while
the lower contact pads perform the grasp.

The contact-rich grasp MP4 artifact is:

```text
videos/class_panda_grasp_collision_eval_5.mp4
```
