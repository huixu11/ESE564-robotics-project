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
