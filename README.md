# Pi 0.7-Inspired Basket Sorting

This folder contains a runnable scaffold for the project proposal:

- language task parsing,
- randomized basket-sorting environment,
- scripted FSM expert,
- differential IK / continuity-aware controller,
- demonstration collection,
- lightweight behavior cloning baseline,
- evaluation and GIF generation.

The current repository does not include the class Panda MuJoCo model, YCB meshes, homework IK wrapper, MuJoCo, PyTorch, or CLIP. For that reason, the implemented default environment is a deterministic kinematic fallback that uses the same interfaces planned for MuJoCo. The MuJoCo integration point is isolated in `src/basket_sorting/envs/mujoco_env.py`.

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

After copying the class Panda MuJoCo model and homework IK wrapper into this folder:

1. Replace or extend `MujocoBasketSortingEnv`.
2. Keep the same environment API as `KinematicBasketSortingEnv`.
3. Reuse `DifferentialIKController` if the wrapper exposes FK/Jacobian functions.
4. Keep the action interface as `(dx, dy, dz, gripper)`.

The FSM and data collection scripts should not need major changes if the MuJoCo environment preserves the same observation and step contracts.
