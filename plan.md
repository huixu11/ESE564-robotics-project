# Full Project Implementation Plan

## Summary

The project is implemented in stages around a reliable scripted baseline first. The current codebase includes a runnable kinematic fallback environment because the class MuJoCo assets and homework IK wrapper are not present in this folder. The fallback uses the same high-level interfaces expected from the final MuJoCo implementation: language task parsing, `(dx, dy, dz, gripper)` actions, continuity-aware differential IK, an FSM expert, demo collection, BC training, evaluation, and GIF output.

## Implemented Steps

1. Create the project package and scripts.
2. Add default YAML configuration.
3. Add task parsing for target object, basket, and speed.
4. Add a randomized basket-sorting environment.
5. Add a differential IK / continuity-aware controller.
6. Add a scripted FSM expert.
7. Add demonstration collection.
8. Add a lightweight state-feature behavior cloning baseline.
9. Add evaluation and video generation scripts.
10. Add unit tests for planner, controller continuity, and FSM rollouts.

## Remaining MuJoCo Steps

1. Copy the class Panda MuJoCo model and homework IK wrapper into `assets/`.
2. Implement `MujocoBasketSortingEnv` using the same API as the kinematic fallback.
3. Connect the homework FK/Jacobian or IK wrapper to `DifferentialIKController`.
4. Replace simple rendered fallback frames with MuJoCo RGB-D camera frames.
5. Re-run collection, BC training, and evaluation on the MuJoCo environment.

## Commands

```powershell
python scripts/run_demo.py --episodes 3 --save-video videos/demo.gif
python scripts/collect_demos.py --episodes 20 --out data/demos
python scripts/train_bc.py --data data/demos --out models/state_linear_bc.npz
python scripts/evaluate.py --policy fsm --episodes 25 --out runs/fsm_eval.json
python -m unittest discover -s tests
```
