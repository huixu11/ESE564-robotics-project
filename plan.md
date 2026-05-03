# Full Project Implementation Plan

## Summary

The project is implemented around a runnable language-conditioned basket-sorting
pipeline. The codebase includes a kinematic fallback, a class Panda MuJoCo scene,
language task parsing, `(dx, dy, dz, gripper)` actions, continuity-aware
differential IK, scripted FSM policies, demonstration collection, BC training,
evaluation, and captioned GIF output.

The final assignment-compliance path is `push_fsm`: it estimates variable object
state from RGB color segmentation and moves the target object through
MuJoCo-integrated pushing rather than manual pose teleportation.

## Implemented Steps

1. Create the project package and scripts.
2. Add default YAML configuration.
3. Add task parsing for target object, basket, and speed.
4. Add randomized basket-sorting environments.
5. Add a differential IK / continuity-aware controller.
6. Add a legacy pick-and-place FSM expert.
7. Add demonstration collection.
8. Add a lightweight state-feature behavior cloning baseline.
9. Add evaluation and captioned video generation scripts.
10. Add class Panda and YCB mesh assets.
11. Add RGB color perception for red/yellow object positions.
12. Add `push_fsm` as the final perception-driven pushing policy.
13. Add MuJoCo push-phase generalized force/velocity integration for object motion.
14. Add unit tests for parsing, controller continuity, perception, push policy, and rollouts.
15. Add an experimental contact-pusher path with a Panda hand pad, mocap contact pusher, base locking, and stabilized object collision geometry.
16. Make the colored baskets visual goal regions so pushed objects can enter them without being blocked by bin-wall collision.

## Assignment Compliance Fix

The assignment PDF requires perception-based world state and physics-based object
motion. The original pick-and-place baseline was useful for development, but it
used simulator object poses and a manual attachment rule. The final submitted
path is therefore:

1. Render RGB frames from the MuJoCo camera.
2. Segment the red cracker box and yellow mustard bottle.
3. Convert image centroids to tabletop coordinates with a calibrated affine map.
4. Use `push_fsm` to choose push waypoints from the language command and perceived object state.
5. Move the object during the push phase through MuJoCo integration, not direct pose setting.
6. Use simulator state only for randomized reset and final success scoring.
7. Report `push_fsm` results and keep the legacy pick-and-place FSM as a comparison path.

## Contact-Dynamics Caveat

I attempted to replace the planar push proxy with fully contact-driven pushing.
The implementation now includes the needed pieces for that path: a physical
Panda hand pad, an optional mocap contact pusher, direct-robot base locking
during MuJoCo stepping, object spawn-height fixes, and visual-only basket goal
regions. In testing, pure contact pushing was stable enough to render but not
reliable enough to solve the basket task; the object either slipped around the
pusher or rebounded during the forward push. Because the final project needs a
runnable evaluation, `configs/class_panda.yaml` keeps the stable planar
force/velocity push proxy enabled. The report should keep this caveat explicit
rather than claiming fully tuned gripper-object contact.

## Current Finalization Status

1. Required 5-task pushing video: `5/5`, average `41.60` steps.
2. Larger randomized pushing smoke check: `20/20`, average `38.75` steps.
3. Explicit language-command pushing videos generated:
   - red/cracker box to blue/left basket,
   - yellow/mustard bottle to green/right basket.
4. README and report now describe `push_fsm` as the final path.
5. Remaining manual step: compile `report/final_project_report.tex` to PDF with Overleaf, MiKTeX, or TeX Live.

## Commands

```powershell
python scripts/run_demo.py --episodes 3 --save-video videos/demo.gif
python scripts/collect_demos.py --episodes 20 --out data/demos
python scripts/train_bc.py --data data/demos --out models/state_linear_bc.npz
python scripts/evaluate.py --policy fsm --episodes 25 --out runs/fsm_eval.json
python scripts/evaluate.py --config configs/class_panda.yaml --policy push_fsm --episodes 5 --out runs/class_panda_push_eval_5.json --save-video videos/class_panda_push_eval_5.gif --video-episodes 5
python -m unittest discover -s tests
```
