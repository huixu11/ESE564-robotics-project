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
15. Add an experimental contact-pusher path with a Panda hand pad, mocap contact pusher, base locking, stabilized object collision geometry, and contact diagnostics.
16. Make the colored baskets visual goal regions so pushed objects can enter them without being blocked by bin-wall collision.
17. Add `configs/class_panda_contact.yaml` for contact-only pushing with the planar proxy disabled.
18. Add overhead color-marker perception and object-specific calibration for contact-level object localization.
19. Add quasi-static settle/lift/replan phases and contact-aware pusher goal offsets to `push_fsm`.

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

The first attempt to replace the planar push proxy with fully contact-driven
pushing failed because the object slipped around the pusher, rebounded during
the forward push, and stale perception caused the second segment to miss. The
current implementation fixes the main structural issues under
`configs/class_panda_contact.yaml`: the planar proxy is disabled, robot link
collisions are disabled except for the dedicated pusher, objects have simple
collision primitives and color markers, the pusher moves through slow MuJoCo
contact, and the FSM lifts and replans between lateral and forward pushes.

This contact-only path now passes the larger randomized acceptance gate
(`18/20`, success rate `0.900`, average `385.25` steps). It remains separate
from the default final path so the stable proxy baseline is still available for
comparison. `configs/class_panda.yaml` remains the stable submission config with
the planar force/velocity push proxy enabled, while
`configs/class_panda_contact.yaml` is the contact-rich no-proxy configuration.

## Research Findings for Contact-Rich Fix

The earlier contact attempt failed for a structural reason, not only because one
gain or friction value was wrong. The useful literature and MuJoCo documentation
point to a slower quasi-static pushing setup with simple collision geometry,
wide pusher contact, and physically stepped control.

1. Quasi-static pushing should be the first target.
   - Lynch and Mason's stable pushing work models pushing as contact mechanics
     with stable contact modes, not as arbitrary object displacement.
   - The pusher should move slowly enough that inertial effects are small and
     the object remains in stable contact during the push.
   - The policy should re-estimate object pose between push phases instead of
     assuming one open-loop push stays valid.

2. Contact geometry should be simple and intentional.
   - MuJoCo contact behavior is much easier to tune with convex primitives than
     with detailed visual meshes.
   - The YCB meshes should remain visual assets; the physical contacts should
     use simple boxes/cylinders matched to each object's footprint.
   - The pusher should be a wide face, fence, or shallow U-shaped guide rather
     than a small point or sphere, because point-like pushing makes side slip
     likely.

3. The controller must not teleport through contact.
   - Directly setting robot joint positions while in contact can inject energy
     and cause bounce, tunneling, or unrealistic object jumps.
   - The contact-only path should use MuJoCo position actuators or a mocap body
     connected through a weld equality, then step the simulation at small
     increments.
   - Arm gains, damping, actuator force limits, timestep, solver iterations,
     and settle steps are part of the contact system and should be tuned
     together.

4. MuJoCo contact settings need explicit tuning.
   - Start from simple friction values on table, object, and pusher rather than
     inheriting defaults.
   - Tune `condim`, friction, `solref`, `solimp`, timestep, and solver
     iterations on one deterministic scene before randomizing.
   - Add logging for contact pairs, object pose, pusher pose, and solver
     warnings so failures are attributable instead of visual guesses.

5. The report should cite the basis for the approach.
   - MuJoCo modeling and contact docs:
     https://mujoco.readthedocs.io/en/stable/modeling.html
   - MuJoCo computation/contact model:
     https://mujoco.readthedocs.io/en/2.1.2/computation.html
   - Lynch and Mason, "Stable Pushing: Mechanics, Controllability, and
     Planning":
     https://www.ri.cmu.edu/publications/stable-pushing-mechanics-controllability-and-planning/
   - Planar pushing survey:
     https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2020.00008/full

## Plan to Solve Fully Contact-Rich Manipulation

The practical way to remove the proxy is to treat contact pushing as its own
staged engineering task, not a small parameter tweak. The current proxy path
should stay as the fallback while this is developed under a separate config such
as `configs/class_panda_contact.yaml`.

1. Create a separate contact-only config.
   - Disable `physics_push.enabled`.
   - Use a separate output path for videos/results so the reliable final
     artifacts are not overwritten.
   - Keep the same language parser, RGB perception, and `push_fsm` interface.

2. Simplify the contact task geometry.
   - Use one stable collision primitive per object, with the YCB mesh kept as a
     visual overlay.
   - Keep baskets as visual goal regions so the hard problem is object transport,
     not catching on basket walls.
   - Add a real collision pusher rigidly attached to the Panda hand or fingers.
   - Prefer a wide rectangular pusher face first; upgrade to a shallow U-shaped
     pusher only if side slip remains the dominant failure.

3. Switch from direct joint teleporting to physically stepped control.
   - Use MuJoCo position actuators or a mocap/welded end-effector target.
   - Do not use direct object forces, direct object velocity writes, or direct
     in-contact joint teleportation in the contact-only config.
   - Tune arm gains, damping, force limits, and timestep/substeps until the
     pusher can approach a stationary object without making it explode or tunnel.
   - Add settle steps after reset, after approach, after lateral alignment, and
     after the forward push.

4. Use axis-aligned multi-stage pushing.
   - First push laterally to align the object with the requested basket x
     coordinate.
   - Re-approach from below the object.
   - Push forward into the basket goal region.
   - Add short settle pauses after each push segment so perception can update
     from the new object pose.
   - Keep pusher speed low enough for quasi-static contact, then increase only
     after deterministic tests are stable.

5. Tune contact parameters with one deterministic seed first.
   - Start with one object and one command, for example mustard to right.
   - Log object pose, pusher pose, contact pairs, and success after every phase.
   - Sweep pusher width, pusher speed, table/object/pusher friction, `condim`,
     solver iterations, and contact softness in small batches.
   - Only expand to randomized starts after the deterministic case succeeds
     repeatedly.

6. Add fallback recovery behaviors.
   - If the object slips around the pusher, re-estimate its RGB position and
     re-plan the next push segment.
   - If the object is near the basket x range but below the basket y range,
     skip lateral alignment and run only the forward push.
   - If the object moves out of workspace bounds, terminate as failure rather
     than hiding the issue.

7. Run acceptance tests before replacing the default.
   - Minimum deterministic gate: same object, same start, same basket succeeds
     `10/10` without the proxy.
   - Required gate: `5/5` video successes with visible contact.
   - Stronger gate: at least `18/20` randomized successes without the proxy.
     Current status: passed with `18/20`.
   - Passing the gate makes contact-only pushing usable as the contact-rich
     evaluation path; keep the proxy config available as a reliability baseline.

## Current Finalization Status

1. Required 5-task pushing video: `5/5`, average `41.60` steps.
2. Larger randomized pushing smoke check: `20/20`, average `38.75` steps.
3. Experimental contact-only randomized gate:
   `18/20`, success rate `0.900`, average `385.25` steps, using
   `configs/class_panda_contact.yaml`.
4. Perception sanity metrics generated:
   - Stable demo-view config: `99.5%` detection, mean `7.04 cm`,
     max `15.49 cm` tabletop error over 100 randomized resets.
   - Contact overhead-marker config: `65.5%` detection, mean `2.11 cm`,
     max `6.03 cm` error over detected objects.
5. Explicit language-command pushing videos generated:
   - red/cracker box to blue/left basket,
   - yellow/mustard bottle to green/right basket.
6. README and report now describe `push_fsm` as the final path.
7. Remaining manual step: compile `report/final_project_report.tex` to PDF with Overleaf, MiKTeX, or TeX Live.
8. Contact-rich MP4 generated for submission:
   `videos/class_panda_contact_eval_5.mp4`.

## Remaining Submission Plan

Since the final PDF will be compiled in Overleaf, the remaining local work should
focus on code/video evidence and packaging.

1. Generate a submission video in a standard format. Completed.
   - Keep the existing GIFs for quick inspection.
   - The contact-rich five-task run has been exported to
     `videos/class_panda_contact_eval_5.mp4`.
   - The report figure snapshots now come from the contact-rich video.

2. Add a perception sanity metric. Completed.
   - `scripts/perception_sanity.py` runs over randomized resets.
   - It compares color-segmentation object estimates against simulator object poses.
   - The report Results section now includes detection rate plus mean and max
     tabletop localization error.
   - This strengthens the claim that the policy uses camera-derived world state
     and identifies perception occlusion as a contact-only failure source.

3. Regenerate final evaluation artifacts after any code/config change.
   - Five-task video: `runs/class_panda_push_eval_5.json` and
     `videos/class_panda_push_eval_5.gif`.
   - Contact-rich five-task video:
     `videos/class_panda_contact_eval_5.mp4`.
   - Randomized smoke evaluation: `runs/class_panda_push_eval_20.json`.
   - Explicit language demos:
     `videos/push_language_red_to_blue.gif` and
     `videos/push_language_yellow_to_green.gif`.

4. Keep the dynamics caveat explicit.
   - State that the stable default remains the proxy baseline, while the
     contact-only config is now a passing no-proxy variant.
   - State that the final default uses camera perception plus MuJoCo-integrated
     planar push proxy.
   - State that the contact-pusher path exists, disables the proxy, and has a
     passing `18/20` randomized gate.

5. Create the final Gradescope zip.
   - Include source code, configs, assets needed to run the class Panda scene,
     final report PDF from Overleaf, and video artifacts.
   - Exclude `.venv`, `__pycache__`, temporary logs, and large unrelated files.
   - Before zipping, run `python -m unittest discover -s tests` and one
     `push_fsm` evaluation command from the README.

## Commands

```powershell
python scripts/run_demo.py --episodes 3 --save-video videos/demo.gif
python scripts/collect_demos.py --episodes 20 --out data/demos
python scripts/train_bc.py --data data/demos --out models/state_linear_bc.npz
python scripts/evaluate.py --policy fsm --episodes 25 --out runs/fsm_eval.json
python scripts/evaluate.py --config configs/class_panda.yaml --policy push_fsm --episodes 5 --out runs/class_panda_push_eval_5.json --save-video videos/class_panda_push_eval_5.gif --video-episodes 5
python -m unittest discover -s tests
```
