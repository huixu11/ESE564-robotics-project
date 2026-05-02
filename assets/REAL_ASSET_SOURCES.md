# Real Asset Sources

The private "class Panda" files are course materials, not public assets. Look for
them in the course LMS, assignment release zip, class GitHub/GitLab repository,
or homework starter code. If they are not posted, ask the instructor/TA for:

- Panda MuJoCo/MJCF scene XML
- Panda mesh folder referenced by that XML
- homework IK wrapper or FK/Jacobian helper
- expected joint, actuator, site, and camera names

Public substitutes:

1. Panda robot model
   - Recommended public source: MuJoCo Menagerie Franka Emika Panda
   - URL: https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda
   - Important files/folders:
     - `franka_emika_panda/scene.xml`
     - `franka_emika_panda/panda.xml`
     - `franka_emika_panda/hand.xml`
     - `franka_emika_panda/assets/`

2. YCB object models
   - Official source: YCB Benchmarks Object Models
   - URL: https://www.ycbbenchmarks.com/object-models/
   - Model database URL listed by YCB:
     - http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/
   - Objects for this project:
     - `003_cracker_box`
     - `006_mustard_bottle`
   - Useful mesh variants inside each object folder:
     - `google_16k/`
     - `poisson/`
     - `tsdf/`
   - Common files:
     - `textured.obj`
     - `textured.mtl`
     - `texture_map.png`
     - `nontextured.stl`
     - `nontextured.ply`

3. Fixed YCB mesh mirror
   - Useful if the official S3 listing is hard to browse.
   - URL: https://huggingface.co/datasets/ll4ma-lab/ycb-fixed-meshes
   - Check the license and cite the original YCB dataset if you use this mirror.

Suggested local layout after downloading:

```text
assets/
  class_panda/
    assets/Panda/
      panda.mjcf
      meshes/
    ikfast/
    meshes/ycb/
      003_cracker_box/
        google_16k/
      006_mustard_bottle/
        google_16k/
```

The current project uses this class-homework layout in
`configs/class_panda.yaml` and `assets/class_panda/basket_scene.mjcf`.

After changing assets, update `configs/mujoco_template.yaml` so these names match
the XML:

- `env.mujoco.model_xml`
- `env.mujoco.ee_site`
- `env.mujoco.camera_name`
- `env.mujoco.arm_joint_names`
- `env.mujoco.arm_actuator_names`
- `env.mujoco.gripper_actuator_names`
- `env.mujoco.object_body_names`
- `env.mujoco.object_free_joint_names`
- `env.mujoco.basket_body_names`
