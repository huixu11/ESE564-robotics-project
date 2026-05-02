# MuJoCo Assets

`scene.xml` is a local Panda-compatible MuJoCo stand-in with YCB-named task
objects. It is not the private class Panda/YCB asset package. It exists so the
project can run end-to-end when the class files are not available locally.

To use the class assets, replace `scene.xml` or update `env.mujoco.model_xml` in
a config override.

The adapter expects these names by default:

- end-effector site: `panda_hand_tcp`
- overhead camera: `overhead`
- arm joints: `panda_joint1` through `panda_joint7`
- arm position actuators: same names as the arm joints
- gripper actuators: `panda_finger_joint1`, `panda_finger_joint2`
- object bodies: `cracker_box`, `mustard_bottle`
- object free joints: `cracker_box_freejoint`, `mustard_bottle_freejoint`
- basket bodies: `left_basket`, `right_basket`

If the class files use different names, edit `configs/mujoco_template.yaml`
instead of changing code.
