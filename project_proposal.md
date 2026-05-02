Pi 0.7-Inspired Language-Conditioned Basket Sorting with a Panda Arm
ESE 564 Final Project Proposal (Revised)
Team member: Hui Xu

Pi 0.7: https://www.pi.website/blog/pi07

1. Problem

The robot must move a target household object from a tabletop into the correct basket based on a natural language instruction. The target object pose is randomized, the basket pose varies across several reachable goal locations, and at least one distractor obstacle is also randomized. A trial is successful if the correct object ends inside the instructed basket within a fixed time limit. This satisfies all four course requirements: the task is physics-based, perception-driven, uses a restricted goal region (the basket interior), and involves mesh-based YCB objects.

2. Robot Platform, Simulator, Object Types, and URDF/XML/MJCF Files

1. Tabletop Franka Panda arm with a parallel-jaw gripper, using the MuJoCo model from class.
2. MuJoCo provides stable contact dynamics and RGB-D rendering.
3. YCB cracker box and mustard bottle are the mesh-based objects. The basket defines the restricted goal region.
4. The existing Panda MJCF will be adapted to include YCB mesh objects, basket goal regions, and distractor obstacles. Each imported object will require scale verification, collision geometry simplification, and mass/friction tuning.

3. Approach

The system is a hybrid two-tier architecture inspired by Pi 0.7's separation of a high-level language planner and a low-level action policy, using diverse conditioning.

Pipeline overview:

- Natural language instruction + speed tag
- LLM high-level planner
- Subtask language commands, such as "grasp cracker box" or "place in left basket"
- BC (Behavior Cloning) low-level policy or scripted FSM (Finite State Machine)
- End-effector delta / target end-effector velocity + gripper command
- Differential IK or continuity-aware IK layer
- Joint-space PD (Proportional-Derivative) control
- Panda in MuJoCo

High-level planner

A GPT-4o API call parses the instruction and speed tag into a sequence of subtask strings, such as "approach", "grasp", and "place". This maps to Pi 0.7's use of language coaching to decompose long-horizon tasks.

Low-level BC policy

A small neural network trained via behavior cloning on automatically generated demonstrations:

- Image encoder: frozen CLIP ViT-B/32 (overhead RGB image)
- Language encoder: frozen CLIP text encoder (current subtask string)
- Metadata input: scalar speed value (0 = careful, 1 = fast), concatenated with embeddings
- Output: 4-dimensional end-effector delta (dx, dy, dz, gripper) per timestep

The CLIP backbone means no vision or language encoder is trained from scratch. This mirrors Pi 0.7's use of language and metadata conditioning to steer a single model toward different behaviors.

Scripted FSM fallback (Expert Demonstrator and Fallback)

The FSM (approach -> grasp -> lift -> transfer -> place) serves two roles: (1) as an expert demonstrator for BC data collection, and (2) as a fallback executor if BC underperforms.

Importantly, the FSM does not directly output motor commands. Each FSM phase uses simple geometric motion primitives to generate target end-effector velocities or end-effector delta actions. These are passed through the same differential IK / continuity-aware IK and low-level controller as the BC policy. The executed end-effector-level actions are recorded as BC training targets.

Control Architecture

Neither the BC policy nor the FSM outputs motor commands directly. The full control hierarchy is:

- BC policy or FSM phase outputs target end-effector delta / end-effector velocity (dx, dy, dz) + gripper command.
- Differential IK or warm-started IK converts the desired end-effector motion into Panda joint targets, seeded from the current joint configuration.
- A continuity check rejects large joint jumps and keeps consecutive solutions on the same local IK branch when possible.
- MuJoCo joint-space PD control / position control tracks the joint targets.
- Gripper open/close is applied as a separate command.

Primary implementation: differential IK or continuity-aware IK-based position control. This addresses the risk that independently solved IK targets can switch between incompatible elbow-up and elbow-down configurations for nearby end-effector poses. OSC (Operational Space Control) may be explored as a comparison if time permits, but continuity-aware IK + joint-space PD is the baseline to keep scope manageable. If this baseline proves unstable, the backup plan is to train the low-level policy to output joint-position deltas directly.

Subtask completion detection

Subtask transitions are rule-based: gripper force threshold triggers grasp completion; end-effector height above basket triggers placement. This is simple and reliable.

4. Learning-Based Components and Data

The BC policy is trained on automatically generated demonstrations, with no manual annotation required.

Data generation procedure:

Run the scripted FSM for 500-1000 trials with full randomization (object pose, basket pose, distractor, instruction, speed tag). Each FSM phase uses geometric motion primitives -> end-effector delta / velocity -> differential IK or continuity-aware IK -> MuJoCo execution.

At each timestep, record: overhead RGB image, current subtask label string, speed metadata value, and the end-effector delta action executed by the IK + controller layer.

Keep only timesteps from successful trials where the FSM reaches the basket and releases the object.

Apply image augmentation (random crop, brightness/contrast jitter) to improve policy generalization.

Training is a supervised regression on the MLP head with frozen CLIP encoders, using MSE loss. Expected training time is under 2 hours on a single GPU.

5. Model-Based Components

Kinematics: Panda MuJoCo model from class. End-effector commands are converted to joint targets using differential IK or warm-started IK through the existing homework wrapper when possible (desired end-effector motion -> joint-position update -> MuJoCo PD controller).

Object meshes: YCB dataset (public). The same meshes are used for MuJoCo rendering and, if needed, for ICP-based pose estimation.

Perception: Depth-based table removal + color/connected-component segmentation from the overhead RGB-D camera. Object pose estimated from segmented point cloud centroid and PCA orientation, or ICP against the known YCB mesh.

6. Evaluation Plan

50-100 randomized trials, reporting:

- Success rate: correct object inside instructed basket within time limit
- Completion time: average wall-clock and simulator time per trial
- BC vs FSM comparison: success rate of each low-level executor under identical conditions
- Speed metadata effect: normal vs. fast trials
- Intermediate diagnostics: segmentation quality, pose estimation error, grasp success rate, placement accuracy

Randomization covers object pose, basket pose, distractor placement, instruction, and speed tag. The 5-video requirement will be satisfied by showing 5 trials varying these parameters.

Fallback trigger: if BC grasp success rate is below 60%, the FSM executor is used for final evaluation. The two-tier architecture is preserved in either case.
