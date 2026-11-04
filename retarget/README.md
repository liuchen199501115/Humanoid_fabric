# G1 机器人动作重定向推理

本仓库提供开箱即用的 Unitree G1 人形机器人动作重定向工具。输入人体动作
H5，即可通过内置模型生成 G1 的根节点轨迹和 29 自由度关节动作。

主要功能：

- 使用内置 checkpoint 在 CPU 或 CUDA 上完成动作重定向
- 导出机器人控制使用的 PKL 和通用 BVH 动作文件
- 使用 MuJoCo 交互查看动作或录制 1920×1080 MP4
- 提供多组可直接运行的 CMU 示例动作
- 支持将 Motion-X、AMASS/CMU、HumanML3D 和 G1 动作转换为所需 H5 格式

## 环境安装

```bash
conda env create -f environment.yml
conda activate robot-retargeting
```

`environment.yml` 通过 `cpuonly` 固定 CPU 版 PyTorch。若要使用 CUDA，请按
本机驱动/CUDA 版本安装对应的 PyTorch GPU 包，然后运行时指定
`--device cuda`。

CUDA 环境示例：

```bash
conda env create -f environment-cuda.yml
conda activate robot-retargeting-cuda
python retarget.py --device cuda
```

如果本机驱动不支持 CUDA 12.1，请把 `environment-cuda.yml` 里的 `pytorch-cuda=12.1` 改成机器支持的版本。

## 可用动作

仓库当前直接提供以下人体动作 H5，可直接传给 `retarget.py --input-h5`：

| 动作 ID | 输入文件 | 内容 |
| --- | --- | --- |
| `cmu_35_01_walk` | `examples/human_motion.h5` | CMU 普通直行步行 |
| `cmu_35_17_run_jog` | `examples/motions/cmu_35_17_run_jog.h5` | CMU 跑步 / 慢跑 |
| `cmu_49_02_jump_hop` | `examples/motions/cmu_49_02_jump_hop.h5` | CMU 连续跳跃 / 单脚跳 |
| `cmu_49_19_one_leg_balance` | `examples/motions/cmu_49_19_one_leg_balance.h5` | CMU 单脚平衡 |
| `cmu_144_20_punch` | `examples/motions/cmu_144_20_punch.h5` | CMU 连续出拳 |

`examples/g1_reference.h5` 是 G1 的目标骨架、T-pose、关节名和尺度参考，不是
人体动作，也不能单独作为演示动作播放。

查看仓库内置的人体动作文件：

```bash
find examples -type f -name '*.h5' ! -name 'g1_reference.h5' -print
```

仓库还支持以下动作来源，但原始数据不随仓库分发，需要先用对应工具转成 H5：

| 动作来源 | 支持的输入 | 转换工具 | 常见内容 |
| --- | --- | --- | --- |
| Motion-X | SMPL-X 322 维 `.npy` | `tools/motionx_to_h5.py` | 舞蹈、武术、健身、日常动作等 |
| AMASS/CMU | SMPL-X `stageii.npz` | `tools/amass_cmu_to_h5.py` | 步行、跑步、转向、跳跃等 |
| MCM-LDM/HumanML3D | `(T, 22, 3)` 关节点 `.npy` | `tools/mcmldm_npy_to_h5.py` | 文本生成或数据集人体动作 |
| PHUMA/G1 | `root_trans/root_ori/dof_pos` `.npy` | `tools/make_g1_h5.py` | G1 目标骨架或机器人参考动作 |

## 快速运行

默认动作是 `cmu_35_17_run_jog`。直接运行：

```bash
python retarget.py --device auto
```

默认会读取 `examples/motions/cmu_35_17_run_jog.h5`，使用
`examples/g1_reference.h5` 作为 G1 目标骨架参考，并输出：

- `outputs/B2A_cmu_35_17_run_jog.pkl`：机器人控制和 MuJoCo 查看格式
- `outputs/B2A_cmu_35_17_run_jog.bvh`：BVH 查看器通用格式

运行其它内置动作时，从上表选择动作 ID 和输入文件。例如：

```bash
ACTION=cmu_49_02_jump_hop
python retarget.py \
  --input-h5 "examples/motions/${ACTION}.h5" \
  --target-h5 examples/g1_reference.h5 \
  --output-dir "outputs/${ACTION}" \
  --tag "${ACTION}" \
  --device auto
```

PKL 导出会把网络产生的局部 SO(3) 旋转投影到 URDF 单自由度关节轴，按
URDF 关节顺序写出，并默认将越界角度限制在 URDF 范围内。若需要检查未经
限位裁剪的原始投影结果，可加 `--no-clamp-limits`。

## MuJoCo 可视化

将内置动作渲染为 MP4：

```bash
python view_motion_follow.py \
  --motion-file outputs/B2A_cmu_35_17_run_jog.pkl \
  --video-path outputs/B2A_cmu_35_17_run_jog.mp4 \
  --width 1920 \
  --height 1080
```

默认录制分辨率是 1920×1080；可用 `--width` 和 `--height` 覆盖。视频默认
继承 PKL 中的动作帧率。可以用 `--fps` 修改视频采样率；播放器会同步修改
采样步长，不会改变动作时长或播放速度。

查看其它动作时，使用同一个动作 ID：

```bash
ACTION=cmu_49_02_jump_hop
python view_motion_follow.py \
  --motion-file "outputs/${ACTION}/B2A_${ACTION}.pkl" \
  --video-path "outputs/${ACTION}/B2A_${ACTION}.mp4"
```

打开交互式 MuJoCo viewer：

```bash
python view_motion_follow.py \
  --motion-file outputs/B2A_cmu_35_17_run_jog.pkl \
  --no-video
```

如果是无显示服务器，可能需要在运行前设置：

```bash
export MUJOCO_GL=egl
```

## 使用自定义动作

把转换后的任意人体 H5 传给 `--input-h5`，并用 `--tag` 设置输出动作名：

```bash
python retarget.py \
  --input-h5 /path/to/human_motion.h5 \
  --target-h5 examples/g1_reference.h5 \
  --output-dir outputs/my_motion \
  --tag my_motion \
  --device auto
```

`--target-h5` 只用于提供机器人骨架拓扑、T-pose、关节名和尺度信息；通常可以一直使用仓库里的 `examples/g1_reference.h5`。
默认继承输入 H5 的 `fps` 属性；旧 H5 没有该属性时回退为 30 FPS。只有需要
覆盖源帧率时才传入例如 `--fps 60`。

生成后使用同一个 tag 查看：

```bash
python view_motion_follow.py \
  --motion-file outputs/my_motion/B2A_my_motion.pkl \
  --video-path outputs/my_motion/B2A_my_motion.mp4
```

如果只想导出 PKL、不需要 BVH，可在推理命令后加 `--no-bvh`。

## 输入 h5 格式

输入人类动作 h5 需要包含以下 dataset：

- `rotat`: `(T, J, 4)`，四元数顺序为 `(x, y, z, w)`
- `joint_translation`: `(T, J, 3)`，第 0 个关节为 root translation
- `joint_parents_idx`: `(J,)`
- `joint_offsets`: `(J, 3)`
- `end_sites`: `(4,)`
- `T_pose`: `(J, 3)`
- `joint_names`: `(J,)`

文件还应包含正数 `fps` 属性，用于保持原始动作速度。

示例中的 MotionX/CMU h5 使用 20 个人体关节；推理时会自动追加 root velocity virtual node。
少于 16 帧的短动作会在网络输入端自动重复末帧补齐，导出前再裁掉补帧，因此
转换工具生成的短片也可以直接推理。

## 动作转 h5 工具

仓库提供常用动作转换脚本，输出格式都可以直接作为
`retarget.py --input-h5` 或 `--target-h5` 使用。

Motion-X SMPL-X 322 维 npy 转 h5：

```bash
python tools/motionx_to_h5.py \
  --motion_root /path/to/Motion-X/motion_data/smplx_322 \
  --smplx_model /path/to/SMPLX_NEUTRAL.npz \
  --out_root data/MotionX_h5 \
  --fps 30 \
  --overwrite
```

AMASS/CMU SMPL-X npz 转 h5：

```bash
python tools/amass_cmu_to_h5.py \
  --amass_root /path/to/AMASS/CMU \
  --smplx_model /path/to/SMPLX_NEUTRAL.npz \
  --out_root data/CMU_h5 \
  --overwrite
```

MCM-LDM / HumanML3D `(T, 22, 3)` 关节点 npy 转 h5：

```bash
python tools/mcmldm_npy_to_h5.py \
  --input /path/to/motion.npy \
  --output data/MCM_LDM_h5 \
  --template_h5 examples/human_motion.h5 \
  --fps 20 \
  --overwrite
```

PHUMA/G1 `root_trans/root_ori/dof_pos` npy 转 G1 h5：

```bash
python tools/make_g1_h5.py \
  --phuma_dir /path/to/PHUMA/data/g1/EgoBody \
  --urdf assets/g1_description/g1_29dof.urdf \
  --out_dir data/g1_h5
```

## 输出 pkl 格式

`B2A_*.pkl` 是一个 Python pickle dict：

- `fps`: float，默认继承输入 H5 的帧率
- `root_pos`: `(T, 3)`，机器人 root 位置
- `root_rot`: `(T, 4)`，root 四元数，顺序为 `(x, y, z, w)`
- `dof_pos`: `(T, 29)`，G1 29 个关节角，单位为弧度
- `dof_names`: 长度为 29 的关节名列表，与 `dof_pos` 最后一维一一对应

## 示例动作来源

内置动作转换自 CMU Graphics Lab Motion Capture Database：

- [Subject 35](https://mocap.cs.cmu.edu/search.php?subjectnumber=35)：Trial 01 普通步行、Trial 17 跑步 / 慢跑
- [Subject 49](https://mocap.cs.cmu.edu/search.php?subjectnumber=49)：Trial 02 连续跳跃 / 单脚跳、Trial 19 单脚平衡
- [Subject 144](https://mocap.cs.cmu.edu/search.php?subjectnumber=144)：Trial 20 连续出拳

CMU 官方说明动作数据可以复制、修改和再分发，但不能直接转售数据（包括转换格式）；许可说明见
[CMU Motion Capture Database FAQ](https://mocap.cs.cmu.edu/faqs.php)。

使用该动作发表结果时，请按 CMU 的建议致谢：

> The data used in this project was obtained from mocap.cs.cmu.edu. The database
> was created with funding from NSF EIA-0196217.

## 目录说明

```text
retarget.py                         # 命令行入口
view_motion_follow.py               # MuJoCo 播放/渲染入口
retargeting/                        # 独立推理包
tools/                              # 动作转 h5 脚本
assets/g1_description/              # G1 URDF、MuJoCo XML 和 mesh
checkpoints/g1_retarget/            # 推理权重、训练配置、归一化统计量
examples/                           # 可直接运行的最小示例
```

## 注意事项

- 运行所需的推理代码、模型权重和 G1 资源均已包含在仓库中。
- `stats.npz` 是该 checkpoint 必需的归一化统计量，不要删除。
- `latest_net_G_B.pth` 约 48 MB，普通 GitHub 仓库可直接保存；如果后续加入更大模型，建议使用 Git LFS。
- `tools/motionx_to_h5.py` 和 `tools/amass_cmu_to_h5.py` 需要用户自行提供 SMPL-X 模型文件。

## 测试

```bash
python -m unittest discover -s tests -v
python retarget.py --device cpu --output-dir /tmp/robot-retarget-smoke --tag smoke
MUJOCO_GL=egl python view_motion_follow.py \
  --motion-file /tmp/robot-retarget-smoke/B2A_smoke.pkl \
  --video-path /tmp/robot-retarget-smoke/B2A_smoke.mp4
```
