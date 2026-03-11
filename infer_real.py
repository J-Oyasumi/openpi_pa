import collections
import time
from dataclasses import dataclass

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import portal
import pyrealsense2 as rs
import tyro


DEFAULT_ROBOT_PORT = 11333


class ClientRobot:
    def __init__(self, port: int = DEFAULT_ROBOT_PORT, host: str = "127.0.0.1"):
        self._client = portal.Client(f"{host}:{port}")

    def get_joint_pos(self) -> np.ndarray:
        return self._client.get_joint_pos().result()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self._client.command_joint_pos(joint_pos)


@dataclass
class Args:
    # Policy server address
    host: str = "0.0.0.0"
    port: int = 8000

    # Image + policy settings
    resize_size: int = 224
    replan_steps: int = 5  # how many steps to execute before re-querying policy
    prompt: str = "do the task"

    # Control loop
    max_steps: int = 500
    hz: float = 5.0

    # Robot connection
    robot_host: str = "127.0.0.1"
    robot_port: int = DEFAULT_ROBOT_PORT


def main(args: Args) -> None:
    """
    Real-world inference loop:
    - 读取 RealSense 彩色图像
    - 从机器人读取当前关节角
    - 调用远程 openpi policy server 得到动作序列
    - 以关节增量的方式下发到机器人
    """
    # 连接 policy server
    client = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )

    # 连接机器人
    robot = ClientRobot(port=args.robot_port, host=args.robot_host)

    # 初始化 RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    # 只用彩色相机，如果你有 wrist 相机，可以在这里额外配置
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    action_plan: collections.deque[np.ndarray] = collections.deque()

    try:
        dt = 1.0 / args.hz
        for step in range(args.max_steps):
            # 读取一帧图像
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            # RealSense 默认是 BGR，转成 RGB
            color_image = color_image[:, :, ::-1]

            # 按训练分辨率做 resize + pad，并转成 uint8
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(color_image, args.resize_size, args.resize_size)
            )
            # 如果现实里暂时没有 wrist camera，用 0 填充
            wrist_img = np.zeros_like(img, dtype=np.uint8)

            # 读取当前关节角作为 state；如果有额外状态，可以在这里拼接
            joint_pos = robot.get_joint_pos().astype(np.float32)
            state = joint_pos

            # 如果 action buffer 已经用完，就重新调用一次 policy
            if not action_plan:
                observation = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": state,
                    "prompt": args.prompt,
                }

                actions = client.infer(observation)["actions"]
                assert (
                    len(actions) >= args.replan_steps
                ), f"Policy returned {len(actions)} steps, but replan_steps={args.replan_steps}"
                action_plan.extend(actions[: args.replan_steps])

            # 取出当前一步的 action
            action = np.asarray(action_plan.popleft(), dtype=np.float32)

            # 假设网络输出的是关节空间增量（长度等于关节数），你可以根据自己的系统改成位姿控制等
            if action.shape[0] >= joint_pos.shape[0]:
                delta_q = action[: joint_pos.shape[0]]
            else:
                # 如果 action 维度比关节数少，就 pad 一些 0
                pad = np.zeros_like(joint_pos, dtype=np.float32)
                pad[: action.shape[0]] = action
                delta_q = pad

            target_q = joint_pos + delta_q
            robot.command_joint_pos(target_q)

            time.sleep(dt)

    finally:
        pipeline.stop()


if __name__ == "__main__":
    tyro.cli(main)
