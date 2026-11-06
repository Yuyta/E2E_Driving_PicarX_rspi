from src.tools.drive_tool import drive_backward, drive_forward, steer_wheels, stop_moving
from src.tools.memory_tool import forget_fact, remember_fact
from src.tools.sensor_tool import read_distance, read_grayscale

# 発話は Agent の返事を Speaker が読み上げる。カメラは認識方式を決めるまで渡さない
ALL_TOOLS = [
    drive_forward,
    drive_backward,
    steer_wheels,
    stop_moving,
    read_distance,
    read_grayscale,
    remember_fact,
    forget_fact,
]

__all__ = [
    "drive_forward",
    "drive_backward",
    "steer_wheels",
    "stop_moving",
    "read_distance",
    "read_grayscale",
    "remember_fact",
    "forget_fact",
    "ALL_TOOLS",
]
