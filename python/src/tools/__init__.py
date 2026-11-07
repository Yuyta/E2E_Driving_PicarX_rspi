from src.tools.drive_tool import drive_backward, drive_forward, steer_wheels, stop_moving
from src.tools.memory_tool import forget_fact, remember_fact
from src.tools.sensor_tool import read_distance, read_grayscale
from src.tools.vision_tool import look

# 発話は Agent の返事を Speaker が読み上げる。カメラは PC の VLM の説明だけを渡す
ALL_TOOLS = [
    drive_forward,
    drive_backward,
    steer_wheels,
    stop_moving,
    read_distance,
    read_grayscale,
    remember_fact,
    forget_fact,
    look,
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
    "look",
    "ALL_TOOLS",
]
