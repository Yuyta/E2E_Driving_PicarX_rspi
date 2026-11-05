from src.tools.camera_tool import take_snapshot
from src.tools.drive_tool import drive_backward, drive_forward, steer_wheels, stop_moving
from src.tools.memory_tool import forget_fact, remember_fact
from src.tools.sensor_tool import read_distance, read_grayscale
from src.tools.voice_tool import speak_text

ALL_TOOLS = [
    drive_forward,
    drive_backward,
    steer_wheels,
    stop_moving,
    read_distance,
    read_grayscale,
    take_snapshot,
    speak_text,
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
    "take_snapshot",
    "speak_text",
    "remember_fact",
    "forget_fact",
    "ALL_TOOLS",
]
