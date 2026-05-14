import os
import sys
from time import sleep
from picarx import Picarx
import readchar

MANUAL = '''
Press keys on keyboard to control PiCar-X!
    w: Forward
    a: Turn left
    s: Backward
    d: Turn right
    i: Head up
    k: Head down
    j: Turn head left
    l: Turn head right
    ctrl + c: Press twice to exit the program
'''

def show_info():
    """ターミナルをクリアして操作説明を表示する"""
    print("\033[H\033[J", end='')
    print(MANUAL)


def initialize_picar():
    """PiCar-Xの初期化を試みる"""
    try:
        return Picarx()
    except Exception as e:
        print(f"Failed to initialize PiCar-X: {e}")
        return None


def handle_movement(px, key, pan_angle, tilt_angle):
    """キー入力に応じて車体とカメラの角度を制御する"""
    # 角度の上限・下限値
    MAX_ANGLE, MIN_ANGLE = 35, -35

    # 1. 車体の進行方向・前後移動の設定
    if key == 'w':
        px.set_dir_servo_angle(0)
        px.forward(80)
    elif key == 's':
        px.set_dir_servo_angle(0)
        px.backward(80)
    elif key == 'a':
        px.set_dir_servo_angle(-35)
        px.forward(80)
    elif key == 'd':
        px.set_dir_servo_angle(35)
        px.forward(80)

    # 2. カメラの首振り角度の計算
    elif key == 'i':
        tilt_angle = min(tilt_angle + 5, MAX_ANGLE)
    elif key == 'k':
        tilt_angle = max(tilt_angle - 5, MIN_ANGLE)
    elif key == 'l':
        pan_angle = min(pan_angle + 5, MAX_ANGLE)
    elif key == 'j':
        pan_angle = max(pan_angle - 5, MIN_ANGLE)

    # 3. カメラ角度の反映と共通のウェイト処理
    px.set_cam_tilt_angle(tilt_angle)
    px.set_cam_pan_angle(pan_angle)
    show_info()

    sleep(0.5)
    px.forward(0)  # 一定時間後に安全のためモーターを停止

    return pan_angle, tilt_angle


def restore_terminal(px):
    """PiCarの安全停止とターミナルの状態を強制復元する"""
    if px is not None:
        try:
            px.set_cam_tilt_angle(0)
            px.set_cam_pan_angle(0)
            px.set_dir_servo_angle(0)
            px.stop()
            sleep(0.2)
        except Exception as e:
            print(f"Error during PiCar stop: {e}")

    # キーボード入力バッファのフラッシュ（暴発・パスワード要求対策）
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except ImportError:
        pass

    # ターミナル表示の復元
    os.system("stty echo")
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()
    print("Terminal restored.")


def main():
    px = initialize_picar()
    if px is None:
        return

    pan_angle = 0
    tilt_angle = 0
    show_info()

    try:
        while True:
            key = readchar.readkey()
            key = key.lower()

            if key in 'wsadikjl':
                # 移動・カメラ制御を行い、更新された角度を受け取る
                pan_angle, tilt_angle = handle_movement(px, key, pan_angle, tilt_angle)

            elif key == readchar.key.CTRL_C:
                print("\n Quit")
                break
    finally:
        restore_terminal(px)


if __name__ == "__main__":
    main()

