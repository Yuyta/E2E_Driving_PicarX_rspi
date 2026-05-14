import sys
import cv2
import numpy as np
from picamera2 import Picamera2

class PiCarCamera:
    """PiCar-Xのカメラ（Picamera2）を管理するクラス"""
    
    def __init__(self, width=640, height=480, fmt="BGR888"):
        self.width = width
        self.height = height
        self.fmt = fmt
        self.picam2 = None
        self._is_running = False

    def start(self):
        """カメラを初期化してストリームを開始する"""
        if self._is_running:
            return

        print("Initializing Picamera2...")
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"format": self.fmt, "size": (self.width, self.height)}
            )
            self.picam2.configure(config)
            self.picam2.start()
            self._is_running = True
            print("Camera system started successfully.")
        except Exception as e:
            print(f"Failed to start camera: {e}")
            sys.exit(1)

    def set_low_light_mode(self, enable=True, exposure_time=80000, analogue_gain=8.0):
        """
        低照度（室内）用のマニュアル露出モードを設定する
        
        Args:
            enable (bool): Trueで手動固定、Falseで自動露出(AE)に戻す
            exposure_time (int): シャッタースピード（マイクロ秒）
            analogue_gain (float): アナログゲイン（1.0 〜 16.0）
        """
        if not self._is_running:
            print("Warning: Camera is not running. Cannot set controls.")
            return

        try:
            if enable:
                print(f"Setting Low-Light Manual Mode (Exposure: {exposure_time}μs, Gain: {analogue_gain})")
                self.picam2.set_controls({"AeEnable": False})
                self.picam2.set_controls({"ExposureTime": exposure_time})
                self.picam2.set_controls({"AnalogueGain": analogue_gain})
            else:
                print("Switching back to Auto Exposure Mode.")
                self.picam2.set_controls({"AeEnable": True})
        except Exception as e:
            print(f"Failed to set sensor controls: {e}")

    def get_frame(self):
        """最新の1フレームをOpenCV(NumPy)形式で取得する"""
        if not self._is_running:
            return None
        return self.picam2.capture_array()

    def close(self):
        """カメラリソースを安全に解放する"""
        if not self._is_running:
            return

        print("Closing camera resources...")
        try:
            self.picam2.stop()
            self.picam2.close()
            self._is_running = False
            print("Camera closed safely.")
        except Exception as e:
            print(f"Error during camera closure: {e}")


# =====================================================================
#  動作テスト用のメイン処理
# =====================================================================
def main():
    # インスタンスの生成
    camera = PiCarCamera(width=640, height=480)
    
    # 開始
    camera.start()
    
    # 先ほど成功した低照度設定を適用（必要に応じて数値を変えて呼べる）
    camera.set_low_light_mode(enable=True, exposure_time=80000, analogue_gain=8.0)

    window_name = "PiCar-X Camera Class Test"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\n--- 操作方法 ---")
    print("  q キー : プログラムを終了")
    print("----------------")

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        # どんな落ち方をしてもfinallyで確実にカメラをクローズする
        camera.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
