# PiCar-X
# RaspbianOS 環境設定周り
- sudoパスワード聞かれるの面倒な場合
  `sudo visudo` で最後の方に下記を追記
  ```
  <USERNAME> ALL=(ALL) NOPASSWD: ALL
  ```
- `/dev/video0`について
  - 最新のRaspbianOS?では`/usr/bin/pipewire`, `/usr/bin/wireplumber` といったプロセスが `/dev/video0` を占有管理する模様
  - このせいで、OpenCVなどからユーザーが自由に `/dev/video0` にアクセスできないっぽい
  - OpenCVなどでカメラデバイスにアクセスする際は以下のように記述すると良いらしい
    ```python
    import cv2

    # 第二引数に cv2.CAP_PIPEWIRE を明示的に指定する
    cap = cv2.VideoCapture(0, cv2.CAP_PIPEWIRE)
    ```
