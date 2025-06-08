from flask import Flask, Response
import cv2
import threading
import time

# ここは LineTracer によって上書きされる
latest_debug_frame = None
frame_lock = threading.Lock()

def update_debug_frame(img):
    global latest_debug_frame
    with frame_lock:
        # ストリーム画像の解像度をカメラ画像に合わせる
        # 例: 320x240 などにリサイズしたい場合はここで変更
        # img.shape: (height, width, ...)
        target_width = img.shape[1]/2
        target_height = img.shape[0]/2
        # 必要ならここで固定サイズにリサイズ（例: 320x240）
        # target_width, target_height = 320, 240
        # img = cv2.resize(img, (target_width, target_height))
        latest_debug_frame = img.copy()

def generate():
    global latest_debug_frame
    while True:
        with frame_lock:
            if latest_debug_frame is None:
                time.sleep(0.05)
                continue
            success, jpeg = cv2.imencode(
                '.jpg',
                latest_debug_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 10]
            )
        if not success:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.05)  # 必要ならここも少し下げてさらに軽く


app = Flask(__name__)

@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_webserver():
    app.run(host='0.0.0.0', port=8000, threaded=True)
