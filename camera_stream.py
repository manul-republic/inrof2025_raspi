from flask import Flask, Response
import cv2

app = Flask(__name__)

# PiカメラをOpenCVで開く（libcamera対応）
cap = cv2.VideoCapture(0)  # Piカメラが /dev/video0 に割り当てられていれば

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        # JPEG形式にエンコード
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        # バイト列にしてストリーム
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
        <head><title>Pi Camera Stream</title></head>
        <body>
            <h1>Pi Camera Live</h1>
            <img src="/video_feed" width="640" height="480">
        </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True)
