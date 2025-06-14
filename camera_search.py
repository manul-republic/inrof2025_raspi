import cv2

for i in range(2, 10):  # /dev/video2 ～ /dev/video9 を確認
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"/dev/video{i} is working")
        else:
            print(f"/dev/video{i} is opened but no frame")
        cap.release()
    else:
        print(f"/dev/video{i} not available")
