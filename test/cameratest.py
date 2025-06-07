import multiprocessing as mp
import cv2
import numpy as np
#teba@raspi:~/Programs/inrof2025/python $ v4l2-ctl --list-devices
#cap = cv2.VideoCapture(4, cv2.CAP_V4L2)
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
#cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'));
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#https://qiita.com/iwatake2222/items/b8c442a9ec0406883950
#v4l2-ctl --list-devices
#v4l2-ctl --list-formats-ext : show list of available resolution, format

while True:
    ret, img = cap.read()

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #lines = cv2.HoughLinesP(gray, rho=1, theta=np.pi/360, threshold=80, minLineLength=400, maxLineGap=5)
    cv2.imshow("test", img)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
#print(lines)

#print(img.shape)