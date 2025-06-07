import multiprocessing as mp
import cv2
import numpy as np
import math

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #gaussianの二値化がいい感じ
        # 板の継ぎ目が見えちゃうけど黒線は2つのラインで表現されるからそれ検出できれば無視できる
        #gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,63,2)
        gray = cv2.GaussianBlur(gray,(3,3),0)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)

        ret,gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #blur = cv2.GaussianBlur(gray,(5,5),0)
        #ret,gray = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        lines, width, prec, nfa = lsd.detect(gray)
        out = img.copy()
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if lines is not None:
            #lsd.drawSegments(out, lines)
            length = np.linalg.norm(lines[:,0,0:2] - lines[:,0,2:4], axis=1)
            lines = lines[length > 100, :]
            for idx in range(len(lines)):
                x1, y1, x2, y2 = lines[idx][0] 
                cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), int(width[idx]))
        cv2.imshow("Test", out)
        #cv2.imshow("test", img)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def line_recog_slide_window():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray,(3,3),0)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)

        lines, width, prec, nfa = lsd.detect(gray)
        #out = img.copy()
        #out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        """
        if lines is not None:
            length = np.linalg.norm(lines[:,0,0:2] - lines[:,0,2:4], axis=1)
            lines = lines[length > 100, :]
            for idx in range(len(lines)):
                x1, y1, x2, y2 = lines[idx][0] 
                cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), int(width[idx]))
        """
        target = img[184:216, 200:440]
        t_gray = gray[184:216, 200:440]
        out = cv2.cvtColor(t_gray, cv2.COLOR_GRAY2BGR)
        lines, width, prec, nfa = lsd.detect(t_gray)
        if lines is not None:
            length = np.linalg.norm(lines[:,0,0:2] - lines[:,0,2:4], axis=1)
            lines = lines[length > 26, :]
            for idx in range(len(lines)):
                x1, y1, x2, y2 = lines[idx][0] 
                cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 1)
        cv2.imshow("Test", out)
        #cv2.imshow("test", img)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def compute_angle(line):
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(dy, dx))  # 角度（度単位）
    return angle #% 180  # 平行性を見るので180度周期でOK

def line_distance(l1, l2):
    p = (l2[0:2] + l2[2:4]) / 2
    ap = p - l1[0:2]
    ab = l1[2:4] - l1[0:2]
    ba = l1[0:2] - l1[2:4]
    bp = p - l1[2:4]
    ai_norm = np.dot(ap, ab)/np.linalg.norm(ab)
    neighbor_point = l1[0:2] + (ab)/np.linalg.norm(ab)*ai_norm
    return np.linalg.norm(p - neighbor_point)

"""    # 線分 line1 の中心点
    cx1 = (line1[0] + line1[2]) / 2
    cy1 = (line1[1] + line1[3]) / 2
    # 線分 line2 の中心点
    cx2 = (line2[0] + line2[2]) / 2
    cy2 = (line2[1] + line2[3]) / 2
    # ユークリッド距離
    return math.hypot(cx2 - cx1, cy2 - cy1)"""

def find_parallel_pairs(lines, angle_threshold=5.0):
    parallel_pairs = []
    distances = []

    lines = [line[0] for line in lines]  # shape (N, 1, 4) → (N, 4)

    for i in range(len(lines)):
        angle_i = compute_angle(lines[i])
        for j in range(i + 1, len(lines)):
            angle_j = compute_angle(lines[j])
            angle_diff = abs(angle_i - angle_j) % 180
            angle_diff = min(angle_diff, 180 - angle_diff)  # 平行性の差（180度考慮）

            if angle_diff <= angle_threshold:
                d = line_distance(lines[i], lines[j])
                parallel_pairs.append((lines[i], lines[j]))
                distances.append(d)

    return parallel_pairs, distances

def test():
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray,(3,3),0)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
    t_gray = gray[184:216, 200:440]
    lines, width, prec, nfa = lsd.detect(t_gray)

    # 平行な線分ペアと距離を取得
    pairs, dists = find_parallel_pairs(lines, angle_threshold=5.0)

    # 出力
    for i, ((l1, l2), d) in enumerate(zip(pairs, dists)):
        print(f"ペア {i+1}: 距離 = {d:.2f}")
        print(f"  線分1: {l1}")
        print(f"  線分2: {l2}")


def line_eval():
    import matplotlib.pyplot as plt
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
    #blur = cv2.GaussianBlur(gray,(5,5),0)
    #ret,gray = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    lines, width, prec, nfa = lsd.detect(gray)
    out = img.copy()
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    bin_edges = np.arange(0, 200, 10)
    bin_labels = [f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(len(bin_edges)-1)]
    length = np.linalg.norm(lines[:,0,0:2] - lines[:,0,2:4], axis=1)
    hist, _ = np.histogram(length, bins=bin_edges) #bins=bin_edges
    plt.bar(bin_labels, hist)
    plt.show()
    #plt.show()

if __name__ == "__main__":
    test()