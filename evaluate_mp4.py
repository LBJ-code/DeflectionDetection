import cv2
import numpy as np
from collections import deque
from DetermineMoving import DetermineMoving
from crop_utils import crop_by_center, detect_center_by_thresh

def trim_endo_movie(frame):
    frame = frame[32:989, 323:1599, :]
    frame = cv2.resize(frame, (480, 352), interpolation=cv2.INTER_LINEAR)
    return frame

if __name__ == '__main__':
    intrinsics_scaled = np.load('./params/intrinsics_scaled.npy')
    dist_coeffs = np.load('./params/dist_coeffs.npy')
    cap = cv2.VideoCapture(r"C:\Users\sato yukiya\Desktop\0.mp4")

    center_que = deque()
    ret, prev_frame = cap.read()
    prev_frame = trim_endo_movie(prev_frame)
    prev_frame = cv2.undistort(prev_frame, intrinsics_scaled, dist_coeffs)
    (prev_center_x, prev_center_y) = detect_center_by_thresh(prev_frame)
    center_que.append((prev_center_x, prev_center_y))

    fps = cap.get(cv2.CAP_PROP_FPS)
    h, w, _ = prev_frame.shape
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))

    freq = 5
    roop_num = 0
    max_que_size = 10
    determiner = DetermineMoving()
    Is_estimated_moving = False

    while True:
        ret, cur_frame = cap.read()
        if not ret:
            break
        cur_frame = trim_endo_movie(cur_frame)
        cur_frame = cv2.undistort(cur_frame, intrinsics_scaled, dist_coeffs)

        (cur_center_x, cur_center_y) = detect_center_by_thresh(cur_frame)
        if cur_center_x is not None:
            center_que.append((cur_center_x, cur_center_y))
            if len(center_que) == max_que_size:
                center_que.popleft()
        center_x_mean, center_y_mean = 0.0, 0.0
        for i in range(len(center_que)):
            center_x_mean += center_que[i][0]
            center_y_mean += center_que[i][1]
        center_x_mean, center_y_mean = int(center_x_mean / len(center_que)), int(center_y_mean / len(center_que))

        show_frame = cur_frame.copy()
        cv2.circle(show_frame, (center_x_mean, center_y_mean), 2, (255, 0, 0), 2)

        if roop_num == freq:
            roop_num -= freq
            cropped_cur_frame, cropped_prev_frame = crop_by_center(prev_frame, cur_frame,
                                                                   (prev_center_x, prev_center_y),
                                                                   (center_x_mean, center_y_mean))
            Is_estimated_moving = determiner(cropped_prev_frame, cropped_cur_frame)
            prev_frame = cur_frame.copy()
            prev_center_x, prev_center_y = center_x_mean, center_y_mean

        cv2.putText(show_frame, "Is_moving=", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        if Is_estimated_moving:
            direction = "forward" if determiner.cur_best_level <= 2 else "backward"
            cv2.putText(show_frame, f"True({direction})", (210, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1,
                        cv2.LINE_AA)
        else:
            cv2.putText(show_frame, "False", (210, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("evaluation", show_frame)
        out.write(show_frame)
        cv2.waitKey(20)
        roop_num += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()