import cv2
import numpy as np
from DetermineMoving import DetermineMoving

def trim_endo_movie(frame):
    frame = frame[32:989, 323:1599, :]
    frame = cv2.resize(frame, (480, 352), interpolation=cv2.INTER_LINEAR)
    return frame

if __name__ == '__main__':
    intrinsics_scaled = np.load('./params/intrinsics_scaled.npy')
    dist_coeffs = np.load('./params/dist_coeffs.npy')
    cap = cv2.VideoCapture(r"C:\Users\sato yukiya\Desktop\0.mp4")
    ret, prev_frame = cap.read()
    prev_frame = trim_endo_movie(prev_frame)
    prev_frame = cv2.undistort(prev_frame, intrinsics_scaled, dist_coeffs)

    fps = cap.get(cv2.CAP_PROP_FPS)
    h, w, _ = prev_frame.shape
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))

    freq = 5
    roop_num = 0
    determiner = DetermineMoving()
    Is_estimated_moving = False

    while True:
        ret, cur_frame = cap.read()
        if not ret:
            break
        cur_frame = trim_endo_movie(cur_frame)
        cur_frame = cv2.undistort(cur_frame, intrinsics_scaled, dist_coeffs)
        show_frame = cur_frame.copy()
        if roop_num == freq:
            roop_num -= freq
            Is_estimated_moving = determiner(prev_frame, cur_frame)
            prev_frame = cur_frame.copy()

        cv2.putText(show_frame, "Is_moving=", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        if Is_estimated_moving:
            direction = "forward" if determiner.cur_best_level <= 2 else "backward"
            cv2.putText(show_frame, f"True({direction})", (210, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1,
                        cv2.LINE_AA)
        else:
            cv2.putText(show_frame, "False", (210, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("evaluation", show_frame)
        out.write(show_frame)
        cv2.waitKey(1)
        roop_num += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()