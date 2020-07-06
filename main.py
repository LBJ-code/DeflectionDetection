import cv2
import time
import csv
from DetermineMoving import DetermineMoving

cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()

fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 3 / 4)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 3 / 4)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))

freq = 5
roop_num = 0
prev_frame = cv2.resize(prev_frame, (w, h), interpolation=cv2.INTER_LINEAR)
deteminer = DetermineMoving()
start = time.time()
seikai = []
yosou = []
Is_real_moving = False

while True:
    ret, cur_frame = cap.read()
    cur_frame = cv2.resize(cur_frame, (w, h), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("cur_frame", cur_frame)
    # cv2.imshow("prev_frame", prev_frame)
    out.write(cur_frame)
    key = cv2.waitKey(10)
    if key == ord('m'):
        Is_real_moving = True if not Is_real_moving else False
    if key == ord('q'):
        break
    if roop_num == freq:
        roop_num -= freq
        now_time = time.time() - start
        # Is_real_moving = True if key == ord('m') else False
        seikai.append([now_time, 1 if Is_real_moving else 0])
        print(f"real_moving = {Is_real_moving}")
        Is_estimated_moving = deteminer(prev_frame, cur_frame)
        yosou.append([now_time, 1 if Is_estimated_moving else 0])
        prev_frame = cur_frame.copy()

    roop_num += 1

cap.release()
out.release()

with open('seikai.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerows(seikai)

with open('yosou.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerows(yosou)

cv2.destroyAllWindows()
