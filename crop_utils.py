import numpy as np
import cv2


def crop_by_center(prev_img, cur_img, prev_center_xy, cur_center_xy):
    if prev_center_xy[0] != cur_center_xy[0]:
        x_diff = abs(prev_center_xy[0] - cur_center_xy[0])
        prev_img, cur_img = (prev_img[:, x_diff:, :], cur_img[:, :-x_diff, :]) if cur_center_xy[0] < prev_center_xy[0] \
            else (prev_img[:, :-x_diff, :], cur_img[:, x_diff:, :])

    if prev_center_xy[1] != cur_center_xy[1]:
        y_diff = abs(prev_center_xy[1] - cur_center_xy[1])
        prev_img, cur_img = (prev_img[y_diff:, :, :], cur_img[:-y_diff, :, :]) if cur_center_xy[1] < prev_center_xy[1] \
            else (prev_img[:-y_diff, :, :], cur_img[y_diff:, :, :])

    return prev_img, cur_img


def detect_center_by_thresh(eso_img, thresh_num=50):
    gray = cv2.cvtColor(eso_img, cv2.COLOR_BGR2GRAY)
    _, bin = cv2.threshold(gray, thresh_num, 255, cv2.THRESH_BINARY_INV)
    if np.count_nonzero(bin) == 0:
        return (None, None)
    mu = cv2.moments(bin, binaryImage=True)
    x, y = int(mu["m10"]/mu["m00"]), int(mu["m01"]/mu["m00"])

    return x, y
