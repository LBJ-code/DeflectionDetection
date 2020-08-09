import numpy as np
import cv2
import time
from skimage.metrics import structural_similarity

class DetermineMoving:
    def __init__(self):
        self.cur_best_level = -1

    def __call__(self, prev_frame, cur_frame):
        prev_pyramid_frame_list = self.create_img_pyramid(prev_frame)
        # for test in prev_pyramid_frame_list:
        #     cv2.imshow("test", test)
        #     cv2.waitKey(0)
        Is_moving = self.determine_moving(prev_pyramid_frame_list, cur_frame)

        return Is_moving


    def create_img_pyramid(self, frame):
        # この辺は適切なサイズを選ぶ
        # resize_ratio_list = [10 / 8, 9 / 8, 1, 7 / 8, 6 / 8]
        resize_ratio_list = [12/10, 11/10, 1, 15/16, 14/16]
        h, w, c = frame.shape
        return_pyramid_img_list = list()

        for ratio in resize_ratio_list:
            resized_img = cv2.resize(frame, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR)
            cropped_img = self.crop_img(resized_img, h, w)
            return_pyramid_img_list.append(cropped_img)

        return return_pyramid_img_list

    def crop_img(self, resized_img, s_h, s_w):
        src_size = s_h * s_w * 3
        r_h, r_w, _ = resized_img.shape

        # リサイズ後の方が大きいとき
        if s_h * s_w < r_h * r_w:
            half_h_diff = int((r_h - s_h) / 2)
            half_w_diff = int((r_w - s_w) / 2)
            cropped_img = resized_img[half_h_diff:-half_h_diff, half_w_diff:-half_w_diff, :]
            cropped_img = cv2.resize(cropped_img, (s_w, s_h),
                                     interpolation=cv2.INTER_LINEAR) if cropped_img.size != src_size else cropped_img

        # elif  r_h * r_w < s_h * s_w:
        #    half_h_diff = int((s_h - r_h) / 2)
        #    half_w_diff = int((s_w - r_w) / 2)
        #    cropped_img = cv2.copyMakeBorder(resized_img, half_h_diff, half_h_diff, half_w_diff, half_w_diff,
        #                                     cv2.BORDER_REPLICATE)

        # その他の大きさのとき
        else:
            cropped_img = resized_img

        return cropped_img

    def determine_moving(self, prev_pyramid_frame_list, cur_frame):
        c_h, c_w, _ = cur_frame.shape
        center_level = int((len(prev_pyramid_frame_list) - 1) / 2 + 1)
        best_level = -1
        best_ssim = -1
        f = open("result.csv", "w")
        f.write("level,ssim\n")
        for i, resized_prev_frame in enumerate(prev_pyramid_frame_list):
            cur_level = i + 1
            # リサイズ後の方が小さいとき
            if resized_prev_frame.size < cur_frame.size:
                r_h, r_w, _ = resized_prev_frame.shape
                half_h_diff = int((c_h - r_h) / 2)
                half_w_diff = int((c_w - r_w) / 2)
                cropped_cur_frame = cur_frame[half_h_diff:-half_h_diff, half_w_diff:-half_w_diff, :]
                cropped_cur_frame = cropped_cur_frame if resized_prev_frame.size == cropped_cur_frame.size else \
                    cv2.resize(cropped_cur_frame, (r_w, r_h))
                cur_ssim = structural_similarity(resized_prev_frame, cropped_cur_frame, multichannel=True)
            else:
                cur_ssim = structural_similarity(resized_prev_frame, cur_frame, multichannel=True)

            f.write(f"{cur_level},{cur_ssim}\n")

            if best_ssim < cur_ssim:
                best_ssim = cur_ssim
                best_level = cur_level

        # cv2.imshow("best_pyramid", prev_pyramid_frame_list[best_level - 1])
        self.cur_best_level = best_level
        Is_moving = False if best_level == center_level else True
        print(f"best_level = {best_level}   best_ssim = {best_ssim}   moving : {Is_moving}")
        return Is_moving
