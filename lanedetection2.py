#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
################################################################################
@File              :   lanedetection.py
@Time              :   2021/07/08 22:49:31
@Author            :   liuyangly
@Email             :   522927317@qq.com
@Desc              :   基于Opencv的车道线检测（优化版）
################################################################################
"""
# Built-in modules
import argparse
import math

# Third-party modules
import cv2
import numpy as np


class LaneDetection:
    r"""车道线检测（优化版）"""

    def __init__(
            self,
            # 增强预处理参数
            ksize=(7, 7),  # 增大高斯核以更好抑制噪声
            sigma=(1.5, 1.5),  # 增加sigma值增强模糊效果
            threshold1=50,  # 降低低阈值保留更多边缘
            threshold2=150,  # 降低高阈值
            aperture_size=3,
            # 改进ROI参数
            direction_point=None,
            roi_top_ratio=0.6,  # ROI顶部位置比例
            # 优化霍夫参数
            rho=1,
            theta=np.pi / 180,
            threshold=30,  # 降低阈值检测更多线段
            min_line_len=100,  # 缩短最小线段长度
            max_line_gap=150,  # 减小线段间隙
            # 直线拟合参数
            x1L=None,
            x2L=None,
            x1R=None,
            x2R=None,
            # 新增参数
            use_white_balance=True,  # 启用白平衡
            use_clahe=True,  # 启用对比度增强
            slope_threshold=0.3  # 更严格的斜率过滤
    ):
        self.ksize = ksize
        self.sigma = sigma
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.aperture_size = aperture_size
        self.direction_point = direction_point
        self.roi_top_ratio = roi_top_ratio
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_len = min_line_len
        self.max_line_gap = max_line_gap
        self.x1L = x1L
        self.x2L = x2L
        self.x1R = x1R
        self.x2R = x2R
        self.use_white_balance = use_white_balance
        self.use_clahe = use_clahe
        self.slope_threshold = slope_threshold

    def __call__(self, img):
        # 新增白平衡预处理
        if self.use_white_balance:
            img = self._white_balance(img)

        gauss = self._image_preprocess(img)

        # 新增CLAHE对比度增强
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gauss = clahe.apply(gauss)

        edge = self._edge_canny(gauss)
        roi = self._roi_trapezoid(edge)
        lines = self._Hough_line_fitting(roi)
        line_img = self._lane_line_fitting(img, lines)
        res = self._weighted_img_lines(img, line_img)
        return res

    def _white_balance(self, img):
        r"""简单白平衡处理，增强不同光照下的稳定性"""
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    def _image_preprocess(self, img):
        r"""预处理优化：增加灰度化前的颜色过滤"""
        # 提取黄色和白色车道线颜色范围
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 黄色范围
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 白色范围
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([255, 30, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # 合并掩码并与原图结合
        combined_mask = cv2.bitwise_or(mask_yellow, mask_white)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_and(gray, combined_mask)

        # 高斯滤波
        gauss = cv2.GaussianBlur(gray, self.ksize, self.sigma[0], self.sigma[1])
        return gauss

    def _edge_canny(self, img):
        r"""Canny边缘检测保持不变"""
        edge = cv2.Canny(img, self.threshold1, self.threshold2, self.aperture_size)
        return edge

    def _roi_trapezoid(self, img):
        r"""优化ROI区域，使其更符合实际车道形状"""
        h, w = img.shape[:2]

        # 动态计算ROI顶部位置
        top_y = int(h * self.roi_top_ratio)
        top_width = int(w * 0.15)  # 顶部宽度

        # 车方向的中心点
        if self.direction_point is None:
            center_x = w // 2
        else:
            center_x = self.direction_point[0]

        # 构建更合理的梯形ROI
        left_top = [center_x - top_width, top_y]
        right_top = [center_x + top_width, top_y]
        left_down = [int(w * 0.1), h]
        right_down = [int(w * 0.9), h]

        self.roi_points = np.array([left_down, left_top, right_top, right_down], np.int32)

        # 填充梯形区域
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [self.roi_points], 255)

        # 目标区域提取：逻辑与
        roi = cv2.bitwise_and(img, mask)
        return roi

    def _Hough_line_fitting(self, img):
        r"""霍夫变换参数优化"""
        lines = cv2.HoughLinesP(
            img, self.rho, self.theta, self.threshold, np.array([]),
            minLineLength=self.min_line_len, maxLineGap=self.max_line_gap
        )
        return lines

    def _lane_line_fitting(self, img, lines, color=(0, 255, 0), thickness=8):
        r"""优化直线拟合逻辑，增加异常值过滤"""
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        if lines is None:
            return line_img

        h, w = img.shape[:2]
        right_x = []
        right_y = []
        left_x = []
        left_y = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                # 计算线段长度，过滤短线段
                length = math.hypot(x2 - x1, y2 - y1)
                if length < self.min_line_len * 0.5:
                    continue

                # 计算斜率，避免除零错误
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)

                # 使用更严格的斜率过滤
                if slope <= -self.slope_threshold:
                    # 过滤偏离合理区域的点
                    if 0 < x1 < w and 0 < y1 < h and 0 < x2 < w and 0 < y2 < h:
                        left_x.extend((x1, x2))
                        left_y.extend((y1, y2))
                elif slope >= self.slope_threshold:
                    if 0 < x1 < w and 0 < y1 < h and 0 < x2 < w and 0 < y2 < h:
                        right_x.extend((x1, x2))
                        right_y.extend((y1, y2))

        # 左车道线拟合：增加异常值检测
        if left_x and left_y:
            # 使用Z-score过滤异常值
            left_points = np.column_stack((left_x, left_y))
            left_points = self._remove_outliers(left_points)
            if len(left_points) >= 2:
                left_fit = np.polyfit(left_points[:, 0], left_points[:, 1], 1)
                left_line = np.poly1d(left_fit)

                # 动态计算插值点，基于图像高度
                if not self.x1L:
                    x1L = int(w * 0.1)
                y1L = int(left_line(x1L))
                # 确保y坐标在图像范围内
                y1L = np.clip(y1L, h // 2, h)

                if not self.x2L:
                    x2L = int(w * 0.4)
                y2L = int(left_line(x2L))
                y2L = np.clip(y2L, h // 2, h)

                cv2.line(line_img, (x1L, y1L), (x2L, y2L), color, thickness)

        # 右车道线拟合：增加异常值检测
        if right_x and right_y:
            right_points = np.column_stack((right_x, right_y))
            right_points = self._remove_outliers(right_points)
            if len(right_points) >= 2:
                right_fit = np.polyfit(right_points[:, 0], right_points[:, 1], 1)
                right_line = np.poly1d(right_fit)

                if not self.x1R:
                    x1R = int(w * 0.6)
                y1R = int(right_line(x1R))
                y1R = np.clip(y1R, h // 2, h)

                if not self.x2R:
                    x2R = int(w * 0.9)
                y2R = int(right_line(x2R))
                y2R = np.clip(y2R, h // 2, h)

                cv2.line(line_img, (x1R, y1R), (x2R, y2R), color, thickness)

        return line_img

    def _remove_outliers(self, points, z_threshold=2.0):
        r"""使用Z-score方法移除异常值点"""
        if len(points) <= 3:
            return points

        mean = np.mean(points, axis=0)
        std = np.std(points, axis=0)
        z_scores = np.abs((points - mean) / std)
        return points[(z_scores < z_threshold).all(axis=1)]

    def _weighted_img_lines(self, img, line_img, α=1, β=1, λ=0.):
        r"""保持不变"""
        res = cv2.addWeighted(img, α, line_img, β, λ)
        return res


def parse_args():
    parser = argparse.ArgumentParser(description="Lane Detection V1.1 (Optimized)")
    parser.add_argument("-i", "--input_path", type=str, default="./assets/1.jpg", help="Input path of image.")
    parser.add_argument("-o", "--output_path", type=str, default="./assets/1_out.jpg", help="Ouput path of image.")
    return parser.parse_args()


def main():
    args = parse_args()

    # 使用优化参数初始化检测器
    lanedetection = LaneDetection(
        ksize=(7, 7),
        sigma=(1.5, 1.5),
        threshold1=50,
        threshold2=150,
        min_line_len=100,
        max_line_gap=150,
        slope_threshold=0.3
    )

    # 图片检测
    if args.input_path.endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(args.input_path, 1)
        if img is None:
            print(f"无法读取图片: {args.input_path}")
            return
        res = lanedetection(img)
        x = np.hstack([img, res])
        cv2.imwrite(args.output_path, x)
        cv2.imshow("Result", x)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 视频检测
    elif args.input_path.endswith('.mp4'):
        video_capture = cv2.VideoCapture(args.input_path)
        if not video_capture.isOpened():
            print('无法打开视频文件!')
            exit()

        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(f"fps: {fps} \nsize: {size}")

        out = cv2.VideoWriter(
            args.output_path,
            fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
            fps=fps,
            frameSize=size
        )

        total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[INFO] 视频总帧数: {total}")

        frameToStart = 0
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frameToStart)

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            res = lanedetection(frame)
            out.write(res)
            cv2.imshow("video", res)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()