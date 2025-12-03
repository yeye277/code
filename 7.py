"这是一段检测车道线"
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import matplotlib.image as mplimg
import matplotlib.pyplot as plt


# 定义感兴趣区域
def interested_region(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)  # 创建掩膜
    return cv2.bitwise_and(img, mask)  # 应用掩膜


# 定义霍夫变换检测车道线
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    if lines is not None:
        right_y_set = []
        right_x_set = []
        right_slope_set = []

        left_y_set = []
        left_x_set = []
        left_slope_set = []

        slope_min = 0.35  # 斜率低阈值
        slope_max = 0.85  # 斜率高阈值
        middle_x = line_img.shape[1] / 2  # 图像中线x坐标
        max_y = line_img.shape[0]  # 最大y坐标

        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 - x1 == 0:  # 避免除以0
                    continue
                fit = np.polyfit((x1, x2), (y1, y2), 1)  # 拟合成直线
                slope = fit[0]  # 斜率

                if slope_min < np.absolute(slope) <= slope_max:
                    # 右边车道线
                    if slope > 0 and x1 > middle_x and x2 > middle_x:
                        right_y_set.append(y1)
                        right_y_set.append(y2)
                        right_x_set.append(x1)
                        right_x_set.append(x2)
                        right_slope_set.append(slope)

                        # 左边车道线
                    elif slope < 0 and x1 < middle_x and x2 < middle_x:
                        left_y_set.append(y1)
                        left_y_set.append(y2)
                        left_x_set.append(x1)
                        left_x_set.append(x2)
                        left_slope_set.append(slope)

                        # 绘制左车道线
        if left_y_set:
            lindex = left_y_set.index(min(left_y_set))  # 最高点
            left_x_top = left_x_set[lindex]
            left_y_top = left_y_set[lindex]
            lslope = np.median(left_slope_set)  # 计算平均值
            left_x_bottom = int(left_x_top + (max_y - left_y_top) / lslope)

            # 绘制线段
            cv2.line(line_img, (left_x_bottom, max_y), (left_x_top, left_y_top), color=(255, 0, 0), thickness=6)

            # 绘制右车道线
        if right_y_set:
            rindex = right_y_set.index(min(right_y_set))  # 最高点
            right_x_top = right_x_set[rindex]
            right_y_top = right_y_set[rindex]
            rslope = np.median(right_slope_set)
            right_x_bottom = int(right_x_top + (max_y - right_y_top) / rslope)

            # 绘制线段
            cv2.line(line_img, (right_x_top, right_y_top), (right_x_bottom, max_y), color=(0, 0, 255), thickness=6)

    return line_img


# 合并图像
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)


# 处理每一帧图像
def process_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    gauss_gray = cv2.GaussianBlur(gray_image, (5, 5), 0)  # 高斯模糊
    canny_edges = cv2.Canny(gauss_gray, 50, 150)  # Canny边缘检测
    plt.imshow(canny_edges), plt.title('canny_edges'), plt.show()

    # 定义感兴趣区域
    imshape = image.shape
    top_left = (0, imshape[0] / 2)
    top_right = (imshape[1], imshape[0] / 2)
    lower_right = (imshape[1], imshape[0])
    lower_left = (0, imshape[0])
    vertices = np.array([top_left, top_right, lower_right, lower_left], dtype=np.int32)
    roi_image = interested_region(canny_edges, [vertices])  # 提取感兴趣区域图像
    plt.imshow(roi_image), plt.title('roi_image'), plt.show()

    # 使用霍夫变换检测车道线
    line_image = hough_lines(roi_image, 1, np.pi / 180, 30, 120, 180)
    result = weighted_img(line_image, image, α=0.8, β=1., λ=0.)  # 合成结果图像
    return result


# 主函数
if __name__ == "__main__":
    img = cv2.imread('6.png')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    img1 = process_image(img)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.show()

    # 以下部分可以用来处理视频
    # white_output = 'output54.mp4'
    # clip1 = VideoFileClip("5.mp4")  # 输入视频文件路径
    # white_clip = clip1.fl_image(process_image)  # 应用处理函数
    # white_clip.write_videofile(white_output, audio=False)