import os
import cv2
import numpy as np
from math import *
from scipy.stats import mode
import time

# 图像矫正类
class ImgCorrect:
    """
    霍夫变换进行线段检索，再根据这些线段算出夹角，利用角度的加权平均值和频率最高的思想作为旋转的最佳角度
    """
    def __init__(self, img):
        self.img = img
        """
        # 图像归一化处理 会造成图像清晰度变低
        self.h, self.w, self.channel = self.img.shape
        if self.w <= self.h:
            self.scale = 700 / self.w
            self.w_scale = 700
            self.h_scale = self.h * self.scale
            self.img = cv2.resize(self.img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        else:
            self.scale = 700 / self.h
            self.h_scale = 700
            self.w_scale = self.w * self.scale
            self.img = cv2.resize(self.img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        """
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # 只对中心区域进行霍夫变换 减少计算量
        h, w = self.gray.shape
        self.gray = self.gray[int(0+h/10):int(h-h/10),int(0+w/10):int(w-w/10)]   # 裁剪坐标为[y0:y1, x0:x1]

    def img_lines(self):
        ret, binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # cv2.imshow("bin",binary)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩形结构
        binary = cv2.dilate(binary, kernel)  # 膨胀
        edges = cv2.Canny(binary, 50, 200)
        # cv2.imshow("edges", edges)
        self.lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=100, maxLineGap=20)
        # print(self.lines)
        if self.lines is None:
            # print("Line segment not found")
            return None
        # lines1 = self.lines[:, 0, :]  # 提取为二维
        # print(lines1)
        # imglines = self.img.copy()
        # for x1, y1, x2, y2 in lines1[:]:
        #     cv2.line(imglines, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # return imglines

    def search_lines(self):
        lines = self.lines[:, 0, :]  # 提取为二维
        # k = [(y2 - y1) / (x2 - x1) for x1, y1, x2, y2 in lines]
        # sorted_k = sorted(lines, key=lambda x:(x[3] - x[1]) / (x[2] - x[0]))
        number_inexistence_k = 0
        sum_positive_k45 = 0
        number_positive_k45 = 0
        sum_positive_k90 = 0
        number_positive_k90 = 0
        sum_negative_k45 = 0
        number_negative_k45 = 0
        sum_negative_k90 = 0
        number_negative_k90 = 0
        sum_zero_k = 0
        number_zero_k = 0
        for x in lines:
            if x[2] == x[0]:
                number_inexistence_k += 1
                continue
            if 0 < degrees(atan((x[3] - x[1]) / (x[2] - x[0]))) < 45:
                number_positive_k45 += 1
                sum_positive_k45 += degrees(atan((x[3] - x[1]) / (x[2] - x[0])))
            # if 45 <= degrees(atan((x[3] - x[1]) / (x[2] - x[0]))) < 90:
            #     number_positive_k90 += 1
            #     sum_positive_k90 += degrees(atan((x[3] - x[1]) / (x[2] - x[0])))
            if -45 < degrees(atan((x[3] - x[1]) / (x[2] - x[0]))) < 0:
                number_negative_k45 += 1
                sum_negative_k45 += degrees(atan((x[3] - x[1]) / (x[2] - x[0])))
            # if -90 < degrees(atan((x[3] - x[1]) / (x[2] - x[0]))) <= -45:
            #     number_negative_k90 += 1
            #     sum_negative_k90 += degrees(atan((x[3] - x[1]) / (x[2] - x[0])))
            if x[3] == x[1]:
                number_zero_k += 1

        max_number = max(number_inexistence_k, number_positive_k45, number_positive_k90, number_negative_k45,
                         number_negative_k90, number_zero_k)
        # print(number_inexistence_k,number_positive_k45, number_positive_k90, number_negative_k45, number_negative_k90,number_zero_k)
        if max_number == number_inexistence_k:
            return 90
        if max_number == number_positive_k45:
            return sum_positive_k45 / number_positive_k45
        if max_number == number_positive_k90:
            return sum_positive_k90 / number_positive_k90
        if max_number == number_negative_k45:
            return sum_negative_k45 / number_negative_k45
        if max_number == number_negative_k90:
            return sum_negative_k90 / number_negative_k90
        if max_number == number_zero_k:
            return 0

    def rotate_image(self, degree):
        """
        正角 逆时针旋转
        :param degree:
        :return:
        """
        # print("degree:", degree)
        if -45 <= degree <= 0:
            degree = degree         # 负角度 顺时针
        if -90 <= degree < -45:
            degree = 90 + degree    # 正角度 逆时针
        if 0 < degree <= 45:
            degree = degree         # 正角度 逆时针
        if 45 < degree < 90:
            degree = degree - 90    # 负角度 顺时针
        # print("rotate degree:", degree)
        # degree = -45
        # # 获取旋转后4角的填充色
        filled_color = -1
        if filled_color == -1:
            filled_color = mode([self.img[0, 0], self.img[0, -1],
                                 self.img[-1, 0], self.img[-1, -1]]).mode[0]
        if np.array(filled_color).shape[0] == 2:
            if isinstance(filled_color, int):
                filled_color = (filled_color, filled_color, filled_color)
        else:
            filled_color = tuple([int(i) for i in filled_color])

        # degree = degree - 90
        height, width = self.img.shape[:2]
        heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))  # 这个公式参考之前内容
        widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

        matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)  # 逆时针旋转 degree

        matRotation[0, 2] += (widthNew - width) / 2  # 因为旋转之后,坐标系原点是新图像的左上角,所以需要根据原图做转化
        matRotation[1, 2] += (heightNew - height) / 2

        imgRotation = cv2.warpAffine(self.img, matRotation, (widthNew, heightNew), borderValue=filled_color)

        return imgRotation

# 图像矫正函数调用
def imgTransform(image_file,img):
    try:
        img_correct = ImgCorrect(img)
        img_correct.img_lines()
        degree = img_correct.search_lines()
        correct_image = img_correct.rotate_image(degree)  # 图像矫正
    except:
        print("矫正失败:"+image_file)
        return img
    else:
        return correct_image

# 图像重命名
def add_prefix_subfolders(input_path,out_path):             # 定义函数名称
    filelist = os.listdir(input_path)        # 取路径下的文件名，生成列表
    i = 0
    for item in filelist:
        if item.endswith('.jpg'):
            src = os.path.join(os.path.abspath(input_path), item)  # 原图的地址
            dst = os.path.join(os.path.abspath(out_path), 'a'+str(i) + '.jpg')  # 新图的地址（这里可以把str(folder) + '_' + str(i) + '.jpg'改成你想改的名称）
            os.rename(src, dst)
            print('converting %s to %s ...' % (src, dst))
            i += 1

# 图像矫正并保存
if __name__ == "__main__":
    time = time.strftime("%Y%m%d", time.localtime())
    input_path = r'I:/Images_OCR/after_image/med2_429'
    out_path = r'I:/Images_OCR/after_image/Voc_med/images'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    imgs_lists = []
    for single_file in os.listdir(input_path):
        file_path = os.path.join(input_path, single_file)
        imgs_lists.append(file_path)
    i = 0
    for image_file in imgs_lists:
        img = cv2.imread(image_file)
        correct_image = imgTransform(image_file,img)  # 矫正
        newPath = os.path.join(os.path.abspath(out_path), time+'_'+str(i) + '.jpg')
        cv2.imwrite(newPath, correct_image)
        print(image_file + "====>" + newPath)
        i += 1