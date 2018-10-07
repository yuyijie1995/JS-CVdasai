# -*- coding=utf-8 -*-
##############################################################
# description:
#     data augmentation for obeject detection
# author:
#     maozezhong 2018-6-27
##############################################################

# 包括:
#     1. 裁剪(需改变bbox)
#     2. 平移(需改变bbox)
#     3. 改变亮度
#     4. 加噪声
#     5. 旋转角度(需要改变bbox)
#     6. 镜像(需要改变bbox)
#     7. cutout
# 注意:   
#     random.seed(),相同的seed,产生的随机数是一样的!!

import time
import random
import cv2
import os
import numpy as np
from skimage.util import random_noise
from skimage import exposure
import xml.dom.minidom
from xml.dom.minidom import Document
import math
import shutil
from PIL import Image
from xml_helper import *
import scipy.misc






def show_pic(img, bboxes=None):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    '''
    cv2.imwrite('./1.jpg', img)
    img = cv2.imread('./1.jpg')
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(img,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,255,0),3) 
    cv2.namedWindow('pic', 0)  # 1表示原图
    cv2.moveWindow('pic', 0, 0)
    cv2.resizeWindow('pic', 1200,800)  # 可视化的图片大小
    cv2.imshow('pic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    os.remove('./1.jpg')

# 图像均为cv2读取
class DataAugmentForObjectDetection():
    def __init__(self, rotation_rate=1, max_rotation_angle=1,
                crop_rate=1, shift_rate=1, change_light_rate=1,
                add_noise_rate=1, flip_rate=1,
                cutout_rate=1, cut_out_length=50, cut_out_holes=1, cut_out_threshold=0.5):
        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.flip_rate = flip_rate
        self.cutout_rate = cutout_rate

        self.cut_out_length = cut_out_length
        self.cut_out_holes = cut_out_holes
        self.cut_out_threshold = cut_out_threshold
    
    # 加噪声
    def _addNoise(self, img,bboxes,filename):
        '''
        输入:
            img:图像array
        输出:
            加噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
        '''
        # random.seed(int(time.time())) 
        # return random_noise(img, mode='gaussian', seed=int(time.time()), clip=True)*255
        noi_img=random_noise(img, mode='gaussian', clip=True) * 255

        noi_bboxes = list()
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]

            # 加入list中
            noi_bboxes.append([xmin, ymin, xmax, ymax])
        #im = Image.fromarray(noi_img)
        #im.save("noi_img/noi%s.jpg"%filename)
        scipy.misc.imsave('newimg/noi%s.jpg' % filename, img)
        #scipy.misc.toimage(noi_img, cmin=0.0, cmax=...).save('noi_img/noi%s.jpg' % filename)
        src_img_path = './newimg/'
        src_xml_path = './newxml'
        # writeXml(anno_new_path, 'P%s_%s' % (angles[i], file_name), w, h, d, gt_new)

        im = Image.open((src_img_path + 'noi%s' % filename + '.jpg'))
        width, height = im.size

        # open the crospronding txt file
        # gt = open(src_txt_dir + '/' + img + '.txt').read().splitlines()
        # gt = open(src_txt_dir + '/gt_' + img + '.txt').read().splitlines()
        gt = noi_bboxes
        # write in xml file
        # os.mknod(src_xml_dir + '/' + img + '.xml')
        xml_file = open((src_xml_path + '/' + 'noi%s' % filename + '.xml'), 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('    <folder>VOC2007</folder>\n')
        xml_file.write('    <filename>' + str('noi%s' % filename) + '.jpg' + '</filename>\n')
        xml_file.write('    <size>\n')
        xml_file.write('        <width>' + str(width) + '</width>\n')
        xml_file.write('        <height>' + str(height) + '</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('    </size>\n')

        # write the region of image on xml file
        for boxes in gt:
            # spt = img_each_label.split(' ')  # 这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
            xml_file.write('    <object>\n')
            # spt[0] = 'person'
            xml_file.write('        <name>' + str('person') + '</name>\n')
            xml_file.write('        <pose>Unspecified</pose>\n')
            xml_file.write('        <truncated>0</truncated>\n')
            xml_file.write('        <difficult>0</difficult>\n')
            xml_file.write('        <bndbox>\n')
            xml_file.write('            <xmin>' + str(int(float(boxes[0]))) + '</xmin>\n')
            xml_file.write('            <ymin>' + str(int(float(boxes[1]))) + '</ymin>\n')
            xml_file.write('            <xmax>' + str(int(float(boxes[2]))) + '</xmax>\n')
            xml_file.write('            <ymax>' + str(int(float(boxes[3]))) + '</ymax>\n')
            xml_file.write('        </bndbox>\n')
            xml_file.write('    </object>\n')

        xml_file.write('</annotation>')
        return noi_img

    
    # 调整亮度
    def _changeLight(self, img,bboxes,filename):
        # random.seed(int(time.time()))
        flag = random.uniform(0.5, 1.5) #flag>1为调暗,小于1为调亮
        lig_img=exposure.adjust_gamma(img, flag)


        lig_bboxes = list()
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]

            # 加入list中
            lig_bboxes.append([xmin, ymin, xmax, ymax])

        scipy.misc.toimage(lig_img, cmin=0.0, cmax=...).save('newimg/lig%s.jpg' % filename)
        src_img_path = './newimg/'
        src_xml_path = './newxml'
        # writeXml(anno_new_path, 'P%s_%s' % (angles[i], file_name), w, h, d, gt_new)

        im = Image.open((src_img_path + 'lig%s' % filename + '.jpg'))
        width, height = im.size

        # open the crospronding txt file
        # gt = open(src_txt_dir + '/' + img + '.txt').read().splitlines()
        # gt = open(src_txt_dir + '/gt_' + img + '.txt').read().splitlines()
        gt = lig_bboxes
        # write in xml file
        # os.mknod(src_xml_dir + '/' + img + '.xml')
        xml_file = open((src_xml_path + '/' + 'lig%s' % filename + '.xml'), 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('    <folder>VOC2007</folder>\n')
        xml_file.write('    <filename>' + str('lig%s' % filename) + '.jpg' + '</filename>\n')
        xml_file.write('    <size>\n')
        xml_file.write('        <width>' + str(width) + '</width>\n')
        xml_file.write('        <height>' + str(height) + '</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('    </size>\n')

        # write the region of image on xml file
        for boxes in gt:
            # spt = img_each_label.split(' ')  # 这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
            xml_file.write('    <object>\n')
            # spt[0] = 'person'
            xml_file.write('        <name>' + str('person') + '</name>\n')
            xml_file.write('        <pose>Unspecified</pose>\n')
            xml_file.write('        <truncated>0</truncated>\n')
            xml_file.write('        <difficult>0</difficult>\n')
            xml_file.write('        <bndbox>\n')
            xml_file.write('            <xmin>' + str(int(float(boxes[0]))) + '</xmin>\n')
            xml_file.write('            <ymin>' + str(int(float(boxes[1]))) + '</ymin>\n')
            xml_file.write('            <xmax>' + str(int(float(boxes[2]))) + '</xmax>\n')
            xml_file.write('            <ymax>' + str(int(float(boxes[3]))) + '</ymax>\n')
            xml_file.write('        </bndbox>\n')
            xml_file.write('    </object>\n')

        xml_file.write('</annotation>')
        return lig_img
    
    # cutout
    def _cutout(self, img, bboxes,filename, length=50, n_holes=1, threshold=0.5):
        '''
        原版本：https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        Randomly mask out one or more patches from an image.
        Args:
            img : a 3D numpy array,(h,w,c)
            bboxes : 框的坐标
            n_holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
        '''
        
        def cal_iou(boxA, boxB):
            '''
            boxA, boxB为两个框，返回iou
            boxB为bouding box
            '''

            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            if xB <= xA or yB <= yA:
                return 0.0

            # compute the area of intersection rectangle
            interArea = (xB - xA + 1) * (yB - yA + 1)

            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            # iou = interArea / float(boxAArea + boxBArea - interArea)
            iou = interArea / float(boxBArea)

            # return the intersection over union value
            return iou

        # 得到h和w
        if img.ndim == 3:
            h,w,c = img.shape
        else:
            _,h,w,c = img.shape
        
        mask = np.ones((h,w,c), np.float32)

        for n in range(n_holes):
            
            chongdie = True    #看切割的区域是否与box重叠太多
            
            while chongdie:
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - length // 2, 0, h)    #numpy.clip(a, a_min, a_max, out=None), clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)

                chongdie = False
                for box in bboxes:
                    if cal_iou([x1,y1,x2,y2], box) > threshold:
                        chongdie = True
                        break
            
            mask[y1: y2, x1: x2, :] = 0.
        
        # mask = np.expand_dims(mask, axis=0)
        img = img * mask
        cut_bboxes = list()
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]

            # 加入list中
            cut_bboxes.append([xmin, ymin, xmax, ymax])

        scipy.misc.imsave('newimg/cut%s.jpg' % filename, img)
        #scipy.misc.toimage(img, cmin=0.0, cmax=...).save('cut_img/cut%s.jpg' % filename)
        src_img_path = './newimg/'
        src_xml_path = './newxml'
        # writeXml(anno_new_path, 'P%s_%s' % (angles[i], file_name), w, h, d, gt_new)

        im = Image.open((src_img_path + 'cut%s' % filename + '.jpg'))
        width, height = im.size

        # open the crospronding txt file
        # gt = open(src_txt_dir + '/' + img + '.txt').read().splitlines()
        # gt = open(src_txt_dir + '/gt_' + img + '.txt').read().splitlines()
        gt = cut_bboxes
        # write in xml file
        # os.mknod(src_xml_dir + '/' + img + '.xml')
        xml_file = open((src_xml_path + '/' + 'cut%s' % filename + '.xml'), 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('    <folder>VOC2007</folder>\n')
        xml_file.write('    <filename>' + str('cut%s' % filename) + '.jpg' + '</filename>\n')
        xml_file.write('    <size>\n')
        xml_file.write('        <width>' + str(width) + '</width>\n')
        xml_file.write('        <height>' + str(height) + '</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('    </size>\n')

        # write the region of image on xml file
        for boxes in gt:
            # spt = img_each_label.split(' ')  # 这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
            xml_file.write('    <object>\n')
            # spt[0] = 'person'
            xml_file.write('        <name>' + str('person') + '</name>\n')
            xml_file.write('        <pose>Unspecified</pose>\n')
            xml_file.write('        <truncated>0</truncated>\n')
            xml_file.write('        <difficult>0</difficult>\n')
            xml_file.write('        <bndbox>\n')
            xml_file.write('            <xmin>' + str(int(float(boxes[0]))) + '</xmin>\n')
            xml_file.write('            <ymin>' + str(int(float(boxes[1]))) + '</ymin>\n')
            xml_file.write('            <xmax>' + str(int(float(boxes[2]))) + '</xmax>\n')
            xml_file.write('            <ymax>' + str(int(float(boxes[3]))) + '</ymax>\n')
            xml_file.write('        </bndbox>\n')
            xml_file.write('    </object>\n')

        xml_file.write('</annotation>')

        return img

    # 旋转
    def _rotate_img_bbox(self, img, bboxes,filename, angle=5, scale=1.):
        '''
        参考:https://blog.csdn.net/u014540717/article/details/53301195crop_rate
        输入:
            img:图像array,(h,w,c)
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            angle:旋转角度
            scale:默认1
        输出:
            rot_img:旋转后的图像array
            rot_bboxes:旋转后的boundingbox坐标list
        '''
        #---------------------- 旋转图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        # 角度变弧度
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(math.sin(rangle)*h) + abs(math.cos(rangle)*w))*scale
        nh = (abs(math.cos(rangle)*h) + abs(math.sin(rangle)*w))*scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0,2] += rot_move[0]
        rot_mat[1,2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        #---------------------- 矫正bbox坐标 ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rot_bboxes = list()
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            point1 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymin, 1]))
            point2 = np.dot(rot_mat, np.array([xmax, (ymin+ymax)/2, 1]))
            point3 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymax, 1]))
            point4 = np.dot(rot_mat, np.array([xmin, (ymin+ymax)/2, 1]))
            # 合并np.array
            concat = np.vstack((point1, point2, point3, point4))
            # 改变array类型
            concat = concat.astype(np.int32)
            # 得到旋转后的坐标
            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = rx
            ry_min = ry
            rx_max = rx+rw
            ry_max = ry+rh
            # 加入list中
            rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])

        scipy.misc.toimage(rot_img, cmin=0.0, cmax=...).save('newimg/rot%s.jpg' % filename)
        src_img_path = './newimg/'
        src_xml_path = './newxml'
        # writeXml(anno_new_path, 'P%s_%s' % (angles[i], file_name), w, h, d, gt_new)

        im = Image.open((src_img_path + 'rot%s' % filename + '.jpg'))
        width, height = im.size

        # open the crospronding txt file
        # gt = open(src_txt_dir + '/' + img + '.txt').read().splitlines()
        # gt = open(src_txt_dir + '/gt_' + img + '.txt').read().splitlines()
        gt = rot_bboxes
        # write in xml file
        # os.mknod(src_xml_dir + '/' + img + '.xml')
        xml_file = open((src_xml_path + '/' + 'rot%s' % filename + '.xml'), 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('    <folder>VOC2007</folder>\n')
        xml_file.write('    <filename>' + str('rot%s' % filename) + '.jpg' + '</filename>\n')
        xml_file.write('    <size>\n')
        xml_file.write('        <width>' + str(width) + '</width>\n')
        xml_file.write('        <height>' + str(height) + '</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('    </size>\n')

        # write the region of image on xml file
        for boxes in gt:
            # spt = img_each_label.split(' ')  # 这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
            xml_file.write('    <object>\n')
            # spt[0] = 'person'
            xml_file.write('        <name>' + str('person') + '</name>\n')
            xml_file.write('        <pose>Unspecified</pose>\n')
            xml_file.write('        <truncated>0</truncated>\n')
            xml_file.write('        <difficult>0</difficult>\n')
            xml_file.write('        <bndbox>\n')
            xml_file.write('            <xmin>' + str(int(float(boxes[0]))) + '</xmin>\n')
            xml_file.write('            <ymin>' + str(int(float(boxes[1]))) + '</ymin>\n')
            xml_file.write('            <xmax>' + str(int(float(boxes[2]))) + '</xmax>\n')
            xml_file.write('            <ymax>' + str(int(float(boxes[3]))) + '</ymax>\n')
            xml_file.write('        </bndbox>\n')
            xml_file.write('    </object>\n')

        xml_file.write('</annotation>')
        
        return rot_img, rot_bboxes

    # 裁剪
    def _crop_img_bboxes(self, img, bboxes,filename):
        '''
        裁剪后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            crop_img:裁剪后的图像array
            crop_bboxes:裁剪后的bounding box的坐标list
        '''
        #---------------------- 裁剪图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w   #裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])
        
        d_to_left = x_min           #包含所有目标框的最小框到左边的距离
        d_to_right = w - x_max      #包含所有目标框的最小框到右边的距离
        d_to_top = y_min            #包含所有目标框的最小框到顶端的距离
        d_to_bottom = h - y_max     #包含所有目标框的最小框到底部的距离

        #随机扩展这个最小框
        crop_x_min = int(x_min - random.uniform(0, d_to_left))
        crop_y_min = int(y_min - random.uniform(0, d_to_top))
        crop_x_max = int(x_max + random.uniform(0, d_to_right))
        crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

        # 随机扩展这个最小框 , 防止别裁的太小
        # crop_x_min = int(x_min - random.uniform(d_to_left//2, d_to_left))
        # crop_y_min = int(y_min - random.uniform(d_to_top//2, d_to_top))
        # crop_x_max = int(x_max + random.uniform(d_to_right//2, d_to_right))
        # crop_y_max = int(y_max + random.uniform(d_to_bottom//2, d_to_bottom))

        #确保不要越界
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        
        #---------------------- 裁剪boundingbox ----------------------
        #裁剪后的boundingbox坐标计算
        crop_bboxes = list()
        for bbox in bboxes:
            crop_bboxes.append([bbox[0]-crop_x_min, bbox[1]-crop_y_min, bbox[2]-crop_x_min, bbox[3]-crop_y_min])
        #crop_bboxes.append()
        scipy.misc.toimage(crop_img, cmin=0.0, cmax=...).save('newimg/crop%s.jpg'%filename)
        src_img_path='./newimg/'
        src_xml_path='./newxml'
        #writeXml(anno_new_path, 'P%s_%s' % (angles[i], file_name), w, h, d, gt_new)

        im = Image.open((src_img_path  + 'crop%s'%filename + '.jpg'))
        width, height = im.size

        # open the crospronding txt file
        #gt = open(src_txt_dir + '/' + img + '.txt').read().splitlines()
        # gt = open(src_txt_dir + '/gt_' + img + '.txt').read().splitlines()
        gt=crop_bboxes
        # write in xml file
        # os.mknod(src_xml_dir + '/' + img + '.xml')
        xml_file = open((src_xml_path + '/' + 'crop%s'%filename + '.xml'), 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('    <folder>VOC2007</folder>\n')
        xml_file.write('    <filename>' + str('crop%s'%filename) + '.jpg' + '</filename>\n')
        xml_file.write('    <size>\n')
        xml_file.write('        <width>' + str(width) + '</width>\n')
        xml_file.write('        <height>' + str(height) + '</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('    </size>\n')

        # write the region of image on xml file
        for boxes in gt:
            #spt = img_each_label.split(' ')  # 这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
            xml_file.write('    <object>\n')
            #spt[0] = 'person'
            xml_file.write('        <name>' + str('person') + '</name>\n')
            xml_file.write('        <pose>Unspecified</pose>\n')
            xml_file.write('        <truncated>0</truncated>\n')
            xml_file.write('        <difficult>0</difficult>\n')
            xml_file.write('        <bndbox>\n')
            xml_file.write('            <xmin>' + str(int(float(boxes[0]))) + '</xmin>\n')
            xml_file.write('            <ymin>' + str(int(float(boxes[1]))) + '</ymin>\n')
            xml_file.write('            <xmax>' + str(int(float(boxes[2]))) + '</xmax>\n')
            xml_file.write('            <ymax>' + str(int(float(boxes[3]))) + '</ymax>\n')
            xml_file.write('        </bndbox>\n')
            xml_file.write('    </object>\n')

        xml_file.write('</annotation>')


        #return crop_img, crop_bboxes
  
    # 平移
    def _shift_pic_bboxes(self, img, bboxes,filename):
        '''
        参考:https://blog.csdn.net/sty945/article/details/79387054
        平移后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            shift_img:平移后的图像array
            shift_bboxes:平移后的bounding box的坐标list
        '''
        #---------------------- 平移图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w   #裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])
        
        d_to_left = x_min           #包含所有目标框的最大左移动距离
        d_to_right = w - x_max      #包含所有目标框的最大右移动距离
        d_to_top = y_min            #包含所有目标框的最大上移动距离
        d_to_bottom = h - y_max     #包含所有目标框的最大下移动距离

        x = random.uniform(-(d_to_left-1) / 3, (d_to_right-1) / 3)
        y = random.uniform(-(d_to_top-1) / 3, (d_to_bottom-1) / 3)
        
        M = np.float32([[1, 0, x], [0, 1, y]])  #x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        #---------------------- 平移boundingbox ----------------------
        shift_bboxes = list()
        for bbox in bboxes:
            shift_bboxes.append([bbox[0]+x, bbox[1]+y, bbox[2]+x, bbox[3]+y])

        scipy.misc.toimage(shift_img, cmin=0.0, cmax=...).save('newimg/shift%s.jpg' % filename)
        src_img_path = './newimg/'
        src_xml_path = './newxml'
        # writeXml(anno_new_path, 'P%s_%s' % (angles[i], file_name), w, h, d, gt_new)

        im = Image.open((src_img_path + 'shift%s' % filename + '.jpg'))
        width, height = im.size

        # open the crospronding txt file
        # gt = open(src_txt_dir + '/' + img + '.txt').read().splitlines()
        # gt = open(src_txt_dir + '/gt_' + img + '.txt').read().splitlines()
        gt = shift_bboxes
        # write in xml file
        # os.mknod(src_xml_dir + '/' + img + '.xml')
        xml_file = open((src_xml_path + '/' + 'shift%s' % filename + '.xml'), 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('    <folder>VOC2007</folder>\n')
        xml_file.write('    <filename>' + str('shift%s' % filename) + '.jpg' + '</filename>\n')
        xml_file.write('    <size>\n')
        xml_file.write('        <width>' + str(width) + '</width>\n')
        xml_file.write('        <height>' + str(height) + '</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('    </size>\n')

        # write the region of image on xml file
        for boxes in gt:
            # spt = img_each_label.split(' ')  # 这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
            xml_file.write('    <object>\n')
            # spt[0] = 'person'
            xml_file.write('        <name>' + str('person') + '</name>\n')
            xml_file.write('        <pose>Unspecified</pose>\n')
            xml_file.write('        <truncated>0</truncated>\n')
            xml_file.write('        <difficult>0</difficult>\n')
            xml_file.write('        <bndbox>\n')
            xml_file.write('            <xmin>' + str(int(float(boxes[0]))) + '</xmin>\n')
            xml_file.write('            <ymin>' + str(int(float(boxes[1]))) + '</ymin>\n')
            xml_file.write('            <xmax>' + str(int(float(boxes[2]))) + '</xmax>\n')
            xml_file.write('            <ymax>' + str(int(float(boxes[3]))) + '</ymax>\n')
            xml_file.write('        </bndbox>\n')
            xml_file.write('    </object>\n')

        xml_file.write('</annotation>')



        #return shift_img, shift_bboxes

    # 镜像
    def _filp_pic_bboxes(self, img, bboxes,filename):
        '''
            参考:https://blog.csdn.net/jningwei/article/details/78753607
            平移后的图片要包含所有的框
            输入:
                img:图像array
                bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            输出:
                flip_img:平移后的图像array
                flip_bboxes:平移后的bounding box的坐标list
        '''
        # ---------------------- 翻转图像 ----------------------
        import copy
        flip_img = copy.deepcopy(img)
        # if random.random() < 0.5:    #0.5的概率水平翻转，0.5的概率垂直翻转
        #     horizon = True
        #if random.random() < 0.5:

        # else:
        #     horizon = False
        #horizon = False
        h,w,_ = img.shape
        # if horizon: #水平翻转
        #     flip_img =  cv2.flip(flip_img, -1)
        # else:
            #flip_img = cv2.flip(flip_img, 0)
        flip_img = cv2.flip(flip_img, 1)

        # ---------------------- 调整boundingbox ----------------------
        flip_bboxes = list()
        for box in bboxes:
            x_min = box[0]
            y_min = box[1]
            x_max = box[2]
            y_max = box[3]
            # if horizon:
            #     flip_bboxes.append([w-x_max, y_min, w-x_min, y_max])
            # else:
            #     flip_bboxes.append([x_min, h-y_max, x_max, h-y_min])
            flip_bboxes.append([w-x_max, y_min, w-x_min, y_max])

        scipy.misc.toimage(flip_img, cmin=0.0, cmax=...).save('newimg/flip%s.jpg' % filename)
        src_img_path = './newimg/'
        src_xml_path = './newxml'
        # writeXml(anno_new_path, 'P%s_%s' % (angles[i], file_name), w, h, d, gt_new)

        im = Image.open((src_img_path + 'flip%s' % filename + '.jpg'))
        width, height = im.size

        # open the crospronding txt file
        # gt = open(src_txt_dir + '/' + img + '.txt').read().splitlines()
        # gt = open(src_txt_dir + '/gt_' + img + '.txt').read().splitlines()
        gt = flip_bboxes
        # write in xml file
        # os.mknod(src_xml_dir + '/' + img + '.xml')
        xml_file = open((src_xml_path + '/' + 'flip%s' % filename + '.xml'), 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('    <folder>VOC2007</folder>\n')
        xml_file.write('    <filename>' + str('flip%s' % filename) + '.jpg' + '</filename>\n')
        xml_file.write('    <size>\n')
        xml_file.write('        <width>' + str(width) + '</width>\n')
        xml_file.write('        <height>' + str(height) + '</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('    </size>\n')

        # write the region of image on xml file
        for boxes in gt:
            # spt = img_each_label.split(' ')  # 这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
            xml_file.write('    <object>\n')
            # spt[0] = 'person'
            xml_file.write('        <name>' + str('person') + '</name>\n')
            xml_file.write('        <pose>Unspecified</pose>\n')
            xml_file.write('        <truncated>0</truncated>\n')
            xml_file.write('        <difficult>0</difficult>\n')
            xml_file.write('        <bndbox>\n')
            xml_file.write('            <xmin>' + str(int(float(boxes[0]))) + '</xmin>\n')
            xml_file.write('            <ymin>' + str(int(float(boxes[1]))) + '</ymin>\n')
            xml_file.write('            <xmax>' + str(int(float(boxes[2]))) + '</xmax>\n')
            xml_file.write('            <ymax>' + str(int(float(boxes[3]))) + '</ymax>\n')
            xml_file.write('        </bndbox>\n')
            xml_file.write('    </object>\n')

        xml_file.write('</annotation>')

        #return flip_img, flip_bboxes

    def dataAugment(self, img, bboxes,filename):
        '''
        图像增强
        输入:
            img:图像array
            bboxes:该图像的所有框坐标
        输出:
            img:增强后的图像
            bboxes:增强后图片对应的box
        '''
        change_num = 0  #改变的次数
        print('------')
        # while change_num < 1:   #默认至少有一种数据增强生效
        #     if random.random() < self.crop_rate:        #裁剪
        #         print('裁剪')
        #         change_num += 1
        #         #img, bboxes = self._crop_img_bboxes(img, bboxes,filename)
        #         self._crop_img_bboxes(img, bboxes, filename)
        #
        #     if random.random() < self.rotation_rate:    #旋转
        #         print('旋转')
        #         change_num += 1
        #         angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
        #         #angle = random.sample([90, 180, 270],1)[0]
        #         scale = random.uniform(0.7, 0.8)
        #         #img, bboxes = self._rotate_img_bbox(img, bboxes,filename, angle, scale)
        #         self._rotate_img_bbox(img, bboxes, filename, angle, scale)
        #
        #     if random.random() < self.shift_rate:        #平移
        #         print('平移')
        #         change_num += 1
        #         #img, bboxes = self._shift_pic_bboxes(img, bboxes,filename)
        #         self._shift_pic_bboxes(img, bboxes, filename)
        #
        #     if random.random() > self.change_light_rate: #改变亮度
        #         print('亮度')
        #         change_num += 1
        #         #img = self._changeLight(img,bboxes,filename)
        #         self._changeLight(img, bboxes, filename)
        #
        #     if random.random() < self.add_noise_rate:    #加噪声
        #         print('加噪声')
        #         change_num += 1
        #         #img = self._addNoise(img,bboxes,filename)
        #         self._addNoise(img, bboxes, filename)
        #
        #     if random.random() < self.cutout_rate:  #cutout将一块位置随机置黑
        #         print('cutout')
        #         change_num += 1
        #         #img = self._cutout(img, bboxes, filename,length=self.cut_out_length, n_holes=self.cut_out_holes, threshold=self.cut_out_threshold)
        #         self._cutout(img, bboxes, filename, length=self.cut_out_length, n_holes=self.cut_out_holes,
        #                      threshold=self.cut_out_threshold)
        #     if random.random() < self.flip_rate:    #翻转
        #         print('翻转')
        #         change_num += 1
        #         #img, bboxes = self._filp_pic_bboxes(img, bboxes,filename)
        #         self._filp_pic_bboxes(img, bboxes, filename)
        #     print('\n')

        self._crop_img_bboxes(img, bboxes, filename)


        angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
        # angle = random.sample([90, 180, 270],1)[0]
        scale = random.uniform(0.7, 0.8)
        #img, bboxes = self._rotate_img_bbox(img, bboxes,filename, angle, scale)
        self._rotate_img_bbox(img, bboxes, filename, angle, scale)


        self._shift_pic_bboxes(img, bboxes, filename)


        self._changeLight(img, bboxes, filename)


        self._addNoise(img, bboxes, filename)


        self._cutout(img, bboxes, filename, length=self.cut_out_length, n_holes=self.cut_out_holes,
                     threshold=self.cut_out_threshold)

        self._filp_pic_bboxes(img, bboxes, filename)
        print('aug success')
        # print('------')
        #return img, bboxes



if __name__ == '__main__':

    ### test ###



    need_aug_num = 300

    dataAug = DataAugmentForObjectDetection()

    source_pic_root_path = './JPEGImages'
    source_xml_root_path = './Annotations'

    
    for parent, _, files in os.walk(source_pic_root_path):
        for file in files:

            pic_path = os.path.join(parent, file)
            xml_path = os.path.join(source_xml_root_path, file[:-4]+'.xml')
            coords = parse_xml(xml_path)        #解析得到box信息，格式为[[x_min,y_min,x_max,y_max,name]]
            coords = [coord[:4] for coord in coords]

            img = cv2.imread(pic_path)
            #show_pic(img, coords)    # 原图
            (filename,filename1)=os.path.splitext(file)
            #auged_img, auged_bboxes = dataAug.dataAugment(img, coords,filename)
            dataAug.dataAugment(img, coords, filename)


                #show_pic(auged_img, auged_bboxes)  # 强化后的图






