# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os
import argparse
import time
from PIL import Image
from scipy import interpolate
# import pylab as pl
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing


def path_arr(path):
    '''

    :param path:
    :return: count list
    '''
    name_list = []
    cnt = 0
    for rt, dirs, files in os.walk(path):
        for file in files:
            # if '_' in file and 'png' in file:
            # if file.endswith(".png"):
            # if 'png' in file:
            name_list.append(file)
            cnt += 1
    return name_list, cnt


def point_zeros(img):
    '''


    :param img: img
    :return: list[]  [x,y]
    '''
    point = []

    [point_tl_x, point_tl_y] = np.where(img[:, :, 0] > 0)
    [point_tr_x, point_tr_y] = np.where(img[:, :, 1] > 0)
    [point_br_x, point_br_y] = np.where(img[:, :, 2] > 0)
    [point_bl_x, point_bl_y] = np.where(img[:, :, 3] > 0)

    # [pointx, pointy] = np.where(img[:, :, 0] > 0)
    #point_tl_x = int(sum(point_tl_x) / len(point_tl_x))
    point_tl_x = np.mean(point_tl_x)
    point_tl_y = np.mean(point_tr_y)
    point_tr_x = np.mean(point_tr_x)
    point_tr_y = np.mean(point_tr_y)
    point_br_x = np.mean(point_br_x)
    point_br_y = np.mean(point_br_y)
    point_bl_x = np.mean(point_bl_x)
    point_bl_y = np.mean(point_bl_y)

    point_tl = [point_tl_y, point_tl_x]
    point_tr = [point_tr_y, point_tr_x]
    point_br = [point_br_y, point_br_x]
    point_bl = [point_bl_y, point_bl_x]
    point.append(point_tl)
    point.append(point_tr)
    point.append(point_br)
    point.append(point_bl)

    return point


def image_boundary(dst):
    '''
    get image boundary
    :param img:  png
    :return: png
    '''
    v_hist = dst[:, :, 3].sum(0)
    all_idx = np.arange(len(v_hist))
    v_non_zero_idx = all_idx[v_hist > 0]

    top = v_non_zero_idx.min()
    bot = v_non_zero_idx.max()

    # abc = dst[:, :, 3]
    # abcd = abc[:, 0]

    # print(abcd)

    h_hist = dst[:, :, 3].sum(1)
    all_idx = np.arange(len(h_hist))
    h_non_zero_idx = all_idx[h_hist > 0]

    lft = h_non_zero_idx.min()
    rht = h_non_zero_idx.max()

    bd_img = dst[lft:rht, top:bot, :]

    return bd_img


def image_point(dst):
    '''
    get four point in the image png
    :param image:
    :return: pts nd.array()  bn_img  nd.array
    '''
    # v_hist = dst
    # if v_hist.shape[2] == 3:
    #     v_hist_n = np.ones((v_hist.shape[0],v_hist.shape[1],v_hist.shape[2] + 1)) * 255
    #     v_hist_n[:,:,0] = v_hist[:,:,0]
    #     v_hist_n[:,:,1] = v_hist[:,:,1]
    #     v_hist_n[:,:,2] = v_hist[:,:,2]
    #     dst = v_hist_n

    v_hist = dst[:, :, 3].sum(1)
    all_idx = np.arange(len(v_hist))
    v_non_zero_idx = all_idx[v_hist > 0]

    top = v_non_zero_idx.min()
    bot = v_non_zero_idx.max()

    # abc = dst[:, :, 3]
    # abcd = abc[:, 0]

    # print(abcd)

    h_hist = dst[:, :, 3].sum(0)
    all_idx = np.arange(len(h_hist))
    h_non_zero_idx = all_idx[h_hist > 0]

    lft = h_non_zero_idx.min()
    rht = h_non_zero_idx.max()

    # bd_img = dst[lft:rht, top:bot, :]
    four_point = np.array([(lft, top), (rht, top), (rht, bot), (lft, bot)])
    return four_point


def image_point_float(dst):
    '''
    get four point in the image png
    :param image:
    :return: pts nd.array()  bn_img  nd.array
    '''
    # v_hist = dst
    # if v_hist.shape[2] == 3:
    #     v_hist_n = np.ones((v_hist.shape[0],v_hist.shape[1],v_hist.shape[2] + 1)) * 255
    #     v_hist_n[:,:,0] = v_hist[:,:,0]
    #     v_hist_n[:,:,1] = v_hist[:,:,1]
    #     v_hist_n[:,:,2] = v_hist[:,:,2]
    #     dst = v_hist_n

    v_hist = dst[:, :, 3].sum(1)
    all_idx = np.arange(len(v_hist))
    v_non_zero_idx = all_idx[v_hist > 0]

    top = v_non_zero_idx.min()
    bot = v_non_zero_idx.max()

    # abc = dst[:, :, 3]
    # abcd = abc[:, 0]

    # print(abcd)

    h_hist = dst[:, :, 3].sum(0)
    all_idx = np.arange(len(h_hist))
    h_non_zero_idx = all_idx[h_hist > 0]

    lft = h_non_zero_idx.min()
    rht = h_non_zero_idx.max()

    # bd_img = dst[lft:rht, top:bot, :]
    four_point = np.array([(lft, top), (rht, top), (rht, bot), (lft, bot)], np.float32)
    return four_point


def rotation(img, pts, rotation_min=-15, rotation_max=15):
    '''

    :param img: input image
    :param pts: input four point
    :param rotation_min:
    :param rotation_max:
    :return: image  png   point ndarray
    '''

    rows, cols, channel = img.shape
    ro_degr = np.random.randint(rotation_min, rotation_max)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ro_degr, 1)
    dst = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT)
    pts = image_point(dst)
    # cv2.imshow('rot_image',dst)
    # cv2.waitKey(0)
    # cv2.imwrite('ttt0.png',dst)
    return dst, pts, ro_degr


def rotation0(img, ro_degr):
    rows, cols, channel = img.shape
    # ro_degr = np.random.randint(rotation_min, rotation_max)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ro_degr, 1)
    dst = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT)
    # pts = image_point(dst)
    # [pointx, pointy] = np.where(dst[:,:,0]>0)
    # pointx = int(sum(pointx)/len(pointx))
    # pointy = int(sum(pointy)/len(pointy))
    # point = [pointy,pointx]
    point = point_zeros(dst)
    # print point, 'rodegr:', ro_degr
    # cv2.imshow('rotat', dst)
    # cv2.waitKey(0)

    # cv2.imwrite('ttt.png',dst)
    # print dst[np.where(dst[:,:,0]>0)]

    return dst, point


def scale(img, pts, scale_min=80, scale_max=120):
    '''
     scale  <140
    :param img: png
    :param pts: ndarray  4 point
    :param scale_min: int
    :param scale_max: int
    :return: image
    '''
    rows, cols, channel = img.shape
    scale_factor = np.random.randint(scale_min, scale_max) * 0.01
    # print("scale_factor=" + str(scale_factor))
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, scale_factor)
    dst = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT)
    pts = image_point(dst)
    # cv2.imshow('image_scal_after', dst)
    # cv2.waitKey(0)

    # print 'scale begin ...'
    return dst, pts, scale_factor


def scale0(img, scale_factor):
    rows, cols, channel = img.shape
    # scale_factor = np.random.randint(scale_min, scale_max) * 0.01
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, scale_factor)
    dst = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT)
    # [pointx,pointy] = np.where(dst[:,:,0]>0)
    # #[pointx, pointy] = np.nonzero(dst[:, :, 0])
    # # cv2.imshow('zero_scal_after',dst)
    # # cv2.waitKey(0)
    # pointx = int(sum(pointx)/len(pointx))
    # pointy = int(sum(pointy)/len(pointy))
    # #print pointy,pointx
    # point =[pointy,pointx]
    point = point_zeros(dst)

    return dst, point


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    size_x, size_y, channel = image.shape
    (tl, tr, br, bl) = image_point(image)
    weight = tr[0] - tl[0]
    height = bl[1] - tl[1]
    top_x = tl[0]
    top_y = tl[1]
    dst = get_four(top_x, top_y, weight, height, size_x, size_y)

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)  # rect  dst
    warped = cv2.warpPerspective(image, M, (size_x, size_y))

    # return the warped image
    return warped, rect, dst


# use four point and order generate img
def get_four(top_x, top_y, weight, height, size_x, size_y):
    # A,B,C,D  top_left top_right, bot_right,bot_left
    a_x = np.random.randint(0, top_x)
    a_y = np.random.randint(0, top_y)
    b_x = np.random.randint(top_x + weight, size_x)
    b_y = np.random.randint(0, top_y)
    c_x = np.random.randint(top_x + weight, size_x)
    c_y = np.random.randint(top_y + height, size_y)
    d_x = np.random.randint(0, top_x)
    d_y = np.random.randint(top_y + height, size_y)
    four_point = np.array([(a_x, a_y), (b_x, b_y), (c_x, c_y), (d_x, d_y)], dtype="float32")

    return four_point


def transform(img, pt_bf):
    '''
    perspect transform  four point
    :param img:
    :param pts:
    :return:
    '''
    # size_x, size_y, channel = img.shape
    # (tl, tr, br, bl) = image_point(img)
    # weight = tr[0]-tl[0]
    # height = bl[1]-tl[1]
    # top_x = tl[0]
    # top_y = tl[1]
    # dst = get_four(top_x, top_y, weight, height, size_x, size_y)
    # rect = image_point(img)
    img, rect, dst = four_point_transform(img, pt_bf)
    # M = cv2.getPerspectiveTransform(rect, dst)  # rect  dst
    # warped = cv2.warpPerspective(img, M, (weight, height))
    pts = image_point(img)
    # cv2.imshow('img_transform_after', img)
    # cv2.waitKey(0)

    # print 'transform begin...'
    return img, pts, rect, dst


def transform0(img, rect, dst):
    '''

    :param img: image
    :param rect: source four point
    :param dst: des  four point
    :return:
    '''
    # img = four_point_transform(img, pts)
    size_x, size_y, channel = img.shape
    M = cv2.getPerspectiveTransform(rect, dst)  # rect  dst
    warped = cv2.warpPerspective(img, M, (size_x, size_y))

    # M = cv2.getPerspectiveTransform(rect, dst)  # rect  dst
    # warped = cv2.warpPerspective(img, M, (weight, height))
    # [pointx, pointy] = np.where(warped[:, :, 0] > 0)
    # # [pointx, pointy] = np.nonzero(dst[:, :, 0])
    # # cv2.imshow('zero_transform_after', warped)
    # # cv2.waitKey(0)
    #
    # pointx = int(sum(pointx) / len(pointx))
    # pointy = int(sum(pointy) / len(pointy))
    # print 'trsport',pointy, pointx,size_x,size_x
    # point = [pointy, pointx]
    point = point_zeros(warped)

    return img, point


def transform_cycle_128(img, re_point, size_x, size_y, count):
    # img, rect, dst = four_point_transform(img, pt_bf)
    # # M = cv2.getPerspectiveTransform(rect, dst)  # rect  dst
    # # warped = cv2.warpPerspective(img, M, (weight, height))
    # pts = image_point(img)
    # # cv2.imshow('img_transform_after', img)
    # # cv2.waitKey(0)
    #
    # # print 'transform begin...'
    img = cv2.resize(img, (size_x, size_y))

    dst = np.array([[re_point[count, 0], re_point[count, 1]], [re_point[count, 2], re_point[count, 3]],
                    [re_point[count, 4], re_point[count, 5]], [re_point[count, 6], re_point[count, 7]]], np.float32)

    rect = image_point_float(img)
    M = cv2.getPerspectiveTransform(rect, dst)  # rect  dst
    warped = cv2.warpPerspective(img, M, (size_x, size_y))

    return warped, dst


def add_background(image, back_image, size_x, size_y, name, path_out, four_zero):
    # image = image_boundary(image)
    # 64*64   0.8     证件
    _x, _y, channel = image.shape
    x_scale = np.random.randint(size_x * 0.5, size_y * 0.99)
    # y_scale = np.random.randint(size_x*0.4, size_y*0.99)
    # x_scale = np.random.randint(size_x * 0.98, size_x * 0.99)
    # x_scale = size_x
    #
    y_scale = int(x_scale * _y / float(_x))

    new_img = cv2.resize(image, (x_scale, y_scale))
    # (tl, tr, br, bl)=image_point(new_img)
    # cv2.imwrite(name_yuan, new_img)
    foreward_name_path = path_out + 'temp/foreward_{}.png'.format(name)
    cv2.imwrite(foreward_name_path, new_img)

    # cv2.imwrite(".png", new_img)

    # 背景
    size = (size_x, size_y)
    # backimage_temp = cv2.imread(back_image, -1)
    backimage0 = cv2.resize(back_image, size)
    # if backimage0.shape!=3:
    #     print
    back_rows, back_cols, back_channel = backimage0.shape
    xx = np.random.randint(0, back_cols - y_scale)  # 起点随机位置
    yy = np.random.randint(0, back_rows - x_scale)

    back_name_path = path_out + 'temp/back_{}.png'.format(name)
    cv2.imwrite(back_name_path, backimage0)

    # cv2.imwrite(path_out+'{}.png'.format(1), new_img)增加背景
    if os.name == 'nt':
        cmd = 'magick convert ' + back_name_path  # linux
    else:
        cmd = 'convert ' + back_name_path  # linux'
    # cmd = 'magick convert ' + back_name_path
    # cmd = 'convert ' + back_name_path  # linux

    to_sub_cmd = ' -compose over '
    to_sub_cmd = to_sub_cmd + foreward_name_path
    to_sub_cmd = to_sub_cmd + ' -geometry' + ' +{}+{} '.format(xx, yy) + '-composite '
    # out_fn = path_out + name + '_{}'.format(ro_degr) + '.png'
    out_fn = path_out + 'image/temp/{}.jpg'.format(name)
    cmd = cmd + to_sub_cmd + out_fn
    # print(cmd)
    os.system(cmd)
    # print 'add_back'
    image = cv2.imread(out_fn, -1)

    # cv2.imshow('zero_back_after', image)
    # cv2.waitKey(0)
    rate_x = x_scale / float(_x)
    rate_y = y_scale / float(_y)
    point = add_back_point(four_zero, xx, yy, rate_x, rate_y)

    # pointx = xx + four_zero[0]*(x_scale/float(_x))
    # pointy = yy + four_zero[1]*float(y_scale/float(_y))
    # print pointx, pointy,'xy',xx,yy,'image',_x,_y,'resi',x_scale,y_scale
    # point = [int(pointx), int(pointy)]
    #
    # pts ='{} {} {} {}'.format(xx, yy, x_scale, y_scale)



    return image, point
def add_background_pil(image, back_image, size_x, size_y, name, path_out, four_zero):
    _x, _y, channel = image.shape
    x_scale = size_x
    y_scale = int(x_scale * _y / float(_x))
    xx=0
    yy=0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)  # change colour
    image = cv2.resize(image, (size_x, size_y))
    image = Image.fromarray(image)

    back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)  # change colour
    back_image = cv2.resize(back_image, (size_x, size_y))
    back_image = Image.fromarray(back_image)

    back_image.paste(image,(xx,yy),image)

    back_image.save(path_out+'image' + os.sep + 'temp/{}.jpg'.format(name))
    rate_x = x_scale / float(_x)
    rate_y = y_scale / float(_y)
    point = add_back_point(four_zero, xx, yy, rate_x, rate_y)
    return back_image, point

def add_back_point(point, xx, yy, rate_x, rate_y):
    '''

    :param point:
    :param xx:
    :param yy:
    :return: point list
    '''
    point[0][0] = int(xx + rate_x * point[0][0])
    point[0][1] = int(yy + rate_y * point[0][1])
    point[1][0] = int(xx + rate_x * point[1][0])
    point[1][1] = int(yy + rate_y * point[1][1])
    point[2][0] = int(xx + rate_x * point[2][0])
    point[2][1] = int(yy + rate_y * point[2][1])
    point[3][0] = int(xx + rate_x * point[3][0])
    point[3][1] = int(yy + rate_y * point[3][1])

    return point


def add_background0(image, back_image, size_x, size_y, name, path_out, pts, four_zero):
    # image = image_boundary(image)
    # 64*64   0.8     证件
    # size_x, size_y, channel = image.shape
    # x_scale = np.random.randint(size_x * 0.7, size_y * 0.8)
    # y_scale = np.random.randint(size_x * 0.7, size_y * 0.8)
    xx, yy, x_scale, y_scale = pts.split(' ')
    xx = int(xx)
    yy = int(yy)
    x_scale = int(x_scale)
    y_scale = int(y_scale)

    cv2.imshow('zero_back_before', image)
    cv2.waitKey(0)
    # (tl, tr, br, bl)=image_point(new_img)
    # cv2.imwrite(name_yuan, new_img)米秒个亿
    foreward_name_path = path_out + 'temp/zero_foreward_{}.png'.format(name)
    cv2.imwrite(foreward_name_path, )

    # cv2.imwrite(".png", new_img)

    # 背景
    size = (size_x, size_y)
    # backimage_temp = cv2.imread(back_image, -1)
    backimage0 = cv2.resize(back_image, size)
    back_rows, back_cols, back_channel = backimage0.shape
    back_name_path = path_out + 'temp/zero_back_{}.png'.format(name)
    cv2.imwrite(back_name_path, backimage0)

    # cv2.imwrite(path_out+'{}.png'.format(1), new_img)增加背景
    cmd = 'magick convert ' + back_name_path
    to_sub_cmd = ' -compose over '
    to_sub_cmd = to_sub_cmd + foreward_name_path
    to_sub_cmd = to_sub_cmd + ' -geometry' + ' +{}+{} '.format(xx, yy) + '-composite '
    # out_fn = path_out + name + '_{}'.format(ro_degr) + '.png'
    out_fn = path_out + 'image/temp/zero_{}.jpg'.format(name)
    cmd = cmd + to_sub_cmd + out_fn
    # print(cmd)
    os.system(cmd)
    # print 'add_back_zero'
    image = cv2.imread(out_fn, -1)

    [pointx, pointy] = np.where(image[:, :, 0] > 0)
    # [pointx, pointy] = np.nonzero(dst[:, :, 0])
    cv2.imshow('zero_back_after', image)
    cv2.waitKey(0)
    pointx = xx + four_zero[0]
    pointy = yy + four_zero[1]
    # print pointy, pointx
    point = [pointy, pointx]
    pts = '{} {} {} {}'.format(xx, yy, x_scale, y_scale)

    return image, point


def get_id_crop_from_std_card(img):
    std_tl = (326, 517)
    std_br = (898, 589)
    std_wh = (1000, 643)

    assert (7.9 < (std_br[0] - std_tl[0]) / (std_br[1] - std_tl[1]) < 8.1)

    tl = (int(std_tl[0] / std_wh[0] * img.shape[1]), int(std_tl[1] / std_wh[1] * img.shape[0]))
    br = (int(std_br[0] / std_wh[0] * img.shape[1]), int(std_br[1] / std_wh[1] * img.shape[0]))

    id_img = img[tl[1]:br[1], tl[0]:br[0], :]

    cv2.imshow("id_img", id_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return id_img


def calc_rand_level_trans_seq(start=0, stop=1, num=256):
    x = np.linspace(start, stop, 4)
    # x=[  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.]
    y = x.copy()

    y[0] += random.randint(0, 100) * 0.001
    y[3] += random.randint(-100, 0) * 0.001

    y[1] += random.randint(-100, 100) * 0.002
    y[2] += random.randint(-100, 100) * 0.002

    xnew = np.linspace(start, stop, num)
    # pl.plot(x, y, "ro")

    # for kind in ["nearest", "zero", "slinear", "quadratic", "cubic"]:  # 插值方式
    for kind in ["quadratic"]:  # 插值方式
        # "nearest","zero"为阶梯插值
        # slinear 线性插值
        # "quadratic","cubic" 为2阶、3阶B样条曲线插值
        f = interpolate.interp1d(x, y, kind=kind)
        # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
        ynew = f(xnew)
        ynew[ynew < 0] = 0
        ynew[ynew > 1] = 1
        # plt.plot(xnew, ynew, label=str(kind))

    # pl.legend(loc="lower right")
    # pl.show()

    ynew = (ynew * 255).astype("uint8")
    return ynew


def trans_level_by_seq(img, seq):
    if len(img.shape) == 3 and img.shape[2] == 4:
        img_3c = img[:, :, 0:3]
    else:
        img_3c = img
    img_st = img_3c.copy()
    for i in range(256):
        img_3c[img_st == i] = seq[i]

    # debug
    if 0:
        img_new = img_st.copy()
        for row in img_new:
            for pt in row:
                for x in range(pt.shape[0]):
                    pt[x] = seq[pt[x]]
        assert (np.array_equal(img, img_new))

    return img


def trans_level_by_rand_seq(img):
    rand_seq = calc_rand_level_trans_seq()
    return trans_level_by_seq(img, rand_seq)


def edge_transparent(img):
    h, w, _ = img.shape

    edge_w = int(w * 0.05)
    edge_h = int(h * 0.05)

    edge_w = 60
    edge_h = edge_w

    mask = np.zeros((h, w), img.dtype)
    # mask = np.ones((h, w), img.dtype) * 255

    # mask[0:edge_h, :]
    mask[0:edge_h, :] = np.random.randint(0, 255, (edge_h, w))
    mask[h - edge_h:h, :] = np.random.randint(0, 255, (edge_h, w))
    mask[:, 0:edge_w] = np.random.randint(0, 255, (h, edge_w))
    mask[:, w - edge_w:w] = np.random.randint(0, 255, (h, edge_w))

    for i in range(edge_h):
        mask[i, :] = np.random.randint(0, 100 - i, (1, w))
        mask[h - 1 - i, :] = np.random.randint(0, 100 - i, (1, w))

    for j in range(edge_w):
        mask[:, j] = np.random.randint(0, 100 - j, (h))
        mask[:, w - 1 - j] = np.random.randint(0, 100 - j, (h))

    thh = 55
    mask[mask > thh] = 255
    mask[mask <= thh] = 0

    cv2.imshow("mask", mask)
    cv2.waitKey()
    cv2.destroyAllWindows()

    m1 = np.random.exponential(1, (edge_h, w)) > 0.5
    a = a[m1]
    b = a.reshape(-1)
    b[b > 0.5] = 1
    a = a


# def transform_cycle():




def main(n=50000, nm=chr(np.random.randint(97, 122)), rotation_key=True, scale_key=True, transform_key=True,
         transform_cycle=False, adjustlevel_key=True,
         back_ground_key=True,
         size_x=128, size_y=128):
    # path = 'E:/kingsoft/image_shenfenzheng/test/lhl_sfz1.png'
    # path = 'E:/kingsoft/image_shenfenzheng/id_card_data/comb_50/comb_50/'
    # path='E:/kingsoft/image_shenfenzheng/id_card_data/comb_502/comb_50/'
    # path = 'E:/kingsoft/image_shenfenzheng/id_card_data/comb_3825/'

    # path = r'E:\kingsoft\image_shenfenzheng\id_card_data\comb_7650/'
    # path_out = 'E:/kingsoft/image_shenfenzheng/test/'
    # back_path = 'E:/kingsoft/image_shenfenzheng/test/background/'

    # path = '/root/lixin/gen_data/comb_7650/'
    # path_out = '/root/lixin/gen_data/20w_0.5_15_0913/'
    # back_path = '/root/lixin/gen_data/background/'
    # re_point_path = '/root/lixin/gen_data/id_card/re1.txt'
    #
    # re_point = np.loadtxt(re_point_path, delimiter=" ")


    path = r'F:\data\kingsoft\comb_7650/'
    path_out = 'F:/data/kingsoft/out/'
    back_path = 'F:/data/kingsoft/background/'

    # image_name_list=[]
    # cnt=0
    # for rt, dirs, files in os.walk(path):
    #     for file in files:
    #         # if '_' in file and 'png' in file:
    #         if file.endswith(".png"):
    #         # if 'png' in file:
    #             image_name_list.append(file)
    #             cnt += 1
    image_name_list, cnt = path_arr(path)
    back_name_list, cn = path_arr(back_path)
    nm = str(os.getpid())

    count = 0
    start = time.time()

    # image = cv2.resize(image, (size_x, size_y))
    # for f in os.listdir(back_path):
    for k in range(n):
        ff = open(path_out + 'text/temp/1234_0913_50w.txt', 'a')
        # try:
        start_simple = time.time()
        back_name_rand = np.random.randint(0, len(back_name_list))
        f = back_name_list[back_name_rand]
        full_file = os.path.join(back_path, f)
        image_name_rand = np.random.randint(0, len(image_name_list))
        path_rand = path + image_name_list[image_name_rand]
        # path_rand = path
        # print path_rand

        image = cv2.imread(path_rand, -1)
        image = cv2.resize(image, (200, 200))
        rows, cols, channel = image.shape
        image_back = cv2.imread(full_file, -1)
        four_points = image_point(image)

        (tl, tr, br, bl) = four_points

        if len(image_back.shape) != 3:
            continue
            print full_file

        # image[tl[0], tl[1], 0] = 255

        # print 'image point', tl
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # print four_points
        image_zeros = np.empty((rows, cols, 4))
        # add four point in 4 channel
        image_zeros[tl[1], tl[0], 0] = 255
        image_zeros[tr[1], tr[0], 1] = 255
        image_zeros[br[1], br[0], 2] = 255
        image_zeros[bl[1], bl[0], 3] = 255

        #       image_zeros[tl[1], tl[0], 3] = 255

        # ss = np.where(image_zeros[:,:,0]>0)
        # print 'zero point  before',tl,'imagepointtl{}'.format(ss)
        # cv2.imshow('zero_before', image_zeros)
        # cv2.waitKey(0)
        # zeros_mat[br[0], bl[1], 3] = 2
        # image_back_zero = np.empty((size_x,size_y,4))
        # cv2.imwrite('b.png', image_back_zero)
        # cv2.imwrite('p.png',zeros_mat)
        # image_back_zero_path = 'F:/data/kingsoft/b.png'
        # print br, image_zeros[tl[1],tl[0],1]

        # print count

        if rotation_key is True:
            image, four_points, ro_degr = rotation(image, four_points)
            image_zeros, four_points_zero = rotation0(image_zeros, ro_degr)

        if scale_key is True:
            image, four_points, scale_factor = scale(image, four_points)
            image_zeros, four_points_zero = scale0(image_zeros, scale_factor)
        if transform_key is True:
            image, four_points, rect, dst = transform(image, four_points)
            image_zeros, four_points_zero = transform0(image_zeros, rect, dst)
        if transform_cycle is True:
            image, four_points_zero = transform_cycle_128(image, re_point, size_x, size_y, count)

        if adjustlevel_key is True:
            image = trans_level_by_rand_seq(image.copy())
        if back_ground_key is True:
            image, four_points_zero = add_background_pil(image, image_back, size_x, size_y, '{}_{}'.format(nm, count),
                                                     path_out, four_points_zero)
            # image, four_points_zero = add_background0(image_zeros, image_back, size_x, size_y, '{}'.format(count), path_out,four_points)

            # four_points = add_background(image,image_back, four_points)
        if back_ground_key is False:
            cv2.imwrite(path_out + 'image/temp/' + '{}_{}.jpg'.format(nm, count), image)
        # ff.write('234')
        ff.write(nm + '_' + str(count) + '.jpg ' + str(four_points_zero[0][0]) + ' ' + str(
            four_points_zero[0][1]) + ' ' + str(four_points_zero[1][0]) + ' ' + str(four_points_zero[1][1]) + ' ' + str(
            four_points_zero[2][0]) + ' ' + str(four_points_zero[2][1]) + ' ' + str(four_points_zero[3][0]) + ' ' + str(
            four_points_zero[3][1]) + '\n')
        ff.close()
        count = count + 1
        if count % 100 == 0:
            print count
            print 'pid:' + str(os.getpid()) + ' ' + 'count:' + str(count) + ' ' + 'time:' + str(time.time() - start)
            # except:
            #     pass


#ff.close()
    print 'all time :', time.time() - start

if __name__ == '__main__':
    # cores = 4
    # for a in range(cores):
    #     # name= str(time.time())+ chr(np.random.randint(97, 122))
    #     time.sleep(2)
    #     p = multiprocessing.Process(target=main, args=(cores,))
    #     p.start()
    #     print "p.pid:", p.pid
    #     print "p.name:", p.name
    #     print "p.is_alive:", p.is_alive()
    main()