import cv2 as cv
import numpy as np
import os

# 增强前的位置
src_folder = 'c:\\zed\\image-enhance\\data_set_img\\'
# 增强后保存的位置
dst_folder = 'c:\\zed\\image-enhance\\result\\'

dx, dy = 50, 100       # 平移像素
angle = 45             # 旋转角度
kernel_size = (3, 3)   # 模糊值，卷积核大小，越大越模糊
scale_percent = 50     # 压缩比例，50表示为原来的50%，不能小于1
crop_x, crop_y, crop_w, crop_h = 100, 100, 300, 300 # 裁剪，距离左上角的位置xy，裁剪的大小wh

# 选择需要执行的处理方法！！！
enhance_methods = [4]
# 处理方法
enhance_map = {
    # 平移
    0: lambda img_file: translate_image(img_file, dx, dy),
    # 旋转
    1: lambda img_file: rotate_image(img_file, angle),
    # 模糊
    2: lambda img_file: blur_image(img_file, kernel_size),
    # 降低像素
    3: lambda img_file: reduce_resolution(img_file, scale_percent),
    # 裁剪
    4: lambda img_file: crop_image(img_file, crop_x, crop_y, crop_w, crop_h),
}

# 图像增强
def enhance():
    # 获取所有图片
    img_files = get_img_files(src_folder)
    for img_file in img_files:
        # 执行需要增强的函数
        for index in enhance_methods:
            enhance_map[index](img_file)
    print('图像增强完成')

# 平移
def translate_image(img_path, dx = 0, dy = 0):
    """
    平移图片函数

    参数:
    img_path: str - 图片路径
    dst_folder: str - 平移后的图片文件夹路径
    dx: int - x轴平移距离
    dy: int - y轴平移距离

    返回:
    None
    """
    # 读取原始图片
    img = cv.imread(img_path)

    # 平移图片
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img_translated = cv.warpAffine(img, M, (cols, rows))

    # 保存平移后的图片
    img_name = os.path.basename(img_path)
    dst_path = os.path.join(dst_folder, f"translate_{img_name}")
    cv.imwrite(dst_path, img_translated)

# 旋转
def rotate_image(img_path, angle = 0):
    """
    旋转图片函数

    参数:
    img_path: str - 图片路径
    dst_folder: str - 旋转后的图片文件夹路径
    angle: float - 旋转角度

    返回:
    None
    """
    # 读取原始图片
    img = cv.imread(img_path)

    # 旋转图片
    rows, cols = img.shape[:2]
    M = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img_rotated = cv.warpAffine(img, M, (cols, rows))

    # 保存旋转后的图片
    img_name = os.path.basename(img_path)
    dst_path = os.path.join(dst_folder, f"rotate_{img_name}")
    cv.imwrite(dst_path, img_rotated)

# 模糊
def blur_image(img_path, kernel_size):
    """
    模糊图片函数

    参数:
    img_path: str - 图片路径
    dst_folder: str - 保存模糊后的图片文件夹路径
    kernel_size: tuple - 模糊核的大小，如(5, 5)

    返回:
    None
    """
    # 读取原始图片
    img = cv.imread(img_path)

    # 模糊图片
    img_blurred = cv.blur(img, kernel_size)

    # 保存模糊后的图片
    img_name = os.path.basename(img_path)
    dst_path = os.path.join(dst_folder, f"blur_{img_name}")
    cv.imwrite(dst_path, img_blurred)

# 降低像素
def reduce_resolution(img_path, scale_percent):
    """
    将图片像素降低函数

    参数:
    img_path: str - 图片路径
    dst_folder: str - 保存降低像素后的图片文件夹路径
    scale_percent: int - 压缩比例

    返回:
    None
    """
    # 读取原始图片
    img = cv.imread(img_path)

    # 计算新图片的尺寸
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # 将图片像素降低
    img_resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)

    # 保存降低像素后的图片
    img_name = os.path.basename(img_path)
    dst_path = os.path.join(dst_folder, f"reduce_{img_name}")
    cv.imwrite(dst_path, img_resized)

# 裁剪
def crop_image(img_path, x, y, w, h):
    """
    该函数用于裁剪指定位置和大小的图片，并将裁剪后的图片保存到指定路径。

    img_path: str, 待裁剪图片的路径。
    x: int, 裁剪区域左上角点的 x 坐标。
    y: int, 裁剪区域左上角点的 y 坐标。
    w: int, 裁剪区域的宽度。
    h: int, 裁剪区域的高度。
    save_path: str, 裁剪后图片的保存路径。

    返回:
    None
    """
    # 读取图像
    img = cv.imread(img_path)

    # 获取图像大小
    height, width = img.shape[:2]

    # 确保裁剪区域在图像范围内
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + w > width:
        w = width - x
    if y + h > height:
        h = height - y

    # 裁剪图像
    crop_img = img[y:y+h, x:x+w]

    # 保存裁剪后的图像
    img_name = os.path.basename(img_path)
    dst_path = os.path.join(dst_folder, f"crop_{img_name}")

    cv.imwrite(dst_path, crop_img)

# 获取文件夹下所有文件
def get_img_files(src_folder):
    """
    平移旋转图片函数

    参数:
    src_folder: str - 原始图片文件夹路径

    返回:
    None
    """

    # 获取源文件夹下的所有图片
    img_files = [os.path.join(src_folder, f) for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]

    return img_files

enhance()