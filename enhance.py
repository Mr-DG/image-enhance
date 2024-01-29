import cv2 as cv
import numpy as np
import os
import urllib.parse
import sys

# 增强前的位置
# src_folder = 'D:\\sofeware\\feiqiu\\gongzuo\\feiq\\Recv Files\\Photos\\'
# src_folder = 'D:\\zed\\project\\image-enhance\\data_set_img\\'
# src_folder = 'D:\\zed\\gongsifile\\dataset\\cardata\\q\\JPEGImages\\'
src_folder = 'D:\\sofeware\\feiqiu\\gongzuo\\feiq\\Recv Files\\huo\\firesmoke\\images\\val\\'
# 增强后保存的位置
# dst_folder = 'D:\\sofeware\\feiqiu\\gongzuo\\feiq\\Recv Files\\gray' # 灰度
dst_folder = 'D:\\sofeware\\feiqiu\\gongzuo\\feiq\\Recv Files\\huo\\firesmoke\\images\\abcd' # 压缩

dx, dy = 50, 100        # 平移像素
angle = 45             # 旋转角度
kernel_size = (3, 3)    # 模糊值，卷积核大小，越大越模糊
scale_percent = 75      # 压缩比例，50表示为原来的50%，不能小于1
crop_x, crop_y, crop_w, crop_h = 100, 100, 300, 300 # 裁剪，距离左上角的位置xy，裁剪的大小wh
flip_direction = 'horizontal'  # 翻转方向，horizontal或者vertical，all为两种都要
factor = 1              # 锐化核为1的时候锐化最高
gray_level = 256        # 灰度级别，取值范围为[1, 256]
brightness = -0.3       # 亮度调整值，取值范围为[-1, 1]
contrast = 200            # 对比度[0, 255]
# saturation_factor = 1   # 饱和度[-1, 1]

# 选择需要执行的处理方法！！！
enhance_methods = [9]
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
    # 水平翻转
    5: lambda img_file: flip_image(img_file, flip_direction),
    # 锐化
    6: lambda img_file: sharpen_image(img_file, factor),
    # 灰度
    7: lambda img_file: gray_scale(img_file, gray_level),
    # 亮度
    8: lambda img_file: brightness_adjust(img_file, brightness),
    # 对比度
    9: lambda img_file: adjust_contrast(img_file, contrast),
}

# 图像增强
def enhance():
    # 获取所有图片
    img_files = get_img_files(src_folder)
    for img_file in img_files:
        try:
            # 执行需要增强的函数
            for index in enhance_methods:
                enhance_map[index](img_file)
        except Exception as e:
            print(f"处理图像时出现错误: {img_file}")
            print(f"错误信息: {str(e)}")
            continue
    print('图像增强完成')

# 平移
def translate_image(img_file, dx = 0, dy = 0):
    """
    平移图片函数

    参数:
    img_file: str - 图片路径
    dst_folder: str - 平移后的图片文件夹路径
    dx: int - x轴平移距离
    dy: int - y轴平移距离

    返回:
    None
    """
    # 读取原始图片
    img = cv.imread(img_file)

    # 平移图片
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img_translated = cv.warpAffine(img, M, (cols, rows))

    # 保存平移后的图片
    img_name = os.path.basename(img_file)
    dst_path = os.path.join(dst_folder, f"translate_{img_name}")
    cv.imwrite(dst_path, img_translated)

# 旋转
def rotate_image(img_file, angle = 0):
    """
    旋转图片函数

    参数:
    img_file: str - 图片路径
    dst_folder: str - 旋转后的图片文件夹路径
    angle: float - 旋转角度

    返回:
    None
    """
    # 读取原始图片
    img = cv.imread(img_file)

    # 旋转图片
    rows, cols = img.shape[:2]
    M = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img_rotated = cv.warpAffine(img, M, (cols, rows))

    # 保存旋转后的图片
    img_name = os.path.basename(img_file)
    dst_path = os.path.join(dst_folder, f"rotate_{img_name}")
    cv.imwrite(dst_path, img_rotated)

# 模糊
def blur_image(img_file, kernel_size):
    """
    模糊图片函数

    参数:
    img_file: str - 图片路径
    dst_folder: str - 保存模糊后的图片文件夹路径
    kernel_size: tuple - 模糊核的大小，如(5, 5)

    返回:
    None
    """
    # 读取原始图片
    img = cv.imread(img_file)

    # 模糊图片, 高斯滤波
    img_blurred = cv.GaussianBlur(img, kernel_size, 0)

    # 保存模糊后的图片
    img_name = os.path.basename(img_file)
    dst_path = os.path.join(dst_folder, f"blur_{img_name}")
    cv.imwrite(dst_path, img_blurred)

# 降低像素
def reduce_resolution(img_file, scale_percent):
    """
    将图片像素降低函数

    参数:
    img_file: str - 图片路径
    dst_folder: str - 保存降低像素后的图片文件夹路径
    scale_percent: int - 压缩比例

    返回:
    None
    """
    with open(img_file, 'rb') as f:
        img_bytes = f.read()

    img_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    # 读取原始图片
    img = cv.imdecode(img_array, cv.IMREAD_COLOR)


    # 计算新图片的尺寸
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # 将图片像素降低
    img_resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)

    # 保存降低像素后的图片
    img_name = os.path.basename(img_file)
    img_name = urllib.parse.quote(img_name, safe='')
    img_name = urllib.parse.unquote(img_name, encoding='utf-8')
    dst_path = os.path.join(dst_folder, f"reduce_{img_name}")
    # 保存图片
    cv.imencode('.jpg', img_resized)[1].tofile(dst_path)

# 裁剪
def crop_image(img_file, x, y, w, h):
    """
    该函数用于裁剪指定位置和大小的图片，并将裁剪后的图片保存到指定路径。

    img_file: str, 待裁剪图片的路径。
    x: int, 裁剪区域左上角点的 x 坐标。
    y: int, 裁剪区域左上角点的 y 坐标。
    w: int, 裁剪区域的宽度。
    h: int, 裁剪区域的高度。
    save_path: str, 裁剪后图片的保存路径。

    返回:
    None
    """
    # 读取图像
    img = cv.imread(img_file)

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
    img_name = os.path.basename(img_file)
    dst_path = os.path.join(dst_folder, f"crop_{img_name}")

    cv.imwrite(dst_path, crop_img)

# 翻转
def flip_image(img_file, flip_direction):
    """
    该函数用于对一张图片进行水平或垂直翻转，并保存翻转后的图片。

    img_file: str, 待翻转图片的路径。
    flip_direction: str, 翻转方向，可选参数为 'horizontal' 或 'vertical'。

    :返回:
    None
    """
    # 读取图像
    img = cv.imread(img_file)

    # 翻转方向
    if flip_direction == 'horizontal':
        flip_code = 1
        img_name = f"horizontal_{os.path.basename(img_file)}"
        flipped_img = cv.flip(img, flip_code)
    elif flip_direction == 'vertical':
        flip_code = 0
        img_name = f"vertical_{os.path.basename(img_file)}"
        flipped_img = cv.flip(img, flip_code)
    elif flip_direction == 'all':
        horizontal_flip = cv.flip(img, 1)
        vertical_flip = cv.flip(img, 0)
        img_name = os.path.basename(img_file)
        dst_path_horizontal = os.path.join(dst_folder, f"flip_horizontal_{img_name}")
        cv.imwrite(dst_path_horizontal, horizontal_flip)
        dst_path_vertical = os.path.join(dst_folder, f"flip_vertical_{img_name}")
        cv.imwrite(dst_path_vertical, vertical_flip)
        return
    else:
        raise ValueError("Invalid flip direction. Only 'horizontal', 'vertical', or 'all' is supported.")

    # 翻转图像
    flipped_img = cv.flip(img, flip_code)

    # 保存翻转后的图像
    dst_path = os.path.join(dst_folder, f"flip_{img_name}")
    cv.imwrite(dst_path, flipped_img)

# 锐化
def sharpen_image(img_file, factor):
    """
    该函数用于对一张图片进行锐化，并保存锐化后的图片。

    :param img_path: str, 待锐化图片的路径。
    :param factor: float, 锐化系数，控制锐化的强度。
    :param save_path: str, 锐化后图片的保存路径。

    :return: None。
    """

    # 读取图像
    img = cv.imread(img_file)
    # 构建锐化核
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    L = cv.filter2D(img, -1, kernel)
    sharpened_img = cv.addWeighted(img, 1, L, factor, 0)
    sharpened_img[sharpened_img > 255] = 255
    sharpened_img[sharpened_img < 0] = 0
    # 保存
    img_name = os.path.basename(img_file)
    dst_path = os.path.join(dst_folder, f"sharpen_{img_name}")
    cv.imwrite(dst_path, sharpened_img)

# 灰度处理
def gray_scale(img_file, gray_level=256):
    """
    将图片转换为灰度图像函数
    参数:
    img_file: str - 图片路径
    gray_level: int - 灰度级别，取值范围为[1, 256]，默认为256

    返回:
    None
    """
    if not 1 <= gray_level <= 256:
        raise ValueError("灰度级别应取值范围为[1, 256]")

    with open(img_file, 'rb') as f:
        img_bytes = f.read()

    img_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    # 读取原始图片

    # img = cv.imdecode(img_array, cv.IMREAD_COLOR)
    img = cv.imread(img_file)

    try:
        # 转换为灰度图像
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    except cv.error:
        print(f"无法转换图像: {img_file}")
        return

    if gray_level != 256:
        # 将灰度值量化到指定级别
        gray_img = np.floor(gray_img / 256 * gray_level) * (256 / gray_level)

    # 保存灰度图像
    img_name = os.path.basename(img_file)
    img_name = urllib.parse.quote(img_name, safe='')
    img_name = urllib.parse.unquote(img_name, encoding='utf-8')
    dst_path = os.path.join(dst_folder, f"gray_{gray_level}_{img_name}")
    # 保存图片
    cv.imencode('.jpg', gray_img)[1].tofile(dst_path)

# 亮度
def brightness_adjust(img_file, brightness):
    """
    调整图像亮度函数

    参数:
    img_file: str - 图像路径
    brightness: float - 亮度增益，取值范围为[-1, 1]，0表示不调整亮度

    返回:
    None
    """

    # 读取图像
    img = cv.imread(img_file)

    # 将图像转换为YCrCb色彩空间
    img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

    # 将亮度调整增益映射到[0, 255]范围内
    brightness = np.clip(brightness, -1, 1) * 255

    # 调整亮度
    img_yuv[:, :, 0] = np.clip(img_yuv[:, :, 0].astype(np.int32) + brightness, 0, 255).astype(np.uint8)

    # 将图像转换回BGR色彩空间
    img_bgr = cv.cvtColor(img_yuv, cv.COLOR_YCrCb2BGR)

    # 保存调整后的图像
    img_name = os.path.basename(img_file)
    dst_path = os.path.join(dst_folder, f"brightness_{brightness:.1f}_{img_name}")
    cv.imwrite(dst_path, img_bgr)

# 对比度
def adjust_contrast(img_file, contrast, brightness = 0):
    """
    调整图像亮度函数

    参数:
    img_file: str - 图像路径
    contrast: float - 对比度增强收益，[0, 255]
    brightness: float - 亮度增益，取值范围为[-1, 1]，0表示不调整亮度

    返回:
    None
    """
    img = cv.imread(img_file)
    output = img * (contrast/127 + 1) - contrast + brightness
    output = np.clip(output, 0, 255)
    output = np.uint8(output)

    # 保存调整后的图像
    img_name = os.path.basename(img_file)
    dst_path = os.path.join(dst_folder, f"contrast_{contrast}_{img_name}")
    cv.imwrite(dst_path, output)

# 饱和度
# def adjust_saturation(img_file, saturation_factor):


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

# 打开一个窗口查看图片
def showImg(img):
    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

enhance()