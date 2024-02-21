from ultralytics import YOLO
import cv2
import os
from PIL import Image, ImageDraw
import numpy as np
def cutimage(path):
    # 打开图像
    image = Image.open(path)

    # 创建一个新的图像，将其尺寸设置为与输入图像相同，三通道RGB模式
    new_image = Image.new('RGB', image.size, (0, 0, 0))

    # 在新图像上绘制圆形区域
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((-20, -40, image.size[0]-30, image.size[1]-40), fill=255)

    # 将原始图像应用到掩码上，进行裁剪
    new_image.paste(image, (0, 0), mask=mask)

    # 将非圆形区域填充为0
    new_image_array = np.array(new_image)
    new_image_array[mask == 0] = 0

    # 创建新的图像对象
    new_image_cropped = Image.fromarray(new_image_array)

    return new_image_cropped


import argparse

# 创建一个ArgumentParser对象
parser = argparse.ArgumentParser(description='参数示例')

# 添加命令行参数
parser.add_argument('--weight', type=str, help='权重文件原')
parser.add_argument('--data-dir', type=str, help='数据地址')
parser.add_argument('--save-dir', type=str, help='保存地址')
parser.add_argument('--device', type=str, help='设备')

# 设置默认值
parser.set_defaults(weight="weights/yolomhead.pt",
                    data_dir="23_2024_01_23_14_31_40_1158_rosbag",
                    sava_dir="/home/lei/pj2/yolodata/result2",
                    device="cuda:0")

# 解析命令行参数
args = parser.parse_args()

# Load a model
model = YOLO(args.weight)
path = args.data_dir
images = [os.path.join(path,file) for file in os.listdir(path) if file.endswith('.jpg')]
annotated_frame = None
save_path = args.sava_dir
i = 0
for image_p in images:
    # 裁剪图像
    image = cutimage(image_p)
    result = model(image, iou=0.5, device=args.device)
    annotated_frame = result[0].plot()
    import os
    print(os.path.join(save_path, os.path.basename(image_p)))
    cv2.imwrite(os.path.join(save_path, os.path.basename(image_p)),annotated_frame)
