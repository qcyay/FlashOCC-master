import os
from PIL import Image

def resize_images(input_folder, output_folder, size):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取指定文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 构造完整的文件路径
        file_path = os.path.join(input_folder, filename)

        # 检查文件是否是图像文件
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 打开图像文件
            with Image.open(file_path) as img:
                # 调整图像大小
                img_resized = img.resize(size)

                # 构造输出文件路径
                output_path = os.path.join(output_folder, filename)

                # 保存调整大小后的图像
                img_resized.save(output_path)
                print(f"Saved resized image to: {output_path}")


# 示例使用
# input_folder = r'F:\数据集\NuScenes\hzy_car_0507\sensor_data\tmp_dir\cam_front'  # 替换为你的输入文件夹路径
# output_folder = r'F:\数据集\NuScenes\hzy_car_0507\sensor_data\tmp_dir\cam_front_resized'  # 替换为你的输出文件夹路径
input_folder = 'data/hzy_car_0507/sensor_data/tmp_dir/CAM_FRONT_RESIZED'  # 替换为你的输入文件夹路径
output_folder = 'data/hzy_car_0507/sensor_data/tmp_dir/CAM_FRONT'  # 替换为你的输出文件夹路径
size = (3840, 2160)  # 替换为你希望的尺寸

resize_images(input_folder, output_folder, size)