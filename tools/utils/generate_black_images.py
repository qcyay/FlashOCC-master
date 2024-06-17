import os
from PIL import Image

# 定义图像所在的文件夹路径
source_folder = 'data/NuScenes-develop_hongqi1280/samples/2023_02_20/2023_02_20_hongqi_8/cam_right_back'  # 更改为实际图像所在的文件夹路径
# 定义保存全黑色图像的文件夹路径
output_folder = 'data/NuScenes-develop_hongqi1280/samples/2023_02_20/2023_02_20_hongqi_8/cam_right_back_black'  # 更改为保存黑色图像的文件夹路径

# 如果保存全黑色图像的文件夹不存在，则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取文件夹下所有文件的列表
files = os.listdir(source_folder)

# 迭代所有文件
for file in files:
    # 获取文件的完整路径
    file_path = os.path.join(source_folder, file)

    try:
        # 打开文件，判断是否是有效图像
        with Image.open(file_path) as img:
            # 获取图像的尺寸
            width, height = img.size
            # 创建一个全黑色图像
            black_image = Image.new('RGB', (width, height), color=(0, 0, 0))
            # 保存全黑色图像，使用与原图像相同的文件名
            output_path = os.path.join(output_folder, file)
            black_image.save(output_path)
            print(f"全黑色图像已保存至: {output_path}")

    except Exception as e:
        # 如果出现异常，可能该文件不是有效的图像
        print(f"跳过非图像文件: {file}. 错误: {e}")