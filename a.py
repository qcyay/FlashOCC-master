import sys
import os
import open3d as o3d
import numpy as np
import cv2
import mmcv
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
import torch
from torchvision import models
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
# sys.path.append(os.getcwd())

warnings.filterwarnings('ignore')

# print(os.getcwd())
# print(os.path.dirname('a/b/c/'))

# ## 加载配置文件、构建数据集并测试

# from mmdet3d.datasets import build_dataloader, build_dataset
# from mmdet3d.models import build_model

# try:
#     from mmdet.utils import compat_cfg
# except ImportError:
#     from mmdet3d.utils import compat_cfg

# #配置文件
# config_file = 'projects/configs/flashocc/flashocc-r50.py'
# cfg = Config.fromfile(config_file)
# cfg = compat_cfg(cfg)
# # plugin_dir = cfg.plugin_dir
# # # print(plugin_dir)
# # _module_dir = os.path.dirname(plugin_dir)
# # # print(_module_dir)
# # print(cfg.dist_params)
# # print(cfg.data.test_dataloader)
#
# #导入本地模块
# import importlib
# if hasattr(cfg, 'plugin_dir'):
#     # projects/mmdet3d_plugin/
#     plugin_dir = cfg.plugin_dir
#     # projects/mmdet3d_plugin
#     _module_dir = os.path.dirname(plugin_dir)
#     _module_dir = _module_dir.split('/')
#     # projects
#     _module_path = _module_dir[0]
#
#     for m in _module_dir[1:]:
#         # projects.mmdet3d_plugin
#         _module_path = _module_path + '.' + m
#     # print(_module_path)
#     #导入一个模块
#     plg_lib = importlib.import_module(_module_path)
#
# # breakpoint()
# cfg.data.test.test_mode = True
# #数据集
# dataset = build_dataset(cfg.data.test)
# # breakpoint()
# # print(type(dataset))
# print(dataset[0].keys())
# print(dataset[0]['img_inputs'][0][0].size())
# # print(type(dataset[0]['points'][0]))
# # print(dataset.CLASSES)
# # #尺寸为[N,3,H,W]
# # imgs = dataset[0].get('img_inputs')[0][0]
# # print(imgs.size())
#
# #数据加载器
# test_loader_cfg = dict(samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
# data_loader = build_dataloader(dataset, **test_loader_cfg)
#
# cfg.model.train_cfg = None
# #构建模型
# model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
# # breakpoint()
# # 加载预训练模型参数
# checkpoint_file = 'ckpts/flashocc-r50-256x704.pth'
# checkpoint = load_checkpoint(model, checkpoint_file, map_location='cpu')
# # print(checkpoint.keys())
#
# #测试
# model = MMDataParallel(model, device_ids=[0])
# model.eval()
# for i, data in enumerate(data_loader):
#     # print(data)
#     print(data.keys())
#     # data['points'] = None
#     # data['img_metas'] = [None]
#     # print(len(data['img_metas']))
#     # breakpoint()
#     with torch.no_grad():
#         result = model(return_loss=False, rescale=True, **data)
#     # breakpoint()
#     break

# img_backbone = model.img_backbone
# print(img_backbone.parameters)
# #尺寸为[N,C,h,w]
# feat = model.img_backbone(imgs)
# print(feat[0].size())

# #测试数据预处理的流程
# from mmdet3d.datasets.pipelines import Compose
# test_pipeline = [
#     dict(type='PrepareImageInputs', data_config=cfg.data_config, sequential=False),
# ]
# test_pipeline = Compose(test_pipeline)


# 尝试快速测试的方法
# # 加载预设置函数
# from mmdetection3d.mmdet3d.apis.inference import init_model, inference_detector
# config_file = 'projects/configs/flashocc/flashocc-r50.py'
# checkpoint_file = 'ckpts/flashocc-r50-256x704.pth'
# # 根据配置文件和 checkpoint 文件构建模型
# model = init_model(config_file, checkpoint_file, device='cuda:0')
# print(model)
# # img = 'vis/scene-0003/6eb8a3ff0abf4f3a9380a48f2a0b87ef/img0.png'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
# # result = inference_detector(model, img)
# # 自定义函数
# def convert_SyncBN(config):
#     """Convert config's naiveSyncBN to BN.
#
#     Args:
#          config (str or :obj:`mmcv.Config`): Config file path or the config
#             object.
#     """
#     if isinstance(config, dict):
#         for item in config:
#             if item == 'norm_cfg':
#                 config[item]['type'] = config[item]['type']. \
#                                     replace('naiveSyncBN', 'BN')
#             else:
#                 convert_SyncBN(config[item])
#
# def init_model(config, checkpoint=None, device='cuda:0'):
#     """Initialize a model from config file, which could be a 3D detector or a
#     3D segmentor.
#
#     Args:
#         config (str or :obj:`mmcv.Config`): Config file path or the config
#             object.
#         checkpoint (str, optional): Checkpoint path. If left as None, the model
#             will not load any weights.
#         device (str): Device to use.
#
#     Returns:
#         nn.Module: The constructed detector.
#     """
#     if isinstance(config, str):
#         config = mmcv.Config.fromfile(config)
#     elif not isinstance(config, mmcv.Config):
#         raise TypeError('config must be a filename or Config object, '
#                         f'but got {type(config)}')
#     config.model.pretrained = None
#     convert_SyncBN(config.model)
#     config.model.train_cfg = None
#     model = build_model(config.model, test_cfg=config.get('test_cfg'))
#     if checkpoint is not None:
#         checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
#         model.CLASSES = config.class_names
#         print(model.CLASSES)
#     model.cfg = config  # save the config in the model for convenience
#     if device != 'cpu':
#         torch.cuda.set_device(device)
#     else:
#         logger = get_root_logger()
#         logger.warning('Don\'t suggest using CPU device. '
#                        'Some functions are not supported for now.')
#     model.to(device)
#     model.eval()
#     return model
#
# config_file = 'projects/configs/flashocc/flashocc-r50.py'
# checkpoint_file = 'ckpts/flashocc-r50-256x704.pth'
# # 根据配置文件和 checkpoint 文件构建模型
# model = init_model(config_file, checkpoint_file, device='cuda:0')
# # print(model)
# from mmdetection3d.mmdet3d.apis.inference import inference_detector
# img = 'vis/scene-0003/6eb8a3ff0abf4f3a9380a48f2a0b87ef/img0.png'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
# result = inference_detector(model, img)
# print(result)


# # 创建网格，在屏幕上显示球体和立方体
# mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
# mesh_sphere.compute_vertex_normals()
# mesh_sphere.paint_uniform_color([1.0, 0.0, 0.0])
#
# mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
# mesh_box.compute_vertex_normals()
# mesh_box.paint_uniform_color([0.0, 0.0, 1.0])
#
# # 网格可视化
# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(mesh_sphere)
# vis.add_geometry(mesh_box)
# view_control = vis.get_view_control()
# view_control.set_lookat(np.array([0, 1, 0]))
# view_control.set_up((0, -1, 0))
# view_control.set_front((-1, 0, 0))
# opt = vis.get_render_option()
# opt.background_color = np.asarray([0, 0, 0])
# vis.run()
# vis.destroy_window()

# net = models.resnet34()
# for name, child in net.named_children():
#     if name=='layer1':
#         for name, child in child.named_children():
#             print(f'name:{name}, child:{child}')

# for i in mmcv.track_iter_progress(range(10)):
#     print(i)

# # 绘制指定颜色的图形
#
# colormap_to_colors = np.array(
#     [
#         [0,   0,   0, 255],  # 0 undefined
#         [112, 128, 144, 255],  # 1 barrier  orange
#         [220, 20, 60, 255],    # 2 bicycle  Blue
#         [255, 127, 80, 255],   # 3 bus  Darkslategrey
#         [255, 158, 0, 255],  # 4 car  Crimson
#         [233, 150, 70, 255],   # 5 cons. Veh  Orangered
#         [255, 61, 99, 255],  # 6 motorcycle  Darkorange
#         [0, 0, 230, 255], # 7 pedestrian  Darksalmon
#         [47, 79, 79, 255],  # 8 traffic cone  Red
#         [255, 140, 0, 255],# 9 trailer  Slategrey
#         [255, 99, 71, 255],# 10 truck Burlywood
#         [0, 207, 191, 255],    # 11 drive sur  Green
#         [175, 0, 75, 255],  # 12 other lat  nuTonomy green
#         [75, 0, 75, 255],  # 13 sidewalk
#         [112, 180, 60, 255],    # 14 terrain
#         [222, 184, 135, 255],    # 15 manmade
#         [0, 175, 0, 255],   # 16 vegeyation
# ], dtype=np.float32)
#
# colors = colormap_to_colors / 255
# idx = 2
# rgb_color = colors[idx][:3]
#
# # 创建一些随机点云
# np.random.seed(42)  # 使结果可重复
# points = np.random.rand(1000, 3)  # 1000 个 3D 点
#
# # # 创建可视化器
# # vis = o3d.visualization.Visualizer()
# # vis.create_window()  # 创建一个交互式窗口
#
# # 将点云转换为 Open3D 点云对象
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# # 为点云设置橙色
# # pcd.paint_uniform_color(rgb_color)
# labels = np.ones((1000)) * idx
# labels = labels.astype(int)
# rgb_colors = colors[labels][:,:3]
# pcd.colors = o3d.utility.Vector3dVector(rgb_colors)
#
# # 绘制点云
# o3d.visualization.draw_geometries([pcd])

# # 将点云添加到窗口
# vis.add_geometry(pcd)
# # 渲染并保存屏幕截图
# vis.poll_events()  # 更新事件
# vis.update_renderer()  # 更新渲染器
# point_canvas = vis.capture_screen_float_buffer(do_render=True)
# #尺寸为[H,W,3]
# point_canvas = np.asarray(point_canvas)
# #尺寸为[H,W,3]
# point_canvas = (point_canvas * 255).astype(np.uint8)
# #尺寸为[H,W,3]
# point_canvas = point_canvas[..., [2, 1, 0]]
# cv2.imwrite('points.png', point_canvas)

# #读取json文件并将旋转矩阵转换为四元数表示形式
# import json
# from pyquaternion import Quaternion
# file_path = '/data/qcyay/Datasets/Task/Occupancy_Prediction/hzy_car_0507/config/CAM_FRONT.json'
# with open(file_path, 'r', encoding='utf-8') as file:
#     data = json.load(file)
# sensor2ego_rotation = np.array(data['rotation']).reshape(3, 3)
# print(sensor2ego_rotation)
# print(Quaternion(matrix=sensor2ego_rotation))
# print(Quaternion(Quaternion(matrix=sensor2ego_rotation)).rotation_matrix)

# filename_list = os.listdir(f'data/hzy_car_0507/sensor_data/tmp_dir/CAM_FRONT')
# # print(filename_list[0])
# smaple_token = filename_list[0].rsplit('.', 1)[0]
# print(smaple_token)

# #显示所有颜色
#
# # 定义颜色数组
# colormap_to_colors = np.array(
#     [
#         [0,   0,   0, 255],  # 0 undefined
#         [112, 128, 144, 255],  # 1 障碍物 barrier  orange
#         [220, 20, 60, 255],    # 2 自行车 bicycle  Blue
#         [255, 127, 80, 255],   # 3 bus 公共汽车 Darkslategrey
#         [255, 158, 0, 255],  # 4 car 汽车 Crimson
#         [233, 150, 70, 255],   # 5 cons. Veh 施工车辆 Orangered
#         [255, 61, 99, 255],  # 6 motorcycle 摩托车 Darkorange
#         [0, 0, 230, 255], # 7 pedestrian 行人 Darksalmon
#         [47, 79, 79, 255],  # 8 traffic cone 交通锥 Red
#         [255, 140, 0, 255],# 9 trailer 拖车 Slategrey
#         [255, 99, 71, 255],# 10 truck 卡车 Burlywood
#         [0, 207, 191, 255],    # 11 drive sur  Green
#         [175, 0, 75, 255],  # 12 other lat  nuTonomy green
#         [75, 0, 75, 255],  # 13 sidewalk
#         [112, 180, 60, 255],    # 14 terrain 地面
#         [222, 184, 135, 255],    # 15 manmade
#         [0, 175, 0, 255],   # 16 vegeyation
# ], dtype=np.float32)
#
# # 类别标签
# labels = [
#     'undefined', 'barrier', 'bicycle', 'bus', 'car', 'cons vehicle',
#     'motorcycle', 'pedestrian', 'traffic cone', 'trailer', 'truck', 'drive surface',
#     'other', 'sidewalk', 'terrain', 'manmade', 'vegetation'
# ]
#
# # 设置每个块的尺寸
# block_width = 60
# block_height = 50
# num_colors = len(colormap_to_colors)
#
# # 创建一个窗口，计算窗口的尺寸
# window_height = block_height + 30  # 增加30像素用于显示标签
# window_width = block_width * num_colors
# window = np.zeros((window_height, window_width, 3), dtype=np.uint8)
#
# # 填充窗口并添加标签
# for i, (color, label) in enumerate(zip(colormap_to_colors, labels)):
#     # 取前3个值作为BGR颜色
#     bgr_color = tuple(map(int, color[:3]))  # 将颜色值转换为整数元组
#     # 计算块的坐标
#     start_x = i * block_width
#     end_x = start_x + block_width
#     # 绘制颜色块
#     cv2.rectangle(window, (start_x, 0), (end_x, block_height), bgr_color, -1)
#     # 添加标签文字
#     cv2.putText(window, label, (start_x + 5, block_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1,
#                 cv2.LINE_AA)
#
# # 显示窗口
# cv2.imshow('Color Blocks with Labels', window)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #将指定目录下文件名根据预先设置的字典进行替换
# # 预先设置的字典，包含要替换的目录名和替换值
# mapping_dict = {
#     "cam_front": "CAM_FRONT",
#     "cam_left_front": "CAM_FRONT_LEFT",
#     "cam_right_front": "CAM_FRONT_RIGHT"
#     # 添加更多的映射关系...
# }
#
# for num in range(1,8):
#     # 特定目录的路径
#     target_directory = f'E:\工作有关\占用网络部署\car_0604\sync_data\\{num}'
#
#     # 获取目录下所有子目录
#     subdirectories = next(os.walk(target_directory))[1]
#
#     # 对每个子目录进行替换
#     for subdir in subdirectories:
#         if subdir in mapping_dict:
#             old_path = os.path.join(target_directory, subdir)
#             new_path = os.path.join(target_directory, mapping_dict[subdir])
#             os.rename(old_path, new_path)
#             print(f"目录 {old_path} 更名为 {new_path}")
