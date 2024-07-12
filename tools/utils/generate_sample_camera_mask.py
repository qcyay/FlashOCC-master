import os

import mmcv
from mmcv import Config, DictAction
import open3d as o3d
import numpy as np
import torch
from pyquaternion import Quaternion
import pickle
import math
from typing import Tuple, List, Dict, Iterable
import argparse
import cv2
import json
from tqdm import tqdm

#通过将栅格投影到图像上生成栅格对应的掩码

VOXEL_SIZE = [0.4, 0.4, 0.4]
SPTIAL_SHAPE = [200, 200, 16]

colormap_to_colors = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [112, 128, 144, 255],  # 1 障碍物 barrier  orange
        [220, 20, 60, 255],    # 2 自行车 bicycle  Blue
        [255, 127, 80, 255],   # 3 bus 公共汽车 Darkslategrey
        [255, 158, 0, 255],  # 4 car 汽车 Crimson
        [233, 150, 70, 255],   # 5 cons. Veh 施工车辆 construction_vehicle Orangered
        [255, 61, 99, 255],  # 6 motorcycle 摩托车 Darkorange
        [0, 0, 230, 255], # 7 pedestrian 行人 Darksalmon
        [47, 79, 79, 255],  # 8 traffic cone 交通锥 Red
        [255, 140, 0, 255],# 9 trailer 拖车 Slategrey
        [255, 99, 71, 255],# 10 truck 卡车 Burlywood
        [0, 207, 191, 255],    # 11 drive sur 可行驶区域 driveable_surface Green
        [175, 0, 75, 255],  # 12 other lat  nuTonomy green
        [75, 0, 75, 255],  # 13 sidewalk 人行道
        [112, 180, 60, 255],    # 14 terrain 地面
        [222, 184, 135, 255],    # 15 manmade
        [0, 175, 0, 255],   # 16 vegeyation
], dtype=np.float32)

# 将占用栅格坐标转换为点坐标
def voxel2points(voxel, occ_show, voxelSize):
    """
    Args:
        voxel: (Dx, Dy, Dz)
        occ_show: (Dx, Dy, Dz)
        voxelSize: (dx, dy, dz)

    Returns:
        points: (N, 3) 3: (x, y, z)
        voxel: (N, ) cls_id
        occIdx: (x_idx, y_idx, z_idx)
    """
    #需要显示的占用栅格的坐标，元组，包含3个元素
    occIdx = np.where(occ_show)
    #占用栅格对应的点坐标，尺寸为[N,3]
    points = np.concatenate((occIdx[0][:, None] * voxelSize[0] + POINT_CLOUD_RANGE[0], \
                             occIdx[1][:, None] * voxelSize[1] + POINT_CLOUD_RANGE[1], \
                             occIdx[2][:, None] * voxelSize[2] + POINT_CLOUD_RANGE[2]),
                             axis=1)      # (N, 3) 3: (x, y, z)
    return points, voxel[occIdx], occIdx

#将占用栅格坐标转换为点坐标并转换到图像坐标系下，并得到有效的点序号（有效意味着点在相机朝向方向并落在图像范围内）
def voxel2pixels(intrinsic, sensor2ego, height, width, gt_camera_mask=None, labels=None):

    # 尺寸为[Dx,Dy,Dz]
    voxel = np.ones(SPTIAL_SHAPE)
    #尺寸为[Dx,Dy,Dz]
    occ_show = np.ones(SPTIAL_SHAPE)
    voxel_size = VOXEL_SIZE
    # pcd，需要显示的占用栅格对应的点坐标，尺寸为[N,3]
    pcd, _, _ = voxel2points(voxel, occ_show, voxel_size)
    # 尺寸为[N,4]
    pcd = np.insert(pcd, 3, 1, 1)
    #尺寸为[3,4]
    cam2img = np.zeros((3,4))
    #尺寸为[3,4]
    cam2img[:3,:3] = intrinsic
    # #变换矩阵可参考文章Interactive Navigation in Environments with Traversable Obstacles Using Large Language and Vision-Language Models中公式2
    # #自车到像素坐标系变换矩阵，尺寸为[3,4]
    # combine = cam2img @ np.linalg.inv(sensor2ego)
    #在相机坐标系下点齐次坐标，尺寸为[4,N]
    pcd = np.linalg.inv(sensor2ego) @ pcd.T
    #尺寸为[3,N]
    points = cam2img @ pcd
    #尺寸为[N,3]
    points = points.T
    #尺寸为[N,3]
    points[:,:2] /= points[:,2:3]
    #尺寸为[N,2]
    points = points[:,:2]

    #尺寸为[N]
    camera_mask = (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height) & (pcd[2, :] > 0)
    if gt_camera_mask is not None:
        #尺寸为[N]
        gt_camera_mask = gt_camera_mask.reshape(-1)
        # 尺寸为[N]
        camera_mask = camera_mask & gt_camera_mask

    #尺寸为[n]
    idxs = np.where(camera_mask)[0]
    #尺寸为[n,2]
    points = points[idxs]
    if labels is not None:
        #尺寸为[n]
        labels = labels.reshape(-1)[idxs]
    # 尺寸为[Dx,Dy,Dz]
    camera_mask = camera_mask.reshape(SPTIAL_SHAPE)

    return points, labels, camera_mask

def parse_args():
    parser = argparse.ArgumentParser(description='Generate a mask for the occupancy grid '
                                                 'based on the camera intrinsics and extrinsics')
    parser.add_argument(
        '--res', help='Path to the predicted result')
    # 配置文件路径
    parser.add_argument('--config', default=None, help='test config file path')
    parser.add_argument(
        '--scene_name',
        type=str,
        default='0',
        help='The name of the scene')
    parser.add_argument(
        '--save_path',
        type=str,
        default=None,
        help='Path to save results')
    parser.add_argument(
        '--load_gt',
        action='store_true',
        help='Whether to project the ground truth onto the image')
    parser.add_argument(
        '--load_pred',
        action='store_true',
        help='Whether to project the prediction results onto the image')
    parser.add_argument(
        '--prefix', type=str, default='camera_mask', help='name of results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # load predicted results
    # 保存占用预测结果路径
    results_dir = args.res

    # 加载配置文件
    # Config类用于操作配置文件，它支持从多种文件格式中加载配置，包括python，json和yaml
    # 对于所有格式的配置文件, 都支持继承。为了重用其他配置文件的字段，需要指定__base__
    cfg = Config.fromfile(args.config)

    global POINT_CLOUD_RANGE
    POINT_CLOUD_RANGE = cfg.point_cloud_range

    # prepare save path and medium
    #保存可视化结果路径
    save_dir = args.save_path
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print('saving generated result to %s' % save_dir)

    #读取图像文件路径
    filename_dict = {}
    for cam in cfg.data_config['cams']:
        filename_dict[cam] = os.listdir(f'{cfg.data_root}/sync_data/{args.scene_name}/{cam}')

    #读取参数文件
    config_dict = {}
    for cam in cfg.data_config['cams']:
        config_dict[cam] = {}
        file_path = f'{cfg.data_root}/config/{cam}.json'
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        config_dict[cam]['cam_intrinsic'] = np.array(data['intrinsic']).reshape(3,3)
        sensor2ego_rotation = np.array(data['rotation']).reshape(3,3)
        config_dict[cam]['sensor2ego_rotation'] = Quaternion(matrix=sensor2ego_rotation)
        config_dict[cam]['sensor2ego_translation'] = np.array(data['translation'])

    print('start generating results')

    keycam = cfg.data_config['cams'][0]
    for cnt, filename in enumerate(tqdm(filename_dict[keycam])):
        if cnt % 10 == 0:
            print('%d/%d' % (cnt, len(filename_dict[keycam])))

        #场景序号
        scene_name = args.scene_name
        #样本数据标记
        sample_token = filename.rsplit('.', 1)[0]

        if args.load_gt:
            # 占用栅格真值结果路径
            gt_occ_path = os.path.join(cfg.data_root, 'gts', scene_name, sample_token, 'labels.npz')
            # 占用栅格真值结果，尺寸为[Dx,Dy,Dz]
            gt_occ = np.load(gt_occ_path)['semantics']
        if args.load_pred:
            # 占用栅格预测结果路径
            pred_occ_path = os.path.join(results_dir, scene_name, sample_token, 'pred.npz')
            #占用栅格预测结果，尺寸为[Dx,Dy,Dz]
            pred_occ = np.load(pred_occ_path)['pred']

        # load imgs and generate camera mask
        # 尺寸为[Dx,Dy,Dz]
        all_camera_masks = np.zeros(SPTIAL_SHAPE).astype(int)
        for view in cfg.data_config['cams']:
            intrinsic = config_dict[view]['cam_intrinsic']
            sensor2ego = np.zeros((4, 4))
            w, x, y, z = config_dict[view]['sensor2ego_rotation']  # 四元数格式
            sensor2ego[:3, :3] = Quaternion(w, x, y, z).rotation_matrix  # (3, 3)
            sensor2ego[:3, 3] = config_dict[view]['sensor2ego_translation']
            sensor2ego[3, 3] = 1

            img_path = f'{cfg.data_root}/sync_data/{scene_name}/{view}/{filename_dict[view][cnt]}'
            img = cv2.imread(img_path)
            pred_img = img.copy()
            height, width = img.shape[:2]

            if args.load_gt:
                # points，在相机朝向方向且投影后在图像范围内的点的坐标，尺寸为[n_pred,2]，camera_mask，点是否有效对应的掩码，尺寸为[Dx,Dy,Dz]
                pred_points, pred_labels, camera_mask = voxel2pixels(intrinsic, sensor2ego, height, width, labels=gt_occ)
            elif args.load_pred:
                # points，在相机朝向方向且投影后在图像范围内的点的坐标，尺寸为[n_pred,2]，camera_mask，点是否有效对应的掩码，尺寸为[Dx,Dy,Dz]
                pred_points, pred_labels, camera_mask = voxel2pixels(intrinsic, sensor2ego, height, width, labels=pred_occ)
            else:
                points, _, camera_mask = voxel2pixels(intrinsic, sensor2ego, height, width)

            # 尺寸为[Dx,Dy,Dz]
            all_camera_masks = all_camera_masks | camera_mask
            # print(f'camera_mask:{len(np.where(camera_mask)[0])},all_camera_masks:{len(np.where(all_camera_masks)[0])}')

            if args.load_gt or args.load_pred:
                colors = colormap_to_colors
                # 尺寸为[n_pred]
                _labels = pred_labels % len(colors)
                # 尺寸为[n_pred,4]
                pred_points_colors = colors[_labels]

            if args.load_gt or args.load_pred:
                for point, point_color in zip(pred_points, pred_points_colors):
                    center_coordinates = (int(point[0]), int(point[1]))
                    point_color = tuple([int(x) for x in point_color])
                    cv2.circle(pred_img, center_coordinates, radius=1, color=point_color[:3], thickness=-1)
            else:
                for point in points:
                    center_coordinates = (int(point[0]), int(point[1]))
                    cv2.circle(pred_img, center_coordinates, radius=1, color=(0, 0, 255), thickness=-1)

            if save_dir is not None:
                out_dir = os.path.join(save_dir, f'{scene_name}', f'{sample_token}')
                mmcv.mkdir_or_exist(out_dir)
                if args.load_gt:
                    cv2.imwrite(os.path.join(out_dir, f'GT_{view}.png'), pred_img)
                elif args.load_pred:
                    cv2.imwrite(os.path.join(out_dir, f'PRED_{view}.png'), pred_img)
                else:
                    cv2.imwrite(os.path.join(out_dir, f'PROJECT_{view}.png'), pred_img)
                cv2.imwrite(os.path.join(out_dir, f'{view}.png'), img)

        if cnt == 0:
            save_path = os.path.join(cfg.data_root, 'mask_camera.npz')
            if not os.path.exists(save_path):
                # 保存到npz文件
                np.savez_compressed(save_path, mask_camera=all_camera_masks)

if __name__ == '__main__':
    main()