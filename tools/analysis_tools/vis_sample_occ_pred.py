import os

import mmcv
from mmcv import Config, DictAction
import open3d as o3d
import numpy as np
import torch
import pickle
import math
from typing import Tuple, List, Dict, Iterable
import argparse
import cv2

#适用于对根据不含有占用栅格真值并不以NuScenes格式组织的图像得到的预测结果的可视化

IMAGE_HEIGHT = 900
IMAGE_WIDTH = 1600

NOT_OBSERVED = -1
FREE = 0
OCCUPIED = 1
FREE_LABEL = 17
BINARY_OBSERVED = 1
BINARY_NOT_OBSERVED = 0

VOXEL_SIZE = [0.4, 0.4, 0.4]
POINT_CLOUD_RANGE = [-40, -40, -1, 40, 40, 5.4]
SPTIAL_SHAPE = [200, 200, 16]
TGT_VOXEL_SIZE = [0.4, 0.4, 0.4]
TGT_POINT_CLOUD_RANGE = [-40, -40, -1, 40, 40, 5.4]

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
        [175, 0, 75, 255],  # 12 other flat  nuTonomy green
        [75, 0, 75, 255],  # 13 sidewalk 人行道
        [112, 180, 60, 255],    # 14 terrain 地面
        [222, 184, 135, 255],    # 15 manmade
        [0, 175, 0, 255],   # 16 植被 vegeyation
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
    occIdx = torch.where(occ_show)
    #占用栅格对应的点坐标，尺寸为[N,3]
    points = torch.cat((occIdx[0][:, None] * voxelSize[0] + POINT_CLOUD_RANGE[0], \
                        occIdx[1][:, None] * voxelSize[1] + POINT_CLOUD_RANGE[1], \
                        occIdx[2][:, None] * voxelSize[2] + POINT_CLOUD_RANGE[2]),
                       dim=1)      # (N, 3) 3: (x, y, z)
    return points, voxel[occIdx], occIdx

#将占用栅格坐标转换为点坐标并转换到图像坐标系下，并得到有效的点序号（有效意味着点在相机朝向方向并落在图像范围内）
def voxel2pixels(voxel, intrinsic, sensor2ego):

    #尺寸为[Dx,Dy,Dz]
    occ_show = np.ones(SPTIAL_SHAPE)
    voxel_size = VOXEL_SIZE
    # pcd，需要显示的占用栅格对应的点坐标，尺寸为[N,3]，labels，需要显示的占用栅格类别，尺寸为[N]，需要显示的占用栅格的坐标
    pcd, labels, _ = voxel2points(voxel, occ_show, voxel_size)
    # 尺寸为[N,4]
    pcd = np.insert(pcd, 3, 1, 1)
    #尺寸为[3,4]
    cam2img = np.zeros((3,4))
    #尺寸为[3,4]
    cam2img[:3,:3] = intrinsic
    # #变换矩阵可参考文章Interactive Navigation in Environments with Traversable Obstacles Using Large Language and Vision-Language Models中公式2
    # #自车到像素坐标系变换矩阵，尺寸为[3,4]
    # combine = cam2img @ np.linalg.inv(sensor2ego)
    #在相机坐标系下点齐次坐标，尺寸为[N,4]
    points = np.linalg.inv(sensor2ego) @ pcd
    #尺寸为[N]
    camera_mask = (points[:,2] > 0)
    #尺寸为[N,3]
    points = cam2img @ pcd
    #尺寸为[N,3]
    points[:,:2] /= points[:,2]
    #尺寸为[N,2]
    points = points[:,:2]
    #尺寸为[N]
    camera_mask = (points[:, 0] >= 0) & (points[:, 0] < 0) & (points[:, 1] >= 0) & (points[:, 1] < 0) & camera_mask
    #尺寸为[n]
    idxs = np.where(camera_mask)[0]
    #尺寸为[n]
    points = points[idxs]
    #尺寸为[Dx,Dy,Dz]
    camera_mask = camera_mask.reshape(SPTIAL_SHAPE)

    return points, camera_mask

def voxel_profile(voxel, voxel_size):
    """
    Args:
        voxel: (N, 3)  3:(x, y, z)
        voxel_size: (vx, vy, vz)

    Returns:
        box: (N, 7) (x, y, z - dz/2, vx, vy, vz, 0)
    """
    #占用栅格中心，尺寸为[N,3]
    centers = torch.cat((voxel[:, :2], voxel[:, 2][:, None] - voxel_size[2] / 2), dim=1)     # (x, y, z - dz/2)
    # centers = voxel
    #尺寸为[N,3]
    wlh = torch.cat((torch.tensor(voxel_size[0]).repeat(centers.shape[0])[:, None],
                     torch.tensor(voxel_size[1]).repeat(centers.shape[0])[:, None],
                     torch.tensor(voxel_size[2]).repeat(centers.shape[0])[:, None]), dim=1)
    #尺寸为[N,1]
    yaw = torch.full_like(centers[:, 0:1], 0)
    return torch.cat((centers, wlh, yaw), dim=1)


def rotz(t):
    """Rotation about the z-axis."""
    c = torch.cos(t)
    s = torch.sin(t)
    return torch.tensor([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

#根据包围框的属性得到框的端点坐标
def my_compute_box_3d(center, size, heading_angle):
    """
    Args:
        center: (N, 3)  3: (x, y, z - dz/2)
        size: (N, 3)    3: (vx, vy, vz)
        heading_angle: (N, 1)
    Returns:
        corners_3d: (N, 8, 3)
    """
    h, w, l = size[:, 2], size[:, 0], size[:, 1]
    center[:, 2] = center[:, 2] + h / 2
    l, w, h = (l / 2).unsqueeze(1), (w / 2).unsqueeze(1), (h / 2).unsqueeze(1)
    #尺寸为[N,8,1]
    x_corners = torch.cat([-l, l, l, -l, -l, l, l, -l], dim=1)[..., None]
    #尺寸为[N,8,1]
    y_corners = torch.cat([w, w, -w, -w, w, w, -w, -w], dim=1)[..., None]
    #尺寸为[N,8,1]
    z_corners = torch.cat([h, h, h, h, -h, -h, -h, -h], dim=1)[..., None]
    #尺寸为[N,8,3]
    corners_3d = torch.cat([x_corners, y_corners, z_corners], dim=2)
    #尺寸为[N,8,3]
    corners_3d[..., 0] += center[:, 0:1]
    #尺寸为[N,8,3]
    corners_3d[..., 1] += center[:, 1:2]
    #尺寸为[N,8,3]
    corners_3d[..., 2] += center[:, 2:3]
    return corners_3d


def show_point_cloud(points: np.ndarray, colors=True, points_colors=None, bbox3d=None, voxelize=False,
                     bbox_corners=None, linesets=None, vis=None, offset=[0,0,0], large_voxel=True, voxel_size=0.4):
    """
    :param points: (N, 3)  3:(x, y, z)
    :param colors: false 不显示点云颜色
    :param points_colors: (N, 4）
    :param bbox3d: voxel grid (N, 7) 7: (center, wlh, yaw=0)
    :param voxelize: false 不显示voxel边界
    :param bbox_corners: (N, 8, 3)  voxel grid 角点坐标, 用于绘制voxel grid 边界.
    :param linesets: 用于绘制voxel grid 边界.
    :return:
    """
    if vis is None:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
    if isinstance(offset, list) or isinstance(offset, tuple):
        offset = np.array(offset)

    pcd = o3d.geometry.PointCloud()
    #open3d.utility.Vector3dVector将形状 (n, 3) 的 float64 numpy 数组转换为 Open3D 格式
    pcd.points = o3d.utility.Vector3dVector(points+offset)
    if colors:
        pcd.colors = o3d.utility.Vector3dVector(points_colors[:, :3])
    #open3d.geometry.TriangleMesh.create_coordinate_frame用于创建坐标系网格的函数。坐标系将以原点为中心。x、y、z轴将分别呈现为红色、绿色和蓝色箭头。
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])

    #open3d.geometry.VoxelGrid.create_from_point_cloud从给定的点云创建体素网格。给定体素的颜色值是落入其中的点的平均颜色值（如果点云有颜色）。创建的体素网格的边界是根据点云计算的。
    voxelGrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    if large_voxel:
        vis.add_geometry(voxelGrid)
    else:
        vis.add_geometry(pcd)

    if voxelize:
        line_sets = o3d.geometry.LineSet()
        #points属性表示线集的顶点
        line_sets.points = o3d.open3d.utility.Vector3dVector(bbox_corners.reshape((-1, 3))+offset)
        #lines属性用于存储每条直线的端点序号
        line_sets.lines = o3d.open3d.utility.Vector2iVector(linesets.reshape((-1, 2)))
        line_sets.paint_uniform_color((0, 0, 0))
        vis.add_geometry(line_sets)

    vis.add_geometry(mesh_frame)

    # ego_pcd = o3d.geometry.PointCloud()
    # ego_points = generate_the_ego_car()
    # ego_pcd.points = o3d.utility.Vector3dVector(ego_points)
    # vis.add_geometry(ego_pcd)

    return vis

# 体素占用情况和类别可视化
def show_occ(occ_state, occ_show, voxel_size, vis=None, offset=[0, 0, 0]):
    """
    Args:
        occ_state: (Dx, Dy, Dz), cls_id
        occ_show: (Dx, Dy, Dz), bool
        voxel_size: [0.4, 0.4, 0.4]
        vis: Visualizer
        offset:

    Returns:

    """
    colors = colormap_to_colors / 255
    #pcd，需要显示的占用栅格对应的点坐标，尺寸为[N,3]，labels，需要显示的占用栅格类别，尺寸为[N]，需要显示的占用栅格的坐标
    pcd, labels, occIdx = voxel2points(occ_state, occ_show, voxel_size)
    # pcd: (N, 3)  3: (x, y, z)
    # labels: (N, )  cls_id
    #尺寸为[N]
    _labels = labels % len(colors)
    #尺寸为[N,4]
    pcds_colors = colors[_labels]   # (N, 4)

    #包围框的位置、尺寸和朝向角，尺寸为[N,7]
    bboxes = voxel_profile(pcd, voxel_size)    # (N, 7)   7: (x, y, z - dz/2, dx, dy, dz, 0)
    #包围框的端点坐标，尺寸为[N,8,3]
    bboxes_corners = my_compute_box_3d(bboxes[:, 0:3], bboxes[:, 3:6], bboxes[:, 6:7])      # (N, 8, 3)

    #尺寸为[N]
    bases_ = torch.arange(0, bboxes_corners.shape[0] * 8, 8)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])  # lines along y-axis
    #尺寸为[N,12,2]
    edges = edges.reshape((1, 12, 2)).repeat(bboxes_corners.shape[0], 1, 1)     # (N, 12, 2)
    # (N, 12, 2) + (N, 1, 1) --> (N, 12, 2)   此时edges中记录的是bboxes_corners的整体id: (0, N*8).
    #尺寸为[N,12,2]
    edges = edges + bases_[:, None, None]

    vis = show_point_cloud(
        points=pcd.numpy(),
        colors=True,
        points_colors=pcds_colors,
        voxelize=True,
        bbox3d=bboxes.numpy(),
        bbox_corners=bboxes_corners.numpy(),
        linesets=edges.numpy(),
        vis=vis,
        offset=offset,
        large_voxel=True,
        voxel_size=0.4
    )
    return vis

def generate_the_ego_car():
    ego_range = [-2, -1, 0, 2, 1, 1.5]
    ego_voxel_size=[0.1, 0.1, 0.1]
    ego_xdim = int((ego_range[3] - ego_range[0]) / ego_voxel_size[0])
    ego_ydim = int((ego_range[4] - ego_range[1]) / ego_voxel_size[1])
    ego_zdim = int((ego_range[5] - ego_range[2]) / ego_voxel_size[2])
    temp_x = np.arange(ego_xdim)
    temp_y = np.arange(ego_ydim)
    temp_z = np.arange(ego_zdim)
    ego_xyz = np.stack(np.meshgrid(temp_y, temp_x, temp_z), axis=-1).reshape(-1, 3)
    ego_point_x = (ego_xyz[:, 0:1] + 0.5) / ego_xdim * (ego_range[3] - ego_range[0]) + ego_range[0]
    ego_point_y = (ego_xyz[:, 1:2] + 0.5) / ego_ydim * (ego_range[4] - ego_range[1]) + ego_range[1]
    ego_point_z = (ego_xyz[:, 2:3] + 0.5) / ego_zdim * (ego_range[5] - ego_range[2]) + ego_range[2]
    ego_point_xyz = np.concatenate((ego_point_y, ego_point_x, ego_point_z), axis=-1)
    ego_points_label =  (np.ones((ego_point_xyz.shape[0]))*16).astype(np.uint8)
    ego_dict = {}
    ego_dict['point'] = ego_point_xyz
    ego_dict['label'] = ego_points_label
    return ego_point_xyz


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize the predicted '
                                     'result of nuScenes')
    parser.add_argument(
        'res', help='Path to the predicted result')
    # 配置文件路径
    parser.add_argument('--config', default=None, help='test config file path')
    parser.add_argument(
        '--canva-size', type=int, default=1000, help='Size of canva in pixel')
    parser.add_argument(
        '--vis-frames',
        type=int,
        default=500,
        help='Number of frames for visualization')
    parser.add_argument(
        '--scale-factor',
        type=int,
        default=4,
        help='Trade-off between image-view and bev in size of '
        'the visualized canvas')
    parser.add_argument(
        '--image_path',
        type=str,
        default='./data/nuscenes',
        help='Path to images')
    parser.add_argument(
        '--vis_id',
        type=int,
        default=-1,
        help='The ordinal number of the displayed occupancy grid category')
    parser.add_argument(
        '--save_path',
        type=str,
        default='./vis',
        help='Path to save visualization results')
    parser.add_argument(
        '--format',
        type=str,
        default='image',
        choices=['video', 'image'],
        help='The desired format of the visualization result')
    parser.add_argument(
        '--fps', type=int, default=10, help='Frame rate of video')
    parser.add_argument(
        '--video_prefix', type=str, default='vis', help='name of video')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # load predicted results
    #保存占用预测结果路径
    results_dir = args.res

    # 加载配置文件
    # Config类用于操作配置文件，它支持从多种文件格式中加载配置，包括python，json和yaml
    # 对于所有格式的配置文件, 都支持继承。为了重用其他配置文件的字段，需要指定__base__
    cfg = Config.fromfile(args.config)

    # prepare save path and medium
    #保存可视化结果路径
    vis_dir = args.save_path
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    print('saving visualized result to %s' % vis_dir)
    scale_factor = args.scale_factor
    canva_size = args.canva_size
    if args.format == 'video':
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        vout = cv2.VideoWriter(
            os.path.join(vis_dir, '%s.mp4' % args.video_prefix), fourcc,
            args.fps, (int(1600 / scale_factor * 3),
                       int(900 / scale_factor * 2 + canva_size)))

    views = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ]
    print('start visualizing results')

    # o3d.visualization.VisualizerWithKeyCallback是open3D中用于渲染和可视化3D场景的类
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # 样本数据标记列表
    sample_token_list = os.listdir(results_dir)
    for cnt, sample_token in enumerate(
            sample_token_list[:min(args.vis_frames, len(sample_token_list))]):
        if cnt % 10 == 0:
            print('%d/%d' % (cnt, min(args.vis_frames, len(sample_token_list))))

        # 占用栅格预测结果路径
        pred_occ_path = os.path.join(results_dir, sample_token, 'pred.npz')

        #占用栅格预测结果，尺寸为[Dx,Dy,Dz]
        pred_occ = np.load(pred_occ_path)['pred']
        # breakpoint()

        # load imgs
        #各视角图像，列表
        imgs = []
        for view in views:
            if view in cfg.data_config['cams']:
                img = cv2.imread(os.path.join(args.image_path, view, f'{sample_token}.jpg'))
                img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
            else:
                img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
            imgs.append(img)

        # occ_canvas
        #表示占据栅格是否被显示的标记，尺寸为[Dx,Dy,Dz]
        if args.vis_id > -1:
            camera_mask = np.zeros(SPTIAL_SHAPE)
            vis_inds = np.where(pred_occ == args.vis_id)
            camera_mask[vis_inds] = 1
        else:
            camera_mask = np.ones(SPTIAL_SHAPE)
            camera_mask[:,:,10:] = 0
        voxel_show = np.logical_and(pred_occ != FREE_LABEL, camera_mask)
        voxel_size = VOXEL_SIZE
        vis = show_occ(torch.from_numpy(pred_occ), torch.from_numpy(voxel_show), voxel_size=voxel_size, vis=vis,
                       offset=[0, pred_occ.shape[0] * voxel_size[0] * 1.2 * 0, 0])

        view_control = vis.get_view_control()

        look_at = np.array([-0.185, 0.513, 3.485])
        front = np.array([-0.974, -0.055, 0.221])
        up = np.array([0.221, 0.014, 0.975])
        zoom = np.array([0.08])

        #set_lookat用于在OpenGL中设置观察矩阵的视点和目标
        view_control.set_lookat(look_at)
        #set_front用于设置物体在视图中的正面朝向
        view_control.set_front(front) # set the positive direction of the x-axis toward you
        #set_up设置可视化工具的向上向量
        view_control.set_up(up) # set the positive direction of the x-axis as the up direction
        #set_zoom用于设置视图控制器的缩放级别
        view_control.set_zoom(zoom)

        #get_render_option用于获取当前可视化窗口的渲染选项
        opt = vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1])
        opt.line_width = 5

        vis.poll_events()
        #update_renderer用于更新当前的渲染器
        vis.update_renderer()
        # vis.run()

        #capture_screen_float_buffer捕获屏幕并将RGB存储在浮动缓冲区
        occ_canvas = vis.capture_screen_float_buffer(do_render=True)
        #尺寸为[H,W,3]
        occ_canvas = np.asarray(occ_canvas)
        #尺寸为[H,W,3]
        occ_canvas = (occ_canvas * 255).astype(np.uint8)
        #尺寸为[H,W,3]
        occ_canvas = occ_canvas[..., [2, 1, 0]]
        occ_canvas_resize = cv2.resize(occ_canvas, (canva_size, canva_size), interpolation=cv2.INTER_CUBIC)

        vis.clear_geometries()

        big_img = np.zeros((900 * 2 + canva_size * scale_factor, 1600 * 3, 3),
                       dtype=np.uint8)
        #显示前方多视角图像
        big_img[:900, :, :] = np.concatenate(imgs[:3], axis=1)
        img_back = np.concatenate(
            [imgs[3][:, ::-1, :], imgs[4][:, ::-1, :], imgs[5][:, ::-1, :]],
            axis=1)
        #显示后方多视角图像
        big_img[900 + canva_size * scale_factor:, :, :] = img_back
        big_img = cv2.resize(big_img, (int(1600 / scale_factor * 3),
                                       int(900 / scale_factor * 2 + canva_size)))
        w_begin = int((1600 * 3 / scale_factor - canva_size) // 2)
        #显示占用栅格
        big_img[int(900 / scale_factor):int(900 / scale_factor) + canva_size,
                w_begin:w_begin + canva_size, :] = occ_canvas_resize

        if args.format == 'image':
            out_dir = os.path.join(vis_dir, f'{sample_token}')
            mmcv.mkdir_or_exist(out_dir)
            for i, img in enumerate(imgs):
                cv2.imwrite(os.path.join(out_dir, f'img{i}.png'), img)
            cv2.imwrite(os.path.join(out_dir, 'occ.png'), occ_canvas)
            cv2.imwrite(os.path.join(out_dir, 'overall.png'), big_img)
        elif args.format == 'video':
            cv2.putText(big_img, f'{cnt:{cnt}}', (5, 15), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                        fontScale=0.5)
            cv2.putText(big_img, f'{sample_token}', (5, 55), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                        fontScale=0.5)
            vout.write(big_img)

    if args.format == 'video':
        vout.release()
    vis.destroy_window()


if __name__ == '__main__':
    main()