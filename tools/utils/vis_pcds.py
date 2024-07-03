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

#对连续帧点云进行可视化

IMAGE_HEIGHT = 900
IMAGE_WIDTH = 1600

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

    return vis

#将点坐标转换到图像坐标系下，并得到有效的点序号（有效意味着点在相机朝向方向并落在图像范围内）
#输出：points，投影后在图像范围内的点坐标，labels，保留的点的标签，camera_mask，点是否在图像范围内的掩码
def point2pixels(pcd, intrinsic, sensor2ego, height, width, labels=None):

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

    #尺寸为[n]
    idxs = np.where(camera_mask)[0]
    #尺寸为[n,2]
    points = points[idxs]
    if labels is not None:
        #尺寸为[n]
        labels = labels.reshape(-1)[idxs]

    return points, labels, camera_mask

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize the points')
    parser.add_argument(
        'data_root', help='Path to the files')
    parser.add_argument(
        '--scene_name',
        type=str,
        default='0',
        help='The name of the scene')
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
        '--point_to_img',
        action='store_true',
        help='Whether to project the point cloud onto the image')
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
    root_dir = args.data_root

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
                       int(900 / scale_factor + canva_size)))

    views = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']
    print('start visualizing results')

    # o3d.visualization.VisualizerWithKeyCallback是open3D中用于渲染和可视化3D场景的类
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    lidar_filename_list = os.listdir(f'{root_dir}/sync_data/{args.scene_name}/lidar_concat')

    if args.point_to_img:
        # 读取参数文件
        config_dict = {}
        for cam in views:
            config_dict[cam] = {}
            file_path = f'{root_dir}/config/{cam}.json'
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            config_dict[cam]['cam_intrinsic'] = np.array(data['intrinsic']).reshape(3, 3)
            sensor2ego_rotation = np.array(data['rotation']).reshape(3, 3)
            config_dict[cam]['sensor2ego_rotation'] = Quaternion(matrix=sensor2ego_rotation)
            config_dict[cam]['sensor2ego_translation'] = np.array(data['translation'])

    for cnt, lidar_filename in enumerate(
            lidar_filename_list[:min(args.vis_frames, len(lidar_filename_list))]):
        if cnt % 10 == 0:
            print('%d/%d' % (cnt, min(args.vis_frames, len(lidar_filename_list))))

        sample_token = lidar_filename.split('.')[0]

        # 激光雷达点云数据保存路径
        lidar_path = os.path.join(root_dir, 'sync_data', args.scene_name, 'lidar_concat', lidar_filename)

        #激光雷达点云
        pcd = o3d.io.read_point_cloud(lidar_path)
        # #设置点云颜色
        # pcd.paint_uniform_color([0, 0, 1])
        if args.point_to_img:
            #尺寸为[N,3]
            all_points = np.asarray(pcd.points)

        # load imgs
        #各视角图像，列表
        imgs = []
        for view in views:
            img = cv2.imread(f'{root_dir}/sync_data/{args.scene_name}/{view}/{sample_token}.jpg')
            if args.point_to_img:
                height, width = img.shape[:2]
                intrinsic = config_dict[view]['cam_intrinsic']
                sensor2ego = np.zeros((4, 4))
                w, x, y, z = config_dict[view]['sensor2ego_rotation']  # 四元数格式
                sensor2ego[:3, :3] = Quaternion(w, x, y, z).rotation_matrix  # (3, 3)
                sensor2ego[:3, 3] = config_dict[view]['sensor2ego_translation']
                sensor2ego[3, 3] = 1
                #尺寸为[n,2]
                points, _, _ = point2pixels(all_points, intrinsic, sensor2ego, height, width)
                for point in points:
                    center_coordinates = (int(point[0]), int(point[1]))
                    cv2.circle(img, center_coordinates, radius=1, color=(255, 0, 0), thickness=-1)
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
            imgs.append(img)

        # 将点云添加到可视化窗口
        vis.add_geometry(pcd)

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

        big_img = np.zeros((900 + canva_size * scale_factor, 1600 * 3, 3),
                       dtype=np.uint8)
        #显示前方多视角图像
        big_img[:900, :, :] = np.concatenate(imgs[:3], axis=1)
        big_img = cv2.resize(big_img, (int(1600 / scale_factor * 3),
                                       int(900 / scale_factor + canva_size)))
        w_begin = int((1600 * 3 / scale_factor - canva_size) // 2)
        #显示占用栅格
        big_img[int(900 / scale_factor):int(900 / scale_factor) + canva_size,
                w_begin:w_begin + canva_size, :] = occ_canvas_resize

        if args.format == 'image':
            out_dir = os.path.join(vis_dir, root_dir.split('/', 1)[1], args.scene_name, sample_token)
            mmcv.mkdir_or_exist(out_dir)
            for i, img in enumerate(imgs):
                cv2.imwrite(os.path.join(out_dir, f'img{i}.png'), img)
            cv2.imwrite(os.path.join(out_dir, 'point_cloud.png'), occ_canvas)
            cv2.imwrite(os.path.join(out_dir, 'overall.png'), big_img)
        elif args.format == 'video':
            cv2.putText(big_img, f'{cnt:{cnt}}', (5, 15), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                        fontScale=0.5)
            cv2.putText(big_img, sample_token, (5, 55), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                        fontScale=0.5)
            vout.write(big_img)

    if args.format == 'video':
        vout.release()
    vis.destroy_window()

if __name__ == '__main__':
    main()