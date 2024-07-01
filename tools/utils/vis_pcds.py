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

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize the points')
    parser.add_argument(
        'res', help='Path to the files')
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

    lidar_filename_list = os.listdir(f'{results_dir}/lidar_concat')

    for cnt, lidar_filename in enumerate(
            lidar_filename_list[:min(args.vis_frames, len(lidar_filename_list))]):
        if cnt % 10 == 0:
            print('%d/%d' % (cnt, min(args.vis_frames, len(lidar_filename_list))))

        sample_token = lidar_filename.split('.')[0]

        # 激光雷达点云数据保存路径
        lidar_path = os.path.join(results_dir, 'lidar_concat', lidar_filename)

        #激光雷达点云
        pcd = o3d.io.read_point_cloud(lidar_path)
        # #设置点云颜色
        # pcd.paint_uniform_color([0, 0, 1])

        # load imgs
        #各视角图像，列表
        imgs = []
        for view in views:
            img = cv2.imread(f'{results_dir}/{view}/{sample_token}.jpg')
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
            out_dir = os.path.join(vis_dir, results_dir.split('/', 1)[1], sample_token)
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