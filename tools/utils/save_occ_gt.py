import os
import numpy as np
from tqdm import tqdm

from mayavi import mlab
import open3d as o3d
import argparse
POINT_CLOUD_RANGE = [-40, -40, -1, 40, 40, 5.4]
VOXEL_SIZE = 0.4
SPTIAL_SHAPE = (int((POINT_CLOUD_RANGE[3]-POINT_CLOUD_RANGE[0])/VOXEL_SIZE),
                int((POINT_CLOUD_RANGE[4]-POINT_CLOUD_RANGE[1])/VOXEL_SIZE),
                int((POINT_CLOUD_RANGE[5]-POINT_CLOUD_RANGE[2])/VOXEL_SIZE))

def remove_far(points, POINT_CLOUD_RANGE):
    mask = (points[:, 0]>POINT_CLOUD_RANGE[0]) & (points[:, 0]<POINT_CLOUD_RANGE[3]) & (points[:, 1]>POINT_CLOUD_RANGE[1]) & (points[:, 1]<POINT_CLOUD_RANGE[4]) \
            & (points[:, 2]>POINT_CLOUD_RANGE[2]) & (points[:, 2]<POINT_CLOUD_RANGE[5])
    return points[mask, :]

def voxelize(voxel: np.array, label_count: np.array):

    for x in range(voxel.shape[0]):
        for y in range(voxel.shape[1]):
            for z in range(voxel.shape[2]):
                if label_count[x, y, z] == 0:
                    continue
                #尺寸为[20]
                labels = voxel[x, y, z]
                try:
                    #将标签设置为体素中最多点对应的标签
                    #np.bincount计算非负整数数组中每个值出现的次数
                    label_count[x, y, z] = np.argmax(np.bincount(labels[labels!=0]))
                except:
                    label_count[x, y, z] = 0
    return label_count

def points2voxel(points, SPTIAL_SHAPE, VOXEL_SIZE, max_points=5, specific_category=None):
    #尺寸为[200,200,16,20]
    voxel = np.zeros((*SPTIAL_SHAPE, max_points), dtype=np.int64)
    #尺寸为[200,100,16]
    label_count = np.zeros((SPTIAL_SHAPE), dtype=np.int64)
    #argsort函数返回数组值从小到大的索引值
    index = points[:, 4].argsort()
    points = points[index]
    for point in points:
      
        x, y, z = point[0], point[1], point[2]
        #获取点的体素坐标
        x = round((x - POINT_CLOUD_RANGE[0]) / VOXEL_SIZE)
        y = round((y - POINT_CLOUD_RANGE[1]) / VOXEL_SIZE)
        z = round((z - POINT_CLOUD_RANGE[2]) / VOXEL_SIZE)

        try:
            #记录体素内各点的标签
            voxel[x, y, z, label_count[x, y, z]] = int(point[4])  # map_label[int(point[4])]
            label_count[x, y, z] += 1
        except:
            continue

    #设置各体素标签
    # 尺寸为[200,200,16]
    voxel = voxelize(voxel, label_count)
    # voxel = voxel.astype(np.float64)
    return voxel


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)
    g_zz = np.arange(0, dims[2] + 1)

    # Obtaining the grid with coords...
    #np.meshgrid从坐标向量返回一个坐标矩阵的元组
    #xx，x轴坐标，尺寸为[512,512,40]，yy，y轴坐标，尺寸为[512,512,40]，zz，z轴坐标，尺寸为[512,512,40]
    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    #所有栅格的坐标，尺寸为[N,3]
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)

    #所有栅格中心点的坐标，尺寸为[N,3]
    coords_grid = (coords_grid * resolution) + resolution / 2

    #尺寸为[N,3]
    temp = np.copy(coords_grid)
    temp[:, 0] = coords_grid[:, 1]
    temp[:, 1] = coords_grid[:, 0]
    coords_grid = np.copy(temp)

    return coords_grid


def draw(
    voxels,
    VOXEL_SIZE=0.2,
):

    # Compute the voxels coordinates
    #计算所有体素中心点的坐标
    #尺寸为[N,3]
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], VOXEL_SIZE
    )

    # Attach the predicted class to every voxel
    #尺寸为[N,4]
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    grid_voxels = grid_coords[
        (grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 255)
    ]

    figure = mlab.figure(size=(1400, 1400), bgcolor=(1, 1, 1))

    #points3d函数基于Numpy数组x、 y、 z提供的三维点坐标，绘制点图形
    plt_plot = mlab.points3d(
        grid_voxels[:, 0],
        grid_voxels[:, 1],
        grid_voxels[:, 2],
        grid_voxels[:, 3],
        colormap="viridis",
        scale_factor=VOXEL_SIZE - 0.05 * VOXEL_SIZE,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=19,
    )

    classname_to_color = {  # RGB.
        "noise": (0, 0, 0),  # Black.
        "animal": (70, 130, 180),  # Steelblue
        "human.pedestrian.adult": (0, 0, 230),  # Blue
        "human.pedestrian.child":(0, 0, 230),  # Skyblue,
        "human.pedestrian.construction_worker":(0, 0, 230),  # Cornflowerblue
        "human.pedestrian.personal_mobility": (0, 0, 230),  # Palevioletred
        "human.pedestrian.police_officer":(0, 0, 230),  # Navy,
        "human.pedestrian.stroller": (0, 0, 230),  # Lightcoral
        "human.pedestrian.wheelchair": (0, 0, 230),  # Blueviolet
        "movable_object.barrier": (112, 128, 144),  # Slategrey
        "movable_object.debris": (112, 128, 144),  # Chocolate
        "movable_object.pushable_pullable":(112, 128, 144),  # Dimgrey
        "movable_object.trafficcone":(112, 128, 144),  # Darkslategrey
        "static_object.bicycle_rack": (188, 143, 143),  # Rosybrown
        "vehicle.bicycle": (220, 20, 60),  # Crimson
        "vehicle.bus.bendy":(255, 158, 0),  # Coral
        "vehicle.bus.rigid": (255, 158, 0),  # Orangered
        "vehicle.car": (255, 158, 0),  # Orange
        "vehicle.construction":(255, 158, 0),  # Darksalmon
        "vehicle.emergency.ambulance":(255, 158, 0),
        "vehicle.emergency.police": (255, 158, 0),  # Gold
        "vehicle.motorcycle": (255, 158, 0),  # Red
        "vehicle.trailer":(255, 158, 0),  # Darkorange
        "vehicle.truck": (255, 158, 0),  # Tomato
        "flat.driveable_surface": (0, 207, 191),  # nuTonomy green
        "flat.other":(0, 207, 191),
        "flat.sidewalk": (75, 0, 75),
        "flat.terrain": (0, 207, 191),
        "static.manmade": (222, 184, 135),  # Burlywood
        "static.other": (0, 207, 191),  # Bisque
        "static.vegetation": (0, 175, 0),  # Green
        "vehicle.ego": (255, 240, 245)
    }
    #尺寸为[N_class,3]
    colors = np.array(list(classname_to_color.values())).astype(np.uint8)
    #尺寸为[N_class,1]
    alpha = np.ones((colors.shape[0], 1), dtype=np.uint8) * 255
    #尺寸为[N_class,4]
    colors = np.hstack([colors, alpha])

    plt_plot.glyph.scale_mode = "scale_by_vector"

    #自定义矢量场中 Glyph 的颜色映射
    plt_plot.module_manager.scalar_lut_manager.lut.table = colors
    plt_plot.module_manager.scalar_lut_manager.data_range = [0, 31]
    mlab.show()

# points = remove_far(points, POINT_CLOUD_RANGE)
def main(path, save_path=None, show=False):

    if args.load_mask:
        # 尺寸为[200,200,16]
        camera_mask = np.load(os.path.join(path, 'mask_camera.npz'))['mask_camera']
        # 尺寸为[200,200,16]
        lidar_mask = np.ones_like(camera_mask)

    scene_name_list = [name for name in os.listdir(os.path.join(path, 'sync_data')) if os.path.isdir(os.path.join(path, 'sync_data', name))]

    for scene_name in scene_name_list:
        print(f'Processing Scene {scene_name}')
        lidar_path_list = [f'{path}/sync_data/{scene_name}/lidar_concat/{name}' for name in os.listdir(f'{path}/sync_data/{scene_name}/lidar_concat')]

        for lidar_path in tqdm(lidar_path_list):
            # 激光雷达点云
            pcd = o3d.io.read_point_cloud(lidar_path)
            # 尺寸为[N,3]
            points = np.asarray(pcd.points)
            # 尺寸为[N,5]
            points = np.insert(points, 3, [[1],[1]], 1)

            # 尺寸为[200,200,16]
            occ_gt = points2voxel(points, SPTIAL_SHAPE, VOXEL_SIZE, 20)
            name = lidar_path.rsplit('/', 1)[1].split('.')[0]
            if save_path is not None:
                file_path = f'{save_path}/gts/{int(scene_name):02d}/{name}'
            else:
                file_path = f'{path}/gts/{int(scene_name):02d}/{name}'
            os.makedirs(file_path, exist_ok=True)
            # 保存到npz文件
            if args.load_mask:
                np.savez_compressed(f'{file_path}/labels.npz', semantics=occ_gt, mask_lidar=lidar_mask, mask_camera=camera_mask)
            else:
                np.savez_compressed(f'{file_path}/labels.npz', semantics=occ_gt)

            if show:
                draw(
                    occ_gt[::-1, ::-1, :],
                    VOXEL_SIZE=0.2,
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pts-path', required=True)
    parser.add_argument('--voxel-size', type=float, default=0.4)
    parser.add_argument('--load_mask', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--save_path', default=None)

    args = parser.parse_args()
    VOXEL_SIZE = args.voxel_size
    SPTIAL_SHAPE=(int((POINT_CLOUD_RANGE[3]-POINT_CLOUD_RANGE[0])/VOXEL_SIZE),
                  int((POINT_CLOUD_RANGE[4]-POINT_CLOUD_RANGE[1])/VOXEL_SIZE),
                  int((POINT_CLOUD_RANGE[5]-POINT_CLOUD_RANGE[2])/VOXEL_SIZE))
    main(args.pts_path, args.save_path, args.show)