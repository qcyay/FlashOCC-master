from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
import numpy as np
from open3d import *
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.geometry_utils import points_in_box
import os.path as osp
from functools import partial
from utils.points_process import *
from sklearn.neighbors import KDTree
import open3d as o3d
import argparse
INTER_STATIC_POINTS = {}
INTER_STATIC_POSE = {}
INTER_STATIC_LABEL = {}

def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--dataroot',
        type=str,
        default='./project/data/nuscenes/',
        help='specify the root path of dataset')
    parser.add_argument(
        '--save_path',
        type=str,
        default='./project/data/nuscenes//occupancy2/',
        required=False,
        help='specify sweeps of lidar per example')
    parser.add_argument(
        '--num_sweeps',
        type=int,
        default=10,
        required=False,
        help='specify sweeps of lidar per example')
    args = parser.parse_args()
    return args

def multi_apply(func, *args, **kwargs):
    #partial固定函数的部分参数，返回一个新的函数
    pfunc = partial(func, **kwargs) if kwargs else func
    #map根据提供的函数对指定序列做映射
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def align_dynamic_thing(box, prev_instance_token, nusc, prev_points, ego_frame_info):
        if prev_instance_token not in ego_frame_info['instance_tokens']:
            box_mask = points_in_box(box,
                                    prev_points[:3, :])
            return np.zeros((prev_points.shape[0], 0)), np.zeros((0, )), box_mask

        #保留在物体框中的点
        box_mask = points_in_box(box,
                                    prev_points[:3, :])
        box_points = prev_points[:, box_mask].copy()
        prev_bbox_center = box.center
        prev_rotate_matrix = box.rotation_matrix

        #将物体框中的点云根据中间帧中物体框的朝向进行逆旋转
        box_points = rotate(box_points, np.linalg.inv(prev_rotate_matrix), center=prev_bbox_center)
        #当前帧中和中间帧物体框对应的序号
        target = ego_frame_info['instance_tokens'].index(prev_instance_token)
        ego_boxes_center = ego_frame_info['boxes'][target].center
        #将中间帧物体框中的点从中间帧移动到当前帧
        box_points = translate(box_points, ego_boxes_center-prev_bbox_center)
        #将物体框中的点根据当前帧中物体框的朝向进行旋转
        box_points = rotate(box_points, ego_frame_info['boxes'][target].rotation_matrix, center=ego_boxes_center)
        #保留中间帧物体框变换后在当前帧物体框中的点
        box_points_mask = filter_points_in_ego(box_points, ego_frame_info, prev_instance_token)
        box_points = box_points[:, box_points_mask]
        box_label = np.full_like(box_points[0], nusc.lidarseg_name2idx_mapping[box.name]).copy()
        return box_points, box_label, box_mask


def get_frame_info(frame, nusc: NuScenes, gt_from='lidarseg'):
    '''
    get frame info
    return: frame_info (Dict):

    '''
    #读取激光雷达信息
    sd_rec = nusc.get('sample_data', frame['data']['LIDAR_TOP'])
    #获取激光雷达路径和物体框信息
    lidar_path, boxes, _ = nusc.get_sample_data(frame['data']['LIDAR_TOP'])
    # lidarseg_labels_filename = os.path.join(nusc.dataroot,
    #                                             nusc.get(gt_from, sd_rec)['filename'])
    #获取激光雷达语义分割标签文件位置
    lidarseg_labels_filename = osp.join(nusc.dataroot,
                                    nusc.get(gt_from, frame['data']['LIDAR_TOP'])['filename'])
    #点云语义分割标签
    points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)
    #激光雷达数据
    pc = LidarPointCloud.from_file(nusc.dataroot+sd_rec['filename']) 

    # pc = LidarPointCloud.from_file(nusc.dataroot+sd_rec['filename']) 
    cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
    #自车位姿
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    velocities = np.array(
                [nusc.box_velocity(token)[:2] for token in frame['anns']])
    velocities = np.concatenate((velocities, np.zeros_like(velocities[:, 0:1])), axis=-1)
    velocities = velocities.transpose(1, 0)
    instance_tokens = [nusc.get('sample_annotation', token)['instance_token'] for token in frame['anns']]
    frame_info = {
        'pc': pc,
        'token': frame['token'],
        'lidar_token': frame['data']['LIDAR_TOP'],
        'cs_record': cs_record,
        'pose_record': pose_record,
        'velocities': velocities,
        'lidarseg': points_label,
        'boxes': boxes,
        'anno_token': frame['anns'],
        'instance_tokens': instance_tokens,
        'timestamp': frame['timestamp'],
    }
    return frame_info


def get_intermediate_frame_info(nusc: NuScenes, prev_frame_info, lidar_rec, flag):
    intermediate_frame_info = dict()
    #当前帧雷达点云数据
    pc = LidarPointCloud.from_file(nusc.dataroot+lidar_rec['filename']) 
    intermediate_frame_info['pc'] = pc
    #移除距原点一定半径范围内太近的点
    intermediate_frame_info['pc'].points = remove_close(intermediate_frame_info['pc'].points)
    intermediate_frame_info['lidar_token'] = lidar_rec['token']
    #当前帧雷达标定信息
    intermediate_frame_info['cs_record'] = nusc.get('calibrated_sensor',
                             lidar_rec['calibrated_sensor_token'])
    #当前帧token
    sample_token = lidar_rec['sample_token']
    #当前帧相关信息
    frame = nusc.get('sample', sample_token)
    #当前帧标注框token
    instance_tokens = [nusc.get('sample_annotation', token)['instance_token'] for token in frame['anns']]
    #当前帧自车位姿相关信息
    intermediate_frame_info['pose_record'] = nusc.get('ego_pose', lidar_rec['ego_pose_token'])
    #lidar_path，激光雷达数据路径，boxes，真值框信息
    lidar_path, boxes, _ = nusc.get_sample_data(lidar_rec['token'])
    intermediate_frame_info['boxes'] = boxes
    intermediate_frame_info['instance_tokens'] = instance_tokens
    assert len(boxes) == len(instance_tokens) , print('erro')
    return intermediate_frame_info

def intermediate_keyframe_align(nusc: NuScenes, prev_frame_info, ego_frame_info, cur_sample_points, cur_sample_labels):
    ''' align prev_frame points to ego_frame
    return: points (np.array) aligned points of prev_frame
            pc_segs (np.array) label of aligned points of prev_frame
    '''
    # 移除距原点一定半径范围内太近的点
    prev_frame_info['pc'].points = remove_close(prev_frame_info['pc'].points, (1, 2))
    #pcs，列表，包含N_box个元素，元素尺寸为[4,n]，labels，列表，包含N_box个元素，元素尺寸为[n]，masks，列表，包含N_box个元素，元素尺寸为[N]
    pcs, labels, masks = multi_apply(align_dynamic_thing, prev_frame_info['boxes'], prev_frame_info['instance_tokens'], nusc=nusc, prev_points=prev_frame_info['pc'].points, ego_frame_info=ego_frame_info)

    # for box, instance_token in zip(prev_frame_info['boxes'], prev_frame_info['instance_tokens']):
    #     align_dynamic_thing(box, instance_token, nusc=nusc, prev_points=prev_frame_info['pc'].points, ego_frame_info=ego_frame_info)

    masks = np.stack(masks, axis=-1)
    masks = masks.sum(axis=-1)
    masks = ~(masks>0)
    prev_frame_info['pc'].points = prev_frame_info['pc'].points[:, masks]

    if prev_frame_info['lidar_token'] in INTER_STATIC_POINTS:
        static_points = INTER_STATIC_POINTS[prev_frame_info['lidar_token']].copy()
        static_points = prev2ego(static_points, INTER_STATIC_POSE[prev_frame_info['lidar_token']], ego_frame_info)
        static_points_label = INTER_STATIC_LABEL[prev_frame_info['lidar_token']].copy()
        assert static_points_label.shape[0] == static_points.shape[1], f"{static_points_label.shape, static_points.shape}"
    else:
        static_points = prev2ego(prev_frame_info['pc'].points, prev_frame_info, ego_frame_info)
        static_points_label = np.full_like(static_points[0], -1)
        #将中间帧静态点标签设置为距离最近的当前帧的点标签（距离需小于预设置的最大距离）
        static_points, static_points_label = search_label(cur_sample_points, cur_sample_labels, static_points, static_points_label)
        INTER_STATIC_POINTS[prev_frame_info['lidar_token']] = static_points.copy()
        INTER_STATIC_LABEL[prev_frame_info['lidar_token']] = static_points_label.copy()
        INTER_STATIC_POSE[prev_frame_info['lidar_token']] = {'cs_record': ego_frame_info['cs_record'],
                                                            'pose_record': ego_frame_info['pose_record'],
                                                            }
    pcs.append(static_points)
    labels.append(static_points_label)
    return np.concatenate(pcs, axis=-1), np.concatenate(labels)

def nonkeykeyframe_align(nusc: NuScenes, prev_frame_info, ego_frame_info, flag='prev', cur_sample_points=None, cur_sample_labels=None):
    ''' align non keyframe points to ego_frame
    return: points (np.array) aligned points of prev_frame
            pc_segs (np.array) seg of aligned points of prev_frame
    '''
    pcs = []
    labels = []
    #获取起始帧相关信息
    start_frame = nusc.get('sample', prev_frame_info['token'])
    #获取起始帧的前一关键帧或后一关键帧相关信息（根据关键帧信息进行数据读取）
    end_frame = nusc.get('sample', start_frame[flag])
    # next_frame_info = get_frame_info(end_frame, nusc)
    #获取起始帧激光雷达相关信息
    start_sd_record = nusc.get('sample_data', start_frame['data']['LIDAR_TOP'])
    #获取起始帧的前一中间帧或后一中间帧激光雷达相关信息（根据激光雷达信息进行数据读取）
    start_sd_record = nusc.get('sample_data', start_sd_record[flag])
    # end_sd_record = nusc.get('sample_data', end_frame['data']['LIDAR_TOP'])
    # get intermediate frame info
    while start_sd_record['token'] != end_frame['data']['LIDAR_TOP']:
        #获取中间帧相关信息（有标注框信息，无语义标注信息）
        intermediate_frame_info = get_intermediate_frame_info(nusc, prev_frame_info, start_sd_record, flag)
        #将中间帧动态点和静态点按照不同方式变换到当前帧坐标系下并获取对应标签
        pc, label = intermediate_keyframe_align(nusc, intermediate_frame_info, ego_frame_info, cur_sample_points, cur_sample_labels)
        # 获取起始帧的前一中间帧或后一中间帧激光雷达相关信息
        start_sd_record = nusc.get('sample_data', start_sd_record[flag])
        pcs.append(pc)
        labels.append(label)
    return np.concatenate(pcs, axis=-1), np.concatenate(labels)


def prev2ego(points, prev_frame_info, income_frame_info, velocity=None, time_gap=0.0):
    ''' translation prev points to ego frame
    '''
    # prev_sd_rec = nusc.get('sample_data', prev_frame_info['data']['LIDAR_TOP'])

    #前帧中激光雷达到自车坐标系的变换矩阵相关信息
    prev_cs_record = prev_frame_info['cs_record']
    #前帧中自车到全局坐标系的变换矩阵相关信息
    prev_pose_record = prev_frame_info['pose_record']

    #将前帧激光雷达点从雷达坐标系变换到自车坐标系
    points = transform(points, Quaternion(prev_cs_record['rotation']).rotation_matrix, np.array(prev_cs_record['translation']))
    #将前帧激光雷达点从自车坐标系变换到全局坐标系
    points = transform(points, Quaternion(prev_pose_record['rotation']).rotation_matrix, np.array(prev_pose_record['translation']))

    if velocity is not None:
        points[:3, :] = points[:3, :] + velocity*time_gap

    #当前帧中激光雷达到自车坐标系的变换矩阵相关信息
    ego_cs_record = income_frame_info['cs_record']
    #当前帧中自车到全局坐标系的变换矩阵相关信息
    ego_pose_record = income_frame_info['pose_record']
    #将前帧激光雷达点从全局坐标系变换到当前帧自车坐标系
    points = transform(points, Quaternion(ego_pose_record['rotation']).rotation_matrix, np.array(ego_pose_record['translation']), inverse=True)
    #将前帧激光雷达点从当前帧自车坐标系变换到雷达坐标系
    points = transform(points, Quaternion(ego_cs_record['rotation']).rotation_matrix, np.array(ego_cs_record['translation']), inverse=True)
    return points.copy()


def filter_points_in_ego(points, frame_info, instance_token):
    '''
    filter points in this frame box
    '''
    index = frame_info['instance_tokens'].index(instance_token)
    box = frame_info['boxes'][index]
    # print(f"ego box pos {box.center}")
    box_mask = points_in_box(box, points[:3, :])
    return box_mask

def keyframe_align(prev_frame_info, ego_frame_info):
    ''' align prev_frame points to ego_frame
    return: points (np.array) aligned points of prev_frame
            pc_segs (np.array) seg of aligned points of prev_frame
    '''
    pcs = []
    pc_segs = []
    #前帧语义分割标签
    lidarseg_prev = prev_frame_info['lidarseg']
    ego_vehicle_mask = (lidarseg_prev == 31) | (lidarseg_prev == 0)
    lidarseg_prev = lidarseg_prev[~ego_vehicle_mask]
    prev_frame_info['pc'].points = prev_frame_info['pc'].points[:, ~ego_vehicle_mask]

    # translation prev static points to ego
    static_mask = (lidarseg_prev >= 24) & (lidarseg_prev <= 30)

    static_points = prev_frame_info['pc'].points[:, static_mask]
    static_seg = lidarseg_prev[static_mask]
    #将前帧静态点云变换到当前帧雷达坐标系
    #尺寸为[4,N_staric]
    static_points = prev2ego(static_points, prev_frame_info, ego_frame_info)
    pcs.append(static_points.copy())
    pc_segs.append(static_seg.copy())
    #保留前帧非静态点云
    prev_frame_info['pc'].points = prev_frame_info['pc'].points[:, ~static_mask].copy()
    lidarseg_prev = lidarseg_prev[~static_mask]
    # translation prev moving points to ego
    for index_anno in range(len(prev_frame_info['boxes'])):
        if prev_frame_info['instance_tokens'][index_anno] not in ego_frame_info['instance_tokens']:
            continue
        #确定前帧中在物体框中的点云
        box_mask = points_in_box(prev_frame_info['boxes'][index_anno],
                                    prev_frame_info['pc'].points[:3, :])
        #前帧中在物体框中的点云
        box_points = prev_frame_info['pc'].points[:, box_mask].copy()
        boxseg_prev = lidarseg_prev[box_mask].copy()
        #前帧中物体框中心
        prev_bbox_center = prev_frame_info['boxes'][index_anno].center

        # 对于动态物体，不通过位姿进行变换，而是根据实例号的对应关系进行变换
        prev_rotate_matrix = prev_frame_info['boxes'][index_anno].rotation_matrix
        # TODO 这里搞清楚原理
        # 将点根据物体框的朝向进行逆变换
        box_points = rotate(box_points, np.linalg.inv(prev_rotate_matrix), center=prev_bbox_center)

        #找到前帧物体在当前帧对应的物体序号
        target = ego_frame_info['instance_tokens'].index(prev_frame_info['instance_tokens'][index_anno])
        #当前帧物体框中心
        ego_boxes_center = ego_frame_info['boxes'][target].center
        #对前帧物体框中点的位置进行变换
        box_points = translate(box_points, ego_boxes_center-prev_bbox_center)
        #对前帧物体框中的点根据当前物体框的朝向进行变换
        box_points = rotate(box_points, ego_frame_info['boxes'][target].rotation_matrix, center=ego_boxes_center)

        #保留前帧物体框中点变换后在当前帧对应物体框中的点
        box_points_mask = filter_points_in_ego(box_points, ego_frame_info, prev_frame_info['instance_tokens'][index_anno])
        box_points = box_points[:, box_points_mask]
        boxseg_prev = boxseg_prev[box_points_mask]

        pcs.append(box_points)
        pc_segs.append(boxseg_prev)
    return np.concatenate(pcs, axis=-1), np.concatenate(pc_segs, axis=-1)


def search_label(points, lidar_seg, intermediate_pcs, intermediate_labels, max_dist=0.5):
    #未标注点掩码
    unlabel_mask = intermediate_labels == -1
    thing_mask = (lidar_seg >= 24) & (lidar_seg <=30)
    #尺寸为[N_thing]
    thing_label = lidar_seg[thing_mask]
    #尺寸为[4,N_thing]
    thing_points = points[:, thing_mask]
    #尺寸为[4,n_static]
    unlabeled_points = intermediate_pcs[:, unlabel_mask]
    #KDTree用于快速广义N点问题
    tree = KDTree(thing_points.transpose(1, 0)[:, :3])
    #尺寸为[n_static,4]
    unlabeled_points = unlabeled_points.transpose(1, 0)
    #query查询树中最近的k个临近点
    #dists，距离，尺寸为[n_static,1]，inds，序号，尺寸为[n_static,1]
    dists, inds = tree.query(unlabeled_points[:, :3], k=1)
    # 尺寸为[n_static]
    inds = np.reshape(inds, (-1,))
    # 尺寸为[n_static]
    dists = np.reshape(dists, (-1,))
    dists = dists<max_dist
    #take_along_axis函数用于由索引矩阵生成新的矩阵
    intermediate_labels[unlabel_mask] = np.take_along_axis(thing_label, inds, axis=-1)
    return intermediate_pcs[:, dists], intermediate_labels[dists]


def generate_occupancy_data(nusc: NuScenes, cur_sample, num_sweeps, save_path='./occupacy/', gt_from: str = 'lidarseg'):
    pcs = [] # for keyframe points
    pc_segs = []

    intermediate_pcs = [] # # for non keyfrme points
    intermediate_labels = []
    #读取激光雷达信息
    lidar_data = nusc.get('sample_data',
                            cur_sample['data']['LIDAR_TOP'])
    #读取激光雷达数据
    pc = LidarPointCloud.from_file(nusc.dataroot+lidar_data['filename'])
    filename = os.path.split(lidar_data['filename'])[-1]
    #当前激光雷达token
    lidar_sd_token = cur_sample['data']['LIDAR_TOP']

    #获取雷达语义分割标签文件位置
    lidarseg_labels_filename = os.path.join(nusc.dataroot,
                                                nusc.get(gt_from, lidar_sd_token)['filename'])
    #获取雷达语义分割标签
    lidar_seg = load_bin_file(lidarseg_labels_filename, type=gt_from)

    # align keyframes
    count_prev_frame = 0
    prev_frame = cur_sample.copy()

    while num_sweeps > 0:
        if prev_frame['prev'] == '':
            break
        prev_frame = nusc.get('sample', prev_frame['prev'])
        count_prev_frame += 1
        if count_prev_frame == num_sweeps:
            break
    #获取当前帧相关信息
    cur_sample_info = get_frame_info(cur_sample, nusc=nusc)
    # convert prev keyframe to ego frame
    if count_prev_frame > 0:
        #获取前帧相关信息
        prev_info = get_frame_info(prev_frame, nusc)
    pc_points = None
    pc_seg = None
    while count_prev_frame > 0:
        income_info = get_frame_info(frame=prev_frame, nusc=nusc)
        prev_frame = nusc.get('sample', prev_frame['next'])
        prev_info = income_info
        #将前帧点云分别根据静态和动态变换到当前帧点云
        pc_points, pc_seg = keyframe_align(prev_info, cur_sample_info)
        pcs.append(pc_points)
        pc_segs.append(pc_seg)
        count_prev_frame -= 1

    # convert next frame to ego frame
    next_frame = cur_sample.copy()
    pc_points = None
    pc_seg = None
    count_next_frame = 0
    while num_sweeps > 0:
        if next_frame['next'] == '':
            break
        next_frame = nusc.get('sample', next_frame['next'])
        count_next_frame += 1
        if count_next_frame == num_sweeps:
            break

    if count_next_frame > 0:
        # 获取后帧相关信息
        prev_info = get_frame_info(next_frame, nusc=nusc)

    while count_next_frame > 0:
        income_info = get_frame_info(frame=next_frame, nusc=nusc)
        prev_info = income_info
        next_frame =  nusc.get('sample', next_frame['prev'])
        #将后帧点云分别根据静态和动态变换到当前帧点云
        pc_points, pc_seg = keyframe_align(prev_info, cur_sample_info)
        pcs.append(pc_points)
        pc_segs.append(pc_seg)
        count_next_frame -= 1
    pcs = np.concatenate(pcs, axis=-1)
    pc_segs = np.concatenate(pc_segs)

    pc.points = np.concatenate((pc.points, pcs), axis=-1)
    lidar_seg = np.concatenate((lidar_seg, pc_segs))

    #保留在一定范围内的点
    range_mask = (pc.points[0,:]<= 60) & (pc.points[0,:]>=-60)\
     &(pc.points[1,:]<= 60) & (pc.points[1,:]>=-60)\
      &(pc.points[2,:]<= 10) & (pc.points[2,:]>=-10)
    pc.points = pc.points[:, range_mask]
    lidar_seg = lidar_seg[range_mask]

    # align nonkeyframe
    count_prev_frame = 0
    prev_frame = cur_sample.copy()

    while num_sweeps > 0:
        if prev_frame['prev'] == '':
            break
        prev_frame = nusc.get('sample', prev_frame['prev'])
        count_prev_frame += 1
        if count_prev_frame == num_sweeps:
            break
    #获取当前帧相关信息
    cur_sample_info = get_frame_info(cur_sample, nusc=nusc)
    # convert prev frame to ego frame
    if count_prev_frame > 0:
        prev_info = get_frame_info(prev_frame, nusc)
    while count_prev_frame > 0:
        income_info = get_frame_info(frame=prev_frame, nusc=nusc)
        prev_frame = nusc.get('sample', prev_frame['next'])
        prev_info = income_info
        # 将关键帧之间所有中间帧的静态点和动态点变换到自车坐标系下并获取对应标签
        intermediate_pc, intermediate_label = nonkeykeyframe_align(nusc, prev_info, cur_sample_info, 'next', pc.points, lidar_seg)
        intermediate_pcs.append(intermediate_pc)
        intermediate_labels.append(intermediate_label)
        count_prev_frame -= 1

    next_frame = cur_sample.copy()
    count_next_frame = 0
    while num_sweeps > 0:
        if next_frame['next'] == '':
            break
        next_frame = nusc.get('sample', next_frame['next'])
        count_next_frame += 1
        if count_next_frame == num_sweeps:
            break

    if count_next_frame > 0:
        prev_info = get_frame_info(next_frame, nusc=nusc)
    while count_next_frame > 0:
        income_info = get_frame_info(frame =next_frame, nusc=nusc)
        prev_info = income_info
        next_frame =  nusc.get('sample', next_frame['prev'])
        # 将关键帧之间所有中间帧的静态点和动态点变换到自车坐标系下并获取对应标签
        intermediate_pc, intermediate_label = nonkeykeyframe_align(nusc, prev_info, cur_sample_info, 'prev', pc.points, lidar_seg)
        intermediate_pcs.append(intermediate_pc)
        intermediate_labels.append(intermediate_label)
        count_next_frame -= 1
    #尺寸为[4,N_inter]
    intermediate_pcs = np.concatenate(intermediate_pcs, axis=-1)
    #尺寸为[N_inter]
    intermediate_labels = np.concatenate(intermediate_labels)
    #尺寸为[1,N_inter]
    intermediate_labels = np.reshape(intermediate_labels, (1, -1))
    #尺寸为[5,N_inter]
    intermediate_pcs = np.concatenate((intermediate_pcs, intermediate_labels), axis=0)
    lidar_seg = np.reshape(lidar_seg, (1, -1))
    pc.points = np.concatenate((pc.points, lidar_seg), axis=0)
    pc.points = np.concatenate((pc.points, intermediate_pcs), axis=1)

    # removed too dense point
    #尺寸为[N,3]
    raw_point = pc.points.transpose(1,0)[:,:3]
    fake_colors = pc.points.transpose(1,0)[:,3:]/255 
    assert pc.points.transpose(1,0)[:,3:].max()<=255
    n, _ = fake_colors.shape
    fake_colors = np.concatenate((fake_colors, np.zeros((n,1))), axis=1)
    pcd=o3d.open3d.geometry.PointCloud()

    #Vector3dVector将形状为 (n, 3) 的 float64 numpy 数组转换为 Open3D 格式。
    pcd.points= o3d.open3d.utility.Vector3dVector(raw_point)
    pcd.colors = o3d.open3d.utility.Vector3dVector(fake_colors)
    #voxel_down_sample使用体素将输入点云下采样为输出点云
    pcd_new = o3d.geometry.PointCloud.voxel_down_sample(pcd, 0.2)
    #尺寸为[n,3]
    new_points = np.asarray(pcd_new.points)
    #尺寸为[n,2]
    fake_colors = np.asarray(pcd_new.colors)[:,:2]*255
    #尺寸为[n,5]
    new_points = np.concatenate((new_points, fake_colors), axis=1)

    range_mask = (new_points[:,0]<= 60) &  (new_points[:,0]>=-60)\
     &(new_points[:,1]<= 60) &  (new_points[:,1]>=-60)\
      &(new_points[:,2]<= 10) &  (new_points[:,2]>=-10)
    new_points = new_points[range_mask]
    new_points = new_points.astype(np.float16)
    new_points.tofile(save_path +filename)
    return pc.points, lidar_seg

def convert2occupy(dataroot,
                        save_path, num_sweeps=10,):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    cnt = 0
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
    for scene in nusc.scene:
        INTER_STATIC_POINTS.clear()
        INTER_STATIC_LABEL.clear()
        INTER_STATIC_POSE.clear()
        #读取场景首个token
        sample_token = scene['first_sample_token']
        #读取当前token数据
        cur_sample = nusc.get('sample', sample_token)
        while True:
            cnt += 1
            print(cnt)
            #将当前帧前后特定范围内所有关键帧和中间帧的点云根据静态点和动态点变换到当前帧坐标系下并得到对应标签
            generate_occupancy_data(nusc, cur_sample, num_sweeps, save_path=save_path)
            if cur_sample['next'] == '':
                break
            #读取当前token的下一个token数据
            cur_sample = nusc.get('sample', cur_sample['next'])

if __name__ == "__main__":
    args = parse_args()
    convert2occupy(args.dataroot, args.save_path, args.num_sweeps)

