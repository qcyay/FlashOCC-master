# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import sys
import os
import warnings
import json
from tqdm import tqdm
sys.path.append(os.getcwd())

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from pyquaternion import Quaternion

import mmdet
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets.pipelines import Compose

if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

warnings.filterwarnings('ignore')

#读取rosbag中的图像以及相关参数并进行测试

def parse_args():
    parser = argparse.ArgumentParser(
        description='test (and eval) a model')
    #配置文件路径
    parser.add_argument('config', help='test config file path')

    #模型文件
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--scene_name', help='the name of the scene')
    parser.add_argument('--save_dir', help='directory where results will be saved')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    #加载配置文件
    #Config类用于操作配置文件，它支持从多种文件格式中加载配置，包括python，json和yaml
    #对于所有格式的配置文件, 都支持继承。为了重用其他配置文件的字段，需要指定__base__
    cfg = Config.fromfile(args.config)
    #修改一些文件以保持配置的兼容性
    cfg = compat_cfg(cfg)

    # 导入模块并初始化自定义的类
    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                # projects/mmdet3d_plugin/
                plugin_dir = cfg.plugin_dir
                # projects/mmdet3d_plugin
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                # projects
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    # projects.mmdet3d_plugin
                    _module_path = _module_path + '.' + m
                # print(_module_path)
                #导入一个模块
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                plg_lib = importlib.import_module(_module_path)

    # 设置 cudnn_benchmark = True 可以加速输入大小固定的模型
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = cfg.class_names

    # # 所有测试数据的标注文件
    # data = mmcv.load(cfg.test_data_config.ann_file, file_format='pkl')
    # # 将数据信息按时间戳顺序排列
    # data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
    # # 标注信息
    # info = data_infos[0]

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

    keycam = cfg.data_config['cams'][0]
    info = {}
    info['cams'] = config_dict
    for idx, filename in enumerate(tqdm(filename_dict[keycam])):
        for cam in cfg.data_config['cams']:
            info['cams'][cam]['data_path'] = f'{cfg.data_root}/sync_data/{args.scene_name}/{cam}/{filename_dict[cam][idx]}'

        input_dict = dict(curr=info)
        pipeline = cfg.test_pipeline
        pipeline = Compose(pipeline)
        # 键值包括img_inputs和img_metas
        data = pipeline(input_dict)

        #配置文件中测试预处理流程需要判别是否直接调用Collect3D类
        if isinstance(data['img_inputs'], tuple):
            img_inputs = list(data['img_inputs'])
        elif isinstance(data['img_inputs'], list):
            img_inputs = list(data['img_inputs'][0])

        for i, t in enumerate(img_inputs):
            if t is not None:
                img_inputs[i] = t.unsqueeze(0).cuda()

        model.cuda()
        model.eval()
        with torch.no_grad():
            result = model.simple_test(img=img_inputs)
            # print(result[0].shape)

        if args.save_dir:
            # print('\nStarting Saving Prediction...')
            mmcv.mkdir_or_exist(args.save_dir)
            sample_token = filename.rsplit('.', 1)[0]
            save_path = os.path.join(args.save_dir, args.scene_name, sample_token)
            mmcv.mkdir_or_exist(save_path)
            save_path = os.path.join(save_path, 'pred.npz')
            np.savez_compressed(save_path, pred=result[0], sample_token=sample_token)

if __name__ == '__main__':
    main()
