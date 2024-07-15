import pickle

map_name_from_general_to_detection = {
    "vehicle.tank": 'tank',
    'movable_object.water_barrier': "water_barrier",
    'pedestrian':'pedestrian',
    'movable_object.horse_barrier': 'horse_barrier',
    'movable_object.trafficcone': 'trafficcone'
}

classes = [
    'pedestrian','tank',"water_barrier",'horse_barrier','trafficcone'
]

def add_ann_adj_info(extra_tag, dataroot):
    for set in ['train', 'val']:
        dataset = pickle.load(
            open('%s/%s_infos_%s.pkl' % (dataroot, extra_tag, set), 'rb'))
        infos = dataset['infos']
        for id in range(len(dataset['infos'])):
            gt_boxes, gt_labels = [], []
            assert infos[id]['gt_boxes'].shape[0] == infos[id]['gt_names'].shape[0], "bbox 与 lables 长度不匹配"
            for bbox_id in range((infos[id]['gt_boxes'].shape[0])):
                gt_box = infos[id]['gt_boxes'][bbox_id, :]
                gt_label = infos[id]['gt_names'][bbox_id]

                gt_boxes.append(gt_box)
                gt_labels.append(
                    classes.index(
                    map_name_from_general_to_detection[gt_label])
                )
            # 添加目标检测标签
            dataset['infos'][id]['ann_infos'] = gt_boxes, gt_labels
            token =  dataset['infos'][0]['token']
            # dataset['infos'][id]['scene_token'] = sample['scene_token']  # nus: 唯一辨识码 'cc8c0bf57f984915a77078b10eb33198'
            token_id = token.split('-')[-1]
            dataset['infos'][id]['scene_token'] = token_id # custom: 唯一辨识码 '001500'

            # dataset['infos'][id]['scene_name'] = scene['name']  # nus: scene['name'] = 'scene-0061'
            scene_id = token.split('-')[2].split('_')[-1]
            dataset['infos'][id]['scene_name'] = scene_id # cus: scene['name'] = '06'
            #添加占用栅格真值的文件路径
            dataset['infos'][id]['occ_path'] = \
                './data/bevdataset_hzy_car/gts/%s/%s'%(scene_id, token_id)
        with open('%s/%s_infos_%s.pkl' % (dataroot, extra_tag, set),
                  'wb') as fid:
            pickle.dump(dataset, fid)

if __name__ == '__main__':

    dataroot = './data/bevdataset_hzy_car/'
    extra_tag = 'bevdetv2-nuscenes'
    add_ann_adj_info(extra_tag, dataroot)