# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet3d.models import DETECTORS
from mmdet3d.models import CenterPoint
from mmdet3d.models import builder


@DETECTORS.register_module()
class BEVDet(CenterPoint):
    def __init__(self, img_backbone, img_neck, img_view_transformer, img_bev_encoder_backbone, img_bev_encoder_neck,
                 pts_bbox_head=None, **kwargs):
        super(BEVDet, self).__init__(img_backbone=img_backbone, img_neck=img_neck, pts_bbox_head=pts_bbox_head,
                                     **kwargs)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone)
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)

    def image_encoder(self, img, stereo=False):
        """
        Args:
            img: (B, N, 3, H, W)
            stereo: bool
        Returns:
            x: (B, N, C, fH, fW)
            stereo_feat: (B*N, C_stereo, fH_stereo, fW_stereo) / None
        """
        #尺寸为[B,N,3,H,W]
        imgs = img
        B, N, C, imH, imW = imgs.shape
        #尺寸为[BN,3,H,W]
        imgs = imgs.view(B * N, C, imH, imW)
        #元组，包含两个张量，x[0]，第三个Res Block输出的特征，尺寸为[BN,C,h,w]，x[1]，第四个Res Block输出的特征，尺寸为[BN,2C,h/2,w/2]
        x = self.img_backbone(imgs)
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            #列表，包含一个张量，尺寸为[BN,256,h,w]
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                #尺寸为[BN,256,h,w]
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        # 尺寸为[B,N,256,h,w]
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat

    @force_fp32()
    def bev_encoder(self, x):
        """
        Args:
            x: (B, C, Dy, Dx)
        Returns:
            x: (B, C', 2*Dy, 2*Dx)
        """
        #列表，包含3个BEV特征，特征尺寸为[B,2C,Dy/2,Dx/2]、[B,4C,Dy/4,Dx/4]和[B,8C,Dy/8,Dx/8]
        x = self.img_bev_encoder_backbone(x)
        #尺寸为[B,C',Dy,Dx]
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        assert len(inputs) == 7 or len(inputs) == 6
        B, N, C, H, W = inputs[0].shape

        # imgs，环视图像，尺寸为[B,N,C,H,W]，sensor2egos，相机到自车坐标系的变换矩阵，尺寸为[B,N,4,4]，ego2globals，自车到全局坐标系的变换矩阵，尺寸为[B,N,4,4]
        # intrins，内参矩阵，尺寸为[B,N,3,3]，post_rots，图像增广旋转，尺寸为[B,N,3,3]，post_trans，图像增广平移，尺寸为[B,N,3]，bda，BEV增广矩阵，尺寸为[B,3,3]
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans = inputs[:6]
        if len(inputs) == 7:
            bda = inputs[6]
        elif len(inputs) == 6:
            bda = torch.eye(3).to(imgs.device)

        #如果变量ego2globals值不为None，则利用ego2globals计算所有相机到关键自车坐标系的变换矩阵，从而使得相机到自车坐标系的变换矩阵更加精确
        if inputs[2] is not None:
            #尺寸为[B,N,4,4]
            sensor2egos = sensor2egos.view(B, N, 4, 4)
            #尺寸为[B,N,4,4]
            ego2globals = ego2globals.view(B, N, 4, 4)

            # calculate the transformation from adj sensor to key ego
            #尺寸为[B,1,4,4]
            keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)    # (B, 1, 4, 4)
            #全局坐标系到自车的变换矩阵，尺寸为[B,1,4,4]
            global2keyego = torch.inverse(keyego2global.double())   # (B, 1, 4, 4)
            # @表示矩阵乘法
            # R_s2k = R_g2k @ R_e2g @ R_s2e
            # 相机到关键自车坐标系的变换矩阵，尺寸为[B,N,4,4]
            sensor2keyegos = \
                global2keyego @ ego2globals.double() @ sensor2egos.double()     # (B, N_views, 4, 4)
            # breakpoint()
            #尺寸为[B,N,4,4]
            sensor2keyegos = sensor2keyegos.float()
        else:
            sensor2keyegos = sensor2egos

        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]

    def extract_img_feat(self, img_inputs, img_metas, **kwargs):

        """ Extract features of images.
        img_inputs:
            imgs:  (B, N_views, 3, H, W)
            sensor2egos: (B, N_views, 4, 4)
            ego2globals: (B, N_views, 4, 4)
            intrins:     (B, N_views, 3, 3)
            post_rots:   (B, N_views, 3, 3)
            post_trans:  (B, N_views, 3)
            bda_rot:  (B, 3, 3)
        Returns:
            x: [(B, C', H', W'), ]
            depth: (B*N, D, fH, fW)
        """
        # img_inputs，处理后的输入，列表，包含7个元素，imgs，环视图像，尺寸为[B,N,C,H,W]，sensor2keyegos，相机到关键自车坐标系的变换矩阵，尺寸为[B,N,4,4]，ego2globals，自车到全局坐标系的变换矩阵，尺寸为[B,N,4,4]
        # intrins，内参矩阵，尺寸为[B,N,3,3]，post_rots，图像增广旋转，尺寸为[B,N,3,3]，post_trans，图像增广平移，尺寸为[B,N,3]，bda，BEV增广矩阵，尺寸为[3,3]
        img_inputs = self.prepare_inputs(img_inputs)
        #尺寸为[B,N,256,h,w]
        x, _ = self.image_encoder(img_inputs[0])    # x: (B, N, C, fH, fW)
        #x，鸟瞰图特征，尺寸为[B,C,Dy,Dx]，depth，特征图每个位置深度的概率，尺寸为[BN,D,fH,fW]
        x, depth = self.img_view_transformer([x] + img_inputs[1:7])
        # x: (B, C, Dy, Dx)
        # depth: (B*N, D, fH, fW)
        #尺寸为[B,C',Dy,Dx]
        x = self.bev_encoder(x)
        return [x], depth

    def extract_feat(self, points, img_inputs, img_metas, **kwargs):
        """Extract features from images and points."""
        """
        points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
        img_inputs:
                imgs:  (B, N_views, 3, H, W)        
                sensor2egos: (B, N_views, 4, 4)
                ego2globals: (B, N_views, 4, 4)
                intrins:     (B, N_views, 3, 3)
                post_rots:   (B, N_views, 3, 3)
                post_trans:  (B, N_views, 3)
                bda_rot:  (B, 3, 3)
        """
        #img_feats，列表，包含1个表示BEV特征的张量，尺寸为[B,C',Dy,Dx]，depth，特征图每个位置深度的概率，尺寸为[BN,D,fH,fW]
        img_feats, depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        pts_feats = None
        return img_feats, pts_feats, depth

    def forward_train(self,
                      points=None,
                      img_inputs=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      img_metas=None,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_inputs:
                imgs:  (B, N_views, 3, H, W)        # N_views = 6 * (N_history + 1)
                sensor2egos: (B, N_views, 4, 4)
                ego2globals: (B, N_views, 4, 4)
                intrins:     (B, N_views, 3, 3)
                post_rots:   (B, N_views, 3, 3)
                post_trans:  (B, N_views, 3)
                bda_rot:  (B, 3, 3)
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, _ = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def forward_test(self,
                     points=None,
                     img_inputs=None,
                     img_metas=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        assert False

    def simple_test(self,
                    points,
                    img_metas,
                    img_inputs=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton.
        Returns:
            bbox_list: List[dict0, dict1, ...]   len = bs
            dict: {
                'pts_bbox':  dict: {
                              'boxes_3d': (N, 9)
                              'scores_3d': (N, )
                              'labels_3d': (N, )
                             }
            }
        """
        img_feats, _, _ = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        # bbox_pts: List[dict0, dict1, ...],  len = batch_size
        # dict: {
        #   'boxes_3d': (N, 9)
        #   'scores_3d': (N, )
        #   'labels_3d': (N, )
        # }
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        img_feats, _, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(img_feats)
        return outs