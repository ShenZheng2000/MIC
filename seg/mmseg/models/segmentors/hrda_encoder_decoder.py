# Obtained from: https://github.com/lhoyer/HRDA
# Modifications:
# - Add return_logits flag
# - Add upscale_pred flag
# - Update debug_output system
# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import numpy as np
import torch
import copy
import torch.nn.functional as F

from mmseg.ops import resize
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder

from ...transforms.fovea import build_grid_net, before_train_json, process_mmseg, read_seg_to_det

# def get_crop_bbox(img_h, img_w, crop_size, divisible=1):
#     """Randomly get a crop bounding box."""
#     assert crop_size[0] > 0 and crop_size[1] > 0
#     if img_h == crop_size[-2] and img_w == crop_size[-1]:
#         return (0, img_h, 0, img_w)
#     margin_h = max(img_h - crop_size[-2], 0)
#     margin_w = max(img_w - crop_size[-1], 0)
#     offset_h = np.random.randint(0, (margin_h + 1) // divisible) * divisible
#     offset_w = np.random.randint(0, (margin_w + 1) // divisible) * divisible
#     crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
#     crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

#     return crop_y1, crop_y2, crop_x1, crop_x2

def get_crop_bbox(img_h, img_w, crop_size, divisible=1):
    """Randomly get a crop bounding box."""
    assert crop_size[0] > 0 and crop_size[1] > 0
    if img_h == crop_size[-2] and img_w == crop_size[-1]:
        return (0, img_h, 0, img_w)

    margin_h = max(img_h - crop_size[-2], 0)
    margin_w = max(img_w - crop_size[-1], 0)

    # TODO: Ensure that the upper bound for randint is greater than the lower bound
    upper_bound_h = max((margin_h + 1) // divisible, 1)
    upper_bound_w = max((margin_w + 1) // divisible, 1)

    offset_h = np.random.randint(0, upper_bound_h) * divisible
    offset_w = np.random.randint(0, upper_bound_w) * divisible

    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

    return crop_y1, crop_y2, crop_x1, crop_x2


def crop(img, crop_bbox):
    """Crop from ``img``"""
    crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
    if img.dim() == 4:
        img = img[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
    elif img.dim() == 3:
        img = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
    elif img.dim() == 2:
        img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    else:
        raise NotImplementedError(img.dim())
    return img


@SEGMENTORS.register_module()
class HRDAEncoderDecoder(EncoderDecoder):
    last_train_crop_box = {}

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 scales=[1],
                 hr_crop_size=None,
                 hr_slide_inference=True,
                 hr_slide_overlapping=True,
                 crop_coord_divisible=1,
                 blur_hr_crop=False,
                 feature_scale=1,
                 # NOTE: hardcode all these configs for now
                 VANISHING_POINT=None, 
                 warp_aug=False,
                 warp_aug_lzu=False, 
                 warp_fovea=False, 
                 warp_fovea_inst=False,
                 warp_fovea_inst_scale=False,
                 warp_fovea_inst_scale_l2=False,
                 warp_fovea_mix=False, 
                 warp_middle=False,
                 warp_debug=False,
                 warp_fovea_center=False,
                 warp_scale=1.0,
                 warp_dataset=[],
                 SEG_TO_DET=None,
                 keep_grid=False,
                 is_seg=True,
                 bandwidth_scale=64,
                 amplitude_scale=1.0,                
                 ):
        self.feature_scale_all_strs = ['all']
        if isinstance(feature_scale, str):
            assert feature_scale in self.feature_scale_all_strs
        scales = sorted(scales)
        decode_head['scales'] = scales
        decode_head['enable_hr_crop'] = hr_crop_size is not None
        decode_head['hr_slide_inference'] = hr_slide_inference
        super(HRDAEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.scales = scales
        self.feature_scale = feature_scale
        self.crop_size = hr_crop_size
        self.hr_slide_inference = hr_slide_inference
        self.hr_slide_overlapping = hr_slide_overlapping
        self.crop_coord_divisible = crop_coord_divisible
        self.blur_hr_crop = blur_hr_crop


        # NOTE: add these stuffs used for warping
        self.warp_aug = warp_aug
        self.warp_aug_lzu = warp_aug_lzu
        self.warp_fovea = warp_fovea
        self.warp_fovea_inst = warp_fovea_inst
        self.warp_fovea_mix = warp_fovea_mix
        self.warp_middle = warp_middle
        self.warp_debug = warp_debug
        self.warp_scale = warp_scale
        self.warp_dataset = warp_dataset
        self.warp_fovea_center = warp_fovea_center
        self.is_seg = is_seg
        self.keep_grid = keep_grid

        self.seg_to_det = read_seg_to_det(SEG_TO_DET)

        self.vanishing_point = before_train_json(VP=VANISHING_POINT)
        self.grid_net = build_grid_net(warp_aug_lzu=warp_aug_lzu,
                                        warp_fovea=warp_fovea,
                                        warp_fovea_inst=warp_fovea_inst,
                                        warp_fovea_mix=warp_fovea_mix,
                                        warp_middle=warp_middle,
                                        warp_scale=warp_scale,
                                        warp_fovea_center=warp_fovea_center,
                                        warp_fovea_inst_scale=warp_fovea_inst_scale,
                                        warp_fovea_inst_scale_l2=warp_fovea_inst_scale_l2,
                                        is_seg=is_seg,
                                        bandwidth_scale=bandwidth_scale,
                                        amplitude_scale=amplitude_scale,)

    # def extract_unscaled_feat(self, img):
    #     x = self.backbone(img)
    #     if self.with_neck:
    #         x = self.neck(x)
    #     return x

    # TODO: need to debug if we have added warping in the correct place. (check shape and vis image)
    def extract_unscaled_feat(self, img, img_metas=None, is_training=True):
        """Extract features from images."""
        
        # Print statements for debugging
        # print("img_metas is", img_metas)
        
        if (self.warp_aug_lzu is True) and (img_metas is not None):
            
            # Check if any source in img_metas' filename matches the warp_dataset
            if any(src in img_metas[0]['filename'] for src in self.warp_dataset) and (is_training is True):
                # Debug print statements
                x, img, img_metas = process_mmseg(img_metas,
                                                        img,
                                                        self.warp_aug_lzu,
                                                        self.vanishing_point,
                                                        self.grid_net,
                                                        self.backbone,
                                                        self.warp_debug,
                                                        seg_to_det=self.seg_to_det,
                                                        keep_grid=self.keep_grid
                                                        )
                
                # Debug print statements
                # print("After, images.shape", img.shape)
            else:
                x = self.backbone(img)
        else:
            x = self.backbone(img)
        
        if self.with_neck:
            x = self.neck(x)
        
        return x

    def extract_slide_feat(self, img, img_metas, is_training):
        if self.hr_slide_overlapping:
            h_stride, w_stride = [e // 2 for e in self.crop_size]
        else:
            h_stride, w_stride = self.crop_size
        h_crop, w_crop = self.crop_size
        bs, _, h_img, w_img = img.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        crop_imgs, crop_feats, crop_boxes = [], [], []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_imgs.append(img[:, :, y1:y2, x1:x2])
                crop_boxes.append([y1, y2, x1, x2])
        crop_imgs = torch.cat(crop_imgs, dim=0)
        crop_feats = self.extract_unscaled_feat(crop_imgs, img_metas, is_training)
        # shape: feature levels, crops * batch size x c x h x w

        return {'features': crop_feats, 'boxes': crop_boxes}

    def blur_downup(self, img, s=0.5):
        img = resize(
            input=img,
            scale_factor=s,
            mode='bilinear',
            align_corners=self.align_corners)
        img = resize(
            input=img,
            scale_factor=1 / s,
            mode='bilinear',
            align_corners=self.align_corners)
        return img

    def resize(self, img, s):
        if s == 1:
            return img
        else:
            with torch.no_grad():
                return resize(
                    input=img,
                    scale_factor=s,
                    mode='bilinear',
                    align_corners=self.align_corners)

    # NOTE: include image warping inside
    def extract_feat(self, img, img_metas=None, is_training=True):
        if self.feature_scale in self.feature_scale_all_strs:
            mres_feats = []
            for i, s in enumerate(self.scales):
                if s == 1 and self.blur_hr_crop:
                    scaled_img = self.blur_downup(img)
                else:
                    scaled_img = self.resize(img, s)
                if self.crop_size is not None and i >= 1:
                    scaled_img = crop(
                        scaled_img, HRDAEncoderDecoder.last_train_crop_box[i])
                print("scaled_img.shape", scaled_img.shape)
                mres_feats.append(
                                self.extract_unscaled_feat(scaled_img, img_metas, is_training)
                                )
            return mres_feats
        else:
            scaled_img = self.resize(img, self.feature_scale)
            return self.extract_unscaled_feat(scaled_img, img_metas, is_training)

    def generate_pseudo_label(self, img, img_metas, is_training):
        self.update_debug_state()
        # print(f"generate_pseudo_label with is_training={is_training}!!!!!!!!!!")
        out = self.encode_decode(img, img_metas, is_training=is_training)
        if self.debug:
            self.debug_output = self.decode_head.debug_output
        return out

    # NOTE: add image warping inside
    def encode_decode(self, img, img_metas, upscale_pred=True,
                      is_training=True
                      ):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""

        # print(f"encoder_decoder with is_training={is_training}!!!!!!!!!!")

        mres_feats = []
        self.decode_head.debug_output = {}
        for i, s in enumerate(self.scales):
            if s == 1 and self.blur_hr_crop:
                scaled_img = self.blur_downup(img)
            else:
                scaled_img = self.resize(img, s)
            # print("is_training is", is_training)
            # print("hr_slide_inference is", self.hr_slide_inference)
            if i >= 1 and self.hr_slide_inference:
                mres_feats.append(
                    self.extract_slide_feat(scaled_img, img_metas, is_training)
                    )
            else:
                mres_feats.append(
                                self.extract_unscaled_feat(scaled_img, img_metas, is_training)
                                )
            if self.decode_head.debug:
                self.decode_head.debug_output[f'Img {i} Scale {s}'] = \
                    scaled_img.detach()
        out = self._decode_head_forward_test(mres_feats, img_metas)
        if upscale_pred:
            out = resize(
                input=out,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        return out

    def _forward_train_features(self, img, img_metas, is_training):
        mres_feats = []
        self.decode_head.debug_output = {}
        assert len(self.scales) <= 2, 'Only up to 2 scales are supported.'
        prob_vis = None
        for i, s in enumerate(self.scales):
            # print("i is", i)
            # print("img is", img.shape) # [2, 3, 512, 1024]
            # print("scale factor of", s) # 0.5, or 1.0
            if s == 1 and self.blur_hr_crop:
                scaled_img = self.blur_downup(img)
            else:
                scaled_img = resize(
                    input=img,
                    scale_factor=s,
                    mode='bilinear',
                    align_corners=self.align_corners)
            # print("scaled_img is", scaled_img.shape) # [2, 3, 512, 1024] or [2, 3, 256, 512]
            # TODO: remove crop_size for now
            if self.crop_size is not None and i >= 1:
                crop_box = get_crop_bbox(*scaled_img.shape[-2:],
                                         self.crop_size,
                                         self.crop_coord_divisible)

                if self.feature_scale in self.feature_scale_all_strs:
                    HRDAEncoderDecoder.last_train_crop_box[i] = crop_box
                self.decode_head.set_hr_crop_box(crop_box)
                scaled_img = crop(scaled_img, crop_box)
            if self.decode_head.debug:
                self.decode_head.debug_output[f'Img {i} Scale {s}'] = \
                    scaled_img.detach()
            mres_feats.append(
                            self.extract_unscaled_feat(scaled_img, img_metas, is_training)
                            )
        return mres_feats, prob_vis

    # NOTE: add image warping inside
    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      seg_weight=None,
                      return_feat=False,
                      return_logits=False,
                      is_training=True
                      ):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        self.update_debug_state()

        losses = dict()

        mres_feats, prob_vis = self._forward_train_features(img, img_metas, is_training)
        for i, s in enumerate(self.scales):
            if return_feat and self.feature_scale in \
                    self.feature_scale_all_strs:
                if 'features' not in losses:
                    losses['features'] = []
                losses['features'].append(mres_feats[i])
            if return_feat and s == self.feature_scale:
                losses['features'] = mres_feats[i]
                break

        loss_decode = self._decode_head_forward_train(mres_feats, img_metas,
                                                      gt_semantic_seg,
                                                      seg_weight,
                                                      return_logits)
        losses.update(loss_decode)

        if self.decode_head.debug and prob_vis is not None:
            self.decode_head.debug_output['Crop Prob.'] = prob_vis

        if self.with_auxiliary_head:
            raise NotImplementedError

        if self.debug:
            self.debug_output.update(self.decode_head.debug_output)
        self.local_iter += 1
        return losses

    # NOTE: seems no use, skip this function for now
    def forward_with_aux(self, img, img_metas):
        assert not self.with_auxiliary_head
        print("forward_with_aux is called!!!!!!!!!!!!!!")
        mres_feats, _ = self._forward_train_features(img, img_metas)
        out = self.decode_head.forward(mres_feats)
        # out = resize(
        #     input=out,
        #     size=img.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        return {'main': out}
    
    def inference(self, img, img_meta, rescale, is_training=False):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        # print("inference is called!!!!!!!!!!!!!!")
        # print("self.test_cfg.mode is", self.test_cfg.mode)
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale, 
                                             is_training=is_training) # NOTE: add is_training here
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale,
                                             is_training=is_training) # NOTE: add is_training here
        if hasattr(self.decode_head, 'debug_output_attention') and \
                self.decode_head.debug_output_attention:
            output = seg_logit
        else:
            output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True, is_training=False):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale, is_training=is_training) # NOTE: add is_training here
        if hasattr(self.decode_head, 'debug_output_attention') and \
                self.decode_head.debug_output_attention:
            seg_pred = seg_logit[:, 0]
        else:
            seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def slide_inference(self, img, img_meta, rescale, 
                        is_training=False
                        ):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batched_slide = self.test_cfg.get('batched_slide', False)
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        if batched_slide:
            crop_imgs, crops = [], []
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_img = img[:, :, y1:y2, x1:x2]
                    crop_imgs.append(crop_img)
                    crops.append((y1, y2, x1, x2))
            crop_imgs = torch.cat(crop_imgs, dim=0)
            crop_seg_logits = self.encode_decode(crop_imgs, img_meta, is_training=is_training) # NOTE: add is_training here
            for i in range(len(crops)):
                y1, y2, x1, x2 = crops[i]
                crop_seg_logit = \
                    crop_seg_logits[i * batch_size:(i + 1) * batch_size]
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        else:
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_img = img[:, :, y1:y2, x1:x2]
                    crop_seg_logit = self.encode_decode(crop_img, img_meta, is_training=is_training) # NOTE: add is_training here
                    preds += F.pad(crop_seg_logit,
                                   (int(x1), int(preds.shape[3] - x2), int(y1),
                                    int(preds.shape[2] - y2)))

                    count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds


    # NOTE: add inference code with image warping inside
    def whole_inference(self, img, img_meta, rescale, 
                        is_training=False
                        ):
        """Inference with full image."""
        # print("whole_inference is called!!!!!!!!!!!!!!")
        seg_logit = self.encode_decode(img, img_meta, 
                                       is_training=is_training # NOTE: add is_training here 
                                       )

        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit
