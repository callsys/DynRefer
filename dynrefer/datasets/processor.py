import copy

import torch
import numpy as np
import cv2
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
from omegaconf import OmegaConf
import torchvision.transforms.v2 as transforms
from torchvision.transforms.functional import InterpolationMode

from lavis.common.registry import registry
from lavis.processors.blip_processors import BlipImageBaseProcessor

import random
@registry.register_processor("dynrefer")
class DynReferProcessor(BlipImageBaseProcessor):
    def __init__(
            self, image_size=364, mean=None, std=None, min_scale=0.5, max_scale=1.0
    ):
        super().__init__(mean=mean, std=std)
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        region_transform = [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToImage(),
                # transforms.ToImageTensor(),
                transforms.ConvertImageDtype(),
                transforms.Normalize(mean, std),
            ]

        self.transform = transforms.Compose(
            region_transform
        )

        expand_ratio = 8
        global_transform = [
                transforms.Resize(
                    (image_size * expand_ratio, image_size * expand_ratio), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToImage(),
                transforms.ConvertImageDtype(),
                transforms.Normalize(mean, std),
            ]

        self.global_transform = transforms.Compose(
            global_transform
        )

        # if False, transform the arbitrary mask into bbox
        self.with_seg = True
        self.num_views = 2

    def seq2mask(self, image, segs):
        w, h = image.size
        bboxes = []
        masks = []
        for seg in segs:
            if isinstance(seg, list):
                seq = []
                for seg_ in seg:
                    seq.extend(seg_)
                x1, y1 = np.array(seq).reshape(-1, 2).min(0)
                x2, y2 = np.array(seq).reshape(-1, 2).max(0)
                if x1 >= x2-1 and x2 - 0 > 0:
                    x1 = x2 - 1
                elif x1 >= x2-1 and w - x1 > 0:
                    x2 = x1 + 1
                if y1 >= y2-1 and y2 - 0 > 0:
                    y1 = y2 -1
                elif y1 >= y2-1 and h - y1 > 0:
                    y2 = y1 + 1
                bbox = [x1, y1, x2, y2]
                mask = np.zeros((h, w), np.uint8)
                for seg_ in seg:
                    mask = cv2.fillPoly(mask, np.array(seg_).reshape(1, -1, 2).astype(np.int64), 1)
                bboxes.append(bbox)
                masks.append(mask)
            else:
                if isinstance(seg["counts"], list):
                    seg = mask_util.frPyObjects(seg, *seg["size"])
                elif not isinstance(seg["counts"], bytes):
                    seg["counts"] = seg["counts"].encode()
                mask = mask_util.decode(seg)
                x1, x2 = np.nonzero(mask.sum(0) != 0)[0][0], np.nonzero(mask.sum(0) != 0)[0][-1]
                y1, y2 = np.nonzero(mask.sum(1) != 0)[0][0], np.nonzero(mask.sum(1) != 0)[0][-1]
                bbox = [x1, y1, x2, y2]
                bboxes.append(bbox)
                masks.append(mask)

        if not self.with_seg:
            masks = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                seg = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                mask = np.zeros((h, w), np.uint8)
                for seg_ in seg:
                    mask = cv2.fillPoly(mask, np.array(seg_).reshape(1, -1, 2).astype(np.int64), 1)
                masks.append(mask)
        return np.array(bboxes), masks

    def cascade_region_process(self, image, cascade_bboxes):
        w, h = image.size
        cascade_bboxes = torch.from_numpy(cascade_bboxes)
        cascade_bboxes[..., 0] = cascade_bboxes[..., 0].clip(min=0, max=w)
        cascade_bboxes[..., 1] = cascade_bboxes[..., 1].clip(min=0, max=h)
        cascade_bboxes[..., 2] = cascade_bboxes[..., 2].clip(min=0, max=w)
        cascade_bboxes[..., 3] = cascade_bboxes[..., 3].clip(min=0, max=h)
        cascade_bboxes = cascade_bboxes.to(torch.int64)
        target_bboxes = cascade_bboxes[0]

        cascade_region_images = []
        cascade_region_bboxes = []

        for bboxes in cascade_bboxes:
            region_images = []
            region_bboxes = []
            for bbox, target_bbox in zip(bboxes, target_bboxes):
                x1, y1, x2, y2 = bbox
                region_image = Image.fromarray(np.array(image)[y1:y2, x1:x2])
                wr, hr = region_image.size
                region_bbox = copy.deepcopy(target_bbox)
                bias = torch.LongTensor([x1, y1, x1, y1])
                region_bbox = region_bbox - bias

                target = {"boxes": torchvision.tv_tensors.BoundingBoxes(region_bbox[None], format="XYXY", canvas_size=(hr, wr))}
                region_image, target = self.transform(region_image, target)
                region_bbox = target["boxes"].data.to(torch.float32)

                region_images.append(region_image)
                region_bboxes.append(region_bbox[0])
            cascade_region_images.append(torch.stack(region_images, 0))
            cascade_region_bboxes.append(torch.stack(region_bboxes, 0))

        return torch.stack(cascade_region_images, 0), torch.stack(cascade_region_bboxes, 0)
    
    def select_regions(self, image, org_bboxes):
        image_arr = np.array(image)
        image_arr = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)
        image_arr = image_arr.astype(np.float32)
        num_views = self.num_views

        # one basic view
        weights = [0]
        w, h = image.size
        oowh = np.array([0, 0, w, h]).astype(np.float32)
        basic_bboxes = []
        basic_phash_codes = []
        for i in range(len(weights)):
            bboxes, phash_codes = [], []
            for bbox in org_bboxes:
                bbox_arr = np.array(bbox).astype(np.float32)
                temp = bbox_arr + (oowh - bbox_arr) * weights[i]
                bboxes.append(temp)
                phash_codes.append(self.phash(image_arr, temp))
            basic_bboxes.append(bboxes)
            basic_phash_codes.append(phash_codes)
        basic_bboxes = np.array(basic_bboxes)
        basic_ratios = np.zeros(basic_bboxes.shape[:2])

        # candidate views
        if self.num_views > 1:
            num_candidate_views = 10
            weights = [((i+1)/(num_candidate_views)) for i in range(num_candidate_views)]
            selected_bboxes = []
            selected_ratios = []
            for i, bbox in enumerate(org_bboxes):
                bbox_arr = np.array(bbox).astype(np.float32)
                candidate_phash_codes = []
                candi_bboxes = []
                for w in weights:
                    candi_bbox = bbox_arr + (oowh - bbox_arr) * w
                    candi_bboxes.append(candi_bbox)
                    candidate_phash_codes.append(self.phash(image_arr, candi_bbox))
                candidate_phash_codes = np.stack(candidate_phash_codes) 

                if self.split == "train":
                    # random selection
                    possible_numbers = list(range(0, num_candidate_views))
                    unique_random_integers = np.array(random.sample(possible_numbers, self.num_views-1))
                    selected_bbox = np.array(candi_bboxes)[unique_random_integers]
                    selected_ratio = np.array(weights)[unique_random_integers]
                else:
                    # greedy search
                    difference = np.zeros(num_candidate_views)
                    difference += (((basic_phash_codes[0][i][None] ^ candidate_phash_codes).sum(-1))) / ((np.array(weights)))
                    selected_ind = []
                    selected_bbox = []
                    selected_ratio = []
                    for _ in range(self.num_views-1):
                        ind = difference.argmax()
                        selected_ind.append(ind)
                        selected_ratio.append(weights[ind])
                        selected_bbox.append(candi_bboxes[ind])
                        difference[np.array(selected_ind).astype(np.int64)] = 0
                    selected_bbox = np.array(selected_bbox)

                selected_bboxes.append(selected_bbox)
                selected_ratios.append(selected_ratio)
            cascade_bboxes = np.concatenate([basic_bboxes[0:1], np.array(selected_bboxes).transpose(1, 0, 2)], 0)
            cascade_ratios = np.concatenate([basic_ratios[0:1], np.array(selected_ratios).transpose(1, 0)], 0)
        else:
            cascade_bboxes = basic_bboxes
            cascade_ratios = basic_ratios
        return cascade_bboxes, cascade_ratios
    
    def phash(self, image_arr, bbox, res=224, epsilon=1e-8):
        x1, y1, x2, y2 = bbox.astype(np.int64)
        region_arr = cv2.resize(image_arr[y1:y2, x1:x2], (res, res))
        region_arr = cv2.dct(region_arr)
        region_arr=region_arr[0:8,0:8]
        bar = region_arr.mean()
        hash_value = np.array([1 if pixel > bar else 0 for pixel in region_arr.ravel()], dtype=np.int8)
        return hash_value
    
    def __call__(self, image, segs, cascade_bboxes=None):
        bboxes, _ = self.seq2mask(image, segs)
        cascade_bboxes, cascade_ratios = self.select_regions(image, bboxes)
        cascade_region_images, cascade_region_bboxes  = self.cascade_region_process(image, cascade_bboxes)

        output = {"cascade_region_images": cascade_region_images.permute(1, 0, 2, 3, 4),
                  "cascade_region_bboxes": cascade_region_bboxes.permute(1, 0, 2),
                  "cascade_region_ratios": torch.from_numpy(cascade_ratios).permute(1, 0),
                  }

        return output

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 364)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )


   