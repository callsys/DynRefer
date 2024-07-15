import copy

import json
import torch
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2
from PIL import Image

import lavis.tasks as tasks
from lavis.common.registry import registry
from dynrefer.common.config import Config


def show_mask(mask, image, random_color=True, img_trans=0.9, mask_trans=0.5, return_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3) * 255], axis=0)
    else:
        color = np.array([30, 144, 255])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    image = cv2.addWeighted(image, img_trans, mask_image.astype('uint8'), mask_trans, 0)
    if return_color:
        return image, mask_image
    else:
        return image

class DynRefer():
    def __init__(self, args, device='cuda'):
        cfg = Config(args)
        task = tasks.setup_task(cfg)
        model = task.build_model(cfg)

        builder = registry.get_builder_class("dynrefer")(cfg.datasets_cfg.refcocog)
        builder.build_processors()

        self.att_thr = 0.3
        self.device = device
        self.model = model
        self.vis_processor = builder.vis_processors['eval']
        self.text_processor = builder.text_processors['eval']
        self.vis_processor.num_views = cfg.datasets_cfg.refcocog.build_info.num_views
        self.vis_processor.split = "test"
        self.model.eval()
        self.model.to(device)

        self.load_meta_files()

    def load_meta_files(self):
        ovad_file = "dynrefer/models/model_configs/ovad2000.json"
        ovad = json.load(open(ovad_file, "r"))
        self.att_list = [el['name'] for el in ovad['attributes']]

        ov_file = "dynrefer/models/model_configs/coco.json"
        ov_list = json.load(open(ov_file, "r"))
        self.ov_list = ov_list

    def seg2seq(self, seg):
        # print(seg.shape)
        x1, x2 = np.nonzero(seg.sum(0) != 0)[0][0], np.nonzero(seg.sum(0) != 0)[0][-1]
        y1, y2 = np.nonzero(seg.sum(1) != 0)[0][0], np.nonzero(seg.sum(1) != 0)[0][-1]
        bbox = [x1, y1, x2, y2]
        print(bbox)
        seq = [x1, y1, x2, y1, x2, y2, x1, y2]
        return [seq]

    def process_controls(self, controls):
        controls = controls.split(",|")
        controls = [control.strip() for control in controls]
        return controls

    def predict(self, image, seg, controls):
        print(controls)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if not isinstance(seg, list):
            seg = self.seg2seq(seg)
        samples = self.vis_processor(image, [seg])
        samples["cascade_region_images"] = samples["cascade_region_images"].to(self.device)
        samples["cascade_region_bboxes"] = samples["cascade_region_bboxes"].to(self.device)
        samples["ids"] = torch.zeros(1).to(torch.int64).to(self.device)
        samples["batch_idx"] = torch.zeros(1).to(torch.int64).to(self.device)
        samples["controls"] = [self.process_controls(controls)]
        with torch.inference_mode():
            output = self.model.predict_answers(samples)
        output = output[0]

        vis_output = dict()

        vis_output['score'] = output['score']
        vis_output['caption'] = output['caption']
        vis_output['attributes'] = [att for score, att in zip(output['attr'], self.att_list) if score > np.mean(output['attr'])*1.2]
        vis_output['class'] = self.ov_list[np.argmax(output['cls'])]
        return vis_output

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    parser.add_argument("--cfg-path",
                        default="demo/demo_dynrefer.yaml",
                        help="path to configuration file.")
    parser.add_argument("--local-rank", default=-1, type=int) # for debug
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

if __name__ == "__main__":
    image_path = "demo/examples/COCO_val2014_000000000074.jpg"
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    segs = np.zeros((h, w))
    bbox = [0.1, 0.6, 0.6, 0.9]
    bbox = [bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h]
    bbox = [int(el) for el in bbox]
    vis_image = np.array(image)
    vis_image = cv2.rectangle(vis_image, [bbox[0], bbox[1]], [bbox[2], bbox[3]], color=[0, 255, 0], thickness=1)
    plt.imshow(vis_image)
    plt.show()
    segs[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
    # segs = [[0, 0, w, 0, w, h, 0, h]]
    args = parse_args()
    dynrefer = DynRefer(args)
    output = dynrefer.predict(image, segs, "")
    print(output)