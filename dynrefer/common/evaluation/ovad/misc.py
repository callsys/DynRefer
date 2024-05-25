import os
import logging
import datetime
import numpy as np
from .ovad_evaluator import AttEvaluator, print_metric_table

# get text representations
object_templates = {
    "none": ["{noun}"],
    "a": ["a {noun}", "a {noun}"],
    "the": ["the {noun}", "the {noun}"],
    "photo": [
        "a photo of a {noun}",
        "a photo of the {noun}",
    ],
}

object_attribute_templates = {
    "has": {
        "none": ["{attr} {dobj} {noun}"],
        "a": ["a {attr} {dobj} {noun}", "a {noun} has {attr} {dobj}"],
        "the": ["the {attr} {dobj} {noun}", "the {noun} has {attr} {dobj}"],
        "photo": [
            "a photo of a {attr} {dobj} {noun}",
            "a photo of an {noun} which has {attr} {dobj}",
            "a photo of the {attr} {dobj} {noun}",
            "a photo of the {noun} which has {attr} {dobj}",
        ],
    },
    "is": {
        "none": ["{attr} {noun}"],
        "a": ["a {attr} {noun}", "a {noun} is {attr}"],
        "the": ["the {attr} {noun}", "the {noun} is {attr}"],
        "photo": [
            "a photo of a {attr} {noun}",
            "a photo of a {noun} which is {attr}",
            "a photo of the {attr} {noun}",
            "a photo of the {noun} which is {attr}",
        ],
    },
}


def check_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def get_logger(logdir, name, evaluate=False):
    # Set logger for saving process experimental information
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    logger.ts = ts
    if evaluate:
        file_path = os.path.join(logdir, "evaluate_{}.log".format(ts))
    else:
        file_path = os.path.join(logdir, "run_{}.log".format(ts))
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    # strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr = logging.StreamHandler()
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)

    return logger


def save_in_log(
    log, save_step, set="", scalar_dict=None, text_dict=None, image_dict=None
):
    if scalar_dict:
        [log.add_scalar(set + "_" + k, v, save_step) for k, v in scalar_dict.items()]
    if text_dict:
        [log.add_text(set + "_" + k, v, save_step) for k, v in text_dict.items()]
    if image_dict:
        for k, v in image_dict.items():
            if k == "sample":
                log.add_images(set + "_" + k, v, save_step)
            elif k == "vec":
                log.add_images(set + "_" + k, v.unsqueeze(1).unsqueeze(1), save_step)
            elif k == "gt":
                log.add_images(
                    set + "_" + k,
                    v.unsqueeze(1).expand(-1, 3, -1, -1).float() / v.max(),
                    save_step,
                )
            elif k == "pred":
                log.add_images(set + "_" + k, v.argmax(dim=1, keepdim=True), save_step)
            elif k == "att":
                assert isinstance(v, list)
                for idx, alpha in enumerate(v):
                    log.add_images(
                        set + "_" + k + "_" + str(idx),
                        (alpha.unsqueeze(1) - alpha.min()) / alpha.max(),
                        save_step,
                    )
            else:
                log.add_images(set + "_" + k, v, save_step)
    log.flush()


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def ovad_validate(att_dict, pred, gt, output_dir, dataset_name="ovad_eval"):

    # Make the dictionaries for evaluator
    attr2idx = {}
    attr_type = {}
    attr_parent_type = {}
    attribute_head_tail = {"head": set(), "medium": set(), "tail": set()}
    attr_multi_synon = {"single": [], "multiple": []}

    for att in att_dict:
        attr2idx[att["name"]] = att["id"]

        if att["type"] not in attr_type.keys():
            attr_type[att["type"]] = set()
        attr_type[att["type"]].add(att["name"])

        if att["parent_type"] not in attr_parent_type.keys():
            attr_parent_type[att["parent_type"]] = set()
        attr_parent_type[att["parent_type"]].add(att["type"])

        attribute_head_tail[att["freq_set"]].add(att["name"])

        if len(att["name"].split(":")[-1].split("/")) > 1:
            attr_multi_synon["multiple"].append(att["name"])
        else:
            attr_multi_synon["single"].append(att["name"])

    attr_type = {key: list(val) for key, val in attr_type.items()}
    attr_parent_type = {key: list(val) for key, val in attr_parent_type.items()}
    attribute_head_tail = {key: list(val) for key, val in attribute_head_tail.items()}

    evaluator = AttEvaluator(
        attr2idx,
        attr_type=attr_type,
        attr_parent_type=attr_parent_type,
        attr_headtail=attribute_head_tail,
        att_seen_unseen=attr_multi_synon,
        dataset_name=dataset_name,
        threshold=np.ceil(pred.mean() * 10) / 10,
        top_k=int((gt == 1).sum(1).mean()),
        exclude_atts=[],
    )

    output_raw = pred
    output_gt = gt
    output_gt[output_gt == -1] = 2
    output_file = os.path.join(output_dir, "ovad_{}.log".format(dataset_name))

    results = evaluator.print_evaluation(
        output_raw.copy(), output_gt.copy(), output_file
    )
    table = print_metric_table(evaluator, results, output_file)

    for key, val in table.items():
        if "ap/all" in key:
            print(f"ALL mAP {round(val, 2)}")
        elif "ap/head" in key:
            print(f"- HEAD: {round(val, 2)}")
        elif "ap/medium" in key:
            print(f"- MEDIUM: {round(val, 2)}")
        elif "ap/tail" in key:
            print(f"- TAIL: {round(val, 2)}")
    return table
