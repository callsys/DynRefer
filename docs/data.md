 | Data                        | Description                                                               | Download                                                                |
  | -------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------- |
  | Visual Genome  | Put images under `DynRefer/data/vg/images`  | [Official](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html)   
| MSCOCO 2014 | Put images under `DynRefer/data/refcoco/images`       | [Official](https://cocodataset.org/#home) |
| MSCOCO 2017 | Put images under `DynRefer/data/coco2017/images/images`       | [Official](https://cocodataset.org/#home) |
| CLIP weights | Put `ckpts/eva_clip_psz14.pt` under `DynRefer/ckpts/` | [Official](https://huggingface.co/BAAI/EVA/blob/main/eva_clip_psz14.pt) |
| Converted annotations | Put `data/*` under `DynRefer/data/`, and unzip the annotations | [OneDrive](https://mailsucasaccn-my.sharepoint.com/:f:/g/personal/zhaoyuzhong20_mails_ucas_ac_cn/EkLua8BRCwRKq_DE8r9SGYABZWrTS1Rr8VXJNMX5FMHa6Q?e=FX4Tgn) |
| Pre-trained DynRefer weights and logs (Optional) | Put `ckpts/*` under `DynRefer/ckpt/` | [OneDrive](https://mailsucasaccn-my.sharepoint.com/:f:/g/personal/zhaoyuzhong20_mails_ucas_ac_cn/EkLua8BRCwRKq_DE8r9SGYABZWrTS1Rr8VXJNMX5FMHa6Q?e=FX4Tgn) |


To train and evaluate DynRefer, download the files in the table and arrange the files according to the file tree below.

```text
    |--DynRefer/
      |--data/
        |--vg/
           |--dynrefer/
           |--images/
              |--1000.jpg
              |--1001.jpg
              ...
        |--refcoco
           |--dynrefer/
           |--images/
              |--COCO_train2014_000000000009.jpg
              |--COCO_train2014_000000000025.jpg
              ...
        |--coco2017
           |--dynrefer/
           |--images/
              |--images/
                 |--000000116207.jpg
                 |--000000116208.jpg
                 ...
              |--train2017/
              |--val2017/
              ...
           ...
      |--ckpts/
         |--eva_clip_psz14.pt
         |--vg1.2_5e.pth
         |--refcocog_ft.pth
      |--configs/
      |--dynrefer/
      |--docs/
      |--scripts/
      |--train.py
      |--eval.py
```
