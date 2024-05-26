 | Data                        | Description                                                               | Download                                                                |
  | -------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------- |
  | Visual Genome  | `ln -s VG/images data/vg/images`  | [Official](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html)   
| MSCOCO 2014 | `ln -s coco2014/train2014 data/refcoco/images`       | [Official](https://cocodataset.org/#home) |
| MSCOCO 2017 | `ln -s coco2017/val2017 data/coco2017/images`       | [Official](https://cocodataset.org/#home) |
| EVA CLIP text encoder | `mv <your_path>/ckpts/eva_clip_psz14.pt ckpts/` | [Official](https://huggingface.co/BAAI/EVA/blob/main/eva_clip_psz14.pt) |
| Converted annotations | `unzip data.zip` | [OneDrive](https://mailsucasaccn-my.sharepoint.com/:f:/g/personal/zhaoyuzhong20_mails_ucas_ac_cn/EkLua8BRCwRKq_DE8r9SGYABZWrTS1Rr8VXJNMX5FMHa6Q?e=FX4Tgn) |
| Meteor package | `unzip meteor.zip; mv meteor dynrefer/common/evaluation/` | [OneDrive](https://mailsucasaccn-my.sharepoint.com/:f:/g/personal/zhaoyuzhong20_mails_ucas_ac_cn/EkLua8BRCwRKq_DE8r9SGYABZWrTS1Rr8VXJNMX5FMHa6Q?e=FX4Tgn) |
| Pre-trained DynRefer weights and logs (Optional) | `mv <your_path>/ckpts/* ckpts/` | [OneDrive](https://mailsucasaccn-my.sharepoint.com/:f:/g/personal/zhaoyuzhong20_mails_ucas_ac_cn/EkLua8BRCwRKq_DE8r9SGYABZWrTS1Rr8VXJNMX5FMHa6Q?e=FX4Tgn) |


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
              |--000000116207.jpg
              |--000000116208.jpg
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
