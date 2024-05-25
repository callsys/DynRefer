 | Data                        | Description                                                               | Download                                                                |
  | -------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------- |
  | Visual Genome  | Visual Genome dataset, put images into `data/vg/images`  | [Official](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html)   
| MSCOCO 2014 | MSCOCO 2014 dataset, put images into `data/refcoco/images`       | [Official](https://cocodataset.org/#home) |
| Converted annotations | Annotations for VG, RefCOCOg, OVAD, COCO, put `data/*` under `data/` | [OneDrive](https://mailsucasaccn-my.sharepoint.com/:f:/g/personal/zhaoyuzhong20_mails_ucas_ac_cn/EkLua8BRCwRKq_DE8r9SGYABZWrTS1Rr8VXJNMX5FMHa6Q?e=FX4Tgn) |
| CLIP weights | Weights of EVA CLIP text encoder | [Official](https://huggingface.co/BAAI/EVA/blob/main/eva_clip_psz14.pt) |


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
           ...
      |--configs/
      |--dynrefer/
      |--docs/
      |--scripts/
      |--train.py
      |--eval.py
```
The annotations are converted using `data.sh`.
