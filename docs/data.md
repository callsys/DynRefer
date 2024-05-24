 | Datasets                        | Description                                                               | Download                                                                |
  | -------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------- |
  | Visual Genome  | Visual Genome dataset, put images into `data/vg/images`  | [Official](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html)   
| MSCOCO 2014  | MSCOCO 2014 dataset, put images into `data/refcoco/images`       | [Official](https://cocodataset.org/#home) |



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
