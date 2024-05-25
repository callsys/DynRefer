### 4.3 Training

Training DynRefer on VG V1.2.
```
bash scripts/train train_vg1.2.sh
```
Finetuning DynRefer on RefCOCOg (Pretrained by VG V1.2).
```
bash scripts/train finetune_refcocog.sh ckpts/vg1.2_5e.pth
```


### 4.4 Evaluation
Evaluating the dense captioning performance of DynRefer on VG V1.2.
```
bash scripts/eval/eval_vg1.2_densecap.sh ckpts/vg1.2_5e.pth
```
Evaluating the region-level captioning performance of DynRefer on VG : `(METEOR 21.2, CIDEr 190.5)`.
```
bash scripts/eval/eval_vg_reg.sh ckpts/vg1.2_5e.pth
```
Evaluating the region-level captioning performance of DynRefer on RefCOCOg.
```
bash scripts/eval/eval_refcocog_reg.sh ckpts/refcocog_ft.pth
```
Evaluating the attribute detection performance of DynRefer on OVAD.
```
bash scripts/eval/eval_ovad.sh ckpts/vg1.2_5e.pth
```
Evaluating the region recognition performance of DynRefer on COCO.
```
bash scripts/eval/eval_coco.sh ckpts/vg1.2_5e.pth
```


