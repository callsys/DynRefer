
export PYTHONPATH=$PYTHONPATH:./

python dynrefer/common/data/vg_processor.py --version vg1.2 --image-root data/vg/images/ --ann-root data/vg/annotations/vg1.2 --save-dir data/vg/dynrefer/vg1.2
python dynrefer/common/data/vg_processor.py --version vg1.0 --image-root data/vg/images/ --ann-root data/vg/annotations/vg1.0 --save-dir data/vg/dynrefer/vg1.0
python dynrefer/common/data/vg_processor.py --version vgcoco --image-root data/vg/images/ --ann-root data/vg/annotations/vg1.2 --save-dir data/vg/dynrefer/vgcoco
python dynrefer/common/data/vg_processor.py --version vg_reg --image-root data/vg/images/ --ann-root data/vg/annotations/glamm/test_caption.json --save-dir data/vg/dynrefer/vg_reg

python dynrefer/common/data/refcoco_processor.py --version refcoco --image-root data/refcoco/images/ --ann-root data/refcoco/annotations/mdetr_annotations --save-dir data/refcoco/dynrefer
