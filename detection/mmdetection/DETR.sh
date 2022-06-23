python tools/train.py  configs/mitotic/detr_r50_8x2_150e_mitotic.py --work-dir checkpoints/CMC_DETR_3
python tools/test.py configs/mitotic/detr_r50_8x2_150e_mitotic.py checkpoints/CMC_DETR_0/epoch_25.pth --out test_pred/CMC_DETR_3.pkl
