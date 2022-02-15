# python tools/train.py  configs/mitotic/mitotic_US.py --work-dir checkpoints/mitotic-US-iter3-bs4-randrepleace2
# python tools/test.py configs/mitotic/mitotic_US.py checkpoints/mitotic-US-iter3-bs4-randrepleace2/epoch_12.pth  --out mitotic-US-iter3-bs4-randrepleace2.pkl

# python tools/train.py  configs/mitotic/mitotic_trainval.py --work-dir checkpoints/mitotic-US-iter3-bs4-trainval
# python tools/test.py configs/mitotic/mitotic_trainval.py checkpoints/mitotic-US-iter3-bs4-trainval/epoch_12.pth  --out mitotic-US-iter3-bs4-trainval.pkl

# python tools/test.py configs/mitotic/mitotic_US.py checkpoints/mitotic-US-iter3-calibrated-bs2/epoch_12.pth  --out mitotic-US-iter3-calibrated-bs2-inference-train.pkl


# python tools/train.py  configs/mitotic/mitotic_US_fold1.py --work-dir checkpoints/mitotic_US_fold1; 
# python tools/test.py configs/mitotic/mitotic_US_fold1.py checkpoints/mitotic_US_fold1/epoch_12.pth  --out mitotic_US_fold1.pkl

# python tools/train.py  configs/mitotic/mitotic_US_fold3.py --work-dir checkpoints/mitotic_US_fold3; python tools/test.py configs/mitotic/mitotic_US_fold3.py checkpoints/mitotic_US_fold3/epoch_12.pth  --out mitotic_US_fold3.pkl
# python tools/train.py  configs/mitotic/mitotic_CMC_base_retina.py --work-dir checkpoints/retina18; 
# python tools/train.py  configs/mitotic/mitotic_CMC_base_101.py --work-dir checkpoints/mitotic_CMC_final_r101_0; 
# python tools/test.py configs/mitotic/mitotic_CMC_base_101.py checkpoints/mitotic_CMC_final_r101_0/epoch_8.pth  --out mitotic_CMC_final_r101_0_test.pkl;

# python tools/train.py  configs/mitotic/mitotic_CMC_base.py --work-dir checkpoints/test; 
# python tools/test.py configs/mitotic/mitotic_CMC_base.py checkpoints/mitotic_CMC_check_0/epoch_8.pth  --out mitotic_CMC_check_0.pkl;
python tools/test.py configs/mitotic/mitotic_CMC_base_retina.py checkpoints/mitotic_CMC_final_retina_0/epoch_8.pth  --out mitotic_CMC_final_retina_0_train.pkl;
python tools/test.py configs/mitotic/mitotic_CMC_base_retina.py checkpoints/mitotic_CMC_final_retina_1/epoch_8.pth  --out mitotic_CMC_final_retina_1_train.pkl;

# python tools/test.py configs/mitotic/mitotic_CMC_base2.py checkpoints/mitotic_CMC_final_0/epoch_8.pth  --out mitotic_CMC_final_0_test_tiling.pkl;
# python tools/test.py configs/mitotic/mitotic_CMC_base2.py checkpoints/mitotic_CMC_final_1/epoch_8.pth  --out mitotic_CMC_final_1_test_tiling.pkl;

# python tools/train.py  configs/mitotic/mitotic_CMC_base_101.py --work-dir checkpoints/mitotic_CMC_base_101_iter1; 
# python tools/test.py configs/mitotic/mitotic_CMC_base_101.py checkpoints/mitotic_CMC_base_101_iter1/epoch_12.pth  --out mitotic_CMC_base_101_iter1_test.pkl;

# python tools/train.py  configs/mitotic/mitotic_CMC_base_101.py --work-dir checkpoints/mitotic_CMC_base_101_iter2; 
# python tools/test.py configs/mitotic/mitotic_CMC_base_101.py checkpoints/mitotic_CMC_base_101_iter2/epoch_12.pth  --out mitotic_CMC_base_101_iter2_test.pkl;
# sleep 420m;
# python tools/train.py  configs/mitotic/mitotic_CCMCT_base.py --work-dir checkpoints/mitotic_CCMCT_base_iter2v3; 
# python tools/test.py configs/mitotic/mitotic_CCMCT_base.py checkpoints/mitotic_CCMCT_base_iter2v3/epoch_8.pth  --out mitotic_CCMCT_base_iter2v3_test.pkl;
# python tools/test.py configs/mitotic/mitotic_CCMCT_base_train.py checkpoints/mitotic_CCMCT_base_iter3v3/epoch_12.pth  --out mitotic_CCMCT_base_iter3v3_train.pkl;

# python tools/test.py configs/mitotic/mitotic_CCMCT_base_train.py checkpoints/mitotic_CCMCT_base_iter2/epoch_6.pth  --out mitotic_CCMCT_base_iter2_train.pkl;

# python tools/train.py  configs/mitotic/mitotic_CMC_base.py --work-dir checkpoints/mitotic_CMC_base_iter2; 
# python tools/test.py configs/mitotic/mitotic_CMC_base.py checkpoints/mitotic_CMC_base_iter2/epoch_12.pth  --out mitotic_CMC_base_iter2_test.pkl;

# python tools/test.py configs/mitotic/mitotic_CMC_base.py checkpoints/mitotic_CMC_base_iter2/epoch_12.pth  --out mitotic_CMC_base_iter2_train.pkl;

# python tools/test.py configs/mitotic/mitotic_CMC_base.py checkpoints/mitotic_CMC_base/epoch_12.pth  --out CMC.pkl

# python tools/train.py  configs/mitotic/mitotic_4_cls.py --work-dir checkpoints/mitotic_4_cls
# python tools/test.py configs/mitotic/mitotic_4_cls.py checkpoints/mitotic_4_cls/epoch_12.pth  --out mitotic_4_cls_train.pkl


# python tools/train.py  configs/CMV/CMV.py --work-dir checkpoints/CMV_pretrain-3000-2
# python tools/test.py configs/CMV/CMV.py checkpoints/CMV_pretrain-3000-2/epoch_12.pth  --out CMV_pretrain-3000-2.pkl

# python tools/train.py  configs/CMV/CMV.py --work-dir checkpoints/CMV_pretrain-3000-2x
# python tools/test.py configs/CMV/CMV.py checkpoints/CMV_pretrain-3000-2x/epoch_12.pth  --out CMV_pretrain-3000-2x.pkl

# python tools/train.py  configs/CMV/CMV.py --work-dir checkpoints/CMV_base-2

# python tools/test.py configs/CMV/CMV.py checkpoints/CMV_base-1/epoch_12.pth  --out CMV_base-1.pkl
# python tools/test.py configs/CMV/CMV.py checkpoints/CMV_base-2/epoch_12.pth  --out CMV_base-2.pkl

# python tools/train.py  configs/mitotic/mitotic_US.py --work-dir checkpoints/e6
# python tools/test.py  configs/mitotic/mitotic_US.py  checkpoints/e6/epoch_11.pth --out e6.pkl
# python tools/train.py  configs/mitotic/mitotic_US.py --work-dir checkpoints/e6

# python tools/test.py configs/mitotic/mitotic_US.py checkpoints/mitotic-US-iter3-calibrated-bs2/epoch_12.pth  --out train_embedding.pkl
# python tools/test.py configs/mitotic/mitotic_US.py checkpoints/mitotic-US-iter3-calibrated-bs2/epoch_12.pth  --out test_embedding.pkl
