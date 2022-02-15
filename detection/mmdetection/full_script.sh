# python tools/train.py  configs/mitotic/mitotic_CMC_base.py --work-dir checkpoints/test2
# python3 tools/data_preparation/inference_detection_stage.py --relocate -d CMC -m checkpoints/test2/epoch_8.pth -o test2r.pkl 
# python3 tools/data_preparation/inference_detection_stage.py  -d CMC -m checkpoints/test2/epoch_8.pth -o test2.pkl 

# python tools/train.py  configs/mitotic/mitotic_CCMCT_base.py --work-dir checkpoints/test5ccmct
# python tools/test.py configs/mitotic/mitotic_CCMCT_base.py checkpoints/test5ccmct/epoch_8.pth --out test_pred/test5ccmct.pkl
# python tools/test.py configs/mitotic/mitotic_CCMCT_base_train.py checkpoints/test5ccmct/epoch_8.pth --out test_pred/train5ccmct.pkl

python tools/test.py configs/mitotic/mitotic_CCMCT_base.py checkpoints/mitotic_CCMCT_base_iter3v2/epoch_8.pth --out test_pred/test7ccmct.pkl
python tools/test.py configs/mitotic/mitotic_CCMCT_base_train.py checkpoints/mitotic_CCMCT_base_iter3v2/epoch_8.pth --out test_pred/train7ccmct.pkl

# python3 tools/data_preparation/inference_detection_stage.py --relocate -d CMC -m checkpoints/test2/epoch_8.pth -o test2r.pkl 
# python3 tools/data_preparation/inference_detection_stage.py  -d CMC -m checkpoints/test2/epoch_8.pth -o test2.pkl 
# python3 tools/data_preparation/inference_detection_stage.py  -d CCMCT -m checkpoints/test5ccmct/epoch_8.pth -o test_pred/test5ccmct.pkl 
# python3 tools/data_preparation/inference_detection_stage.py -d CCMCT -s train -m checkpoints/test2ccmct/epoch_8.pth -o test_pred/train1ccmct.pkl