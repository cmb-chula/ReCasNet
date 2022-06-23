
python3 tools/train.py configs/mitotic/yolof_r50_mitotic_CCMCT.py --work-dir checkpoints/CCMCT_YOLOF --seed 42
python3 tools/test.py configs/mitotic/yolof_r50_mitotic_CCMCT.py checkpoints/CCMCT_YOLOF/epoch_16.pth --out test_pred/CCMCT_YOLOF.pkl


python3 tools/data_preparation/format_output.py \
        -d CCMCT -i test_pred/CCMCT_YOLOF.pkl \
        -o test_pred/CCMCT_YOLOF_base.pkl 


python3 tools/data_preparation/format_output.py \
        -d CCMCT -i test_pred/CCMCT_YOLOF.pkl \
        -c configs/mitotic/yolof_r50_mitotic_CCMCT_refocus.py \
        -m checkpoints/CCMCT_YOLOF/epoch_16.pth -o test_pred/CCMCT_YOLOF_relocated.pkl -r
