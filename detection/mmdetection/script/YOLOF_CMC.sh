
python3 tools/train.py configs/mitotic/yolof_r50_mitotic_CMC.py --work-dir checkpoints/CMC_YOLOF --seed 42
python3 tools/test.py configs/mitotic/yolof_r50_mitotic_CMC.py checkpoints/CMC_YOLOF/epoch_16.pth --out test_pred/CMC_YOLOF.pkl


python3 tools/data_preparation/format_output.py \
        -d CMC -i test_pred/CMC_YOLOF.pkl \
        -o test_pred/CMC_YOLOF_base.pkl 


python3 tools/data_preparation/format_output.py \
        -d CMC -i test_pred/CMC_YOLOF.pkl \
        -c configs/mitotic/yolof_r50_mitotic_CMC_refocus.py \
        -m checkpoints/CMC_YOLOF/epoch_16.pth -o test_pred/CMC_YOLOF_relocated.pkl -r

