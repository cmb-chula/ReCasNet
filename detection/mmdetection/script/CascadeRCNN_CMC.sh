python3 tools/train.py configs/mitotic/cascade_rcnn_r50_fpn_mitotic_CMC.py --work-dir checkpoints/CMC_cascade --seed 0
python tools/test.py configs/mitotic/cascade_rcnn_r50_fpn_mitotic_CMC.py checkpoints/CMC_cascade/epoch_8.pth --out test_pred/CMC_cascade.pkl

python3 tools/data_preparation/format_output.py \
        -d CCMCT -i test_pred/CMC_cascade.pkl \
        -o test_pred/CMC_cascade.pkl 

python3 tools/data_preparation/format_output.py \
        -d CCMCT -i test_pred/CMC_cascade.pkl \
        -c configs/mitotic/cascade_rcnn_r50_fpn_mitotic_CMC_refocus.py \
        -m checkpoints/CMC_cascade/epoch_8.pth -o test_pred/CMC_cascade_relocated.pkl -r


