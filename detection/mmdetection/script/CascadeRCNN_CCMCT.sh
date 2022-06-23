python3 tools/train.py configs/mitotic/cascade_rcnn_r50_fpn_mitotic_CCMCT.py --work-dir checkpoints/CCMCT_cascade --seed 0
python tools/test.py configs/mitotic/cascade_rcnn_r50_fpn_mitotic_CCMCT.py checkpoints/CCMCT_cascade/epoch_8.pth --out test_pred/CCMCT_cascade.pkl

python3 tools/data_preparation/format_output.py \
        -d CCMCT -i test_pred/CCMCT_cascade.pkl \
        -o test_pred/CCMCT_cascade_base.pkl 

python3 tools/data_preparation/format_output.py \
        -d CCMCT -i test_pred/CCMCT_cascade.pkl \
        -c configs/mitotic/cascade_rcnn_r50_fpn_mitotic_CCMCT_refocus.py \
        -m checkpoints/CCMCT_cascade/epoch_8.pth -o test_pred/CCMCT_cascade_relocated.pkl -r


