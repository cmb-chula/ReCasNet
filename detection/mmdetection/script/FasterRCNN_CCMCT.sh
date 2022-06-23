python3 tools/train.py configs/mitotic/mitotic_CCMCT_base.py --work-dir checkpoints/CCMCT_faster --seed 42
python tools/test.py configs/mitotic/mitotic_CCMCT_base.py checkpoints/CCMCT_faster/epoch_8.pth --out test_pred/CCMCT_faster.pkl

python3 tools/data_preparation/format_output.py \
        -d CCMCT -i test_pred/CCMCT_faster.pkl \
        -o test_pred/CCMCT_faster_base.pkl 

python3 tools/data_preparation/format_output.py \
        -d CCMCT -i test_pred/CCMCT_faster.pkl \
        -c configs/mitotic/mitotic_CCMCT_refocus.py \
        -m checkpoints/CCMCT_faster/epoch_8.pth -o test_pred/CCMCT_faster_relocated.pkl -r


