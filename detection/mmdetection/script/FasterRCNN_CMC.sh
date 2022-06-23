python3 tools/train.py configs/mitotic/mitotic_CMC_base.py --work-dir checkpoints/CMC_faster --seed 42
python tools/test.py configs/mitotic/mitotic_CMC_base.py checkpoints/CMC_faster/epoch_8.pth --out test_pred/CMC_faster.pkl

python3 tools/data_preparation/format_output.py \
        -d CMC -i test_pred/CMC_faster.pkl \
        -o test_pred/CMC_faster_base.pkl 

python3 tools/data_preparation/format_output.py \
        -d CMC -i test_pred/CMC_faster.pkl \
        -c configs/mitotic/mitotic_CMC_refocus.py \
        -m checkpoints/CMC_faster/epoch_8.pth -o test_pred/CMC_faster_relocated.pkl -r


