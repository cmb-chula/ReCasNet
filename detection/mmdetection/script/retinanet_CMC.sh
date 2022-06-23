python3 tools/train.py configs/mitotic/mitotic_CMC_retina.py --work-dir checkpoints/CMC_retina --seed 42
python tools/test.py configs/mitotic/mitotic_CMC_retina.py checkpoints/CMC_retina/epoch_8.pth --out test_pred/CMC_retina.pkl

python3 tools/data_preparation/format_output.py \
        -d CMC -i test_pred/CMC_retina.pkl \
        -o test_pred/CMC_retina_base.pkl 

python3 tools/data_preparation/format_output.py \
        -d CMC -i test_pred/CMC_retina.pkl \
        -c configs/mitotic/mitotic_CMC_retina_refocus.py \
        -m checkpoints/CMC_retina/epoch_8.pth -o test_pred/CMC_retina_relocated.pkl -r


