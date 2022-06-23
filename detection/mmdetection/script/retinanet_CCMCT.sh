python3 tools/train.py configs/mitotic/mitotic_CCMCT_retina.py --work-dir checkpoints/CCMCT_retina --seed 42
python tools/test.py configs/mitotic/mitotic_CCMCT_retina.py checkpoints/CCMCT_retina/epoch_8.pth --out test_pred/CCMCT_retina.pkl

python3 tools/data_preparation/format_output.py \
        -d CCMCT -i test_pred/CCMCT_retina.pkl \
        -o test_pred/CCMCT_retina_base.pkl 

python3 tools/data_preparation/format_output.py \
        -d CCMCT -i test_pred/CCMCT_retina.pkl \
        -c configs/mitotic/mitotic_CCMCT_retina_refocus.py \
        -m checkpoints/CCMCT_retina/epoch_8.pth -o test_pred/CCMCT_retina_relocated.pkl -r


