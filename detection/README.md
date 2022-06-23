# Detection Stage

Implementation of the detection, and window relocation stage.



## Environment Setup
We have provided a docker build script for each directory. Run the command below to setup docker container for training and evaluation environment:

```
docker build -t detection detection/mmdetection/docker/ # build docker image
docker run --gpus all -it -v path_to_ReCasNet_git/:/work/ --shm-size=20g detection #start docker container
cd detection/mmdetection/
sh post_install.sh
```

## Data 

The WSI of the CCMCT and CMC dataset an its annotation could be download by using the following command:
```
python3 tools/data_preparation/download_WSI.py --dataset CMC #change CMC to CCMCT to download CCMCT dataset
wget -P data/database/ https://github.com/DeepPathology/MITOS_WSI_CMC/raw/master/databases/MITOS_WSI_CMC_CODAEL_TR_ROI.sqlite
wget -P data/database/ https://github.com/DeepPathology/MITOS_WSI_CCMCT/raw/master/databases/MITOS_WSI_CCMCT_ODAEL.sqlite
```

## Training

Training and inference script of other models and datasets could be found in `script/`

### Detection Stage Training

```
python3 tools/train.py configs/mitotic/mitotic_CMC_base.py --work-dir checkpoints/CMC_faster --seed 42
```

## Inference

`-d`, `-i`, `-c`, `-m`, `-o` are dataset name, prediction result path, config path, model path, and output destination path, respectively. Remove `-r` flag to remove a window relocation stage. 

The output is in a pickle form that could be directy used for evaluation in the `classification` directory.

```
#generate raw prediction result
python tools/test.py configs/mitotic/mitotic_CMC_base.py checkpoints/CMC_faster/epoch_8.pth --out test_pred/CMC_faster.pkl

#generate formatted prediction result (w/o relocation stage)
python3 tools/data_preparation/format_output.py \
        -d CMC 
        -i test_pred/CMC_faster.pkl \
        -o test_pred/CMC_faster_base.pkl 

#generate formatted prediction result (w/ relocation stage)
python3 tools/data_preparation/format_output.py \
        -d CMC -i test_pred/CMC_faster.pkl \
        -c configs/mitotic/mitotic_CMC_refocus.py \
        -m checkpoints/CMC_faster/epoch_8.pth -o test_pred/CMC_faster_relocated.pkl -r

```

