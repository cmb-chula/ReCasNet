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
### Detection Stage Training

```
python tools/train.py  configs/mitotic/mitotic_CMC_base.py --work-dir checkpoints/example
```

## Inference

`-m`, and `-o` is model checkpoint path, and output path respectively. Remove `--relocate` flag to remove a window relocation stage. The output is in a pickle form that could be directy used for evaluation in the `classification` directory.

```
python3 tools/data_preparation/inference_detection_stage.py --relocate -d CMC -m checkpoints/example/epoch_8.pth -o test.pkl 
```