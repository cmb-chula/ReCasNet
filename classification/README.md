# Classification Stage

This directory contains an implementation of the Object Center Adjustment stage, classification stage, data selection, and evaluation code.


## Environment Setup
We have provided a docker build script for each directory. Run the command below to setup docker container for training and evaluation environment:

```
docker build -t classification classification/docker/ # build docker image
docker run --gpus all -it -v path_to_ReCasNet_git/:/work/ --shm-size=20g classification #start docker container
cd classification/
```
## Data 

The extracted patch for classification stage training of CCMCT and CMC datasets could be downloaded <a href="https://chula-my.sharepoint.com/:f:/g/personal/6372025021_student_chula_ac_th/ErP3lSm3r_JHthDCkMMFr5oB52c1I0bTyVsgKbECDoEm4w?e=RcRnWw" title="">HERE</a>. The download destination should be the as same the path in the datset config file (see `config/dataset/mitotic_CMC_base.py` for example).

## Training
### Classification Stage Training
`-i`, and `-o` is config path, and output path respectively. The training dataset could be changed by modifying the config file.

```
python3 tools/train/train.py -i config/mitotic_CMC_base.py -o CMC/example; #start training process
python3 tools/utils/create_inference_model.py -i config/mitotic_CMC_base.py -o CMC/example; #the model is in converted_model/output path 
```

### Object Center Adjustment Stage Training
`-i`, and `-o`, `-a` is config path, output path, and relocation loss weight (l_reg) , respectively.

```
python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/example2 -a 0.9; 
python3 tools/utils/create_inference_model.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/example2; 
```

## Evaluation

### Cell-level Evalutaion
The cell-level performance of the pipeline could be evaluated using the following command:

```
#inference Object Center Adjustment stage
python3 tools/eval/inference_relcoation.py --dataset CMC -ip ../detection/mmdetection/test_pred/test1.pkl -op ./adj.pkl  -m converted_model/CMC/example2 -c 1 

#inference classifcaiton stage
python3 tools/eval/inference_classification.py --dataset CMC -ip ./adj.pkl -op ./cls.pkl  -m converted_model/CMC/example -c 1 

#calculate F1
python3 tools/eval/calculate_F1.py -ip ./cls.pkl 
```

The first and second line could be skipped to perform detection stage performance calculation.
```
python3 tools/eval/calculate_F1.py -ip ../detection/mmdetection/test_pred/test1.pkl
```

### End-to-End Evalutaion
The end-to-end performance of the pipeline (MAPE, MAE) could be evaluated using the following command:
```
python3 tools/eval/end2end.py -ip ./cls.pkl -d CMC
```
