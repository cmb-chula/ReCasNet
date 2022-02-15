###########training###################
python3 tools/train/train.py -i config/mitotic_CMC_base.py -o CMC/publish/publish_base_0;
python3 tools/utils/create_inference_model.py -i config/mitotic_CMC_base.py -o CMC/publish/publish_base_0;
#####################################

python3 tools/train/train.py -i config/mitotic_CMC_proposed.py -o CMC/publish/classifier_data_selection_1;
python3 tools/utils/create_inference_model.py -i config/mitotic_CMC_proposed.py -o CMC/publish/classifier_data_selection_1;

###########inference#################
python3 tools/eval/inference_classification.py -d CMC -ip ../detection/mmdetection/test_pred/CMC_test_final_0.pkl -op ./cls.pkl  -m CMC/publish/classifier_data_selection_1 -c 1

##########evaluation#################
python3 tools/eval/calculate_F1.py -ip ./cls.pkl
