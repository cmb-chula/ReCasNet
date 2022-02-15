###########training###################
# python3 tools/train/train.py -i config/mitotic_CMC_all.py -o CMC/publish/classifier_proposed_all_0;
# python3 tools/utils/create_inference_model.py -i config/mitotic_CMC_all.py -o CMC/publish/classifier_proposed_all_0;
# python3 tools/train/train.py -i config/mitotic_CMC_all.py -o CMC/publish/classifier_proposed_all_1;
# python3 tools/utils/create_inference_model.py -i config/mitotic_CMC_all.py -o CMC/publish/classifier_proposed_all_1;
#####################################


# python3 tools/train/train.py -i config/mitotic_CMC_US.py -o CMC/publish/classifier_proposed_US_0;
# python3 tools/utils/create_inference_model.py -i config/mitotic_CMC_US.py -o CMC/publish/classifier_proposed_US_0;
# python3 tools/train/train.py -i config/mitotic_CMC_US.py -o CMC/publish/classifier_proposed_US_1;
# python3 tools/utils/create_inference_model.py -i config/mitotic_CMC_US.py -o CMC/publish/classifier_proposed_US_1;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/publish/relocation_2 -a 0.9;
# python3 tools/utils/create_inference_model.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/publish/relocation_2;
# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/publish/relocation_1 -a 0.9;
# python3 tools/utils/create_inference_model.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/publish/relocation_1;

# python3 tools/train/train.py -i config/mitotic_CMC_proposed_kcenter.py -o CMC/publish/classifier_data_selection_kencter_24k_1;
# python3 tools/utils/create_inference_model.py -i config/mitotic_CMC_proposed_kcenter.py -o CMC/publish/classifier_data_selection_kencter_24k_1;

###########inference#################
# python3 tools/eval/inference_classification.py -d CMC -ip ../detection/mmdetection/test_pred/test1r.pkl -op ./cls.pkl  -m CMC/publish/classifier_data_selection_kencter_24k_0 -c 1
# python3 tools/eval/calculate_F1.py -ip ./cls.pkl

python3 tools/eval/inference_center_adjustment.py -d CMC -ip ../detection/mmdetection/test_pred/test1r.pkl -op ./cls1r.pkl  -m CMC/publish/relocation_1 -t 0.2
# python3 tools/eval/inference_classification.py -d CMC -ip ../detection/mmdetection/test_pred/test2.pkl -op ./cls.pkl  -m CMC/publish/classifier_data_selection_0 -c 1
python3 tools/eval/inference_classification.py -d CMC -ip ./cls1r.pkl -op ./cls1r.pkl  -m CMC/publish/classifier_data_selection_0 -c 0.6
python3 tools/eval/calculate_F1.py -ip ./cls1r.pkl
