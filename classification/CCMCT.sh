

# python3 tools/train/train.py -i config/mitotic_CCMCT_base.py -o CCMCT/publish/classifier_base_CCMCT_0;
# python3 tools/utils/create_inference_model.py -i config/mitotic_CCMCT_base.py -o CCMCT/publish/classifier_base_CCMCT_0;

# python3 tools/train/train.py -i config/mitotic_CCMCT_base.py -o CCMCT/publish/classifier_base_CCMCT_1;
# python3 tools/utils/create_inference_model.py -i config/mitotic_CCMCT_base.py -o CCMCT/publish/classifier_base_CCMCT_1;



# python3 tools/train/train.py -i config/mitotic_CCMCT_proposed.py -o CCMCT/publish/classifier_proposed_CCMCT_5;
# python3 tools/utils/create_inference_model.py -i config/mitotic_CCMCT_proposed.py -o CCMCT/publish/classifier_proposed_CCMCT_5;

# python3 tools/train/train.py -i config/mitotic_CCMCT_base.py -o CCMCT/publish/classifier_base_CCMCT_1;
# python3 tools/utils/create_inference_model.py -i config/mitotic_CCMCT_base.py -o CCMCT/publish/classifier_base_CCMCT_1;

# python3 tools/train/train_mitotic_translate_aux.py -d CCMCT -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/publish/relocation_CCMCT_4 -a 0.9;
# python3 tools/utils/create_inference_model.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/publish/relocation_CCMCT_4;
# python3 tools/train/train_mitotic_translate_aux.py -d CCMCT -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/publish/relocation_CCMCT_1 -a 0.9;
# python3 tools/utils/create_inference_model.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/publish/relocation_CCMCT_1;

# python3 tools/train/train.py -i config/mitotic_CMC_proposed_kcenter.py -o CMC/publish/classifier_data_selection_kencter_24k_1;
# python3 tools/utils/create_inference_model.py -i config/mitotic_CMC_proposed_kcenter.py -o CMC/publish/classifier_data_selection_kencter_24k_1;

python3 tools/eval/inference_center_adjustment.py -d CCMCT -ip ../detection/mmdetection/test_pred/baseline_refocus_detector.pkl -op ./cls1CCMCT.pkl  -m CCMCT/publish/relocation_CCMCT_0 -c 1 -t 0.5
python3 tools/eval/inference_classification.py -d CCMCT -ip ./cls1CCMCT.pkl -op ./cls1CCMCTt.pkl  -m CCMCT/publish/classifier_proposed_CCMCT_1 -c 0.6
# python3 tools/eval/inference_classification.py -d CCMCT -ip ../detection/mmdetection/test_pred/baseline_simple_detector.pkl -op ./cls1CCMCTt.pkl  -m CCMCT/publish/classifier_base_CCMCT_1 -c 1
python3 tools/eval/calculate_F1.py -ip ./cls1CCMCTt.pkl
python3 tools/eval/end2end.py

# python3 tools/eval/inference_classification.py -d CCMCT -s train -ip ../detection/mmdetection/test_pred/train1ccmct.pkl -op ./cls1CCMCT_trainv2.pkl  -m CCMCT/publish/classifier_base_CCMCT_1 -c 1
