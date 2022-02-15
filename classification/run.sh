
# python3 tools/train/train_mitotic_translate.py -i config/mitotic_CMC_proposed_translate.py -o CMC/k_center/CMC_relcoation_k_center_1
# python3 tools/train/train_mitotic_translate.py -i config/mitotic_CMC_proposed_translate.py -o CMC/k_center/CMC_relcoation_k_center_2

# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate.py -o CMC/k_center/CMC_relcoation_k_center_1
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate.py -o CMC/k_center/CMC_relcoation_k_center_2

# python3 tools/train/train.py -i config/mitotic_CMC_proposed.py -o CMC/k_center/CMC_classification_k_center_1 -a 0 -b 0
# python3 tools/train/train.py -i config/mitotic_CMC_proposed.py -o CMC/k_center/CMC_classification_k_center_2 -a 0 -b 0

# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed.py -o CMC/k_center/CMC_classification_k_center_1
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed.py -o CMC/k_center/CMC_classification_k_center_2



# python3 tools/train/train_mitotic_translate.py -i config/mitotic_CMC_proposed_translate.py -o CMC/relocation/CMC_relocation_propose_3
# python3 tools/train/train_mitotic_translate.py -i config/mitotic_CMC_proposed_translate.py -o CMC/relocation/CMC_relocation_propose_4

# python3 tools/train/train.py -i config/mitotic_CMC_proposed.py -o CMC/var_3model/CMC_classification_propose_1_var_3model -a 0 -b 0
# python3 tools/train/train.py -i config/mitotic_CMC_proposed.py -o CMC/var_3model/CMC_classification_propose_2_var_3model -a 0 -b 0
# python3 tools/train/train.py -i config/mitotic_CMC_proposed.py -o CMC/var_3model/CMC_classification_propose_3_var_3model -a 0 -b 0


# python3 tools/train/train_mitotic_translate.py -i config/mitotic_CMC_proposed_translate.py -o CMC/var_3model/CMC_relocation_propose_1_var_3model
# python3 tools/train/train_mitotic_translate.py -i config/mitotic_CMC_proposed_translate.py -o CMC/var_3model/CMC_relocation_propose_2_var_3model
# python3 tools/train/train_mitotic_translate.py -i config/mitotic_CMC_proposed_translate.py -o CMC/var_3model/CMC_relocation_propose_3_var_3model

# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed.py -o CMC/var_3model/CMC_classification_propose_1_var_3model
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed.py -o CMC/var_3model/CMC_classification_propose_2_var_3model
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed.py -o CMC/var_3model/CMC_classification_propose_3_var_3model


# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate.py -o CMC/var_3model/CMC_relocation_propose_1_var_3model
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate.py -o CMC/var_3model/CMC_relocation_propose_2_var_3model
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate.py -o CMC/var_3model/CMC_relocation_propose_3_var_3model


# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/proposed_aux/CMC_relocation_propose_1_aux

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/proposed_aux_var/CMC_relocation_propose_1_aux_0.95 -a 0.95; 
# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/proposed_aux_var/CMC_relocation_propose_2_aux_0.95 -a 0.95;
# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/proposed_aux_var/CMC_relocation_propose_3_aux_0.95 -a 0.95;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/proposed_aux_var/CMC_relocation_propose_1_aux_0.9 -a 0.9;  
# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/proposed_var_aux_ignore/CMC_relocation_propose_1_var_ignore_aux_0.8 -a 0.8;
# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/proposed_var_aux_ignore/CMC_relocation_propose_2_var_ignore_aux_0.8 -a 0.8;

# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/proposed_var_aux_ignore/CMC_relocation_propose_1_var_ignore_aux_0.8 ;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/proposed_var_aux_ignore/CMC_relocation_propose_2_var_ignore_aux_0.8 ;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/proposed_aux_var/CMC_relocation_propose_3_aux_0.95;


# python3 tools/train/train.py -i config/mitotic_CMC_proposed.py -o CMC/resnet50_proposed/test1 -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed.py -o CMC/resnet50_proposed/test1;

# python3 tools/train/train.py -i config/mitotic_CMC_proposed.py -o CMC/resnet50_proposed/test2 -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed.py -o CMC/resnet50_proposed/test2;

# python3 tools/train/train_mitotic_translate.py -i config/mitotic_slide_extra.py -o CCMCT/relocation_base/relocation_base_1 -a 0 -b 0
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate.py -o CCMCT/relocation_base/relocation_base_1;

# python3 tools/train/train_mitotic_translate.py -i config/mitotic_slide_extra.py -o CCMCT/relocation_base/relocation_base_2 -a 0 -b 0
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate.py -o CCMCT/relocation_base/relocation_base_2;

# python3 tools/train/train_mitotic_translate.py -i config/mitotic_CMC_proposed_translate.py -o CMC/retina/relocation_retina_1;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate.py -o CMC/retina/relocation_retina_1;

# python3 tools/train/train_mitotic_translate.py -i config/mitotic_CMC_proposed_translate.py -o CMC/retina/relocation_retina_1;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate.py -o CMC/retina/relocation_retina_1;

# python3 tools/train/train_mitotic_translate.py -i config/mitotic_CMC_proposed_translate.py -o CMC/retina/relocation_retina_2;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate.py -o CMC/retina/relocation_retina_2;

# python3 tools/train/train.py -i config/mitotic_CMC_proposed.py -o CMC/retina/classification_retina_iter2_1 -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed.py -o CMC/retina/classification_retina_iter2_1;

# python3 tools/train/train.py -i config/mitotic_CMC_proposed.py -o CMC/retina/classification_retina_iter2_2 -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed.py -o CMC/retina/classification_retina_iter2_2;

# python3 tools/train/train.py -i config/mitotic_CMC_proposed_2x.py -o CMC/retina/classification_retina_iter2_2_2x -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed.py -o CMC/retina/classification_retina_iter2_2_2x;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/retina/relocation_aux_ignore_neg_1_v2_0.9 -a 0.9;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/retina/relocation_aux_ignore_neg_1_v2_0.9 ;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/retina/relocation_aux_ignore_neg_2_v2_0.9 -a 0.9;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/retina/relocation_aux_ignore_neg_2_v2_0.9;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/retina/relocation_aux_ignore_neg_1_0.8 -a 0.8;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/retina/relocation_aux_ignore_neg_1_0.8;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/retina/relocation_aux_ignore_neg_1_0.95 -a 0.95;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/retina/relocation_aux_ignore_neg_1_0.95;


# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/densenet/densenet_relocation_aux_ignore_neg_3_v3_0.9 -a 0.9;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/densenet/densenet_relocation_aux_ignore_neg_3_v3_0.9 ;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/densenet/densenet_relocation_aux_ignore_neg_4_v3_0.9 -a 0.9;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/densenet/densenet_relocation_aux_ignore_neg_4_v3_0.9;


# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/relocation/effnet_relocation_aux_ignore_neg_0.9_2 -a 0.9;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/relocation/effnet_relocation_aux_ignore_neg_0.9_2 ;

# python3 tools/train/train.py -i config/mitotic_simple_extend.py -o CCMCT/classification/classification_proposed_1_extend -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_simple_extend.py -o CCMCT/classification/classification_proposed_1_extend;

# python3 tools/train/train.py -i config/mitotic_simple_extend.py -o CCMCT/classification/classification_proposed_2_extend -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_simple_extend.py -o CCMCT/classification/classification_proposed_2_extend;

# # sleep 150m;
# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/relocation_noextra/effnet_relocation_aux_CMC_noextra_0.9_1 -a 0.9;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/relocation_noextra/effnet_relocation_aux_CMC_noextra_0.9_1 ;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/relocation_noextra/effnet_relocation_aux_CMC_noextra_0.9_2 -a 0.9;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/relocation_noextra/effnet_relocation_aux_CMC_noextra_0.9_2 ;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/relocation_noextra/effnet_relocation_aux_CMC_noextra_0.9_3 -a 0.9;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/relocation_noextra/effnet_relocation_aux_CMC_noextra_0.9_3 ;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/relocation_noextra/effnet_relocation_aux_CMC_noextra_0.95_1 -a 0.95;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/relocation_noextra/effnet_relocation_aux_CMC_noextra_0.95_1 ;


# python3 tools/train/train_mitotic_translate.py -i config/mitotic_CMC_proposed_translate.py -o CMC/relocation_noaux/effnet_relocation_noaux_1;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate.py -o CMC/relocation_noaux/effnet_relocation_noaux_1 ;


# python3 tools/train/train_mitotic_translate.py -i config/mitotic_CMC_proposed_translate.py -o CMC/relocation_noaux/effnet_relocation_noaux_2;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate.py -o CMC/relocation_noaux/effnet_relocation_noaux_2 ;


# python3 tools/train/train_mitotic_translate.py -i config/mitotic_CMC_proposed_translate.py -o CMC/relocation_noaux/effnet_relocation_noaux_3;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate.py -o CMC/relocation_noaux/effnet_relocation_noaux_3 ;


# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/relocation_final/CMC_relocation_095_final_0 -a 0.95;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/relocation_final/CMC_relocation_095_final_0 ;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/relocation_noextra/effnet_relocation_aux_CMC_noextra_0.99_3 -a 0.99;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/relocation_noextra/effnet_relocation_aux_CMC_noextra_0.99_3 ;


# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/relocation_extra/effnet_relocation_aux_CCMCT_extra_0.95_1_fix -a 0.95;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/relocation_extra/effnet_relocation_aux_CCMCT_extra_0.95_1_fix ;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/relocation_extra/effnet_relocation_aux_CCMCT_extra_0.95_0_fix -a 0.95;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/relocation_extra/effnet_relocation_aux_CCMCT_extra_0.95_0_fix ;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/relocation_extra/effnet_relocation_aux_CCMCT_extra_0.95_2_fix -a 0.95;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/relocation_extra/effnet_relocation_aux_CCMCT_extra_0.95_2_fix ;

# python3 tools/train/train.py -i config/mitotic_CMC_proposed.py -o CMC/CMC_proposed_9_final -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed.py -o CMC/CMC_proposed_9_final;

# python3 tools/train/train.py -i config/mitotic_CMC_proposed.py -o CMC/CMC_proposed_9_final -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed.py -o CMC/CMC_proposed_9_final;

# python3 tools/train/train.py -i config/mitotic_CMC_proposed.py -o CMC/CMC_proposed_10_final -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed.py -o CMC/CMC_proposed_10_final;

# python3 tools/train/train.py -i config/mitotic_CMC_proposed.py -o CMC/CMC_proposed_11_final -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed.py -o CMC/CMC_proposed_11_final;

# python3 tools/train/train.py -i config/mitotic_CMC_proposed.py -o CMC/CMC_proposed_6_final -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed.py -o CMC/CMC_proposed_6_final;

# python3 tools/train/train.py -i config/mitotic_CMC_proposed.py -o CMC/CMC_proposed_2_final -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed.py -o CMC/CMC_proposed_2_final;

# python3 tools/train/train.py -i config/mitotic_CMC_proposed.py -o CMC/CMC_proposed_2 -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed.py -o CMC/CMC_proposed_2;


# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/relocation_extra/effnet_CMC_aux_extra_090_1 -a 0.9;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/relocation_extra/effnet_CMC_aux_extra_090_1 ;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/relocation_extra/effnet_CMC_aux_extra_090_2 -a 0.9;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/relocation_extra/effnet_CMC_aux_extra_090_2 ;


# python3 tools/train/train.py -i config/mitotic_CMC_proposed.py -o CMC/CMC_translation_test3 -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed.py -o CMC/CMC_translation_test3;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/relocation_extra/effnet_CCMCT_aux_no_relocation_0 -a 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/relocation_extra/effnet_CCMCT_aux_no_relocation_0 ;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/relocation_extra/effnet_CCMCT_aux_no_relocation_1 -a 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/relocation_extra/effnet_CCMCT_aux_no_relocation_1 ;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/relocation_extra/effnet_CCMCT_aux_no_relocation_2 -a 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/relocation_extra/effnet_CCMCT_aux_no_relocation_2 ;

# python3 tools/train/train.py -i config/mitotic_CCMCT_proposed.py -o CCMCT/classification/add_query/CCMCT_addquery_1_fixv9_with_translate -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CCMCT_proposed.py -o CCMCT/classification/add_query/CCMCT_addquery_1_fixv9_with_translate;

# python3 tools/train/train.py -i config/mitotic_CCMCT_proposed.py -o CCMCT/classification/add_query/CCMCT_addquery_2_fixv9_with_translate -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CCMCT_proposed.py -o CCMCT/classification/add_query/CCMCT_addquery_2_fixv9_with_translate;

# python3 tools/train/train.py -i config/mitotic_CCMCT_proposed.py -o CCMCT/classification/add_query/CCMCT_addquery_3_fixv9_with_translate -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CCMCT_proposed.py -o CCMCT/classification/add_query/CCMCT_addquery_3_fixv9_with_translate;

# python3 tools/train/train.py -i config/mitotic_CCMCT_proposed.py -o CCMCT/classification/add_query/CCMCT_addquery_2 -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CCMCT_proposed.py -o CCMCT/classification/add_query/CCMCT_addquery_2;


# python3 tools/train/train.py -i config/mitotic_CCMCT_base.py -o CCMCT/classification/base/CCMCT_base_0_fix -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CCMCT_base.py -o CCMCT/classification/base/CCMCT_base_0_fix;

# python3 tools/train/train.py -i config/mitotic_CCMCT_base.py -o CCMCT/classification/base/CCMCT_base_1_fix -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CCMCT_base.py -o CCMCT/classification/base/CCMCT_base_1_fix;
# python3 tools/train/train.py -i config/mitotic_CCMCT_base.py -o CCMCT/classification/base/CCMCT_base_2 -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CCMCT_base.py -o CCMCT/classification/base/CCMCT_base_2;

# python3 tools/train/train.py -i config/mitotic_CMC_dump.py -o CMC/ablation/full_query_0 -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_dump.py -o CMC/ablation/full_query_0;

python3 tools/train/train.py -i config/mitotic_CMC_dump2.py -o CMC/ablation/full_query_3 -a 0 -b 0;
python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_dump2.py -o CMC/ablation/full_query_3;

python3 tools/train/train.py -i config/mitotic_CMC_dump2.py -o CMC/ablation/full_query_2 -a 0 -b 0;
python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_dump2.py -o CMC/ablation/full_query_2;

python3 tools/train/train.py -i config/mitotic_CMC_dump.py -o CMC/ablation/US_2 -a 0 -b 0;
python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_dump.py -o CMC/ablation/US_2;

python3 tools/train/train.py -i config/mitotic_CMC_dump3.py -o CMC/ablation/D_0 -a 0 -b 0;
python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_dump3.py -o CMC/ablation/D_0;

python3 tools/train/train.py -i config/mitotic_CMC_dump3.py -o CMC/ablation/D_1 -a 0 -b 0;
python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_dump3.py -o CMC/ablation/D_1;

python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/appendix/relocation_final_long1_0 -a 1;
python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/appendix/relocation_final_long1_0 ;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/appendix/relocation_final_long099_0 -a 0.99;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/appendix/relocation_final_long099_0 ;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/appendix/relocation_final_long099_1 -a 0.99;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/appendix/relocation_final_long099_1 ;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/appendix/relocation_final_long080_0 -a 0.8;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/appendix/relocation_final_long080_0 ;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/appendix/relocation_final_long080_1 -a 0.8;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/appendix/relocation_final_long080_1 ;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/appendix/relocation_final_long070_0 -a 0.7;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/appendix/relocation_final_long070_0 ;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/appendix/relocation_final_long070_1 -a 0.7;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/appendix/relocation_final_long070_1 ;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/appendix/relocation_final_long060_0 -a 0.6;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/appendix/relocation_final_long060_0 ;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/appendix/relocation_final_long060_1 -a 0.6;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux.py -o CMC/appendix/relocation_final_long060_1 ;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CMC_proposed_translate_aux2.py -o CMC/long_test/relocation_final_long095_1 -a 0.95;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed_translate_aux2.py -o CMC/long_test/relocation_final_long095_1 ;
# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/relocation_noextra2/effnet_relocation_aux_CCMCT_noextra_0.9_3 -a 0.9;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/relocation_noextra2/effnet_relocation_aux_CCMCT_noextra_0.9_3 ;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/relocation_noextra/effnet_relocation_aux_CCMCT_noextra_0.9_1 -a 0.9;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/relocation_noextra/effnet_relocation_aux_CCMCT_noextra_0.9_1 ;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/relocation_noextra/effnet_relocation_aux_CCMCT_noextra_0.9_2 -a 0.9;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/relocation_noextra/effnet_relocation_aux_CCMCT_noextra_0.9_2 ;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/relocation_noextra/effnet_relocation_aux_CCMCT_noextra_0.9_3 -a 0.9;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CCMCT/relocation_noextra/effnet_relocation_aux_CCMCT_noextra_0.9_3 ;




# python3 tools/train/train.py -i config/mitotic_CMC_base.py -o CMC/final/classification_base_1_full -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_base.py -o CMC/final/classification_base_1_full;

# python3 tools/train/train.py -i config/mitotic_CMC_base.py -o CMC/final/classification_base_2_full -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_base.py -o CMC/final/classification_base_2_full;

# python3 tools/train/train_.py -i config/mitotic_CCMCT_proposed.py -o CCMCT/final2/classification_proposed_1 -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CCMCT_proposed.py -o CCMCT/final2/classification_proposed_1;

# python3 tools/train/train.py -i config/mitotic_CCMCT_proposed.py -o CCMCT/final2/classification_proposed_2 -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CCMCT_proposed.py -o CCMCT/final2/classification_proposed_2;

# python3 tools/train/train.py -i config/mitotic_CCMCT_proposed.py -o CCMCT/final2/classification_proposed_3 -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CCMCT_proposed.py -o CCMCT/final2/classification_proposed_3;

# python3 tools/train/train.py -i config/mitotic_CMC_proposed.py -o CMC/densenet121/classification_densenet121_1 -a 0 -b 0;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CMC_proposed.py -o CMC/densenet121/classification_densenet121_1;

# python3 tools/train/train_mitotic_translate_aux.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CMC/densenet/densenet_relocation_aux_ignore_neg_4_v3_0.9 -a 0.9;
# python3 tools/utils/convert_to_onnx.py -i config/mitotic_CCMCT_proposed_translate_aux.py -o CMC/densenet/densenet_relocation_aux_ignore_neg_4_v3_0.9;


