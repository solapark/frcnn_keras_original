#!/bin/bash

GPU=0
VAL_START_IDX=20
VAL_INTERVAL=5
DATASET_PATH="/data1/sap/MessyTable/labels/test.json"
RPN_PICKLE_DIR="pickle/messytable/rpn/test/sv_messytable_cam3_resume_110_model"
VEN_PICKLE_DIR="/data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn59_th.1/reid_output/test"

SAVE_DIRS=(
    "tmvreid_messytable_rpn59_th.1_pos3_classifier_only"
    "tmvreid_messytable_rpn59_th.1_roi64_pos48_classifier_only"
    "tmvreid_messytable_rpn59_th.1_unique_sample_classifier_only"
    "tmvreid_messytable_rpn59_th.1_roi64_pos48+fix_num_pos+unique_sample_classifier_only"
    "tmvreid_messytable_rpn59_th.1_roi32_pos24+fix_num_pos+unique_sample_classifier_only"
    "tmvreid_messytable_rpn59_th.1_roi16_pos12+fix_num_pos+unique_sample_classifier_only"
    "tmvreid_messytable_rpn59_th.1_roi8_pos6+fix_num_pos+unique_sample_classifier_only"
    "tmvreid_messytable_rpn59_th.1_roi64_pos48+fix_num_pos+unique_sample+gtmxot.2_classifier_only"
    "tmvreid_messytable_rpn59_th.1_roi64_pos48+fix_num_pos+unique_sample+gtmxot.3_classifier_only"
    "tmvreid_messytable_rpn59_th.1_roi32_pos24+fix_num_pos+unique_sample+gtmxot.3_classifier_only"
    "tmvreid_messytable_rpn59_th.1_roi32_pos24+fix_num_pos+unique_sample+small16ot.1_classifier_only"
    "tmvreid_messytable_rpn59_th.1_roi32_pos24+fix_num_pos+unique_sample+small16ot.15_classifier_only"
    "tmvreid_messytable_rpn59_th.1_roi32_pos24+fix_num_pos+unique_sample+small14ot.1_classifier_only"
    "tmvreid_messytable_rpn59_th.1_roi32_pos24+fix_num_pos+unique_sample+small14ot.15_classifier_only"
)

NUM_ROIS_VALUES=(
    4
    64
    4
    64
    32
    16
    8
    64
    64
    32
    32
    32
    32
    32
)

NUM_POS_VALUES=(
    3
    48
    3
    48
    24
    12
    6
    48
    48
    24
    24
    24
    24
    24
)

# Loop over SAVE_DIR values
for ((i=0; i<${#SAVE_DIRS[@]}; i++)); do
    SAVE_DIR="${SAVE_DIRS[i]}"
    NUM_ROIS="${NUM_ROIS_VALUES[i]}"
    NUM_POS="${NUM_POS_VALUES[i]}"
    
    # Loop over sv_thresh values
    for sv_thresh in $(seq 0.1 0.1 0.9); do
        # Run first command
        CUDA_VISIBLE_DEVICES=$GPU python main.py \
            --mode val_models \
            --save_dir "$SAVE_DIR" \
            --val_start_idx $VAL_START_IDX \
            --val_interval $VAL_INTERVAL \
            --dataset_path "$DATASET_PATH" \
            --is_use_epipolar \
            --freeze_rpn \
            --freeze_ven \
            --rpn_pickle_dir "$RPN_PICKLE_DIR" \
            --ven_pickle_dir "$VEN_PICKLE_DIR" \
            --classifier_nms_thresh "$sv_thresh" \
            --unique_sample \
            --num_rois "$NUM_ROIS" \
            --num_pos "$NUM_POS" \
            --val_models_log_name "svnms$sv_thresh"

'<<COMMENT
        # Loop over inter_cls_thresh values
        for inter_cls_thresh in $(seq 0.1 0.1 0.9); do
            # Run second command
            CUDA_VISIBLE_DEVICES=$GPU python main.py \
                --mode val_models \
                --save_dir "$SAVE_DIR" \
                --val_start_idx $VAL_START_IDX \
                --val_interval $VAL_INTERVAL \
                --dataset_path "$DATASET_PATH" \
                --is_use_epipolar \
                --freeze_rpn \
                --freeze_ven \
                --rpn_pickle_dir "$RPN_PICKLE_DIR" \
                --ven_pickle_dir "$VEN_PICKLE_DIR" \
                --classifier_nms_thresh "$sv_thresh" \
                --unique_sample \
                --num_rois "$NUM_ROIS" \
                --num_pos "$NUM_POS" \
                --inter_cls_mv_nms \
                --classifier_inter_cls_mv_nms_thresh "$inter_cls_thresh" \
                --val_models_log_name "svnms${sv_thresh}_interclsnms${inter_cls_thresh}"
        done
COMMENT'
    done
done
