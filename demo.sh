#train mv_interpark18
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --reset  --mode train --dataset INTERPARK18 --save_dir interpark18 --input_weight_path /data3/sap/frcnn_keras_original/model/interpark18_rpn_only.hdf5 --num_valid_cam 3 --dataset_path /data3/sap/frcnn_keras_original/data/INTERPARK18/train.json 

#train mv_interpark18(resume)
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --resume  --mode train --dataset INTERPARK18 --save_dir interpark18 --input_weight_path /data3/sap/frcnn_keras_original/experiment/interpark18/model/model_9.hdf5 --num_valid_cam 3 --dataset_path /data3/sap/frcnn_keras_original/data/INTERPARK18/train.json

#val models mv_interpark18
CUDA_VISIBLE_DEVICES='' python -m pdb main.py --mode val_models --dataset INTERPARK18 --save_dir interpark18 --num_valid_cam 3 --val_start_idx 1 --val_interval 1 --dataset_path /data3/sap/frcnn_keras_original/data/INTERPARK18/val.json

#train messytable
CUDA_VISIBLE_DEVICES=2 python -m pdb main.py --reset --mode train --dataset MESSYTABLE --save_dir messytable_epi --input_weight_path /data3/sap/frcnn_keras_original/model/messytable_rpn_only.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar

#train messytable (freeze rpn)
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --reset --mode train --dataset MESSYTABLE --save_dir messytable_epi_freeze_rpn_debug --input_weight_path /data3/sap/frcnn_keras_original/model/messytable_rpn_only.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn

#train messytable (rpn+reid only)
CUDA_VISIBLE_DEVICES=3 python -m pdb main.py --reset --mode train --dataset MESSYTABLE --save_dir messytable_rpn_reid_only --input_weight_path /data3/sap/frcnn_keras_original/model/messytable_rpn_only.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn

#train messytable (rpn+reid only, ven_alph .6)
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --reset --mode train --dataset MESSYTABLE --save_dir messytable_rpn_reid_only_alpha6 --input_weight_path /data3/sap/frcnn_keras_original/model/messytable_rpn_only.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --ven_loss_alpha .6

#val messytable
CUDA_VISIBLE_DEVICES='' python -m pdb main.py --mode val --dataset MESSYTABLE --save_dir mv_messytable_classifier_only --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/val.json --input_weight_path /data3/sap/frcnn_keras_original/experiment/mv_messytable_classifier_only/model/model_216.hdf5 --is_use_epipolar --freeze_rpn --freeze_ven --rpn_dir rpn_pickle_val --ven_dir ven_pickle_val

#val_models messytable
CUDA_VISIBLE_DEVICES='' python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir messytable --num_valid_cam 3 --val_start_idx 1 --val_interval 1 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_dir rpn_pickle_val --ven_dir ven_pickle_val
 
#test messytable 
CUDA_VISIBLE_DEVICES='' python -m pdb main.py  --mode test --dataset MESSYTABLE --save_dir messytable --input_weight_path /data3/sap/frcnn_keras_original/experiment/messytable/model/model_9.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --is_use_epipolar

# train ven
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --reset --mode train --dataset MESSYTABLE --save_dir mv_messytable_reid_only --input_weight_path /data3/sap/frcnn_keras_original/model/sv_messytable_cam3_resume_110_model.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir pickle/messytable/rpn/train/sv_messytable_cam3_resume_110_model --freeze_classifier

# train classifier
CUDA_VISIBLE_DEVICES=3 python -m pdb main.py --reset --mode train --dataset MESSYTABLE --save_dir mv_messytable_classifier_only_strict_pos --input_weight_path /data3/sap/frcnn_keras_original/experiment/mv_messytable_reid_only/model/model_15.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir pickle/messytable/rpn/train/sv_messytable_cam3_resume_110_model --freeze_ven --ven_pickle_dir pickle/messytable/ven/train/mv_messytable_reid_only_model_15

# fine tuning
CUDA_VISIBLE_DEVICES=2 python -m pdb main.py --reset --mode train --dataset MESSYTABLE --save_dir mv_messytable_fine_tuning --input_weight_path /data3/sap/frcnn_keras_original/experiment/mv_messytable_classifier_only_real/model/model_151.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar

# save_rpn_feature(train)
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode save_rpn_feature --dataset MESSYTABLE --input_weight_path /data3/sap/frcnn_keras_original/model/sv_messytable_cam3_resume_110_model.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --rpn_pickle_dir pickle/messytable/rpn/train/sv_messytable_cam3_resume_110_model 

# save_rpn_feature(val)
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode save_rpn_feature --dataset MESSYTABLE --input_weight_path /data3/sap/frcnn_keras_original/model/sv_messytable_cam3_resume_110_model.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/val.json --rpn_pickle_dir pickle/messytable/rpn/val/sv_messytable_cam3_resume_110_model 

# save_ven_feature(train)
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode save_ven_feature --dataset MESSYTABLE --input_weight_path /data3/sap/frcnn_keras_original/experiment/mv_messytable_reid_only/model/model_15.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir pickle/messytable/rpn/train/sv_messytable_cam3_resume_110_model --ven_pickle_dir pickle/messytable/ven/train/mv_messytable_reid_only_model_15

# save_ven_feature(val)
CUDA_VISIBLE_DEVICES=3 python -m pdb main.py --mode save_ven_feature --dataset MESSYTABLE --input_weight_path /data3/sap/frcnn_keras_original/experiment/mv_messytable_reid_only/model/model_15.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir pickle/messytable/rpn/val/sv_messytable_cam3_resume_110_model --ven_pickle_dir pickle/messytable/ven/val/mv_messytable_reid_only_model_15

#messytable demo classifier only
CUDA_VISIBLE_DEVICES=2 python -m pdb main.py --mode demo --dataset MESSYTABLE --save_dir mv_messytable_classifier_only_real --input_weight_path /data3/sap/frcnn_keras_original/experiment/mv_messytable_classifier_only_real/model/model_87.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir pickle/messytable/rpn/val/sv_messytable_cam3_resume_110_model --freeze_ven --ven_pickle_dir pickle/messytable/ven/val/mv_messytable_reid_only_model_15 

#save_sv_wgt 
CUDA_VISIBLE_DEVICES=2 python -m pdb main.py --mode save_sv_wgt  --dataset MESSYTABLE --input_weight_path /data3/sap/frcnn_keras/model/sv_messytable_cam3_best/sv_messytable_cam3_resume_110_model.hdf5 --output_weight_path /data3/sap/frcnn_keras_original/model/sv_messytable_cam3_resume_110_model.hdf5 --num_valid_cam 3

#draw_json
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode draw_json --dataset MESSYTABLE --save_dir drawing/sv_messytable/svdet+asnet+majority --num_valid_cam 3 --dataset_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet+majority.json 
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode draw_json --dataset MESSYTABLE --save_dir drawing/mv_messytable/mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05_model_341_fine_tunning_model_80 --num_valid_cam 3 --dataset_path  /data3/sap/frcnn_keras_original/experiment/mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05_model_341_fine_tunning/model_80.json

#val_json_json
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/svdet+asnet+majority --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --pred_dataset_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet+majority.json
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05_model_341_fine_tunning_model_80 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --pred_dataset_path /data3/sap/frcnn_keras_original/experiment/mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05_model_341_fine_tunning/model_80.json 

# write json
CUDA_VISIBLE_DEVICES=1 python -m pdb main.py --mode write_json --dataset MESSYTABLE --save_dir mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05_model_341_fine_tunning  --input_weight_path /data3/sap/frcnn_keras_original/experiment/mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05_model_341_fine_tunning/model/model_80.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --result_json_path /data3/sap/frcnn_keras_original/experiment/mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05_model_341_fine_tunning/model_80.json --is_use_epipolar --max_dist_epiline_to_box .05


CUDA_VISIBLE_DEVICES=3 python -m pdb main.py --reset --mode train --dataset MESSYTABLE --save_dir mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05 --input_weight_path /data3/sap/frcnn_keras_original/experiment/mv_messytable_reid_only/model/model_15.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir pickle/messytable/rpn/train/sv_messytable_cam3_resume_110_model --freeze_ven --ven_pickle_dir pickle/messytable/ven/train/mv_messytable_reid_only_model_15_max_dist_epiline_to_box.05 --max_dist_epiline_to_box .05

# val 341
CUDA_VISIBLE_DEVICES=1 python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --is_use_epipolar --val_start_idx 341 --val_end_idx 341 --val_interval 1

# train fine tunning
CUDA_VISIBLE_DEVICES=2 python -m pdb main.py --reset --mode train --dataset MESSYTABLE --save_dir tmp --input_weight_path /data3/sap/frcnn_keras_original/experiment/mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05/model/model_341.hdf5 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --max_dist_epiline_to_box .05

# val fine tunning
CUDA_VISIBLE_DEVICES=1 python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05_model_341_fine_tunning --dataset_path /data1/sap/MessyTable/labels/test.json --is_use_epipolar --val_start_idx 1 --val_end_idx 3 --val_interval 1 --max_dist_epiline_to_box .05

#demo 341
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode demo --dataset MESSYTABLE --save_dir mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05/model_341 --input_weight_path /data3/sap/frcnn_keras_original/experiment/mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05/model/model_341.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --is_use_epipolar --max_dist_epiline_to_box .05


-----------------------------------------------------------
#svdet#

# val svdet and generate csv file (log.csv)
cd ~/frcnn_keras

CUDA_VISIBLE_DEVICES=3 python -m pdb test.py  --save_name sv_messytable_cam3_resume --test_path /data1/sap/MessyTable/labels/test_3cam.txt --model_idx 110 --val --progbar --log_output

# conver svdet csv to json(log.json)
python script/csv2json.py --src_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/log.csv --dst_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/log.json

# val svdet log.json
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/test_3cam --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --pred_dataset_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/log.json

# generate svdet.json (assign gt inst id to svdet result)
python script/assign_inst_id_to_frcnn_result.py --gt_path /data1/sap/MessyTable/labels/test.json --src_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/log.json --dst_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet.json

# draw svdet.json
*draw svdet.json
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode draw_json --dataset MESSYTABLE --save_dir drawing/svdet.json --num_valid_cam 3 --dataset_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet.json
-------------------------------------------------------------


