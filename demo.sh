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

# save_rpn_feature(test)
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode save_rpn_feature --dataset MESSYTABLE --input_weight_path /data3/sap/frcnn_keras_original/model/sv_messytable_cam3_resume_110_model.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --rpn_pickle_dir pickle/messytable/rpn/test/sv_messytable_cam3_resume_110_model 

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

# write json_rpn_only
CUDA_VISIBLE_DEVICES=2 python -m pdb main.py --mode write_json --dataset MESSYTABLE --save_dir write_json_rpn_only --input_weight_path /data3/sap/frcnn_keras_original/experiment/mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05_model_341_fine_tunning/model/model_80.hdf5 --dataset_path /data1/sap/MessyTable/labels/test.json --result_json_path /data3/sap/frcnn_keras_original/experiment/mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05_model_341_fine_tunning/test_model_80.json --write_rpn_only --is_use_epipolar


-----------------------------------------------------------
#svdet#

# val svdet and generate csv file (log.csv)
cd ~/frcnn_keras

CUDA_VISIBLE_DEVICES=3 python -m pdb test.py  --save_name sv_messytable_cam3_resume --test_path /data1/sap/MessyTable/labels/test_3cam.txt --model_idx 110 --val --progbar --log_output

# convert svdet csv to json(log.json)
python script/csv2json.py --src_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/log.csv --dst_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/log.json

# val log.json
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/test_3cam --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --pred_dataset_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/log.json

# generate svdet.json (assign gt inst id to svdet result to eval reid performance)
python script/assign_inst_id_to_frcnn_result_past.py --gt_path /data1/sap/MessyTable/labels/test.json --src_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/log.json --dst_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet.json

# draw svdet.json
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode draw_json --dataset MESSYTABLE --save_dir drawing/svdet.json --num_valid_cam 3 --dataset_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet.json

-------------------------------------------------------------
#val model on test.josn#

# val
CUDA_VISIBLE_DEVICES=3 python -m pdb main.py --mode val --dataset MESSYTABLE --save_dir mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05_model_341_fine_tunning --dataset_path /data1/sap/MessyTable/labels/test.json --input_weight_path /data3/sap/frcnn_keras_original/experiment/mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05_model_341_fine_tunning/model/model_80.hdf5 --is_use_epipolar

# write json
CUDA_VISIBLE_DEVICES=2 python -m pdb main.py --mode write_json --dataset MESSYTABLE --save_dir mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05_model_341_fine_tunning  --input_weight_path /data3/sap/frcnn_keras_original/experiment/mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05_model_341_fine_tunning/model/model_80.hdf5 --dataset_path /data1/sap/MessyTable/labels/test.json --result_json_path /data3/sap/frcnn_keras_original/experiment/mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05_model_341_fine_tunning/test_model_80.json --is_use_epipolar 

# val_json_json
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05_model_341_fine_tunning --dataset_path /data1/sap/MessyTable/labels/test.json --pred_dataset_path /data3/sap/frcnn_keras_original/experiment/mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05_model_341_fine_tunning/test_model_80.json

-------------------------------------------------------------
###how to deal with asnet network
[train asnet]
#train asnet on cam3 messytable
CUDA_VISIBLE_DEVICES=3 python -m pdb train.py --config_dir asnet

[generate result_img_pairs]
#test asnet on test.json
cp -r asnet gt+asnet

CUDA_VISIBLE_DEVICES=3 python test.py --config_dir gt+asnet --eval_json /data1/sap/MessyTable/labels/test.json --save_features --eval_model
CUDA_VISIBLE_DEVICES=3 python test.py --config_dir gt+asnet --eval_json /data1/sap/MessyTable/labels/test.json --save_features --eval_model_esc

#test asnet on svdet.json
cp -r asnet svdet+asnet

CUDA_VISIBLE_DEVICES=0 python test.py --config_dir svdet+asnet --eval_json /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet_gt_aligned.json  --save_features  --eval_model
CUDA_VISIBLE_DEVICES=0 python test.py --config_dir svdet+asnet --eval_json /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet_gt_aligned.json --save_features  --eval_model_esc

[reid]
python script/svdet+reid.py --reid_img_pairs_path /data3/sap/Messytable/models/svdet+asnet/results_img_pairs.json --src_json_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet_gt_aligned.json --dst_json_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet.json

[majority]
python script/svdet+reid+majority.py --src /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet.json --dst /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet+majority.json

[nms]
python script/nms.py --src /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet+majority.json --dst /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet+majority+nms.json


[eval reid]
assign gt id
#majority
python script/assign_inst_id_to_frcnn_result.py --gt_path /data1/sap/MessyTable/labels/test.json --src_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet+majority.json --dst_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet+majority.json

#nms
python script/assign_inst_id_to_frcnn_result.py --gt_path /data1/sap/MessyTable/labels/test.json --src_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet+majority+nms.json --dst_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet+majority+nms.json

cp dir
#majority
cp -r /data3/sap/Messytable/models/asnet /data3/sap/Messytable/models/svdet+asnet+majority

#nms
cp -r /data3/sap/Messytable/models/asnet /data3/sap/Messytable/models/svdet+asnet+majority+nms

run test.py
#majority
CUDA_VISIBLE_DEVICES=-1 python test.py --config_dir svdet+asnet+majority --eval_json /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet+majority.json  --eval_frcnn

#nms
CUDA_VISIBLE_DEVICES=-1 python test.py --config_dir svdet+asnet+majority+nms --eval_json /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet+majority+nms.json  --eval_frcnn


[eval map]
#majority
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/svdet+asnet+majority.json --dataset_path /data1/sap/MessyTable/labels/test.json --pred_dataset_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet+majority.json

#nms
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/svdet+asnet+majority+nms.json --dataset_path /data1/sap/MessyTable/labels/test.json --pred_dataset_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet+majority+nms.json


================================
###how to deal with triplenet network
[train triplenet]
#train triplenet on cam3 messytable
CUDA_VISIBLE_DEVICES=3 python -m pdb train.py --config_dir triplenet

[generate result_img_pairs]
#test triplenet on test.json
cp -r triplenet gt+triplenet

CUDA_VISIBLE_DEVICES=3 python test.py --config_dir gt+triplenet --eval_json /data1/sap/MessyTable/labels/test.json --save_features --eval_model
CUDA_VISIBLE_DEVICES=3 python test.py --config_dir gt+triplenet --eval_json /data1/sap/MessyTable/labels/test.json --save_features --eval_model_esc

#test triplenet on svdet.json
cp -r triplenet svdet+triplenet

CUDA_VISIBLE_DEVICES=0 python test.py --config_dir svdet+triplenet --eval_json /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet_gt_aligned.json  --save_features  --eval_model
CUDA_VISIBLE_DEVICES=0 python test.py --config_dir svdet+triplenet --eval_json /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet_gt_aligned.json --save_features  --eval_model_esc

[reid]
python script/svdet+reid.py --reid_img_pairs_path /data3/sap/Messytable/models/svdet+triplenet/results_img_pairs.json --src_json_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet_gt_aligned.json --dst_json_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+triplenet.json

[majority]
python script/svdet+reid+majority.py --src /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+triplenet.json --dst /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+triplenet+majority.json

[nms]
python script/nms.py --src /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+triplenet+majority.json --dst /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+triplenet+majority+nms.json


[eval reid]
assign gt id
#majority
python script/assign_inst_id_to_frcnn_result.py --gt_path /data1/sap/MessyTable/labels/test.json --src_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+triplenet+majority.json --dst_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+triplenet+majority.json

#nms
python script/assign_inst_id_to_frcnn_result.py --gt_path /data1/sap/MessyTable/labels/test.json --src_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+triplenet+majority+nms.json --dst_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+triplenet+majority+nms.json

cp dir
#majority
cp -r /data3/sap/Messytable/models/triplenet /data3/sap/Messytable/models/svdet+triplenet+majority

#nms
cp -r /data3/sap/Messytable/models/triplenet /data3/sap/Messytable/models/svdet+triplenet+majority+nms

run test.py
#majority
CUDA_VISIBLE_DEVICES=-1 python test.py --config_dir svdet+triplenet+majority --eval_json /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+triplenet+majority.json  --eval_frcnn

#nms
CUDA_VISIBLE_DEVICES=-1 python test.py --config_dir svdet+triplenet+majority+nms --eval_json /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+triplenet+majority+nms.json  --eval_frcnn


[eval map]
#majority
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/svdet+triplenet+majority.json --dataset_path /data1/sap/MessyTable/labels/test.json --pred_dataset_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+triplenet+majority.json

#nms
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/svdet+triplenet+majority+nms.json --dataset_path /data1/sap/MessyTable/labels/test.json --pred_dataset_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+triplenet+majority+nms.json

===================================================
[draw]
#draw svdet
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode draw_json --dataset MESSYTABLE --save_dir drawing/svdet.json --num_valid_cam 3 --dataset_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet.json

#draw svdet+asnet.json
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode draw_json --dataset MESSYTABLE --save_dir drawing/svdet+asnet.json --num_valid_cam 3 --dataset_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet.json

#draw svdet+asnet+majority.json
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode draw_json --dataset MESSYTABLE --save_dir drawing/svdet+asnet+majority.json --num_valid_cam 3 --dataset_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet+majority.json

#draw svdet+asnet+majority+nms.json
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode draw_json --dataset MESSYTABLE --save_dir drawing/svdet+asnet+majority+nms.json --num_valid_cam 3 --dataset_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet+majority+nms.json

#draw svdet+triplenet.json
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode draw_json --dataset MESSYTABLE --save_dir drawing/svdet+triplenet.json --num_valid_cam 3 --dataset_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+triplenet.json

#draw svdet+triplenet+majority.json
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode draw_json --dataset MESSYTABLE --save_dir drawing/svdet+triplenet+majority.json --num_valid_cam 3 --dataset_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+triplenet+majority.json

#draw svdet+triplenet+majority+nms.json
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode draw_json --dataset MESSYTABLE --save_dir drawing/svdet+triplenet+majority+nms.json --num_valid_cam 3 --dataset_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+triplenet+majority+nms.json


----------------------------------------------------------
#220516
#comp_json

CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode comp_json --reset --dataset MESSYTABLE --save_dir comp_json/org_vs_remove_invalid_inst --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --pred_dataset_path1 /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/test_model_18.json --pred_dataset_path2 /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/test_model_18_remove_invalid_inst.json

CUDA_VISIBLE_DEVICES=2 python -m pdb main.py --mode write_json --dataset MESSYTABLE --save_dir 220516/mv_messytable_fine_tunning_from_model9  --input_weight_path  /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --result_json_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/test_model_18_remove_invalid_inst.json --is_use_epipolar 


CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/220516/svdet_1041.json --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --pred_dataset_path /data3/sap/frcnn_keras/result/result-messytable_model_1041/test/log.json

CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/220516/test_model_18.json --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --pred_dataset_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/test_model_18.json

CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/220516/test_model_18_inter_class_nms.json --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --pred_dataset_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/test_model_18_inter_class_nms.json

CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/220516/test_model_18_remove_invalid_inst.json --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --pred_dataset_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/test_model_18_remove_invalid_inst.json


CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/220516/test_model_18_remove_invalid_inst.json --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --pred_dataset_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/test_model_18_remove_invalid_inst.json


CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/220516/test_model_18_remove_invalid_inst_nms1.json --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --pred_dataset_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/test_model_18_remove_invalid_inst_nms1.json

CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/220516/test_model_18_remove_invalid_inst_inter_class_nms.json --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --pred_dataset_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/test_model_18_remove_invalid_inst_inter_class_nms.json

CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/sep_mvdet/svdet+asnet+majority.json --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --pred_dataset_path /data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet+majority.json


----------------------------------
230921 for multiview reid in transformer
#mvdet check
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode write_json --dataset MESSYTABLE --save_dir 230921/mvdet  --input_weight_path  /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --result_json_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet.json --is_use_epipolar

CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/230921/mvdet --dataset_path /data1/sap/MessyTable/labels/test.json --pred_dataset_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet.json

###assign gt inst id to rpn result
python script/assign_inst_id_to_frcnn_result.py --gt_path /data1/sap/MessyTable/labels/train.json --src_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_train.json --dst_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_train.json

### draw mvdet rpn_only
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode draw_json --dataset MESSYTABLE --save_dir drawing/mvdet_rpn_only300/train --num_valid_cam 3 --dataset_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_train.json --draw_inst_by_inst

### draw mvdet rpn_only aligned
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode draw_json --dataset MESSYTABLE --save_dir drawing/mvdet_rpn_only300_aligned/train --num_valid_cam 3 --dataset_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_train_aligned.json --draw_inst_by_inst

###align rpn with gt id (include results which don't belong to any gt)
python script/align_with_gt_id.py --gt_path /data1/sap/MessyTable/labels/train.json --src_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_train.json --dst_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_train_aligned.json 

----------------------------------
231009 save reid input to generate dataset formatted pickle for training trmvried
### 1. save reid input pickle
#train
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode save_reid_input --dataset MESSYTABLE --input_weight_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --reid_input_pickle_dir pickle/messytable/mvdet/reid_input/train
#val
CUDA_VISIBLE_DEVICES=2 python -m pdb main.py --mode save_reid_input --dataset MESSYTABLE --input_weight_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --reid_input_pickle_dir pickle/messytable/mvdet/reid_input/val
#test
CUDA_VISIBLE_DEVICES=3 python -m pdb main.py --mode save_reid_input --dataset MESSYTABLE --input_weight_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --is_use_epipolar --reid_input_pickle_dir pickle/messytable/mvdet/reid_input/test

### 2. write mvdet rpn_only
#train
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode write_json --dataset MESSYTABLE --save_dir 230921/mvdet/train --input_weight_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --dataset_path /data1/sap/MessyTable/labels/train.json --result_json_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_train.json --write_rpn_only --is_use_epipolar
#val
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode write_json --dataset MESSYTABLE --save_dir 230921/mvdet/val --input_weight_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --dataset_path /data1/sap/MessyTable/labels/val.json --result_json_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_val.json --write_rpn_only --is_use_epipolar
#test
CUDA_VISIBLE_DEVICES=1 python -m pdb main.py --mode write_json --dataset MESSYTABLE --save_dir 230921/mvdet/test --input_weight_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --dataset_path /data1/sap/MessyTable/labels/test.json --result_json_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_test.json --write_rpn_only --is_use_epipolar

### 3. align rpn with gt id. exclude results which don't belong to any gt. (assign gt class)
# gt-rpn 1-to-1 matching
#train
python script/align_with_gt_id.py --gt_path /data1/sap/MessyTable/labels/train.json --src_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_train.json --dst_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_train_aligned_core.json --save_core --assign_gt_class
#val
python script/align_with_gt_id.py --gt_path /data1/sap/MessyTable/labels/val.json --src_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_val.json --dst_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_val_aligned_core.json --save_core --assign_gt_class
#test
python script/align_with_gt_id.py --gt_path /data1/sap/MessyTable/labels/test.json --src_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_test.json --dst_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_test_aligned_core.json --save_core --assign_gt_class

# gt-rpn 1-to-N matching
#train
python script/align_with_gt_id.py --gt_path /data1/sap/MessyTable/labels/train.json --src_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_train.json --dst_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_1toN_train.json --one2N --iou_thresh .5
#val
python script/align_with_gt_id.py --gt_path /data1/sap/MessyTable/labels/val.json --src_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_val.json --dst_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_1toN_val.json --one2N --iou_thresh .5
#test
python script/align_with_gt_id.py --gt_path /data1/sap/MessyTable/labels/test.json --src_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_test.json --dst_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_1toN_test.json --one2N --iou_thresh .5


### 4. reid input to nuscenes
#train
python -m pdb script/Messytable2Nuscenes_withDLT.py --json_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_train_aligned_core.json --type train --pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/mvdet/reid_input/train  --save_dir /data3/sap/VEDet/data/Messytable/rpn --rpn
#val
python -m pdb script/Messytable2Nuscenes_withDLT.py --json_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_val_aligned_core.json --type val --pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/mvdet/reid_input/val  --save_dir /data3/sap/VEDet/data/Messytable/rpn --rpn
#test
python -m pdb script/Messytable2Nuscenes_withDLT.py --json_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_test_aligned_core.json --type test --pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/mvdet/reid_input/test  --save_dir /data3/sap/VEDet/data/Messytable/rpn --rpn


231016 save ASNet pickle to generate dataset formatted pickle for training trmvried
#train
python -m pdb script/Messytable2Nuscenes_withDLT.py --json_path /data1/sap/MessyTable/labels/train.json --type train --pickle_dir /data3/sap/VEDet/data/Messytable_ASNet/train  --save_dir /data3/sap/VEDet/data/Messytable_ASNet
#val
python -m pdb script/Messytable2Nuscenes_withDLT.py --json_path /data1/sap/MessyTable/labels/val.json --type val --pickle_dir /data3/sap/VEDet/data/Messytable_ASNet/val  --save_dir /data3/sap/VEDet/data/Messytable_ASNet
#test
python -m pdb script/Messytable2Nuscenes_withDLT.py --json_path /data1/sap/MessyTable/labels/test.json --type test --pickle_dir /data3/sap/VEDet/data/Messytable_ASNet/test  --save_dir /data3/sap/VEDet/data/Messytable_ASNet

231016 fix extrinsic parameter error in cam '5' and '6'
python -m pdb script/calribrate_cam.py --src_json_path /data1/sap/MessyTable/labels/test.json --dst_json_path /data1/sap/MessyTable/labels/test_correction.json --target_cam_id 5 
python -m pdb script/Messytable2Nuscenes_withDLT.py --json_path /data1/sap/MessyTable/labels/test_correction.json --type test_correction --pickle_dir /data3/sap/VEDet/data/Messytable_ASNet/test  --save_dir /data3/sap/VEDet/data/Messytable_ASNet --debug

231024 assign multiple rpn to a single gt
### 3. align multiple rpn with gt id (exclude results which don't belong to any gt) (assign gt class)
#train
python script/align_with_gt_id.py --gt_path /data1/sap/MessyTable/labels/train.json --src_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_train.json --dst_path /data3/sap/frcnn_keras_original/experiment/231024/mvdet/mvdet_multiple_rpn_only_train_aligned_core.json --save_core --assign_gt_class --save_gt_pos --iou_thresh .3 --multiple_pred

231227 save rpn and gt at the same time. multiple rpn for a gt
#train
python -m pdb script/Messytable2Nuscenes_withDLT.py --json_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_1toN_train.json --type train --pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/mvdet/reid_input/train  --save_dir /data3/sap/VEDet/data/Messytable/rpn1toN --rpn1toN
#val
python -m pdb script/Messytable2Nuscenes_withDLT.py --json_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_1toN_val.json --type val --pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/mvdet/reid_input/val  --save_dir /data3/sap/VEDet/data/Messytable/rpn1toN --rpn1toN
#test
python -m pdb script/Messytable2Nuscenes_withDLT.py --json_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_1toN_test.json --type test --pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/mvdet/reid_input/test  --save_dir /data3/sap/VEDet/data/Messytable/rpn1toN --rpn1toN



### 5. save reid output
#train
python -m pdb tools/test.py projects/configs/tmvreid_messytable_rpn17.py work_dirs/tmvreid_messytable_rpn17/epoch_200.pth --save_reid_pickle
#

### 6. train classifier
#trmreid rpn17
CUDA_VISIBLE_DEVICES=3 python -m pdb main.py --reset --mode train --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn17_classifier_only --input_weight_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir pickle/messytable/rpn/train/sv_messytable_cam3_resume_110_model --freeze_ven --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn17/reid_output/train --num_epochs 300
#trmreid rpn18
CUDA_VISIBLE_DEVICES=3 python -m pdb main.py --reset --mode train --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn18_classifier_only --input_weight_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir pickle/messytable/rpn/train/sv_messytable_cam3_resume_110_model --freeze_ven --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn18/reid_output/train --num_epochs 100 --save_interval 5
#trmreid rpn18 + fair_classifier_gt_choice
CUDA_VISIBLE_DEVICES=3 python -m pdb main.py --reset --mode train --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn18_classifier_only_fair_gt --input_weight_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir pickle/messytable/rpn/train/sv_messytable_cam3_resume_110_model --freeze_ven --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn18/reid_output/train --num_epochs 100 --save_interval 5 --fair_classifier_gt_choice
#trmreid rpn23
CUDA_VISIBLE_DEVICES=3 python -m pdb main.py --reset --mode train --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn23_classifier_only --input_weight_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir pickle/messytable/rpn/train/sv_messytable_cam3_resume_110_model --freeze_ven --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn23/reid_output/train --num_epochs 100 --save_interval 5 
#trmreid rpn34_new
CUDA_VISIBLE_DEVICES=3 python -m pdb main.py --reset --mode train --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn34_new_classifier_only --input_weight_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir pickle/messytable/rpn/train/sv_messytable_cam3_resume_110_model --freeze_ven --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn34_new/reid_output/train --num_epochs 100 --save_interval 5 --fair_classifier_gt_choice
#trmreid rpn38
CUDA_VISIBLE_DEVICES=3 python -m pdb main.py --reset --mode train --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn38_classifier_only --input_weight_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir pickle/messytable/rpn/train/sv_messytable_cam3_resume_110_model --freeze_ven --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn38/reid_output/train --num_epochs 100 --save_interval 5 --fair_classifier_gt_choice
#trmreid rpn43
CUDA_VISIBLE_DEVICES=3 python -m pdb main.py --reset --mode train --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn43_classifier_only --input_weight_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir pickle/messytable/rpn/train/sv_messytable_cam3_resume_110_model --freeze_ven --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn43/reid_output/train --num_epochs 100 --save_interval 5 --fair_classifier_gt_choice
#trmreid rpn45
CUDA_VISIBLE_DEVICES=3 python main.py --reset --mode train --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn45_classifier_only --input_weight_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir pickle/messytable/rpn/train/sv_messytable_cam3_resume_110_model --freeze_ven --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn45/reid_output/train --num_epochs 100 --save_interval 5 --fair_classifier_gt_choice
#trmreid rpn47
CUDA_VISIBLE_DEVICES=2 python main.py --reset --mode train --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn47_classifier_only --input_weight_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir pickle/messytable/rpn/train/sv_messytable_cam3_resume_110_model --freeze_ven --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn47/reid_output/train --num_epochs 100 --save_interval 5 --fair_classifier_gt_choice
#trmreid rpn49
CUDA_VISIBLE_DEVICES=2 python main.py --reset --mode train --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn49_classifier_only --input_weight_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir pickle/messytable/rpn/train/sv_messytable_cam3_resume_110_model --freeze_ven --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn49/reid_output/train --num_epochs 100 --save_interval 5 --fair_classifier_gt_choice
#trmreid rpn50
CUDA_VISIBLE_DEVICES=2 python main.py --reset --mode train --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn50_classifier_only --input_weight_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir pickle/messytable/rpn/train/sv_messytable_cam3_resume_110_model --freeze_ven --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn50/reid_output/train --num_epochs 100 --save_interval 5 --fair_classifier_gt_choice
#trmreid rpn59
CUDA_VISIBLE_DEVICES=2 python main.py --reset --mode train --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn59_classifier_only --input_weight_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir pickle/messytable/rpn/train/sv_messytable_cam3_resume_110_model --freeze_ven --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn59/reid_output/train --num_epochs 100 --save_interval 5 --fair_classifier_gt_choice
#trmreid rpn59 th.1
CUDA_VISIBLE_DEVICES=2 python main.py --reset --mode train --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn59_th.1_classifier_only --input_weight_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir pickle/messytable/rpn/train/sv_messytable_cam3_resume_110_model --freeze_ven --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn59_th.1/reid_output/train --num_epochs 100 --save_interval 5 --fair_classifier_gt_choice
#trmreid rpn68 ep20 th.1
CUDA_VISIBLE_DEVICES=2 python main.py --reset --mode train --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn68_ep20_th.1_classifier_only --input_weight_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir pickle/messytable/rpn/train/sv_messytable_cam3_resume_110_model --freeze_ven --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn68_ep20_th.1/reid_output/train --num_epochs 100 --save_interval 5 --fair_classifier_gt_choice
#trmreid rpn68 ep20 th.01
CUDA_VISIBLE_DEVICES=2 python main.py --reset --mode train --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn68_ep20_th.01_classifier_only --input_weight_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir pickle/messytable/rpn/train/sv_messytable_cam3_resume_110_model --freeze_ven --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn68_ep20_th.01/reid_output/train --num_epochs 100 --save_interval 5 --fair_classifier_gt_choice
#trmreid rpn75
CUDA_VISIBLE_DEVICES=2 python main.py --reset --mode train --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn75_classifier_only --input_weight_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir pickle/messytable/rpn/train/sv_messytable_cam3_resume_110_model --freeze_ven --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn75/reid_output/train --num_epochs 100 --save_interval 5 --fair_classifier_gt_choice
#trmreid rpn75 nms100
CUDA_VISIBLE_DEVICES=2 python main.py --reset --mode train --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn75_nms100_classifier_only --input_weight_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir pickle/messytable/rpn/train/sv_messytable_cam3_resume_110_model --freeze_ven --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn75_nms100/reid_output/train --num_epochs 100 --save_interval 5 --fair_classifier_gt_choice


### 6. val classifier
#trmreid rpn17
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn17_classifier_only --val_start_idx 10 --val_interval 5 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/val/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn17/reid_output/val
#trmreid rpn18
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn18_classifier_only --val_start_idx 10 --val_interval 5 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/val/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn18/reid_output/val
#trmreid rpn18+ fair_classifier_gt_choice
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn18_classifier_only_fair_gt --val_start_idx 10 --val_interval 5 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/val/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn18/reid_output/val
#trmreid rpn23
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn23_classifier_only --val_start_idx 10 --val_interval 5 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/val/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn23/reid_output/val
#trmreid rpn34_new
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn34_new_classifier_only --val_start_idx 10 --val_interval 5 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/val/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn34_new/reid_output/val
#trmreid rpn34_new_nms2
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn34_new_classifier_only --val_start_idx 10 --val_interval 5 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/val/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn34_new/reid_output/val
#trmreid rpn38
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn38_classifier_only --val_start_idx 10 --val_interval 5 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/val/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn38/reid_output/val
#trmreid rpn43
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn43_classifier_only --val_start_idx 10 --val_interval 5 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/val/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn43/reid_output/val
#trmreid rpn45
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn45_classifier_only --val_start_idx 10 --val_interval 5 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/val/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn45/reid_output/val
#trmreid rpn47
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn47_classifier_only --val_start_idx 10 --val_interval 5 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/val/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn47/reid_output/val
#trmreid rpn49
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn49_classifier_only --val_start_idx 10 --val_interval 5 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/val/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn49/reid_output/val
#trmreid rpn50
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn50_classifier_only --val_start_idx 10 --val_interval 5 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/val/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn50/reid_output/val
#trmreid rpn59
#val
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn59_classifier_only --val_start_idx 10 --val_interval 5 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/val/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn59/reid_output/val
#test
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn59_classifier_only --val_start_idx 10 --val_interval 5 --dataset_path /data1/sap/MessyTable/labels/test.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/val/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn59/reid_output/test
#trmreid rpn59_th.1
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn59_th.1_classifier_only --val_start_idx 10 --val_interval 5 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/val/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn59_th.1/reid_output/val
#trmreid rpn59_th.1 mv nms + ep filter
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn59_th.1_classifier_only --val_start_idx 10 --val_interval 5 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/val/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn59_th.1/reid_output/val --mv_nms --use_epipolar_filter --max_dist_epiline_to_box .09
CUDA_VISIBLE_DEVICES=1 python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn59_th.1_classifier_only --val_start_idx 10 --val_interval 5 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/val/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn59_th.1/reid_output/val --mv_nms --use_epipolar_filter --max_dist_epiline_to_box .08
CUDA_VISIBLE_DEVICES=2 python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn59_th.1_classifier_only --val_start_idx 10 --val_interval 5 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/val/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn59_th.1/reid_output/val --mv_nms --use_epipolar_filter --max_dist_epiline_to_box 1.

#trmreid rpn68 ep20 th.1
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn68_ep20_th.1_classifier_only --val_start_idx 10 --val_interval 5 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/val/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn68_ep20_th.1/reid_output/val
#trmreid rpn68 ep20 th.01
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn68_ep20_th.01_classifier_only --val_start_idx 10 --val_interval 5 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/val/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn68_ep20_th.01/reid_output/val
#trmreid rpn75
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn75_classifier_only --val_start_idx 10 --val_interval 5 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/val/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn75/reid_output/val
#trmreid rpn75_nms100
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn75_nms100_classifier_only --val_start_idx 10 --val_interval 5 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/val/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn75_nms100/reid_output/val

# write json
CUDA_VISIBLE_DEVICES=1 python -m pdb main.py --mode write_json --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn59_classifier_only  --input_weight_path /data3/sap/frcnn_keras_original/experiment/tmvreid_messytable_rpn59_classifier_only/model/model_35.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --result_json_path /data3/sap/frcnn_keras_original/experiment/tmvreid_messytable_rpn59_classifier_only/model_35.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/test/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn59/reid_output/test
CUDA_VISIBLE_DEVICES=1 python -m pdb main.py --mode write_json --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn59_classifier_only  --input_weight_path /data3/sap/frcnn_keras_original/experiment/tmvreid_messytable_rpn59_classifier_only/model/model_35.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --result_json_path /data3/sap/frcnn_keras_original/experiment/tmvreid_messytable_rpn59_classifier_only/model_35.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/test/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn59/reid_output/test
#mv_nms
CUDA_VISIBLE_DEVICES=1 python -m pdb main.py --mode write_json --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn59_classifier_only  --input_weight_path /data3/sap/frcnn_keras_original/experiment/tmvreid_messytable_rpn59_classifier_only/model/model_35.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --result_json_path /data3/sap/frcnn_keras_original/experiment/tmvreid_messytable_rpn59_classifier_only/model_35_mv_nms.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/test/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn59/reid_output/test --mv_nms
#mv_nms+epi filter
CUDA_VISIBLE_DEVICES=1 python -m pdb main.py --mode write_json --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn59_classifier_only  --input_weight_path /data3/sap/frcnn_keras_original/experiment/tmvreid_messytable_rpn59_classifier_only/model/model_35.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --result_json_path /data3/sap/frcnn_keras_original/experiment/tmvreid_messytable_rpn59_classifier_only/model_35_mv_nms_epipolar_filter.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/test/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn59/reid_output/test --mv_nms --use_epipolar_filter --max_dist_epiline_to_box .08
CUDA_VISIBLE_DEVICES=1 python -m pdb main.py --mode write_json --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn59_classifier_only  --input_weight_path /data3/sap/frcnn_keras_original/experiment/tmvreid_messytable_rpn59_classifier_only/model/model_35.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --result_json_path /data3/sap/frcnn_keras_original/experiment/tmvreid_messytable_rpn59_classifier_only/model_35_mv_nms_epipolar_filter.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/test/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn59/reid_output/test --mv_nms --use_epipolar_filter --max_dist_epiline_to_box .07
CUDA_VISIBLE_DEVICES=1 python -m pdb main.py --mode write_json --dataset MESSYTABLE --save_dir tmvreid_messytable_rpn59_th.1_classifier_only  --input_weight_path /data3/sap/frcnn_keras_original/experiment/tmvreid_messytable_rpn59_th.1_classifier_only/model/model_85.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --result_json_path /data3/sap/frcnn_keras_original/experiment/tmvreid_messytable_rpn59_th.1_classifier_only/model_85_mv_nms_epipolar_filter.08.json --is_use_epipolar --freeze_rpn --freeze_ven --rpn_pickle_dir pickle/messytable/rpn/test/sv_messytable_cam3_resume_110_model --ven_pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/tmvreid_messytable_rpn59/reid_output/test --mv_nms --use_epipolar_filter --max_dist_epiline_to_box .08

#val_json_json
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/tmvreid_messytable_rpn59_classifier_only --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --pred_dataset_path /data3/sap/frcnn_keras_original/experiment/tmvreid_messytable_rpn59_classifier_only/model_35_no_nms.json
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/tmvreid_messytable_rpn59_classifier_only --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --pred_dataset_path /data3/sap/frcnn_keras_original/experiment/tmvreid_messytable_rpn59_classifier_only/model_35_nms2.json
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/tmvreid_messytable_rpn59_classifier_only --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --pred_dataset_path /data3/sap/frcnn_keras_original/experiment/tmvreid_messytable_rpn59_classifier_only/model_35_mv_nms.json
#mv_nms+epi filter
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/tmvreid_messytable_rpn59_classifier_only --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/test.json --pred_dataset_path /data3/sap/frcnn_keras_original/experiment/tmvreid_messytable_rpn59_classifier_only/model_35_mv_nms_epipolar_filter.json

#draw json
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode draw_json --dataset MESSYTABLE --save_dir drawing/tmvreid_messytable_rpn59_classifier_only --num_valid_cam 3 --dataset_path /data3/sap/frcnn_keras_original/experiment/tmvreid_messytable_rpn59_classifier_only/model_35.json --draw_inst_by_inst
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode draw_json --dataset MESSYTABLE --save_dir drawing/tmvreid_messytable_rpn59_classifier_only --num_valid_cam 3 --dataset_path /data3/sap/frcnn_keras_original/experiment/tmvreid_messytable_rpn59_classifier_only/model_35_mv_nms.json --draw_inst_by_inst
#mv_nms+epi filter
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode draw_json --dataset MESSYTABLE --save_dir drawing/tmvreid_messytable_rpn59_th.1_model_85_mv_nms_epipolar_filter.08 --num_valid_cam 3 --dataset_path /data3/sap/frcnn_keras_original/experiment/tmvreid_messytable_rpn59_th.1_classifier_only/model_85_mv_nms_epipolar_filter.08.json --draw_inst_by_inst

