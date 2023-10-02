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

# write mvdet rpn_only
#train
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode write_json --dataset MESSYTABLE --save_dir 230921/mvdet/train --input_weight_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --dataset_path /data1/sap/MessyTable/labels/train.json --result_json_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_train.json --write_rpn_only --is_use_epipolar

#assign gt inst id to rpn result
python script/assign_inst_id_to_frcnn_result.py --gt_path /data1/sap/MessyTable/labels/train.json --src_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_train.json --dst_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_train.json

#align rpn with gt id (include results which don't belong to any gt)
python script/align_with_gt_id.py --gt_path /data1/sap/MessyTable/labels/train.json --src_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_train.json --dst_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_train_aligned.json 

#align rpn with gt id (exclude results which don't belong to any gt) (assign gt class)
python script/align_with_gt_id.py --gt_path /data1/sap/MessyTable/labels/train.json --src_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_train.json --dst_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_train_aligned_core.json --save_core --assign_gt_class

# draw mvdet rpn_only
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode draw_json --dataset MESSYTABLE --save_dir drawing/mvdet_rpn_only300/train --num_valid_cam 3 --dataset_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_train.json --draw_inst_by_inst

# draw mvdet rpn_only aligned
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode draw_json --dataset MESSYTABLE --save_dir drawing/mvdet_rpn_only300_aligned/train --num_valid_cam 3 --dataset_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_train_aligned.json --draw_inst_by_inst

#val
#test

# save reid input pickle
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode save_reid_feature --dataset MESSYTABLE --input_weight_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_18.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --reid_input_pickle_dir pickle/messytable/mvdet/reid_input/train

# reid input to nuscenes
python -m pdb script/Messytable2Nuscenes_withDLT.py --json_path /data3/sap/frcnn_keras_original/experiment/230921/mvdet/mvdet_rpn_only_train_aligned_core.json --type train --pickle_dir /data3/sap/frcnn_keras_original/pickle/messytable/mvdet/reid_input/train  --save_dir /data3/sap/VEDet/data/Messytable/rpn




