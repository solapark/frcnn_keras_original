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
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --reset --mode train --dataset MESSYTABLE --save_dir mv_messytable_reid_only --input_weight_path /data3/sap/frcnn_keras_original/model/messytable_rpn_only.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir rpn_pickle --freeze_classifier

# train classifier
CUDA_VISIBLE_DEVICES=3 python -m pdb main.py --reset --mode train --dataset MESSYTABLE --save_dir mv_messytable_classifier_only_real --input_weight_path /data3/sap/frcnn_keras_original/experiment/mv_messytable_reid_only/model/model_67.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir rpn_pickle --freeze_ven --ven_pickle_dir ven_pickle_train

# fine tuning
CUDA_VISIBLE_DEVICES=2 python -m pdb main.py --reset --mode train --dataset MESSYTABLE --save_dir mv_messytable_fine_tuning --input_weight_path /data3/sap/frcnn_keras_original/experiment/mv_messytable_classifier_only_real/model/model_151.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar

# save_rpn_feature
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode save_rpn_feature --dataset MESSYTABLE --save_dir tmp --input_weight_path /data3/sap/frcnn_keras_original/model/messytable_rpn_only.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --rpn_pickle_dir rpn_pickle 

# save_ven_feature
#CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode save_ven_feature --dataset MESSYTABLE --input_weight_path /data3/sap/frcnn_keras_original/experiment/messytable_rpn_reid_only_alpha6/model/model_248.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir rpn_pickle --ven_pickle_dir ven_pickle
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --mode save_ven_feature --dataset MESSYTABLE --input_weight_path /data3/sap/frcnn_keras_original/experiment/messytable_rpn_reid_only_alpha6/model/model_248.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir rpn_pickle_val --ven_pickle_dir ven_pickle_val

#messytable demo classifier only
CUDA_VISIBLE_DEVICES=2 python -m pdb main.py --mode demo --dataset MESSYTABLE --save_dir mv_messytable_classifier_only_real --input_weight_path /data3/sap/frcnn_keras_original/experiment/mv_messytable_classifier_only_real/model/model_151.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/val.json --is_use_epipolar --freeze_rpn --rpn_pickle_dir rpn_pickle_val --freeze_ven --ven_pickle_dir ven_pickle_val --classifier_nms_thresh .3

#save_sv_wgt 
CUDA_VISIBLE_DEVICES=2 python -m pdb main.py --mode save_sv_wgt  --dataset MESSYTABLE --input_weight_path /data3/sap/frcnn_keras/model/sv_messytable_cam3_best/sv_messytable_cam3_resume_110_model.hdf5 --output_weight_path /data3/sap/frcnn_keras_original/model/sv_messytable_cam3_resume_110_model.hdf5 --num_valid_cam 3

