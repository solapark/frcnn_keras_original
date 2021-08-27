#train mv_interpark18
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --reset  --mode train --dataset INTERPARK18 --save_dir interpark18 --input_weight_path /data3/sap/frcnn_keras_original/model/interpark18_rpn_only.hdf5 --num_valid_cam 3 --train_path /data3/sap/frcnn_keras_original/data/INTERPARK18/train.json 

#train mv_interpark18(resume)
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --resume  --mode train --dataset INTERPARK18 --save_dir interpark18 --input_weight_path /data3/sap/frcnn_keras_original/experiment/interpark18/model/model_9.hdf5 --num_valid_cam 3 --train_path /data3/sap/frcnn_keras_original/data/INTERPARK18/train.json

#val models mv_interpark18
CUDA_VISIBLE_DEVICES='' python -m pdb main.py --mode val_models --dataset INTERPARK18 --save_dir interpark18 --num_valid_cam 3 --val_start_idx 1 --val_interval 1 --val_path /data3/sap/frcnn_keras_original/data/INTERPARK18/val.json

#train messytable
CUDA_VISIBLE_DEVICES=2 python -m pdb main.py --reset --mode train --dataset MESSYTABLE --save_dir messytable_epi --input_weight_path /data3/sap/frcnn_keras_original/model/messytable_rpn_only.hdf5 --num_valid_cam 3 --train_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar

#train messytable (freeze rpn)
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --reset --mode train --dataset MESSYTABLE --save_dir messytable_epi_freeze_rpn_debug --input_weight_path /data3/sap/frcnn_keras_original/model/messytable_rpn_only.hdf5 --num_valid_cam 3 --train_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn

#train messytable (rpn+reid only)
CUDA_VISIBLE_DEVICES=3 python -m pdb main.py --reset --mode train --dataset MESSYTABLE --save_dir messytable_rpn_reid_only --input_weight_path /data3/sap/frcnn_keras_original/model/messytable_rpn_only.hdf5 --num_valid_cam 3 --train_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn

#train messytable (rpn+reid only, ven_alph .6)
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --reset --mode train --dataset MESSYTABLE --save_dir messytable_rpn_reid_only_alpha6 --input_weight_path /data3/sap/frcnn_keras_original/model/messytable_rpn_only.hdf5 --num_valid_cam 3 --train_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn --ven_loss_alpha .6

#val messytable
CUDA_VISIBLE_DEVICES='' python -m pdb main.py --mode val_models --dataset MESSYTABLE --save_dir messytable --num_valid_cam 3 --val_start_idx 1 --val_interval 1 --val_path /data1/sap/MessyTable/labels/val.json
 
#test messytable 
CUDA_VISIBLE_DEVICES='' python -m pdb main.py  --mode test --dataset MESSYTABLE --save_dir messytable --input_weight_path /data3/sap/frcnn_keras_original/experiment/messytable/model/model_9.hdf5 --num_valid_cam 3 --test_path /data1/sap/MessyTable/labels/test.json


# train reid from shared feat
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --reset --mode train --dataset MESSYTABLE --save_dir reid_from_shared_feat --input_weight_path /data3/sap/frcnn_keras_original/model/messytable_rpn_only.hdf5 --num_valid_cam 3 --train_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn

### debug
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --reset --mode train --dataset MESSYTABLE --save_dir tmp --input_weight_path /data3/sap/frcnn_keras_original/experiment/messytable_rpn_reid_only/model/model_99.hdf5 --num_valid_cam 3 --train_path /data1/sap/MessyTable/labels/train.json --is_use_epipolar --freeze_rpn
