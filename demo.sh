CUDA_VISIBLE_DEVICES=3 python -m pdb main.py --reset --mode train --save_dir mv_messytable_valid_cam3 --input_weight_path /data3/sap/frcnn_keras_original/experiment/sv_messytable_original/model/model_frcnn_0639.hdf5

CUDA_VISIBLE_DEVICES=3 python -m pdb main.py --reset --mode train --save_dir mv_interpark2 --input_weight_path /data3/sap/frcnn_keras_original/experiment/sv_messytable_original/model/model_frcnn_0639.hdf5 --print_every 1 --num_cam 3 

CUDA_VISIBLE_DEVICES=3 python -m pdb main.py --reset --mode train --save_dir mv_messytable_valid_cam3 --input_weight_path /data3/sap/frcnn_keras_original/experiment/sv_messytable_original/model/model_frcnn_0639.hdf5 --num_valid_cam 3 --train_path /data1/sap/MessyTable/labels/train.json
