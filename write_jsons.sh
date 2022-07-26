for i in `seq 3 1 9`
do
    cd ~/frcnn_keras_original
	
    CUDA_VISIBLE_DEVICES=3 python main.py --mode write_json --dataset MESSYTABLE --save_dir 220516/mv_messytable_fine_tunning_from_model9/val  --input_weight_path  /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/model/model_$i.hdf5 --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/val.json --result_json_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/val/model_$i.json --is_use_epipolar 

    CUDA_VISIBLE_DEVICES=-1 python main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/220516/mv_messytable_fine_tunning_from_model9/val/MODA/model_$i --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/val.json --pred_dataset_path /data3/sap/frcnn_keras_original/experiment/220516/mv_messytable_fine_tunning_from_model9/val/model_$i.json

    done
