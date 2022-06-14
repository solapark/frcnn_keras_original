for i in `seq 526 1 1000`
do
    cd ~/frcnn_keras_original
	
    python script/csv2json.py --src_path /data3/sap/frcnn_keras/result/result-messytable_model_$i/val/log.csv --dst_path /data3/sap/frcnn_keras/result/result-messytable_model_$i/val/log.json 

    CUDA_VISIBLE_DEVICES=-1 python main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/sv_messytable_$i --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/val.json --pred_dataset_path /data3/sap/frcnn_keras/result/result-messytable_model_$i/val/log.json 

done
