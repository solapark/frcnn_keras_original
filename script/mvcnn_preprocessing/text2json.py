import json
import argparse 
from tqdm import tqdm

if __name__ == '__main__' :
    parser=argparse.ArgumentParser()
    parser.add_argument('--src_json', default = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet.json')
    parser.add_argument('--src_txt', default = '/data3/sap/mvcnn/svdet+asnet+mvcnn.txt')
    parser.add_argument('--dst_json', default = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet+mvcnn.json')
    args = parser.parse_args()

    f = open(args.src_txt,'r')

    with open(args.src_json,'r') as before_json:
        json_data_before = json.load(before_json)

    image_list = []
    for image_name in json_data_before['scenes'].keys():
        image_list.append(image_name)

    for line in tqdm(f.readlines()):
        line = line.strip()
        scene_name = line.rsplit('-',2)[0].rsplit('/',1)[-1]
        instance_name = line.rsplit('-',1)[-1].split(',')[0]
        pred = line.split(',')[-1]
        index = image_list.index(scene_name)

        if pred == '120' :
            if str(instance_name) in json_data_before['scenes'][image_list[index]]['instance_summary'] :
                json_data_before['scenes'][image_list[index]]['instance_summary'].pop(str(instance_name))

            for cam_idx in json_data_before['scenes'][scene_name]['cameras'] :   
                if str(instance_name) in json_data_before['scenes'][scene_name]['cameras'][cam_idx]['instances'] :
                    #print(cam_idx) 
                    json_data_before['scenes'][scene_name]['cameras'][cam_idx]['instances'].pop(str(instance_name))
            continue

        json_data_before['scenes'][image_list[index]]['instance_summary'][str(instance_name)] = int(pred)
        for cam_idx in json_data_before['scenes'][scene_name]['cameras'] :   
            if str(instance_name) in json_data_before['scenes'][scene_name]['cameras'][cam_idx]['instances'] :
                json_data_before['scenes'][scene_name]['cameras'][cam_idx]['instances'][str(instance_name)]['subcls'] = int(pred)

        #before_json.seek(0)
        #json.dump(json_data_before,before_json, indent = 7)

    with open(args.dst_json,'w',encoding = 'utf-8') as after_json:
        json.dump(json_data_before,after_json,indent=4)

#print(len(image_list))
