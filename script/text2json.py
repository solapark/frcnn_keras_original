import json

src_json = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/reid.json'
src_txt = '/data3/sjyang/MVCNN/reid_result.txt'
dst_json ='/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet+mvcnn.json'

f = open(src_txt,'r')

with open(src_json,'r') as before_json:
    json_data_before = json.load(before_json)

image_list = []
for image_name in json_data_before['scenes'].keys():
    image_list.append(image_name)

for line in f.readlines():
    line = line.strip()
    scene_name = line.rsplit('-',2)[0].rsplit('/',1)[-1]
    instance_name = line.rsplit('-',1)[-1].split(',')[0]
    pred = line.split(',')[-1]
    index = image_list.index(scene_name)
    json_data_before['scenes'][image_list[index]]['instance_summary'][str(instance_name)] = int(pred)
    for cam_idx in json_data_before['scenes'][scene_name]['cameras'] :   
        if str(instance_name) in json_data_before['scenes'][scene_name]['cameras'][cam_idx]['instances'] :
            json_data_before['scenes'][scene_name]['cameras'][cam_idx]['instances'][str(instance_name)]['subcls'] = int(pred)

    #before_json.seek(0)
    #json.dump(json_data_before,before_json, indent = 7)

with open(dst_json,'w',encoding = 'utf-8') as after_json:
    json.dump(json_data_before,after_json,indent=4)



#print(len(image_list))
