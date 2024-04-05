import os

def find_max_performance(file_path):
    # Dictionary to store max performance for each metric
    max_performance = {
        "MODA": {"epoch": 0, "value": [0,0,0], "perform": []},
        "MODP": {"epoch": 0, "value": [0,0,0], "perform": []},
        "F1": {"epoch": 0, "value": [0,0,0], "perform": []}
    }

    # Iterate over log files in the folder
    with open(file_path, "r") as file:
        model = ""
        epoch = 0
        for line in file:
            if line.startswith("model"):
                model = line.strip().split(" : ")[1]
                epoch = int(model.split("_")[-1].split(".")[0])
            elif line.startswith("ALL"):
                _, moda, modp, f1, recall, precision = line.strip().split()
                moda, modp, f1, recall, precision = map(float, [moda, modp, f1, recall, precision])
                perform = [moda, modp, f1, recall, precision]
                moda_value = [moda, f1, modp]
                f1_value = [f1, moda, modp]
                modp_value = [modp, f1, moda]
                if moda_value > max_performance["MODA"]["value"]:
                    max_performance["MODA"]["epoch"] = epoch
                    max_performance["MODA"]["value"] = moda_value
                    max_performance["MODA"]["perform"] = perform
                if modp_value > max_performance["MODP"]["value"]:
                    max_performance["MODP"]["epoch"] = epoch
                    max_performance["MODP"]["value"] = modp_value
                    max_performance["MODP"]["perform"] = perform
                if f1_value > max_performance["F1"]["value"]:
                    max_performance["F1"]["epoch"] = epoch
                    max_performance["F1"]["value"] = f1_value
                    max_performance["F1"]["perform"] = perform
    return max_performance

root_path = '/data3/sap/frcnn_keras_original/experiment'
folders = [
    "tmvreid_messytable_rpn59_th.1_pos3_classifier_only",
    "tmvreid_messytable_rpn59_th.1_roi64_pos48_classifier_only",
    "tmvreid_messytable_rpn59_th.1_unique_sample_classifier_only",
    "tmvreid_messytable_rpn59_th.1_roi64_pos48+fix_num_pos+unique_sample_classifier_only",
    "tmvreid_messytable_rpn59_th.1_roi32_pos24+fix_num_pos+unique_sample_classifier_only",
    "tmvreid_messytable_rpn59_th.1_roi16_pos12+fix_num_pos+unique_sample_classifier_only",
    "tmvreid_messytable_rpn59_th.1_roi8_pos6+fix_num_pos+unique_sample_classifier_only",
    "tmvreid_messytable_rpn59_th.1_roi64_pos48+fix_num_pos+unique_sample+gtmxot.2_classifier_only",
    "tmvreid_messytable_rpn59_th.1_roi64_pos48+fix_num_pos+unique_sample+gtmxot.3_classifier_only",
    "tmvreid_messytable_rpn59_th.1_roi32_pos24+fix_num_pos+unique_sample+gtmxot.3_classifier_only",
    "tmvreid_messytable_rpn59_th.1_roi32_pos24+fix_num_pos+unique_sample+small16ot.1_classifier_only",
    "tmvreid_messytable_rpn59_th.1_roi32_pos24+fix_num_pos+unique_sample+small16ot.15_classifier_only",
    "tmvreid_messytable_rpn59_th.1_roi32_pos24+fix_num_pos+unique_sample+small14ot.1_classifier_only",
    "tmvreid_messytable_rpn59_th.1_roi32_pos24+fix_num_pos+unique_sample+small14ot.15_classifier_only",
]

folders = [os.path.join(root_path, folder) for folder in folders]
for folder in folders:
    for filename in sorted(os.listdir(folder)):
        if filename.startswith("log_val_models") and filename.endswith(".txt"):
            file_path = os.path.join(folder, filename)
            max_performance = find_max_performance(file_path)
            print(folder.split('/')[-1], filename)
            print("Max MODA Model: ", max_performance["MODA"]["epoch"]) 
            print(' '.join(map(str, max_performance["MODA"]["perform"])))
            print("Max F1 Model: ", max_performance["F1"]["epoch"]) 
            print(' '.join(map(str, max_performance["F1"]["perform"])))
            print("Max MODP Model: ", max_performance["MODP"]["epoch"]) 
            print(' '.join(map(str, max_performance["MODP"]["perform"])))
            print()
