from importlib import import_module
import utility

def get_dataset_info(args):
    m = import_module('data.' + args.dataset.lower())
    obj = getattr(m, args.dataset)()
    args.num_cls = obj.num_cls
    args.num_cls_with_bg = obj.num_cls + 1
    args.class_mapping = obj.class_mapping
    args.class_mapping_without_bg = obj.class_mapping_without_bg
    args.sv_num2cls_with_bg = obj.sv_class_mapping_with_bg
    args.num2cls_with_bg = list(args.class_mapping)
    args.num2cls = args.num2cls_with_bg[:-1] # without bg
    args.cls2num = {cls:i for i, cls in enumerate(args.num2cls)} # without bg
    args.width, args.height = obj.width, obj.height
    args.resized_width, args.resized_height, args.resize_ratio = utility.get_new_img_size(args.width, args.height, args.im_size)
