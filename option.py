import datetime
import argparse 
import math
from template import get_dataset_info

parser = argparse.ArgumentParser()

#mode
parser.add_argument('--mode', type=str, default='test', choices=['train', 'val', 'test', 'val_models'])

#model
parser.add_argument('--model', type=str, default='frcnn')
parser.add_argument('--base_net', type=str, default='frcnn_resnet')
parser.add_argument('--mv', type=bool, default=False)


parser.add_argument('--num_cam', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=1)

#dataset
parser.add_argument("--dataset", default = 'INTERPARK18', help="Path to training data.")
parser.add_argument("--train_path", help="Path to training data.")
parser.add_argument("--val_path", help="Path to training data.")
parser.add_argument("--val_models_path", help="Path to val models.")
parser.add_argument("--test_path", help="Path to training data.")
parser.add_argument("--demo_file", default = '', help="Path to demo.")
parser.add_argument("--parser_type", help="Parser to use. One of simple or pascal_voc", default="simple")

#augment
parser.add_argument("-hf", '--use_horizontal_flips', help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
parser.add_argument("-vf", '--use_vertical_flips', help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
parser.add_argument("-rot", "--rot_90", help="Augment with 90 degree rotations in training. (Default=false).", action="store_true", default=False)

#log
parser.add_argument("--loss_log", nargs=3, default=['rpn_cls', 'rpn_grer', 'cls_cls', 'cls_regr', 'all'], help="names of losses to log")

#train_spec
parser.add_argument("--num_epochs", type=int, help="Number of epochs.", default=2000)
parser.add_argument("--epoch_length", type=int, help="iters per epoch")
parser.add_argument("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.")
parser.add_argument("--model_path", help= "model_base_name",  default="model.hdf5")
parser.add_argument("--model_load_path", help= "model load path")
parser.add_argument("--base_path", help= "base path",  default="/data3/sap/frcnn_keras_original")
parser.add_argument("--save_dir", help= "save_dir",  default=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
parser.add_argument("--reset", help="reset save_dir", action="store_true", default=False)
parser.add_argument('--resume', action="store_true", default=False, help='resume from last checkpoint')

#anchor
parser.add_argument("--anchor_box_scales", nargs=3, default=[128, 256, 512], help="anchor box scales")
parser.add_argument('--anchor_box_ratios', type=str, default=[1, 1, 1./math.sqrt(2), 2./math.sqrt(2), 2./math.sqrt(2), 1./math.sqrt(2)],help="anchor box ratios")

parser.add_argument('--im_size', type=int, default=416,
                    help='Size to resize the smallest side of the image')

#img normalize
parser.add_argument('--img_channel_mean', nargs=3, default=[103.939, 116.779, 123.68],
                    help='image channel-wise mean to subtract')
parser.add_argument('--img_scaling_factor', type=float, default=1.0,
                    help='')

#network
parser.add_argument("--network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')

#rpn
parser.add_argument('--rpn_stride', type=int, default=16,
                    help='')
parser.add_argument('--std_scaling', type=float, default=4.0,
                    help='scaling the standard deviation')
parser.add_argument('--rpn_max_overlap', type=float, default=.7,
                    help='threshold for positive sample')
parser.add_argument('--rpn_min_overlap', type=float, default=.3,
                    help='threshold for negative sample')

#classifier
parser.add_argument("--num_rois", type=int, help="Number of RoIs to process at once.", default=4)
parser.add_argument('--classifier_regr_std', nargs=4, default=[8.0, 8.0, 4.0, 4.0],
                    help='scaling the standard deviation. x1, x2, y1, y2')
parser.add_argument('--classifier_max_overlap', type=float, default=.5,
                    help='threshold for positive sample')
parser.add_argument('--classifier_min_overlap', type=float, default=.1,
                    help='threshold for negative sample')

args = parser.parse_args()

if(args.mode == 'train' and  args.resume == args.reset):
    print('options.resume(', args.resume, ') == options.reset(', args.reset, ') in train mode')
    exit(1)

args.anchor_box_ratios = [[args.anchor_box_ratios[i*2], args.anchor_box_ratios[i*2+1]] for i in range(3)]
args.num_anchors = len(args.anchor_box_scales) * len(args.anchor_box_ratios)

get_dataset_info(args)
args.class_list = list(args.class_mapping.keys())
args.base_dir = args.base_path
