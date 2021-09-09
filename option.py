import datetime
import argparse 
import math
from template import get_dataset_info

parser = argparse.ArgumentParser()

#mode
parser.add_argument('--mode', type=str, default='demo', choices=['train', 'val', 'demo', 'val_models', 'save_rpn_feature', 'save_ven_feature', 'save_sv_wgt'])
parser.add_argument('--fast_val', action="store_true")

#model
parser.add_argument('--model', type=str, default='mv_frcnn')
parser.add_argument('--base_net', type=str, default='frcnn_vgg')


parser.add_argument('--num_cam', type=int, default=3)
parser.add_argument('--num_valid_cam', type=int, default=3)
parser.add_argument('--num_nms', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=1)

#dataset
parser.add_argument("--dataset", choices = ['INTERPARK18', 'MESSYTABLE'])
parser.add_argument("--dataset_path", help="Path to dataset.")
parser.add_argument("--val_start_idx", default = 1, type=int, help="start idx of model to be validated")
parser.add_argument("--val_end_idx", default = 10000, type=int, help="end idx of model to be validated")
parser.add_argument("--val_interval", default = 1, type=int, help="intervalof models to be validated")
parser.add_argument("--demo_file", default = '', help="Path to demo.")
parser.add_argument("--parser_type", help="Parser to use. One of simple or pascal_voc", default="simple")

#messytable
parser.add_argument("--messytable_img_dir", default="/data1/sap/MessyTable/images")

#augment
parser.add_argument("-hf", '--use_horizontal_flips', help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
parser.add_argument("-vf", '--use_vertical_flips', help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
parser.add_argument("-rot", "--rot_90", help="Augment with 90 degree rotations in training. (Default=false).", action="store_true", default=False)

#log
parser.add_argument("--loss_log", default=['rpn_cls', 'rpn_regr', 'ven', 'cls_cls', 'cls_regr', 'all'], help="names of losses to log")
parser.add_argument("--print_every", default=10, type=int, help="print every iter")

#train_spec
parser.add_argument("--num_epochs", type=int, help="Number of epochs.", default=2000)
parser.add_argument("--epoch_length", type=int, help="iters per epoch")
parser.add_argument("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.")
parser.add_argument("--output_weight_path", help="Ouput path for weights of sv model.")
parser.add_argument("--model_path", help= "model_base_name",  default="model.hdf5")
parser.add_argument("--model_load_path", help= "model load path")
parser.add_argument("--base_path", help= "base path",  default="/data3/sap/frcnn_keras_original")
parser.add_argument("--save_dir", help= "save_dir",  default=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
parser.add_argument("--reset", help="reset save_dir", action="store_true", default=False)
parser.add_argument('--resume', action="store_true", default=False, help='resume from last checkpoint')
parser.add_argument("--freeze_rpn", action="store_true", default=False)
parser.add_argument("--freeze_ven", action="store_true", default=False)
parser.add_argument("--freeze_classifier", action="store_true", default=False)
parser.add_argument("--rpn_pickle_dir", default='rpn_pickle')
parser.add_argument("--ven_pickle_dir", default='ven_pickle')

#anchor
parser.add_argument("--anchor_box_scales", nargs=3, default=[64, 128, 256], help="anchor box scales")
parser.add_argument('--anchor_box_ratios', type=str, default=[1, 1, 1./math.sqrt(2), 2./math.sqrt(2), 2./math.sqrt(2), 1./math.sqrt(2)],help="anchor box ratios")

parser.add_argument('--im_size', type=int, default=600,
                    help='Size to resize the smallest side of the image')

#img normalize
parser.add_argument('--img_channel_mean', nargs=3, default=[103.939, 116.779, 123.68],
                    help='image channel-wise mean to subtract')
parser.add_argument('--img_scaling_factor', type=float, default=1.0,
                    help='')

#network
parser.add_argument("--network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')

#shared_layer
parser.add_argument('--shared_layer_channels', type=int, default=512)

#rpn
parser.add_argument('--rpn_body_channels', type=int, default=512)
parser.add_argument('--rpn_max_num_sample', type=int, default=256,
                    help='rpn maximum number of samples')
parser.add_argument('--rpn_stride', type=int, default=16,
                    help='')
parser.add_argument('--std_scaling', type=float, default=4.0,
                    help='scaling the standard deviation')
parser.add_argument('--rpn_max_overlap', type=float, default=.7,
                    help='threshold for positive sample')
parser.add_argument('--rpn_min_overlap', type=float, default=.3,
                    help='threshold for negative sample')

#ven
parser.add_argument('--num_max_ven_samples', type=int, default=16, help='size of reid embedding')
parser.add_argument('--view_invar_feature_size', type=int, default=128, help='size of reid embedding')
parser.add_argument('--ven_loss_alpha', type=float, default=.3, help='reid_loss_alpha')

#reid
parser.add_argument('--is_use_epipolar', action="store_true", default=False)
parser.add_argument('--reid_min_emb_dist', type=float, default=.5, help='minimum embedding distance to match two pred boxes')
parser.add_argument('--max_dist_epiline_to_box', type=float, default=.01, help='valid maximum distance from target box to epipolar line')
parser.add_argument('--max_dist_epiline_cross_to_box', type=float, default=.05, help='valid maximum distance from target box to epipolar cross line')


#reid_gt
parser.add_argument('--reid_gt_min_overlap', type=float, default=.3, help='minimum iou to match pred box and gt box')


#classifier
parser.add_argument("--num_rois", type=int, help="Number of RoIs to process at once.", default=4)
parser.add_argument('--classifier_num_input_features', type=int, default=512, help='number of input features of classifier')
parser.add_argument('--classifier_nms_thresh', type=float, default=.5)

#classifier_gt
parser.add_argument('--classifier_std_scaling', nargs=4, default=[8.0, 8.0, 4.0, 4.0], help='scaling the standard deviation. x1, x2, y1, y2')
parser.add_argument('--classifier_max_overlap', type=float, default=.4, help='threshold for positive sample')
parser.add_argument('--classifier_min_overlap', type=float, default=.1, help='threshold for negative sample')

args = parser.parse_args()

if(args.mode == 'train' and  args.resume == args.reset):
    print('options.resume(', args.resume, ') == options.reset(', args.reset, ') in train mode')
    exit(1)

args.rpn_std_scaling = args.std_scaling
args.anchor_box_ratios = [[args.anchor_box_ratios[i*2], args.anchor_box_ratios[i*2+1]] for i in range(3)]
args.anchor_wh = [(scale * ratio[0], scale*ratio[1]) for scale in args.anchor_box_scales for ratio in args.anchor_box_ratios]
args.num_anchors = len(args.anchor_box_scales) * len(args.anchor_box_ratios)

get_dataset_info(args)
args.grid_rows = args.resized_height//args.rpn_stride
args.grid_cols = args.resized_width//args.rpn_stride
args.class_list = list(args.class_mapping.keys())
args.base_dir = args.base_path
