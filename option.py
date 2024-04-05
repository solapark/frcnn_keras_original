import datetime
import argparse 
import math
from template import get_dataset_info

parser = argparse.ArgumentParser()

#mode
parser.add_argument('--mode', type=str, default='demo', choices=['train', 'val', 'demo', 'val_models', 'save_rpn_feature', 'save_ven_feature', 'save_reid_input', 'save_sv_wgt', 'draw_json', 'val_json_json', 'write_json', 'comp_json'])
parser.add_argument('--fast_val', action="store_true")

#model
parser.add_argument('--model', type=str, default='mv_frcnn')
parser.add_argument('--base_net', type=str, default='frcnn_vgg')


parser.add_argument('--num_cam', type=int, default=3)
parser.add_argument('--num_valid_cam', type=int, default=3)
parser.add_argument('--num_nms', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=1)

#dataset
parser.add_argument("--dataset", choices = ['INTERPARK18', 'MESSYTABLE'], default='MESSYTABLE')
parser.add_argument("--dataset_path", help="Path to dataset.")
parser.add_argument("--pred_dataset_path", help="Path to dataset.")
parser.add_argument("--pred_dataset_path1", help="Path to dataset for comp json.")
parser.add_argument("--pred_dataset_path2", help="Path to dataset for comp json.")
parser.add_argument("--result_json_path", help="Path to save detection results.")
parser.add_argument("--val_start_idx", default = 1, type=int, help="start idx of model to be validated")
parser.add_argument("--val_end_idx", default = 10000, type=int, help="end idx of model to be validated")
parser.add_argument("--val_interval", default = 1, type=int, help="intervalof models to be validated")
parser.add_argument("--demo_file", default = '', help="Path to demo.")
parser.add_argument("--parser_type", help="Parser to use. One of simple or pascal_voc", default="simple")

#messytable
parser.add_argument("--messytable_img_dir", default="/data1/sap/MessyTable/images")

#augment
parser.add_argument("--augment", help="Augment. (Default=false).", action="store_true", default=False)
parser.add_argument("--hf", help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
parser.add_argument("--vf", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
parser.add_argument("--rot", help="Augment with 90 degree rotations in training. (Default=false).", action="store_true", default=False)

#log
parser.add_argument("--loss_log", default=['rpn_cls', 'rpn_regr', 'ven', 'cls_cls', 'cls_regr', 'all'], help="names of losses to log")
parser.add_argument("--print_every", default=100, type=int, help="print every iter")
parser.add_argument("--val_models_log_name", default='', type=str, help="val models log file name")

#train_spec
parser.add_argument("--num_epochs", type=int, help="Number of epochs.", default=100)
parser.add_argument("--save_interval", type=int, help="Interval of epochs to save.", default=5)
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
parser.add_argument("--reid_input_pickle_dir", default='reid_input_pickle')

parser.add_argument("--write_rpn_only", action="store_true", default=False)
parser.add_argument("--write_reid", action="store_true", default=False)
parser.add_argument("--write_is_valid", action="store_true", default=False)
parser.add_argument("--write_emb_dist", action="store_true", default=False)
parser.add_argument("--write_bg", action="store_true", default=False)

parser.add_argument("--eval_rpn_only", action="store_true", default=False)

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
#parser.add_argument('--reid_vague_emb_dist', type=float, default=.5, help='vague_embedding distance to match two pred boxes')
#parser.add_argument('--max_dist_epiline_to_box', type=float, default=.01, help='valid maximum distance from target box to epipolar line')
parser.add_argument('--max_dist_epiline_to_box', type=float, default=.05, help='valid maximum distance from target box to epipolar line')
parser.add_argument('--max_dist_epiline_cross_to_box', type=float, default=.05, help='valid maximum distance from target box to epipolar cross line')
parser.add_argument('--use_epipolar_filter', action="store_true", default=False)


#reid_gt
parser.add_argument('--reid_gt_min_overlap', type=float, default=.3, help='minimum iou to match pred box and gt box')


#classifier
parser.add_argument("--num_rois", type=int, help="Number of RoIs to process at once.", default=4)
parser.add_argument("--num_pos", type=int, help="Number of pos RoIs to process at once.", default=2)
parser.add_argument("--fix_num_pos", action="store_true", help="Number of pos RoIs is fixed to num_pos for every iterations.", default=False)
parser.add_argument('--classifier_num_input_features', type=int, default=512, help='number of input features of classifier')
parser.add_argument('--classifier_poolsize', type=int, default=7, help='pooling size')
parser.add_argument('--classifier_view_pooling', type=str, default='reduce_concat', help='how to view pooling')
parser.add_argument('--classifier_nms_thresh', type=float, default=.3)
parser.add_argument('--classifier_mv_nms_thresh', type=float, default=.3)
parser.add_argument('--classifier_inter_cls_mv_nms_thresh', type=float, default=.3)
parser.add_argument("--mv_nms", action="store_true", default=False)
parser.add_argument("--inter_cls_mv_nms", action="store_true", default=False)
parser.add_argument("--json_nms", action="store_true", default=False)
parser.add_argument("--sv_nms", action="store_true", default=False)

#classifier_gt
parser.add_argument('--classifier_std_scaling', nargs=4, default=[8.0, 8.0, 4.0, 4.0], help='scaling the standard deviation. x1, x2, y1, y2')
parser.add_argument('--classifier_max_overlap', type=float, default=.4, help='threshold for positive sample')
parser.add_argument('--classifier_min_overlap', type=float, default=0.0, help='threshold for negative sample')
parser.add_argument('--small_obj_size', type=float, default=1000., help='size of small gt')
parser.add_argument('--classifier_small_obj_max_overlap', type=float, default=.1, help='threshold for positive sample of small gt')
parser.add_argument("--fair_classifier_gt_choice", help="fair choie of gt", action="store_true", default=False)
parser.add_argument("--unique_sample", help="remove repeated samples", action="store_true", default=False)

#drawing
parser.add_argument("--draw_inst_by_inst", help="draw one inst in one image.", action="store_true", default=False)
parser.add_argument("--draw_thresh", type=float, default=0.0)
parser.add_argument("--draw_num", type=float, default=300)
parser.add_argument("--is_draw_emb_dist", action="store_true", default=False)
parser.add_argument("--is_draw_is_valid", action="store_true", default=False)

#eval_thresh
parser.add_argument('--eval_thresh', type=float, default=0)

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
