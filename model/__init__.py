import os
from importlib import import_module
import utility

class Model :
    def __init__(self, args):
        print('Making model...')
        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args)
        #print(self.model.summary())
        if args.input_weight_path and args.mode != 'save_sv_wgt':
            self.load(args.input_weight_path)
        self.apath = os.path.join(args.base_path, 'experiment/model')

        if(args.mode == 'train'):
            path_manager = utility.Model_path_manager(args)

        self.args = args

    def load_sv_wgt(self, path):
        self.model.load_sv_wgt(path)

    def load(self, path=None):
        if(path):
            model_path = path
        elif(self.args.mode == 'train'):
            if(self.args.resume) :
                model_path = path_manager.get_resume_path()
            elif(self.args.input_weight_path) :
                model_path = self.args.input_weight_path
            else :
                model_path = self.model.get_weight_path()
        else : 
            print('model path should be given')
            exit(1)
        print('Loading model from', model_path)
        self.model.load(model_path)

    def save(self, path=''):
        if(path):
            model_path = path 
        elif(args.mode == 'train'):
            model_path = path_manager.get_save_path()
        else : 
            print('model path should be given in val or test mode')
            exit(1)
        print('Saving model to', model_path)
        self.model.save(model_path)

    def rpn_predict_batch(self, X, debug_images) :
        return self.model.rpn_predict_batch(X, debug_images)

    def ven_predict_batch(self, X, debug_images, extrins, rpn_results) :
        return self.model.ven_predict_batch(X, debug_images, extrins, rpn_results)

    def predict_batch(self, X, images, extrins, rpn_results, ven_results):
        return self.model.predict_batch(X, images, extrins, rpn_results, ven_results) 

    def train_batch(self, X, images, labels, image_paths, extrins, rpn_results, ven_results):
        return self.model.train_batch(X, images, labels, image_paths, extrins, rpn_results, ven_results)
