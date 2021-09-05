import math
import utility

class PICKLE_DATALOADER:
    def __init__(self, args, img_path_list, pickle_dir):        
        self.pickle_saver = utility.Pickle_result_saver(args, pickle_dir)
        self.path_list = [self.pickle_saver.get_general_file_name(img_paths) for img_paths in img_path_list]
        self.batch_size = args.batch_size

        del self.pickle_saver

    def __len__(self):
        return math.ceil(len(self.path_list) / self.batch_size)

    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch = [utility.pickle_load(self.path_list[i]) for i in indices] 
        return batch

    def set_indices(self, indices):
        self.indices = indices
