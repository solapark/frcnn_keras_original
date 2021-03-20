import math

class LABEL_DATALOADER:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch = [self.data[i] for i in indices] 
        return batch

    def set_indices(self, indices):
        self.indices = indices
