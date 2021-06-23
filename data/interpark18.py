import copy

class INTERPARK18:
    def __init__(self):
        self.class_mapping_without_bg = {'seagram': 0, 'pebriz': 1, 'boxtape': 2, 'cass': 3, 'detergent': 4, 'kantata': 5, 'kechap': 6, 'myzzo': 7, 'papercup': 8, 'parisredbean': 9, 'pringles': 10, 'sausage': 11, 'seoulmilk': 12, 'shrimpsnack': 13, 'sunuporange': 14, 'tunacan': 15, 'vita500': 16, 'vitajelly': 17}
        self.class_mapping = copy.deepcopy(self.class_mapping_without_bg)
        self.class_mapping['bg'] = 18
        self.width = 640
        self.height = 360
        self.num_cls = len(self.class_mapping) - 1
