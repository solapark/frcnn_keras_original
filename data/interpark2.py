class INTERPARK2:
    def __init__(self):
        self.class_mapping = {'myzzo': 0, 'tunacan': 1}
        self.width = 640
        self.height = 360
        self.num_cls = len(self.class_mapping) - 1
