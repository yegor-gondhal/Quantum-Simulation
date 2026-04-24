import numpy as np

class SimParams:
    def __init__(self, filepath):
        data = np.load(filepath)
        for key in data:
            setattr(self, key, data[key])
