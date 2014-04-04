import numpy as np
cimport numpy as np

from . import base

class Manager(base.Null):
    def __init__(self, *args, **kwargs):
        super(Manager, self).__init__()

    @classmethod
    def load_dict(cls, fm_info):
        obj = cls()
        return obj

    def write_fm(self, json_fm={}):
        if 'feature_list' not in json_fm:
            json_fm['feature_list'] = []
        json_fm['feature_list'].append('stage')
        json_fm['stage'] = {}
        return json_fm

    def create_node_cache(self, g, n):
        return np.array([1])

    def update_node_cache(self, g, n1, n2, dst, src):
        dst += src

    def compute_node_features(self, g, n, cache=None):
        if cache is None: 
            cache = g.node[n][self.default_cache]
        return cache

    def compute_difference_features(self,g, n1, n2, cache1=None, cache2=None):
        if cache1 is None:
            cache1 = g.node[n1][self.default_cache]
        if cache2 is None:
            cache2 = g.node[n2][self.default_cache]
        return cache1-cache2 
