import numpy as np
cimport numpy as np

from numpy import zeros as np_zeros
from . import base
from moments import central_moments_from_noncentral_sums, ith_root
from moments import Manager as MomentsManager

class Manager(base.Null):
    def __init__(self, radii=[],  nmoments=4, 
            normalize=False, *args, **kwargs):
        super(Manager, self).__init__(*args, **kwargs)
        self.radii = radii
        self.nmoments = nmoments
        self.normalize = normalize
        self.lazy_cache = {}

    def lazy_cache_get(self,k1,k2):
        try: return self.lazy_cache[k1][k2]
        except KeyError: pass
        try: return self.lazy_cache[k2][k1]
        except KeyError: return None

    def lazy_cache_set(self,k1,k2,val):
        if k2 < k1: k1,k2 = k2,k1
        if k1 not in self.lazy_cache:
            self.lazy_cache[k1] = {}
        self.lazy_cache[k1][k2] = val

    @classmethod
    def load_dict(cls, fm_info):
        obj = cls(fm_info['radii'], fm_info['nmoments'],
                    fm_info['oriented'], fm_info['normalize'])
        return obj

    def write_fm(self, json_fm={}):
        if 'feature_list' not in json_fm:
            json_fm['feature_list'] = []
        json_fm['feature_list'].append('moments')
        json_fm['moments'] = {
            'radii' : self.radii,
            'nmoments' : self.nmoments,
            'oriented' : self.oriented,
            'normalize' : self.normalize
        }
        return json_fm

    def compute_moment_sums(self, ar, idxs):
        values = ar[idxs][...,np.newaxis]
        return (values ** np.arange(self.nmoments+1)).sum(axis=0).T

    def compute_edge_features(self, g, n1, n2, cache=None):
        cached = self.lazy_cache_get(n1,n2)
        if cached != None:return cached
        single_len = self.nmoments+1
        total_len = single_len*2*len(self.radii)
        vector = np.zeros(total_len,)
        for ii, radius in enumerate(self.radii):
            #idx_sets = g.voxels_at_intersection(n1, n2, radius, as_coords=False)
            #s1a = self.compute_moment_sums(g.non_oriented_probabilities_r, idx_sets[0])
            #s2a = self.compute_moment_sums(g.non_oriented_probabilities_r, idx_sets[1])
            matrices = g.voxels_at_intersection(n1,n2,radius,as_coords=True)
            s1 = compute_moment_sums(g.probabilities, matrices[0], self.nmoments)
            s2 = compute_moment_sums(g.probabilities, matrices[1], self.nmoments)
            #if (s1 != s1a).any() or (s2 != s2a).any():
            #    print "s1a",s1a
            #    print "s2a",s2a
            #    print "s1",s1
            #    print "s2",s2
            m1 = central_moments_from_noncentral_sums(s1)
            m2 = central_moments_from_noncentral_sums(s2)
            if self.normalize: m1, m2 = map(ith_root, [m1, m2])
            md = abs(m1-m2)
            f1 = m1.ravel()[0]
            f2 = m2.ravel()[0]
            pos = ii*(2*single_len)
            vector[pos:(pos+single_len)] = np.concatenate(([f1], m1[1:].T.ravel()))
            vector[(pos+single_len):(pos+2*single_len)] = np.concatenate(([f2], m2[1:].T.ravel()))
            if not np.isfinite(vector).all():
                print "nan or inf for %d to %d" % (n1,n2)
                print "s1",s1
                print "s2",s2
                print "m1",m1
                print "m2",m2
        self.lazy_cache_set(n1,n2,vector)
        return vector

def compute_moment_sums(ar, ps, nmoments):
    if isinstance(ar, np.ndarray) and ar.ndim == 4: 
        return _compute_moment_sums(ar, ps, nmoments)
    values = ar[ps][...,np.newaxis]
    return (values ** np.arange(nmoments+1)).sum(axis=0).T

cdef _compute_moment_sums(double[:,:,:,:] ar, long[:,:] points, long nmoments):
    cdef np.ndarray[np.double_t, ndim=2] vals = np.zeros((nmoments+1,ar.shape[3]), dtype=np.double)
    cdef int power,pp,chan
    for chan in range(ar.shape[3]):
        for power in range(nmoments+1):
            for pp in range(points.shape[0]):
                vals[power,chan] += ar[points[pp,0],points[pp,1],points[pp,2],chan] ** power
    return vals 
        
