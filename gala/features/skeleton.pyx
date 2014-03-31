import numpy as np
cimport numpy as np
from . import base

class Manager(base.Null):
    def __init__(self, z_resolution_factor, *args, **kwargs):
        super(Manager, self).__init__()
        self.z_resolution_factor = z_resolution_factor
        self.null_feature_vector = np.ones(8) * -1

    @classmethod
    def load_dict(cls, fm_info):
        obj = cls(fm_info['z_resolution_factor'])
        return obj

    def write_fm(self, json_fm={}):
        if 'feature_list' not in json_fm:
            json_fm['feature_list'] = []
        json_fm['feature_list'].append('skeleton')
        json_fm['contact'] = {
            'z_resolution_factor': self.z_resolution_factor
        }
        return json_fm

    def create_node_cache(self, g, n):
        points = self.points_from_idxs(list(g.node[n]['extent']), 
                                            g.probabilities.shape)
        return compute_flat_centroids(points, 0)

    def points_from_idxs(self, idxs, cube_shape):
        if len(cube_shape) == 3:
            zs, ys, xs = np.unravel_index(idxs, cube_shape)
        else:
            zs, ys, xs, cs = np.unravel_index(idxs, cube_shape)
        unsquished_zs = zs * self.z_resolution_factor
        return np.concatenate((unsquished_zs[:,np.newaxis],
                ys[:,np.newaxis],xs[:,np.newaxis]), axis=1).astype(np.integer)

    def compute_difference_features(self, g, n1, n2, n1_centroids=None, n2_centroids=None):
        if n1_centroids is None: n1_centroids = g.node[n1][self.default_cache]
        if n2_centroids is None: n2_centroids = g.node[n2][self.default_cache]
        if len(n1_centroids) < 2 or len(n2_centroids) < 2:
            return self.null_feature_vector
        n1_closest_centroid, n2_closest_centroid = closest_pair(n1_centroids,
                                                                 n2_centroids)
        if len(n1_closest_centroid) < 1 or len(n2_closest_centroid) < 1:
            return self.null_feature_vector
        n1_next_closest = closest_point(n1_closest_centroid, n1_centroids)
        n2_next_closest = closest_point(n2_closest_centroid, n2_centroids)
        if len(n1_next_closest) < 1 or len(n2_next_closest) < 1:
            return self.null_feature_vector
        return compute_feature_vector(n1_closest_centroid, n2_closest_centroid,
                                    n1_next_closest, n2_next_closest)

    def update_node_cache(self, g, n1, n2, dst, src):
        dst += src


def compute_flat_centroids(points, dim):
    zs = []
    centroids = []
    for pp in range(points.shape[0]):
        if points[pp,dim] in zs: continue
        zs.append(points[pp,dim])
        count = 0
        for qq in range(points.shape[0]):
            if points[qq,dim] == points[pp,dim]: count += 1
        relevant_points = np.zeros((count, points.shape[1]))
        rr = 0
        for qq in range(points.shape[0]):
            if points[qq,dim] == points[pp,dim]:
                relevant_points[rr, :] = points[qq, :]
                rr += 1
        if rr+1 < count: 
            raise IndexError("Did not compute centroid correctly for pp=%d on %s" % (
                    pp, str(points)))
        centroids.append(relevant_points.mean(axis=0))
    if len(zs) > 1:
        print "zs:",zs
        print "centroids:",centroids
    return centroids

def compute_feature_vector(p1, p2, s1, s2):
    """ p1 and s1 are the first and second closest points in segment 1 to 
    p2 and s2, the first and second closest points in segment2"""
    p1_to_p2 = _norm((p1-p2))
    p1_to_s1 = _norm((p1-s1))
    p2_to_s2 = _norm((p2-s2))
    feature_vector = np.zeros(8)
    feature_vector[0] = np.dot(p1_to_p2, p1_to_s1)
    feature_vector[1] = np.dot(p1_to_p2, p2_to_s2)
    feature_vector[2] = feature_vector[0] - feature_vector[1]
    # coplanarity checks
    feature_vector[3] = (p1[0] == p2[0])
    feature_vector[4] = (p1[0] == s1[0])
    feature_vector[5] = (p1[0] == s2[0])
    feature_vector[6] = (p2[0] == s1[0])
    feature_vector[7] = (p2[0] == s2[0])
    return feature_vector

def closest_pair(set1, set2, no_coplanar=False):
    """ set1 and set2 are lists of ndarrays """
    champ_distance = np.inf
    champ_s1, champ_s2 = (), ()
    for p1 in set1:
        for p2 in set2:
            if p1[0] == p2[0] and no_coplanar: continue
            distance = euclidean_distance(p1, p2)
            if distance > 0 and distance < champ_distance:
                champ_distance = distance
                champ_s1, champ_s2 = p1, p2
    return champ_s1, champ_s2

def closest_point(p1, set2, no_coplanar=False):
    champ_distance = np.inf
    champ_s2 = ()
    for p2 in set2:
        if p1[0] == p2[0] and no_coplanar: continue
        distance = euclidean_distance(p1, p2)
        if distance > 0 and distance < champ_distance:
            champ_distance = distance
            champ_s2 = p2
    return champ_s2

def euclidean_distance(p1, p2):
    v1, v2 = np.array(p1), np.array(p2)
    return np.sqrt(np.sum((v1-v2)**2))

cdef _norm(double[:] vector):
    cdef double total = 0
    for ii in range(vector.shape[0]):
        total += vector[ii] * vector[ii]
    cdef double length = np.sqrt(total)
    cdef np.ndarray[np.double_t, ndim=1] normed = np.zeros(vector.shape[0], dtype=np.double)
    if total == 0: return normed
    for ii in range(vector.shape[0]):
        normed[ii] = vector[ii] / length
    return normed
