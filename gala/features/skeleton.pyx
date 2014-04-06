import numpy as np
cimport numpy as np
from . import base
cdef extern from "math.h":
    double sqrt(double x)

NULL_VALUE = -1
BASE_FEATURE_VECTOR_LEN = 9

class Manager(base.Null):
    def __init__(self, z_resolution_factor, neighbors_considered=1, *args, **kwargs):
        super(Manager, self).__init__()
        self.z_resolution_factor = z_resolution_factor
        self.neighbors_considered = neighbors_considered
        print "--- considering %d neighbors" % (neighbors_considered)
        self.unit_null_feature_vector = np.ones(BASE_FEATURE_VECTOR_LEN) * NULL_VALUE
        self.null_feature_vector = np.ones(BASE_FEATURE_VECTOR_LEN*neighbors_considered) * NULL_VALUE

    @classmethod
    def load_dict(cls, fm_info):
        obj = cls(fm_info['z_resolution_factor'])
        return obj

    def write_fm(self, json_fm={}):
        if 'feature_list' not in json_fm:
            json_fm['feature_list'] = []
        json_fm['feature_list'].append('skeleton')
        json_fm['contact'] = {
            'z_resolution_factor': self.z_resolution_factor,
            'neighbors_considered': self.neighbors_considered
        }
        return json_fm

    def create_node_cache(self, g, n):
        points = self.points_from_idxs(list(g.extent(n)), 
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

    def compute_difference_features(self, g, n1, n2, n1_centroids_l=None, n2_centroids_l=None):
        if n1_centroids_l is None: n1_centroids_l = g.node[n1][self.default_cache]
        if n2_centroids_l is None: n2_centroids_l = g.node[n2][self.default_cache]
        if len(n1_centroids_l) < 2 or len(n2_centroids_l) < 2:
            return self.null_feature_vector
        n1_centroids, n2_centroids = np.vstack(n1_centroids_l), np.vstack(n2_centroids_l)
        n1_closest_centroid, n2_closest_centroid = _closest_pair(n1_centroids,n2_centroids)
        if (n1_closest_centroid==NULL_VALUE).all() or (
            n2_closest_centroid==NULL_VALUE).all():
            return self.null_feature_vector
        feature_vector = np.zeros(BASE_FEATURE_VECTOR_LEN * self.neighbors_considered,)
        points_used = np.ones((2*self.neighbors_considered, n1_centroids.shape[1]))*NULL_VALUE
        for ii in range(self.neighbors_considered):
            n1_next_closest = _closest_point(n1_closest_centroid, n1_centroids, points_used)
            n2_next_closest = _closest_point(n2_closest_centroid, n2_centroids, points_used)
            for jj in range(points_used.shape[1]):
                points_used[(ii*2),jj] = n1_next_closest[jj]
                points_used[(ii*2+1),jj] = n2_next_closest[jj]
            if (n1_next_closest==NULL_VALUE).all() or (
                n2_next_closest==NULL_VALUE).all():
                feature_vector[ii*BASE_FEATURE_VECTOR_LEN:(ii+1)*BASE_FEATURE_VECTOR_LEN] = self.unit_null_feature_vector
            else:
                feature_vector[ii * BASE_FEATURE_VECTOR_LEN:(ii+1)*BASE_FEATURE_VECTOR_LEN] = \
                    compute_feature_vector(n1_closest_centroid, n2_closest_centroid, n1_next_closest, n2_next_closest)
        #print "fvec:",feature_vector
        return feature_vector

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
    feature_vector = np.zeros(BASE_FEATURE_VECTOR_LEN,)
    feature_vector[0] = np.dot(p1_to_p2, p1_to_s1)
    feature_vector[1] = np.dot(p1_to_p2, p2_to_s2)
    feature_vector[2] = feature_vector[0] - feature_vector[1]
    # coplanarity checks
    feature_vector[3] = (p1[0] == p2[0])
    feature_vector[4] = (p1[0] == s1[0])
    feature_vector[5] = (p1[0] == s2[0])
    feature_vector[6] = (p2[0] == s1[0])
    feature_vector[7] = (p2[0] == s2[0])
    feature_vector[8] = (s1[0] == s2[0])
    #print "p1:",p1,"p2:",p2,"s1:",s1,"s2:",s2
    return feature_vector

cdef _closest_pair(double[:,:] set1, double[:,:] set2):
    cdef double distance
    cdef double champ_distance = np.inf
    cdef np.ndarray[np.double_t, ndim=1] champ_s1 = np.ones(set1.shape[1], dtype=np.double)*NULL_VALUE
    cdef np.ndarray[np.double_t, ndim=1] champ_s2 = np.ones(set2.shape[1], dtype=np.double)*NULL_VALUE
    for p1 in range(set1.shape[0]):
        for p2 in range(set2.shape[0]):
            distance = _euclidean_distance_sq(set1[p1,0], set1[p1,1], set1[p1,2],
                                              set2[p2,0], set2[p2,1], set2[p2,2])
            if distance > 0 and distance < champ_distance:
                champ_distance = distance
                for ii in range(set1.shape[1]): champ_s1[ii] = set1[p1,ii]
                for ii in range(set2.shape[1]): champ_s2[ii] = set2[p2,ii]
    return champ_s1, champ_s2

cdef _closest_point(double[:] v1, double[:,:] set2, double[:,:] forbidden_points):
    cdef int p2,ii,jj,ff,match,any_match
    cdef double distance
    cdef double champ_distance = np.inf
    cdef np.ndarray[np.double_t, ndim=1] champ_s2 = np.ones(set2.shape[1], dtype=np.double)*NULL_VALUE
    for p2 in range(set2.shape[0]):
        any_match = 0
        # skip if its a forbidden point
        for ff in range(forbidden_points.shape[0]):
            match = 1
            for jj in range(forbidden_points.shape[1]):
                if (forbidden_points[ff,jj] != set2[p2,jj]):
                    match = 0
                    break
            if (match == 1):
                any_match = 1
                break
        if any_match == 1: continue
        distance = _euclidean_distance_sq(v1[0], v1[1], v1[2],
                                          set2[p2,0], set2[p2,1], set2[p2,2])
        if distance > 0 and distance < champ_distance:
            champ_distance = distance
            for ii in range(set2.shape[1]): champ_s2[ii] = set2[p2,ii]
    return champ_s2

cdef inline _euclidean_distance_sq(double z1, double y1, double x1, double z2, double y2, double x2):
    return (z1-z2)*(z1-z2) + (y1-y2)*(y1-y2) + (x1-x2)*(x1-x2)

cdef _norm(double[:] vector):
    cdef double total = 0
    for ii in range(vector.shape[0]):
        total += vector[ii] * vector[ii]
    cdef double length = sqrt(total)
    cdef np.ndarray[np.double_t, ndim=1] normed = np.zeros(vector.shape[0], dtype=np.double)
    if total == 0: return normed
    for ii in range(vector.shape[0]):
        normed[ii] = vector[ii] / length
    return normed
