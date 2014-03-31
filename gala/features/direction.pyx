import numpy as np
cimport numpy as np
from . import base

NULL_VALUE = -1

class Manager(base.Null):
    def __init__(self, z_resolution_factor, box_radius=110, 
                subsample_target=500, min_z_extent=10, min_sample_stride=10,
                *args, **kwargs):
        super(Manager, self).__init__()
        self.z_resolution_factor = z_resolution_factor
        self.box_radius = box_radius
        self.subsample_target = subsample_target
        self.min_z_extent = min_z_extent
        self.min_sample_stride = min_sample_stride

    @classmethod
    def load_dict(cls, fm_info):
        obj = cls(fm_info['z_resolution_factor'], fm_info['box_radius'],
                fm_info['subsample_target'], fm_info['min_z_extent'],
                fm_info['min_sample_stride'])
        return obj

    def write_fm(self, json_fm={}):
        if 'feature_list' not in json_fm:
            json_fm['feature_list'] = []
        json_fm['feature_list'].append('direction')
        json_fm['direction'] = {
            'z_resolution_factor' : self.z_resolution_factor,
            'box_radius' : self.box_radius,
            'subsample_target' : self.subsample_target,
            'min_z_extent' : self.min_z_extent,
            'min_sample_stride' : self.min_sample_stride
        }
        return json_fm

    def points_from_idxs(self, idxs, cube_shape):
        if len(cube_shape) == 3:
            zs, ys, xs = np.unravel_index(idxs, cube_shape)
        else:
            zs, ys, xs, cs = np.unravel_index(idxs, cube_shape)
        unsquished_zs = zs * self.z_resolution_factor
        return np.concatenate((unsquished_zs[:,np.newaxis],
                ys[:,np.newaxis],xs[:,np.newaxis]), axis=1).astype(np.integer)

    def compute_edge_features(self, g, n1, n2, cache=None):
        if cache == None: cache = g[n1][n2][self.default_cache]
        cube_shape = g.probabilities.shape
        edge_points = self.points_from_idxs(list(g[n1][n2]['boundary']), cube_shape)
        edge_mean = np.mean(edge_points, axis=0).astype(np.integer)
        all_n1_points = self.points_from_idxs(list(g.node[n1]['extent']), cube_shape)
        all_n2_points = self.points_from_idxs(list(g.node[n2]['extent']), cube_shape)
        z_max = cube_shape[0] * self.z_resolution_factor 
        if _dim_extent(all_n1_points, 0, z_max) < self.min_z_extent or \
           _dim_extent(all_n2_points, 0, z_max) < self.min_z_extent:
            return compute_feature_vector(np.ones((3,3)) * NULL_VALUE)

        vectors = compute_pc_vectors(all_n1_points, all_n2_points, 
                        edge_mean, self.box_radius, self.subsample_target,
                        self.min_sample_stride)
        return compute_feature_vector(vectors)


def compute_feature_vector(pc_vectors):
    feature_vector = np.ones(5, dtype=np.double)
    if (pc_vectors == NULL_VALUE).all():
        return feature_vector * NULL_VALUE

    # we assume that pc_vectors is normalized
    between_segments = np.dot(pc_vectors[0,:],pc_vectors[1,:])
    between_pc_and_centers_1 = np.dot(pc_vectors[0,:],pc_vectors[2,:])
    between_pc_and_centers_2 = np.dot(pc_vectors[1,:],pc_vectors[2,:])
    feature_vector[0] = between_segments
    feature_vector[1] = between_pc_and_centers_1
    feature_vector[2] = between_pc_and_centers_2
    feature_vector[3] = between_pc_and_centers_1 + between_pc_and_centers_2
    feature_vector[4] = between_pc_and_centers_1 * between_pc_and_centers_2
    return feature_vector


def compute_pc_vectors(all_n1_points, all_n2_points, center, radius, 
                                    max_points, min_sample_stride):
    n1_points = _limit_to_radius(all_n1_points, center, radius, 
                    max_points, min_sample_stride)
    n2_points = _limit_to_radius(all_n2_points, center, radius,
                    max_points, min_sample_stride)
   
    # catch having no points in extent 
    if n1_points.shape[0] == 0 or n2_points.shape[0] == 0:
        return np.ones((3,3)) * NULL_VALUE

    n1_mean = np.mean(n1_points, axis=0)
    n2_mean = np.mean(n2_points, axis=0)
    n1_meanless = n1_points - n1_mean 
    n2_meanless = n2_points - n2_mean
    between_means = n1_mean - n2_mean
    try:
        u1,d1,v1 = np.linalg.svd(n1_meanless)
        u2,d2,v2 = np.linalg.svd(n2_meanless)
    except Exception as e:
        print "Error: %s " % (e.message)
        print e.args
        print "n1_meanless:",n1_meanless
        print "n2_meanless:",n2_meanless
        print "all_n1_points:",all_n1_points
        print "n1_points.shape:",n1_points.shape
        print "n2_points.",n2_points.shape
        return np.ones((3,3)) * NULL_VALUE

    return np.concatenate((_norm(v1[0])[np.newaxis,:], _norm(v2[0])[np.newaxis,:], 
            _norm(between_means)[np.newaxis,:]), axis=0)


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

def dim_extent(points, dim, upper_bound):
    return _dim_extent(points, dim, upper_bound)

cdef _dim_extent(long[:,:] points, int dim, int upper_bound):
    cdef int pp, extent
    cdef np.ndarray[np.int_t, ndim=1] seen = np.zeros(upper_bound, dtype=np.integer)
    extent = 0
    for pp in range(points.shape[0]):
        seen[points[pp, dim]] = 1
    for pp in range(seen.shape[0]):
        extent += seen[pp]
    return extent

def limit_to_radius(points, center, radius, max_count, stride):
    return _limit_to_radius(points, center, radius, max_count, stride)

cdef _limit_to_radius(long[:,:] points, long[:] center, int radius,
                             int max_count, int min_sample_stride):
    cdef int pp, gg, ss, valid_count, valid, skip_indicator, stride_len, offset
    skip_indicator = -1 # indices, so negative will never occur naturally
    valid_count = 0
    
    # count all points that are within the radius, and mark all others
    for pp in range(points.shape[0]):
        valid = 1
        for dd in range(points.shape[1]):
            if (points[pp,dd] < center[dd] - radius) or (
                points[pp,dd] > center[dd] + radius):
                valid = 0
                break
        if valid: valid_count += 1
        else: points[pp,0] = skip_indicator

    # determine how many of the valid we have to skip to stay under max_count
    if valid_count > max_count:
        new_length = max_count
        stride_len = valid_count / max_count + 1
        if stride_len < min_sample_stride: stride_len = min_sample_stride
    else:
        new_length = valid_count
        stride_len = min_sample_stride
    cdef np.ndarray[np.int_t, ndim=2] good_points = np.zeros([new_length,
                                        points.shape[1]],dtype=np.integer)
    gg = 0
    offset = 0
    while gg < good_points.shape[0]:
        ss = stride_len - offset
        for pp in range(points.shape[0]):
            if gg >= good_points.shape[0]: break
            if points[pp,0] == skip_indicator: continue
            if ss > 0: ss -= 1; continue
            ss = stride_len
            for dd in range(points.shape[1]):
                good_points[gg,dd] = points[pp,dd]
            gg += 1
        offset += 1 
    return good_points
