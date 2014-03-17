import sys, os
import cPickle as pck
from copy import deepcopy as copy

import numpy as np
from numpy.testing import (assert_allclose, assert_approx_equal,
                           assert_equal)

rundir = os.path.dirname(__file__)
sys.path.append(rundir)

from gala import agglo, features


def feature_profile(g, f, n1=1, n2=2):
    out = []
    out.append(copy(g[n1][n2]['feature-cache']))
    out.append(copy(g.node[n1]['feature-cache']))
    out.append(copy(g.node[n2]['feature-cache']))
    out.append(f(g, n1, n2))
    out.append(f(g, n1))
    out.append(f(g, n2))
    return out


def list_of_feature_arrays(g, f, edges=[(1,2)], merges=[]):
    e1, edges = edges[0], edges[1:]
    out = feature_profile(g, f, *e1)
    for edge, merge in zip(edges, merges):
        g.merge_nodes(*merge)
        out.extend(feature_profile(g, f, *edge))
    return out


def assert_equal_lists_or_arrays(a1, a2, eps=1e-3):
    """Return True if ls1 and ls2 are arrays equal within eps or equal lists.
    
    The equality relationship can be nested. For example, lists of lists of 
    arrays that have identical structure will match.
    """
    if type(a1) == list and type(a2) == list:
        [assert_equal_lists_or_arrays(i1, i2, eps) for i1, i2 in zip(a1,a2)]
    elif type(a1) == np.ndarray and type(a2) == np.ndarray:
        assert_allclose(a1, a2, atol=eps)
    elif type(a1) == float and type(a2) == float:
        assert_approx_equal(a1, a2, int(-np.log10(eps)))
    else:
        assert_equal(a1, a2)


probs2 = np.load('toy-data/test-04-probabilities.npy')
probs1 = probs2[..., 0]
wss1 = np.loadtxt('toy-data/test-04-watershed.txt', np.uint32)
f1, f2, f3 = (features.moments.Manager(2, False),
              features.histogram.Manager(3, compute_percentiles=[0.5]),
              features.squiggliness.Manager(ndim=2))
f4 = features.base.Composite(children=[f1, f2, f3])


def run_matched(f, fn, c=1,
                edges=[(1, 2), (6, 3), (7, 4)],
                merges=[(1, 2), (6, 3)]):
    p = probs1 if c == 1 else probs2
    g = agglo.Rag(wss1, p, feature_manager=f)
    o = list_of_feature_arrays(g, f, edges, merges)
    r = pck.load(open(fn, 'r'))
    assert_equal_lists_or_arrays(o, r)


def test_1channel_moment_features():
    f = f1
    run_matched(f, 'toy-data/test-04-moments-1channel-12-13.pck')

def test_2channel_moment_features():
    f = f1
    run_matched(f, 'toy-data/test-04-moments-2channel-12-13.pck', 2)

def test_1channel_histogram_features():
    f = f2
    run_matched(f, 'toy-data/test-04-histogram-1channel-12-13.pck')

def test_2channel_histogram_features():
    f = f2
    run_matched(f, 'toy-data/test-04-histogram-2channel-12-13.pck', 2)

def test_1channel_squiggliness_feature():
    f = f3
    run_matched(f, 'toy-data/test-04-squiggle-1channel-12-13.pck')

def test_1channel_composite_feature():
    f = f4
    run_matched(f, 'toy-data/test-04-composite-1channel-12-13.pck')

def test_2channel_composite_feature():
    f = f4
    run_matched(f, 'toy-data/test-04-composite-2channel-12-13.pck', 2)

def test_central_moments():
    a = np.array([[1,2,3,4,5]]).T
    b = np.array([[10,20,30,40,50],
                  [500, 10, 2, -5, 12]]).T
    c = np.array([[10,20,30,40,50, -5],
                  [500, 10, 2, -5, 12, 70],
                  [70, 0.12, 14, -5, 12,0]]).T
    d = np.array([[  1.95600000e+03,   1.95600000e+03],
                 [  1.55505882e+03,   1.01972053e+02],
                 [  1.32785123e+03,  1.14825759e+01],
                 [  1.18178212e+03,  1.70749482e+00],
                 [  1.07847585e+03,  2.99198871e-01]])
    e = np.array([[  6.60000000e+01,  6.60000000e+01],
                [  2.07960784e+01,  3.12759092e+00],
                [  6.66117647e+00,  2.10134564e-01],
                [  2.16794186e+00,  1.82360668e-02],
                [  7.16501937e-01,  1.86370658e-03]])
    pymo = features.moments.central_moments_from_noncentral_sums_py
    cymo = features.moments.central_moments_from_noncentral_sums
    for example in [a,b,c,d,e]:
        assert_equal_lists_or_arrays(pymo(example), cymo(example))

def test_histogram_percentiles():
    desired_percentiles = np.array([0.1, 0.5, 0.9])
    manager = features.histogram.Manager(25, 0, 1, desired_percentiles)
    h1 = np.array([[ 0.,0.,0.,0.,0.,0., 0.,0.,0.00205339,  0.01026694,  0.04106776,  0.04722793,  0.04312115,
                       0.04928131,  0.02669405,  0.0513347,   0.03696099,  0.02874743,  0.0349076,
                       0.03901437,  0.06570842,  0.09650924,  0.137577,    0.20533881,  0.08418891],
                     [ 0.52977413,  0.25462012,  0.14784394,  0.06365503,  0.00410678,  0., 0., 0., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    for h in [h1]:
        cyh = manager.percentiles(h, desired_percentiles)
        pyh = manager.percentiles_py(h, desired_percentiles)
        assert_equal_lists_or_arrays(pyh, cyh)

if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()

