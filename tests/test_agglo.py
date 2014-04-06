import numpy as np
from numpy.testing import assert_equal, assert_allclose

from gala import agglo
from gala import evaluate as ev


test_idxs = range(4)
num_tests = len(test_idxs)
fns = ['toy-data/test-%02i-probabilities.txt' % i for i in test_idxs]
probs = map(np.loadtxt, fns)
fns = ['toy-data/test-%02i-watershed.txt' % i for i in test_idxs]
wss = [np.loadtxt(fn, dtype=np.uint32) for fn in fns]
fns = ['toy-data/test-%02i-groundtruth.txt' % i for i in test_idxs]
results = map(np.loadtxt, fns)

landscape = np.array([1,0,1,2,1,3,2,0,2,4,1,0])

def test_8_connectivity():
    p = np.array([[0,0.5,0],[0.5,1.0,0.5],[0,0.5,0]])
    ws = np.array([[1,0,2],[0,0,0],[3,0,4]], np.uint32)
    g = agglo.Rag(ws, p, connectivity=2)
    assert_equal(agglo.boundary_mean(g, 1, 2), 0.75)
    assert_equal(agglo.boundary_mean(g, 1, 4), 1.0)

def test_empty_rag():
    g = agglo.Rag()
    assert_equal(g.nodes(), [])
    assert_equal(g.copy().nodes(), [])

def test_agglomeration():
    i = 1
    g = agglo.Rag(wss[i], probs[i], agglo.boundary_mean, 
        normalize_probabilities=True)
    g.agglomerate(0.51)
    assert_allclose(ev.vi(g.get_segmentation(), results[i]), 0.0,
                    err_msg='Mean agglomeration failed.')

def test_ladder_agglomeration():
    i = 2
    g = agglo.Rag(wss[i], probs[i], agglo.boundary_mean,
        normalize_probabilities=True)
    g.agglomerate_ladder(2)
    g.agglomerate(0.5)
    assert_allclose(ev.vi(g.get_segmentation(), results[i]), 0.0,
                    err_msg='Ladder agglomeration failed.')

def test_no_dam_agglomeration():
    i = 3
    g = agglo.Rag(wss[i], probs[i], agglo.boundary_mean, 
        normalize_probabilities=True)
    g.agglomerate(0.75)
    assert_allclose(ev.vi(g.get_segmentation(), results[i]), 0.0,
                    err_msg='No dam agglomeration failed.')

def points_to_cube(points, shape):
    out = np.zeros(shape)
    for point in points:
        out[point] = 1
    return out

def test_flood_fill():
    fail_message = 'Flood fill failed.'
    example = np.array([[[0,1,2,5],
                         [0,0,2,4],
                         [1,0,1,2]],
                        [[0,0,5,5],
                         [1,1,1,5],
                         [1,2,1,5]]])
    t1 = agglo.flood_fill(example, (0,0,0), [0], True)
    assert_equal(set(t1), set([0,4,5,9,12,13]), fail_message)
    t2 = agglo.flood_fill(example, (0,0,0), [0], False)
    assert_equal(set(t2), set([(0,0,0), (0,1,0), (0,1,1), (0,2,1), (1,0,0),
                            (1,0,1)]), fail_message)
    t3 = agglo.flood_fill(example, (0,1,3), [2], True)
    assert_equal(set(t3), set([]), fail_message)
    t4 = agglo.flood_fill(example, (0,0,3), [5], True)
    assert_equal(set(t4), set([3,14,15,19,23]), fail_message)
    t5 = agglo.flood_fill(example, (1,1,1), [1,4], True)
    assert_equal(set(t5), set([8,10,16,17,18,20,22]), fail_message)
    t6 = agglo.flood_fill(example, (0,1,2), [2,5], True)
    assert_equal(set(t6), set([2,3,6,11,14,15,19,23]), fail_message)


    shape = (10,10,10)
    example2 = np.random.randint(1, size=shape)
    example2[5,5,5] = 0
    t7 = agglo.flood_fill(example2, (5,5,5), [0], False)
    assert_equal(len(t7), (example2==0).sum())

    example3 = np.random.randint(6, size=(200,200,200))
    example3[2,2,:] = 6
    example3[:,2,150] = 6
    example3[45,:,:] = 6
    t8 = agglo.flood_fill(example3, (2,2,0), [6], True)
    assert_equal(len(t8), (example3==6).sum())

if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()

