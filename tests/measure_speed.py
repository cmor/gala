from timeit import Timer
import numpy as np

def measure_nchoosek_speed():
    TRIALS = 100000
    print "Timing nchoosek()"
    for imp in ["from gala.features.moments import nchoosek",
                 "from scipy.misc import comb as nchoosek"]:
        timer = Timer("nchoosek(10000,20)", setup=imp)
        time = timer.timeit(TRIALS)
        print "-- %s time across %i trials: %f seconds" % (imp, TRIALS, time)

def measure_central_moments_speed():
    TRIALS = 10000
    print "Timing central_moments_from_noncentral_sums()"
    for imp,label in [("from gala.features.moments import central_moments_from_noncentral_sums as target","cython"),
                 ("from gala.features.moments import central_moments_from_noncentral_sums_py as target","python")]:
        imp += """; import numpy as np; ex = np.array([[  6.60000000e+01,  6.60000000e+01],
                [  2.07960784e+01,  3.12759092e+00],
                [  6.66117647e+00,  2.10134564e-01],
                [  2.16794186e+00,  1.82360668e-02],
                [  7.16501937e-01,  1.86370658e-03]])"""
        timer = Timer("target(ex)", setup=imp)
        time = timer.timeit(TRIALS)
        print "-- %s time across %i trials: %f seconds" % (label, TRIALS, time)

def measure_histogram_speed():
    TRIALS = 10000
    print "Timing percentiles()"
    setup = """import numpy as np; from gala.features import histogram
h = np.array([[ 0.,0.,0.,0.,0.,0., 0.,0.,0.00205339,  0.01026694,  0.04106776,  0.04722793,  0.04312115,
                   0.04928131,  0.02669405,  0.0513347,   0.03696099,  0.02874743,  0.0349076,
                   0.03901437,  0.06570842,  0.09650924,  0.137577,    0.20533881,  0.08418891],
                 [ 0.52977413,  0.25462012,  0.14784394,  0.06365503,  0.00410678,  0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
dp = np.array([0.1, 0.5, 0.9])
manager = histogram.Manager(25, 0, 1, dp)"""
    for action, label in [("manager.percentiles(h,dp)", "cython"), ("manager.percentiles_py(h,dp)", "python")]:
        timer = Timer(action, setup=setup)
        time = timer.timeit(TRIALS)
        print "-- %s time across %i trials: %f seconds" % (label, TRIALS, time)

if __name__ == '__main__':
    measure_nchoosek_speed()
    measure_histogram_speed()
    measure_central_moments_speed()
