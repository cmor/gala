# gala: segmentation of nD images [![Picture](https://raw.github.com/janelia-flyem/janelia-flyem.github.com/master/images/gray_janelia_logo.png)](http://janelia.org/)

Gala is a python library for performing and evaluating image segmentation,
distributed under the open-source [Janelia Farm license](http://janelia-flyem.github.com/janelia_farm_license.html). It implements the algorithm
described in [Nunez-Iglesias *et al*.](http://arxiv.org/abs/1303.6163), PLOS
ONE, 2013.

Gala supports n-dimensional images (images, volumes, videos, videos of 
volumes...) and multiple channels per image.

[![Build Status](https://travis-ci.org/janelia-flyem/gala.png?branch=master)](https://travis-ci.org/janelia-flyem/gala)

## Requirements

* Python 2.7
* numpy 1.7+
* scipy 0.10+
* Image (a.k.a. Python Imaging Library or PIL) 1.1.7
* networkx 1.6+
* HDF5 and h5py 1.5+
* nose 1.3+
* cython 0.17+
* scikit-learn 0.10+, preferably 0.14+
* matplotlib 1.2+
* scikit-image 0.9+

### Optional dependencies

* progressbar 2.3-dev
* [vigra/vigranumpy](hci.iwr.uni-heidelberg.de/vigra/) (1.9.0)

For vigra, you are on your own. It is used for the random forest classifier,
but if you don't install it you can use any of the scikit-learn classifiers,
including their newly-excellent random forest.

## Installation

### Installing gala

Gala is a pure python library and can be installed in two ways:
* Add the gala directory to your PYTHONPATH environment variable, or
* Use distutils to install it into your preferred python environment:

```bash
$ python setup.py install
```

### Installing requirements

You can install all the requirements yourself: most are available in
the Python Package Index (PyPI) and can be installed with simple commands:

```bash
$ pip install scikit-learn
```

Alternatively, a number of Python distributions include all the above
dependencies and a bunch more for good measure. Two examples are
[Continuum Anaconda](http://www.continuum.io/downloads) and
[Enthought Canopy](https://www.enthought.com/products/canopy/).

Finally, you can use Janelia's own
[buildem system](http://github.com/janelia-flyem/buildem#readme) to
automatically download, compile, test, and install requirements into a
specified buildem prefix directory. (You will need CMake.) 

```
$ cmake -D BUILDEM_DIR=/path/to/platform-specific/build/dir <gala directory>
$ make
```

You might have to run the above steps twice if this is the first time you are
using the buildem system.

On Mac, you might have to install compilers (such as gcc, g++, and gfortran).


### Testing

The test coverage is rather tiny, but it is still a nice way to check you
haven't completely screwed up your installation. Note: the test scripts
*must* be run from the `tests` directory.

```bash
$ cd tests
$ python test_agglo.py
$ python test_features.py
$ python test_watershed.py
$ python test_gala.py
```

## Usage

An example script, `example.py`, exists in the `tests/example-data`
directory. We step through it here for a quick rundown of gala's capabilities.

First, import gala's submodules:

```python
from gala import imio, classify, features, agglo, evaluate as ev
```

Next, read in the training data: a ground truth volume (`gt_train`), a
probability map (`pr_train`) and a superpixel or watershed map (`ws_train`).

```python
gt_train, pr_train, ws_train = (map(imio.read_h5_stack,
                                ['train-gt.lzf.h5', 'train-p1.lzf.h5',
                                 'train-ws.lzf.h5']))
```

A _feature manager_ is a callable object that computes feature vectors from
graph edges. The object has the following responsibilities, which it can inherit
from `classify.base.Null`:

* create a (possibly empty) _feature cache_ on each edge and node, precomputing
  some of the calculations needed for feature computation;
* maintain the feature cache throughout node merges during agglomeration; and,
* compute the feature vector from the feature caches when called with the
  inputs of a graph and two nodes.

Feature managers can be chained through the `features.Composite` class.

```python
fm = features.moments.Manager()
fh = features.histogram.Manager()
fc = features.base.Composite(children=[fm, fh])
```

With the feature manager, and the above data, we can create a *region adjacency
graph* or *RAG*, and use it to train the agglomeration process:

```python
g_train = agglo.Rag(ws_train, pr_train, feature_manager=fc)
(X, y, w, merges) = g_train.learn_agglomerate(gt_train, fc)[0]
y = y[:, 0] # gala has 3 truth labeling schemes, pick the first one
```

`X` and `y` above have the now-standard scikit-learn [supervised dataset
format](http://scikit-learn.org/stable/tutorial/statistical_inference/settings.html#datasets).
This means we can use any classifier that satisfies the scikit-learn API.
Below, we use a simple wrapper around the scikit-learn
`RandomForestClassifier`.

```python
rf = classify.DefaultRandomForest().fit(X, y)
```

The composition of a feature map and a classifier defines a *policy* or
*merge priority function*, which will determine the agglomeration of a volume
of hereby unseen data (the *test* volume).

```python
learned_policy = agglo.classifier_probability(fc, rf)

pr_test, ws_test = (map(imio.read_h5_stack,
                        ['test-p1.lzf.h5', 'test-ws.lzf.h5']))
g_test = agglo.Rag(ws_test, pr_test, learned_policy, feature_manager=fc)
```

The best expected segmentation is obtained at a threshold of 0.5, when a
merge has even odds of being correct or incorrect, according to the trained
classifier.

```python
g_test.agglomerate(0.5)
```

The RAG is a *model* for the segmentation. To extract the segmentation itself,
use the `get_segmentation` function. This is a map of labels of the same shape
as the original image.

```python
seg_test1 = g_test.get_segmentation()
```

Gala transparently supports multi-channel probability maps. In the case of EM
images, for example, one channel may be the probability that a given pixel is
part of a cell boundary, while the next channel may be the probability that it
is part of a mitochondrion. The feature managers work identically with single
and multi-channel features.

```python
# p4_train and p4_test have 4 channels
p4_train = imio.read_h5_stack('train-p4.lzf.h5')
# the existing feature manager works transparently with multiple channels!
g_train4 = agglo.Rag(ws_train, p4_train, feature_manager=fc)
(X4, y4, w4, merges4) = g_train4.learn_agglomerate(gt_train, fc)[0]
y4 = y4[:, 0]
rf4 = classify.DefaultRandomForest().fit(X4, y4)
learned_policy4 = agglo.classifier_probability(fc, rf4)
p4_test = imio.read_h5_stack('test-p4.lzf.h5')
g_test4 = agglo.Rag(ws_test, p4_test, learned_policy4, feature_manager=fc)
g_test4.agglomerate(0.5)
seg_test4 = g_test4.get_segmentation()
```

For comparison, gala allows the implementation of many agglomerative
algorithms, including mean agglomeration (below) and
[LASH](http://www.mit.edu/people/sturaga/papers/JainNIPS2011.pdf).

```python
g_testm = agglo.Rag(ws_test, pr_test,
                    merge_priority_function=agglo.boundary_mean)
g_testm.agglomerate(0.5)
seg_testm = g_testm.get_segmentation()
```

### Evaluation

The gala library contains numerous evaluation functions, including edit
distance, Rand index and adjusted Rand index, and our personal favorite, the
variation of information (VI):

```python
gt_test = imio.read_h5_stack('test-gt.lzf.h5')
import numpy as np
results = np.vstack((
    ev.split_vi(ws_test, gt_test),
    ev.split_vi(seg_testm, gt_test),
    ev.split_vi(seg_test1, gt_test),
    ev.split_vi(seg_test4, gt_test)
    ))
print(results)
```

This should print something like:

```
[[ 0.1845286   1.64774412]
 [ 0.18719817  1.16091003]
 [ 0.38978567  0.28277887]
 [ 0.39504714  0.2341758 ]]
```

Each row is an evaluation, with the first number representing the
undersegmentation error or false merges, and the second representing the
oversegmentation error or false splits, both measured in bits.

(Results may vary since there is some randomness involved in training a random
forest, and the datasets are small.)

### Threshold-dependent evaluation

An ultrametric contour map (UCM) can be thresholded to provide the segmentation
at any threshold of agglomeration. (It may, however, result in a split when a
segment becomes thinner than three pixels, because gala uses a pixel-level
approximation for the boundary between segments, which is ultimately a subpixel
property.)

To get the UCM, agglomerate to infinity, and then use the `get_ucm` function.

With the UCM, you can test threshold-dependent segmentation performance, using,
for example, the split VI plot:

```python
g_test.agglomerate(np.inf)
g_test4.agglomerate(np.inf)
g_testm.agglomerate(np.inf)
ucms = [g.get_ucm() for g in [g_test, g_test4, g_testm]]
vis = [ev.vi_by_threshold(u, gt_test, [0], [0])[1:] for u in ucms]
colors = ['deepskyblue', 'orange', 'black']
from matplotlib import pyplot as plt
plt.figure(figsize=(5,5))
from gala import viz
viz.plot_split_vi(vis, colors=colors)
plt.xlim(0, 1); plt.ylim(0, 1)
```

And, as mentioned earlier, many other evaluation functions are available. See
the documentation for the `evaluate` package for more information.

```python
# rand index and adjusted rand index
ri = ev.rand_index(seg_test1, gt_test)
ari = ev.adj_rand_index(seg_test1, gt_test)
# Fowlkes-Mallows index
fm = ev.fm_index(seg_test1, gt_test)
# pixel-wise precision-recall
pwprs = [ev.pixel_wise_precision_recall(u, gt_test) for u in ucms]
```

### Other options

Gala supports a wide array of merge priority functions to explore your data.
We can specify the median boundary probability with the
`merge_priority_function` argument to the RAG constructor:

```python
g_testM = agglo.Rag(ws_test, pr_test,
                    merge_priority_function=agglo.boundary_median)
```

A user can specify their own merge priority function. A valid merge priority
function is a callable Python object that takes as input a graph and two nodes,
and returns a real number.

### To be continued...

That's a quick summary of the capabilities of Gala. There are of course many
options under the hood, many of which are undocumented... Feel free to push me
to update the documentation of your favorite function!

