from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases

install_aliases()

import numpy as np
import os
import gzip
import struct
import array

import matplotlib.pyplot as plt
import matplotlib.image
from urllib.request import urlretrieve


def download(url, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    out_file = os.path.join('data', filename)
    if not os.path.isfile(out_file):
        urlretrieve(url, out_file)


def mnist():
    base_url = 'http://yann.lecun.com/exdb/mnist/'

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz',
                     'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz']:
        download(base_url + filename, filename)

    train_images = parse_images('data/train-images-idx3-ubyte.gz')
    train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
    test_images = parse_images('data/t10k-images-idx3-ubyte.gz')
    test_labels = parse_labels('data/t10k-labels-idx1-ubyte.gz')

    return train_images, train_labels, test_images, test_labels


def load_mnist():
    partial_flatten = lambda x: np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    train_images, train_labels, test_images, test_labels = mnist()
    train_images = train_images[:10000]
    train_labels = train_labels[:10000]
    test_labels = test_labels[:10000]
    train_images = partial_flatten(train_images) / 255.0
    test_images = partial_flatten(test_images) / 255.0
    train_images = np.round(train_images)
    test_images = np.round(test_images)
    N_data = train_images.shape[0]

    return N_data, train_images, train_labels, test_images, test_labels


def plot_images(images, ax, padding=5,
                cmap=matplotlib.cm.binary, vmin=None, vmax=None):
    """Images should be a (n_rows, n_cols, height, width) matrix."""
    N_rows = images.shape[0]
    N_cols = images.shape[1]
    digit_dimensions = images.shape[2:]
    pad_value = np.min(images.ravel())
    concat_images = np.full(((digit_dimensions[0] + padding) * N_rows + padding,
                             (digit_dimensions[1] + padding) * N_cols + padding), pad_value)
    for row_ix in range(N_rows):
        for col_ix in range(N_cols):
            cur_image = images[row_ix, col_ix]
            row_start = padding + (padding + digit_dimensions[0]) * row_ix
            col_start = padding + (padding + digit_dimensions[1]) * col_ix
            concat_images[row_start: row_start + digit_dimensions[0],
            col_start: col_start + digit_dimensions[1]] = cur_image
    cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    return cax


def show_images_plot(images, size=None):
    f, ax = plt.subplots()
    if size is not None:
        f.set_size_inches(size[0], size[1])
    plot_images(images, ax, vmin=0.0, vmax=1.0)
    plt.show()
