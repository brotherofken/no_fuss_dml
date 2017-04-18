#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:56:38 2017

Stolen from https://github.com/breze-no-salt/breze/blob/master/breze/arch/component
"""

import lasagne
import theano.tensor as T

def lookup(what, where, default=None):
    """Return ``where.what`` if what is a string, otherwise what. If not found
    return ``default``."""
    if isinstance(what, (str, unicode)):
        res = getattr(where, what, default)
    else:
        res = what
    return res


def l2(arr, axis=None):
    """Return the L2 norm of a tensor.
    Parameters
    ----------
    arr : Theano variable.
        The variable to calculate the norm of.
    axis : integer, optional [default: None]
        The sum will be performed along this axis. This makes it possible to
        calculate the norm of many tensors in parallel, given they are organized
        along some axis. If not given, the norm will be computed for the whole
        tensor.
    Returns
    -------
    res : Theano variable.
        If ``axis`` is ``None``, this will be a scalar. Otherwise it will be
        a tensor with one dimension less, where the missing dimension
        corresponds to ``axis``.
    Examples
    --------
    >>> v = T.vector()
    >>> this_norm = l2(v)
    >>> m = T.matrix()
    >>> this_norm = l2(m, axis=1)
    >>> m = T.matrix()
    >>> this_norm = l2(m)
    """
    return T.sqrt((arr ** 2).sum(axis=axis) + 1e-8)

def pairwise_diff(X, Y=None):
    """Given two arrays with samples in the row, compute the pairwise
    differences.
    Parameters
    ----------
    X : Theano variable
        Has shape ``(n, d)``. Contains one item per first dimension.
    Y : Theano variable, optional [default: None]
        Has shape ``(m, d)``.  If not given, defaults to ``X``.
    Returns
    -------
    res : Theano variable
        Has shape ``(n, d, m)``.
    """
    Y = X if Y is None else Y
    diffs = X.T.dimshuffle(1, 0, 'x') - Y.T.dimshuffle('x', 0, 1)
    return diffs


def distance_matrix(X, Y=None, norm=l2):
    """Return an expression containing the distances given the norm of up to two
    arrays containing samples.
    Parameters
    ----------
    X : Theano variable
        Has shape ``(n, d)``. Contains one item per first dimension.
    Y : Theano variable, optional [default: None]
        Has shape ``(m, d)``.  If not given, defaults to ``X``.
    norm : string or callable
        Either a string pointing at a function in ``breze.arch.component.norm``
        or a function that has the same signature as these.
    Returns
    -------
    res : Theano variable
        Has shape ``(n, m)``.
    """
    diff = pairwise_diff(X, Y)
    return distance_matrix_by_diff(diff, norm=norm)


def distance_matrix_by_diff(diff, norm=l2):
    """Return an expression containing the distances given the norm ``norm``
    arrays containing samples.
    Parameters
    ----------
    D : Theano variable
        Has shape ``(n, d, m)`` and represents differences between two
        collections of the same set.
    norm : string or callable
        Either a string pointing at a function in ``breze.arch.component.norm``
        or a function that has the same signature as these.
    Returns
    -------
    res : Theano variable
        Has shape ``(n, m)``.
    """
#    if isinstance(norm, (str, unicode)):
#        norm = lookup(norm, norm_)
    dist_comps = norm(diff, axis=1)
    return dist_comps


def nca_loss(targets, embeddings, proxies_layer):
    """Return the NCA loss for No Fuss DML.
    For details, we refer you to 'Neighbourhood Component Analysis' by
    J Goldberger, S Roweis, G Hinton, R Salakhutdinov (2004).
    Parameters
    ----------
    targets : Theano variable
        An array of shape ``(n,)`` where ``n`` is the number of samples. Each
        entry of the array should be an integer between ``0`` and ``k - 1``,
        where ``k`` is the number of classes.
    embeddings : Theano variable
        An array of shape ``(n, d)`` where each row represents a point in``d``-dimensional space.
    proxies_layer : Lasagne embedding layer
        Should contain An array of shape ``(m, d)`` where each row represents a proxy-point in``d``-dimensional space.
    Returns
    -------
    res : Theano variable
        Scalar containing loss value.
    """

    from misc import distance_matrix
    from misc import l2


    def normalize_rowwise(mat):
        return mat / (l2(mat, axis=1).reshape((mat.shape[0], 1)) + 1e-8)

    embeddings_normed = normalize_rowwise(embeddings)
    # Positive distances
    target_proxies = lasagne.layers.get_output(proxies_layer, targets)[0]
    target_proxies_normed = normalize_rowwise(target_proxies)

    positive_distances = T.diagonal(distance_matrix(target_proxies_normed, embeddings_normed))
    numerator = T.exp(-positive_distances)

    # Negative distances
    proxies_normed = normalize_rowwise(proxies_layer.W)
    all_distances = distance_matrix(embeddings_normed, proxies_normed)
    denominator = T.sum(T.exp(-all_distances), axis=1) - numerator

    # Compute ratio
    loss_vector = -T.log(numerator / (denominator + 1e-8))
    loss = T.mean(loss_vector)
    return loss
