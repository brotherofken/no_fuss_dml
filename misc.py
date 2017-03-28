#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:56:38 2017

Stolen from https://github.com/breze-no-salt/breze/blob/master/breze/arch/component
"""

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
