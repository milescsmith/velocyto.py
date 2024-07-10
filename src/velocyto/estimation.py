from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.optimize
from loguru import logger

from velocyto.speedboosted import (
    _colDeltaCor,
    _colDeltaCorLog10,
    _colDeltaCorLog10partial,
    _colDeltaCorpartial,
    _colDeltaCorSqrt,
    _colDeltaCorSqrtpartial,
)


def colDeltaCor(emat: npt.ArrayLike, dmat: npt.ArrayLike, threads: int | None = None) -> npt.ArrayLike:
    """Calculate the correlation between the displacement (d[:,i])
    and the difference between a cell and every other (e - e[:, i])

    Parallel cython+OpenMP implemetation

    Arguments
    ---------
    emat: npt.ArrayLike (ngenes, ncells)
        gene expression matrix
    dmat: npt.ArrayLike (ngenes, ncells)
        gene velocity/displacement matrix
    threads: int
        number of parallel threads to use
    """
    import multiprocessing

    if threads is None:
        num_threads = int(multiprocessing.cpu_count() / 2)
    else:
        num_threads = max(threads, multiprocessing.cpu_count())
    out = np.zeros((emat.shape[1], emat.shape[1]))
    _colDeltaCor(emat, dmat, out, num_threads)
    return out


def colDeltaCorpartial(
    emat: npt.ArrayLike, dmat: npt.ArrayLike, ixs: npt.ArrayLike, threads: int | None = None
) -> npt.ArrayLike:
    """Calculate the correlation between the displacement (d[:,i])
    and the difference between a cell and every other (e - e[:, i])

    Parallel cython+OpenMP implemetation

    Arguments
    ---------
    emat: npt.ArrayLike (ngenes, ncells)
        gene expression matrix
    dmat: npt.ArrayLike (ngenes, ncells)
        gene velocity/displacement matrix
    ixs: the neighborhood matrix (ncells, nneighbours)
        ixs[i, k] is the kth neighbour to the cell i
    threads: int
        number of parallel threads to use
    """
    import multiprocessing

    if threads is None:
        num_threads = int(multiprocessing.cpu_count() / 2)
    else:
        num_threads = max(threads, multiprocessing.cpu_count())
    out = np.zeros((emat.shape[1], emat.shape[1]))
    emat = np.require(emat, requirements="C")
    ixs = np.require(ixs, requirements="C").astype(np.intp)
    _colDeltaCorpartial(emat, dmat, out, ixs, num_threads)
    return out


def colDeltaCorLog10(
    emat: npt.ArrayLike, dmat: npt.ArrayLike, threads: int | None = None, psc: float = 1.0
) -> npt.ArrayLike:
    """Calculate the correlation between the displacement (d[:,i])
    and the difference between a cell and every other (e - e[:, i])

    Parallel cython+OpenMP implemetation

    Arguments
    ---------
    emat: npt.ArrayLike (ngenes, ncells)
        gene expression matrix
    dmat: npt.ArrayLike (ngenes, ncells)
        gene velocity/displacement matrix
    threads: int
        number of parallel threads to use
    """
    import multiprocessing

    if threads is None:
        num_threads = int(multiprocessing.cpu_count() / 2)
    else:
        num_threads = max(threads, multiprocessing.cpu_count())
    out = np.zeros((emat.shape[1], emat.shape[1]))
    _colDeltaCorLog10(emat, dmat, out, num_threads, psc)
    return out


def colDeltaCorLog10partial(
    emat: npt.ArrayLike,
    dmat: npt.ArrayLike,
    ixs: npt.ArrayLike,
    threads: int | None = None,
    psc: float = 1.0,
) -> npt.ArrayLike:
    """Calculate the correlation between the displacement (d[:,i])
    and the difference between a cell and every other (e - e[:, i])

    Parallel cython+OpenMP implemetation

    Arguments
    ---------
    emat: npt.ArrayLike (ngenes, ncells)
        gene expression matrix
    dmat: npt.ArrayLike (ngenes, ncells)
        gene velocity/displacement matrix
    ixs: the neighborhood matrix (ncells, nneighbours)
        ixs[i, k] is the kth neighbour to the cell i
    threads: int
        number of parallel threads to use
    """
    import multiprocessing

    if threads is None:
        num_threads = int(multiprocessing.cpu_count() / 2)
    else:
        num_threads = max(threads, multiprocessing.cpu_count())
    out = np.zeros((emat.shape[1], emat.shape[1]))
    emat = np.require(emat, requirements="C")
    ixs = np.require(ixs, requirements="C").astype(np.intp)
    _colDeltaCorLog10partial(emat, dmat, out, ixs, num_threads, psc)
    return out


def colDeltaCorSqrt(
    emat: npt.ArrayLike, dmat: npt.ArrayLike, threads: int | None = None, psc: float = 0.0
) -> npt.ArrayLike:
    """Calculate the correlation between the displacement (d[:,i])
    and the difference between a cell and every other (e - e[:, i])

    Parallel cython+OpenMP implemetation

    Arguments
    ---------
    emat: npt.ArrayLike (ngenes, ncells)
        gene expression matrix
    dmat: npt.ArrayLike (ngenes, ncells)
        gene velocity/displacement matrix
    threads: int
        number of parallel threads to use
    """
    import multiprocessing

    if threads is None:
        num_threads = int(multiprocessing.cpu_count() / 2)
    else:
        num_threads = max(threads, multiprocessing.cpu_count())
    out = np.zeros((emat.shape[1], emat.shape[1]))
    _colDeltaCorSqrt(emat, dmat, out, num_threads, psc)
    return out


def colDeltaCorSqrtpartial(
    emat: npt.ArrayLike,
    dmat: npt.ArrayLike,
    ixs: npt.ArrayLike,
    threads: int | None = None,
    psc: float = 0.0,
) -> npt.ArrayLike:
    """Calculate the correlation between the displacement (d[:,i])
    and the difference between a cell and every other (e - e[:, i])

    Parallel cython+OpenMP implemetation

    Arguments
    ---------
    emat: npt.ArrayLike (ngenes, ncells)
        gene expression matrix
    dmat: npt.ArrayLike (ngenes, ncells)
        gene velocity/displacement matrix
    ixs: the neighborhood matrix (ncells, nneighbours)
        ixs[i, k] is the kth neighbour to the cell i
    threads: int
        number of parallel threads to use
    """
    import multiprocessing

    if threads is None:
        num_threads = int(multiprocessing.cpu_count() / 2)
    else:
        num_threads = max(threads, multiprocessing.cpu_count())
    out = np.zeros((emat.shape[1], emat.shape[1]))
    emat = np.require(emat, requirements="C")
    ixs = np.require(ixs, requirements="C").astype(np.intp)
    _colDeltaCorSqrtpartial(emat, dmat, out, ixs, num_threads, psc)
    return out


def _fit1_slope(y: npt.ArrayLike, x: npt.ArrayLike) -> float:
    """Simple function that fit a linear regression model without intercept"""
    if not np.any(x):
        return np.NAN
    elif not np.any(y):
        return 0
    else:
        result, _ = scipy.optimize.nnls(x[:, None], y)  # Fastest but costrains result >= 0
        return result[0]
        # Second fastest: m, _ = scipy.optimize.leastsq(lambda m: x*m - y, x0=(0,))
        # Third fastest: m = scipy.optimize.minimize_scalar(lambda m: np.sum((x*m - y)**2 )).x
        # Before I was doinf fastest: scipy.optimize.minimize_scalar(lambda m: np.sum((y - m * x)**2), bounds=(0, 3), method="bounded").x
        # Optionally one could clip m if high value make no sense
        # m = np.clip(m,0,3)


def _fit1_slope_weighted(
    y: npt.ArrayLike,
    x: npt.ArrayLike,
    w: npt.ArrayLike,
    limit_gamma: bool = False,
    bounds: tuple[float, float] = (0, 20),
) -> float:
    """Simple function that fit a weighted linear regression model without intercept"""
    if not np.any(x):
        m = np.NAN  # It is definetelly not at steady state!!!
    elif not np.any(y):
        m = 0
    elif limit_gamma:
        if np.median(y) > np.median(x):
            high_x = x > np.percentile(x, 90)
            up_gamma = np.percentile(y[high_x], 10) / np.median(x[high_x])
            up_gamma = np.maximum(1.5, up_gamma)
        else:
            up_gamma = 1.5  # Just a bit more than 1
        m = scipy.optimize.minimize_scalar(
            lambda m: np.sum(w * (x * m - y) ** 2),
            bounds=(1e-8, up_gamma),
            method="bounded",
        ).x
    else:
        m = scipy.optimize.minimize_scalar(lambda m: np.sum(w * (x * m - y) ** 2), bounds=bounds, method="bounded").x
    return m


def _fit1_slope_weighted_offset(
    y: npt.ArrayLike,
    x: npt.ArrayLike,
    w: npt.ArrayLike,
    fixperc_q: bool = False,
    limit_gamma: bool = False,
) -> tuple[float, float]:
    """Function that fits a weighted linear regression model with intercept
    with some adhoc
    """
    if not np.any(x):
        m = (np.NAN, 0)  # It is definetelly not at steady state!!!
    elif not np.any(y):
        m = (0, 0)
    elif fixperc_q:
        m1 = np.percentile(y[x <= np.percentile(x, 1)], 50)
        m0 = scipy.optimize.minimize_scalar(
            lambda m: np.sum(w * (x * m - y + m1) ** 2),
            bounds=(0, 20),
            method="bounded",
        ).x
        m = (m0, m1)
    else:
        # m, _ = scipy.optimize.leastsq(lambda m: np.sqrt(w) * (-y + x * m[0] + m[1]), x0=(0, 0))  # This is probably faster but it can have negative slope
        # NOTE: The up_gamma is to deal with cases where consistently y > x. Those should have positive velocity everywhere
        if limit_gamma:
            if np.median(y) > np.median(x):
                high_x = x > np.percentile(x, 90)
                up_gamma = np.percentile(y[high_x], 10) / np.median(x[high_x])
                up_gamma = np.maximum(1.5, up_gamma)
            else:
                up_gamma = 1.5  # Just a bit more than 1
        else:
            up_gamma = 20
        up_q = 2 * np.sum(y * w) / np.sum(w)
        m = scipy.optimize.minimize(
            lambda m: np.sum(w * (-y + x * m[0] + m[1]) ** 2),
            x0=(0.1, 1e-16),
            method="L-BFGS-B",
            bounds=[(1e-8, up_gamma), (0, up_q)],
        ).x  # If speedup is needed either the gradient or numexpr could be used
    return m[0], m[1]


def _fit1_slope_offset(y: npt.ArrayLike, x: npt.ArrayLike, fixperc_q: bool = False) -> tuple[float, float]:
    """Simple function that fit a linear regression model with intercept"""
    if not np.any(x):
        m = (np.NAN, 0)  # It is definetelly not at steady state!!!
    elif not np.any(y):
        m = (0, 0)
    elif fixperc_q:
        m1 = np.percentile(y[x <= np.percentile(x, 1)], 50)
        m0 = scipy.optimize.minimize_scalar(
            lambda m: np.sum((x * m - y + m1) ** 2),
            bounds=(0, 20),
            method="bounded",
        ).x
        m = (m0, m1)
    else:
        m, _ = scipy.optimize.leastsq(lambda m: -y + x * m[0] + m[1], x0=(0, 0))
    return m[0], m[1]


def fit_slope(Y: npt.ArrayLike, X: npt.ArrayLike) -> npt.ArrayLike:
    """Loop through the genes and fits the slope

    Y: npt.ArrayLike, shape=(genes, cells)
        the dependent variable (unspliced)
    X: npt.ArrayLike, shape=(genes, cells)
        the independent variable (spliced)
    """
    # NOTE this could be easily parallelized
    return np.fromiter(
        (_fit1_slope(Y[i, :], X[i, :]) for i in range(Y.shape[0])),
        dtype="float32",
        count=Y.shape[0],
    )


def fit_slope_offset(Y: npt.ArrayLike, X: npt.ArrayLike, fixperc_q: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Loop through the genes and fits the slope

    Y: npt.ArrayLike, shape=(genes, cells)
        the dependent variable (unspliced)
    X: npt.ArrayLike, shape=(genes, cells)
        the independent variable (spliced)
    """
    # NOTE this could be easily parallelized
    slopes = np.zeros(Y.shape[0], dtype="float32")
    offsets = np.zeros(Y.shape[0], dtype="float32")
    for i in range(Y.shape[0]):
        m, q = _fit1_slope_offset(Y[i, :], X[i, :], fixperc_q)
        slopes[i] = m
        offsets[i] = q
    return slopes, offsets


def fit_slope_weighted(
    Y: npt.ArrayLike,
    X: npt.ArrayLike,
    W: npt.ArrayLike,
    return_R2: bool = False,
    limit_gamma: bool = False,
    bounds: tuple[float, float] = (0, 20),
) -> npt.ArrayLike | tuple[np.ndarray, np.ndarray]:
    """Loop through the genes and fits the slope

    Y: npt.ArrayLike, shape=(genes, cells)
        the dependent variable (unspliced)
    X: npt.ArrayLike, shape=(genes, cells)
        the independent variable (spliced)
    W: npt.ArrayLike, shape=(genes, cells)
        the weights that will scale the square residuals
    """
    # NOTE this could be easily parallelized
    # slopes = np.fromiter((_fit1_slope_weighted(Y[i, :], X[i, :], W[i, :], bounds=bounds) for i in range(Y.shape[0])),
    #                     dtype="float32",
    #                     count=Y.shape[0])

    slopes = np.zeros(Y.shape[0], dtype="float32")
    # offsets = np.zeros(Y.shape[0], dtype="float32")
    if return_R2:
        R2 = np.zeros(Y.shape[0], dtype="float32")
    for i in range(Y.shape[0]):
        m = _fit1_slope_weighted(Y[i, :], X[i, :], W[i, :], limit_gamma)
        slopes[i] = m
        if return_R2:
            # NOTE: the coefficient of determination is not weighted but the fit is
            with np.errstate(divide="ignore", invalid="ignore"):
                SSres = np.sum((m * X[i, :] - Y[i, :]) ** 2)
                SStot = np.sum((Y[i, :].mean() - Y[i, :]) ** 2)
                _R2 = 1 - (SSres / SStot)
            R2[i] = _R2 if np.isfinite(_R2) else -1e16
    return (slopes, R2) if return_R2 else slopes


def fit_slope_weighted_offset(
    Y: npt.ArrayLike,
    X: npt.ArrayLike,
    W: npt.ArrayLike,
    fixperc_q: bool = False,
    return_R2: bool = True,
    limit_gamma: bool = False,
) -> Any:
    """Loop through the genes and fits the slope

    Y: npt.ArrayLike, shape=(genes, cells)
        the dependent variable (unspliced)
    X: npt.ArrayLike, shape=(genes, cells)
        the independent variable (spliced)
    """
    # NOTE this could be easily parallelized
    slopes = np.zeros(Y.shape[0], dtype="float32")
    offsets = np.zeros(Y.shape[0], dtype="float32")
    if return_R2:
        R2 = np.zeros(Y.shape[0], dtype="float32")
    for i in range(Y.shape[0]):
        m, q = _fit1_slope_weighted_offset(Y[i, :], X[i, :], W[i, :], fixperc_q, limit_gamma)
        slopes[i] = m
        offsets[i] = q
        if return_R2:
            # NOTE: the coefficient of determination is not weighted but the fit is
            with np.errstate(divide="ignore", invalid="ignore"):
                SSres = np.sum((m * X[i, :] + q - Y[i, :]) ** 2)
                SStot = np.sum((Y[i, :].mean() - Y[i, :]) ** 2)
                _R2 = 1 - (SSres / SStot)
            R2[i] = _R2 if np.isfinite(_R2) else -1e16
    return (slopes, offsets, R2) if return_R2 else (slopes, offsets)


def clusters_stats(
    U: npt.ArrayLike,
    S: npt.ArrayLike,
    clusters_uid: npt.ArrayLike,
    cluster_ix: npt.ArrayLike,
    size_limit: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the averages per cluster

    If the cluster is too small (size<size_limit) the average of the toal is reported instead
    """
    U_avgs = np.zeros((S.shape[0], len(clusters_uid)))
    S_avgs = np.zeros((S.shape[0], len(clusters_uid)))
    # avgU_div_avgS = np.zeros((S.shape[0], len(clusters_uid)))
    # slopes_by_clust = np.zeros((S.shape[0], len(clusters_uid)))
    for i, uid in enumerate(clusters_uid):
        cluster_filter = cluster_ix == i
        n_cells = np.sum(cluster_filter)
        logger.info(f"Cluster: {uid} ({n_cells} cells)")
        if n_cells > size_limit:
            U_avgs[:, i], S_avgs[:, i] = U[:, cluster_filter].mean(1), S[:, cluster_filter].mean(1)
        else:
            U_avgs[:, i], S_avgs[:, i] = U.mean(1), S.mean(1)

    return U_avgs, S_avgs
