"""Region x region connectivity measures computed from a BOLD window.

Every function takes `window` -- a [num_regions, num_timepoints] array -- and returns a
[num_regions, num_regions] matrix, so any of them can be dropped into the custom-attention
construction in place of Pearson FC.

Some measures are symmetric (correlation, phase synchrony), some are directed and therefore
asymmetric (Granger causality). Attention itself is asymmetric, so the directed ones are worth
trying even though the original FC hypothesis was symmetric.

Gaussian estimators are used for the information-theoretic measures: for a Gaussian the entropy
depends only on the covariance, so mutual information and O-information have closed forms in the
correlation coefficients. That is cheap and stable at 424 regions x 200 timepoints, at the cost
of only seeing linear dependence.
"""
import os
from functools import partial

import numpy as np
from scipy.io import loadmat
from scipy.signal import hilbert
from scipy.stats import rankdata


def _standardize(window):
    """Zero mean, unit variance per region."""
    centered = window - window.mean(axis=1, keepdims=True)
    std = centered.std(axis=1, keepdims=True)
    std[std == 0] = 1e-12
    return centered / std


def _normal_scores(window):
    """Rank-transform each region to Gaussian marginals (the Gaussian-copula transform)."""
    from scipy.special import ndtri
    ranks = np.apply_along_axis(rankdata, 1, window)
    return ndtri(ranks / (window.shape[1] + 1))


def pearson(window):
    """Standard functional connectivity."""
    return np.corrcoef(window)


def spearman(window):
    """Rank correlation -- picks up monotone but non-linear coupling, and is outlier-robust."""
    return np.corrcoef(np.apply_along_axis(rankdata, 1, window))


def partial_correlation(window, shrinkage=0.1):
    """Correlation between two regions with all other regions regressed out, i.e. direct rather
    than indirect coupling. With 424 regions and only 200 timepoints the covariance is singular,
    so it is shrunk toward its diagonal before inversion."""
    cov = np.cov(_standardize(window))
    cov = (1 - shrinkage) * cov + shrinkage * np.eye(len(cov)) * np.trace(cov) / len(cov)
    precision = np.linalg.inv(cov)
    d = np.sqrt(np.diag(precision))
    partial = -precision / np.outer(d, d)
    np.fill_diagonal(partial, 1.0)
    return partial


def mutual_information(window):
    """Gaussian-copula mutual information, -0.5 * log(1 - r^2) on normal-scored data.

    Rank-transforming first makes this invariant to any monotone rescaling of the signal, but it
    is still a monotone function of the copula correlation -- so expect it to rank region pairs
    much like Spearman does, just on a different scale.
    """
    r = np.corrcoef(_normal_scores(window))
    mi = -0.5 * np.log(np.clip(1 - r ** 2, 1e-12, None))
    # Self-MI is infinite; leaving it clipped would put ~300x the off-diagonal weight on the
    # diagonal and, once rows are normalized, collapse attention to the identity.
    np.fill_diagonal(mi, mi[~np.eye(len(mi), dtype=bool)].max())
    return mi


def _phases(window):
    return np.angle(hilbert(_standardize(window), axis=1))


def phase_locking_value(window):
    """PLV: how consistent the phase difference between two regions is, ignoring amplitude.
    1 = perfectly locked at some constant offset, 0 = uniformly distributed phase difference."""
    z = np.exp(1j * _phases(window))
    return np.abs(z @ z.conj().T) / window.shape[1]


def phase_lag_index(window):
    """PLI: consistency of the *sign* of the phase difference. Unlike PLV it ignores zero-lag
    coupling entirely, so it responds to lead-lag relationships rather than instantaneous
    synchrony."""
    phases = _phases(window)
    n = len(phases)
    out = np.empty((n, n))
    for i in range(n):
        out[i] = np.abs(np.mean(np.sign(np.sin(phases[i] - phases)), axis=1))
    np.fill_diagonal(out, 1.0)
    return out


def granger_causality(window, lag=1):
    """Directed: G[i, j] = log(var of j predicted from its own past / var of j predicted from its
    own past plus i's past). Larger means i's history helps forecast j. Order-1 bivariate, solved
    in closed form for every pair at once."""
    z = _standardize(window)
    past, future = z[:, :-lag], z[:, lag:]
    past = past - past.mean(axis=1, keepdims=True)
    future = future - future.mean(axis=1, keepdims=True)
    n, t = future.shape

    gram = past @ past.T                     # uu, vv and the uv cross terms
    cross = past @ future.T                  # cross[i, j] = past_i . future_j
    own_var = np.diag(gram)                  # uu for each region
    own_cross = np.diag(cross)               # uy for each region
    target_var = np.einsum("it,it->i", future, future)

    restricted = target_var - own_cross ** 2 / np.maximum(own_var, 1e-12)

    # Two-predictor least squares for every (source i, target j) pair.
    uu = own_var[None, :]                    # target's own past
    vv = own_var[:, None]                    # source's past
    uv = gram                                # uv[i, j] = past_i . past_j
    uy = own_cross[None, :]
    vy = cross                               # vy[i, j] = past_i . future_j
    det = uu * vv - uv ** 2
    det = np.where(np.abs(det) < 1e-12, np.nan, det)
    a = (vv * uy - uv * vy) / det
    b = (uu * vy - uv * uy) / det
    full = target_var[None, :] - a * uy - b * vy

    gc = np.log(np.maximum(restricted[None, :], 1e-12) / np.maximum(full, 1e-12))
    gc = np.nan_to_num(gc, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(gc, 0.0)
    return np.maximum(gc, 0.0)


def random_baseline(window):
    """Control: random values where a connectivity matrix would go.

    Symmetric, like the real measures. The seed is derived from the window so every subject gets
    its own independent draw (no single lucky matrix) while staying reproducible.
    """
    n = len(window)
    rng = np.random.default_rng(int(np.abs(window).sum()) % (2 ** 32))
    a = rng.standard_normal((n, n))
    return (a + a.T) / 2


def uniform_baseline(window):
    """Control: every region pair weighted identically.

    Once rows are normalized, each token spreads its attention evenly over all regions sharing
    its time patch. Carries no connectivity information whatsoever, so it isolates what the
    block geometry alone is worth -- the floor any real measure has to beat.
    """
    return np.ones((len(window), len(window)))


_SC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "input", "SC_D_AAL424.mat")
_sc_cache = {}


def structural_connectivity(window=None, path=_SC_PATH):
    """Structural connectivity from diffusion tractography, AAL424 template.

    The odd one out: it does not depend on `window` at all. It is a single anatomical template
    shared by every subject, so it asks how much of attention a generic wiring prior explains
    with no functional and no subject-specific information. Heavy-tailed (median ~0.0006 vs mean
    ~0.05) and, unlike FC, it has no homotopic band -- its strongest off-diagonal is at lag 1.
    """
    if path not in _sc_cache:
        if not os.path.exists(path):
            raise FileNotFoundError(f"structural connectivity matrix not found at {path}")
        _sc_cache[path] = np.asarray(loadmat(path)["SC"], dtype=np.float64)
    return _sc_cache[path].copy()


def o_information_triplet(window, max_third=None, seed=0):
    """Mean O-information of the triplet (i, j, k) over third regions k.

    A cheap pairwise reduction of O-information: score each pair by averaging over the third
    variable. Positive means redundancy-dominated, negative synergy-dominated.

    Under a Gaussian model the triplet expression collapses to a closed form in the pairwise
    correlations:

        O(i,j,k) = 0.5 * [ log|R_ijk| - log(1-r_ij^2) - log(1-r_ik^2) - log(1-r_jk^2) ]
        |R_ijk|  = 1 + 2 r_ij r_ik r_jk - r_ij^2 - r_ik^2 - r_jk^2

    Uses the same Gaussian-copula correlation as the greedy version, so the two differ only in
    how they summarize sets onto pairs (mean over third variables vs. min/max over a search).

    `max_third` subsamples the third variable for speed; None uses all regions.
    """
    r = np.corrcoef(_normal_scores(window))
    n = len(r)
    ks = np.arange(n)
    if max_third is not None and max_third < n:
        ks = np.random.default_rng(seed).choice(n, size=max_third, replace=False)

    r2 = np.clip(r ** 2, 0, 1 - 1e-9)
    log_pair = np.log(1 - r2)                       # log(1 - r_ij^2) for every pair

    total = np.zeros((n, n))
    count = np.zeros((n, n))
    for k in ks:
        rik = r[:, k][:, None]                      # r_ik, broadcast over j
        rjk = r[:, k][None, :]                      # r_jk, broadcast over i
        det = 1 + 2 * r * rik * rjk - r2 - rik ** 2 - rjk ** 2
        omega = 0.5 * (np.log(np.clip(det, 1e-12, None))
                       - log_pair - np.log(np.clip(1 - rik ** 2, 1e-12, None))
                       - np.log(np.clip(1 - rjk ** 2, 1e-12, None)))
        valid = np.ones((n, n), dtype=bool)
        valid[k, :] = False
        valid[:, k] = False
        total += np.where(valid, omega, 0.0)
        count += valid
    out = total / np.maximum(count, 1)
    np.fill_diagonal(out, 0.0)
    return out


# --- O-information, Herzog et al. 2025 (bioRxiv 2025.06.19.660516) -------------------------
#
# For a system X^n of n variables (Rosas et al. 2019):
#
#     Omega(X^n) = (n-2) H(X^n) + sum_j [ H(X_j) - H(X^n_-j) ]
#
# Omega > 0 is redundancy-dominated, Omega < 0 synergy-dominated. Entropies use the Gaussian
# copula estimator, as in the paper: rank-transform to Gaussian marginals, then every entropy is
# a log-determinant. The (2*pi*e) constants cancel exactly, and using |R_-j| = |R| (R^-1)_jj the
# whole expression collapses to
#
#     Omega = -log|R| - 0.5 * sum_j log (R^-1)_jj
#
# (verified against the direct definition to machine precision).


def _omega_from_correlation(R):
    """O-information of one set, from its copula correlation matrix."""
    _, logdet = np.linalg.slogdet(R)
    diag_precision = np.diag(np.linalg.inv(R))
    return -logdet - 0.5 * np.sum(np.log(np.clip(diag_precision, 1e-12, None)))


def _omega_of_extensions(R, sets, inv, logdet, diag_inv):
    """O-information of each set extended by each candidate region, all at once.

    Growing a set by one region is a rank-1 bordered update, so instead of factorizing every
    candidate matrix from scratch we reuse the current set's inverse:

        s      = 1 - b' A^-1 b                    (Schur complement of the new region)
        |M|    = |A| * s
        diag(M^-1) = [ diag(A^-1) + (A^-1 b)^2 / s ,  1/s ]

    which turns the whole step into a matrix multiply. Exact, ~15x faster than refactorizing.
    Returns [n_sets, n_regions].
    """
    b = R[sets, :]                                        # set <-> candidate correlations
    u = np.einsum("cij,cjn->cin", inv, b)                 # A^-1 b
    s = np.clip(1.0 - np.einsum("cin,cin->cn", b, u), 1e-10, None)

    logdet_extended = logdet[:, None] + np.log(s)
    diag_extended = diag_inv[:, :, None] + u ** 2 / s[:, None, :]
    sum_log_diag = np.sum(np.log(np.clip(diag_extended, 1e-12, None)), axis=1) + np.log(1.0 / s)
    return -logdet_extended - 0.5 * sum_log_diag


def _set_statistics(R, sets, ridge=1e-6):
    submatrix = R[sets[:, :, None], sets[:, None, :]] + ridge * np.eye(sets.shape[1])
    inv = np.linalg.inv(submatrix)
    _, logdet = np.linalg.slogdet(submatrix)
    return inv, logdet, np.einsum("...ii->...i", inv)


def o_information(window, max_order=10, mode="min", chunk=2048):
    """O-information via the paper's greedy search, as a region x region matrix.

    The paper's O-information scores a *set* of regions, not a pair, so making a matrix from it
    requires a choice. We use the greedy search itself: Herzog et al. seed the search from every
    possible pair, grow the set one region at a time -- each step adding whichever region
    minimizes (or maximizes) the O-information -- and then report the best value over all seed
    pairs. Here we simply keep the per-seed value instead of collapsing it, so entry (i, j) is
    the O-information of the set grown from the pair (i, j). Symmetric by construction.

    mode="min" chases synergy-dominated sets (the paper's focus), "max" chases redundancy.
    max_order caps the set size: the paper grows to 30, where synergy has died out in every
    parcellation, and finds the most synergy-dominated order around 7-12 depending on
    granularity. Cost grows with max_order -- roughly 2 minutes per subject at the default.
    """
    if mode not in {"min", "max"}:
        raise ValueError("mode must be 'min' or 'max'")
    if max_order < 3:
        raise ValueError("O-information is identically zero below order 3")

    R = np.corrcoef(_normal_scores(window))
    n = len(R)
    source, target = np.triu_indices(n, k=1)
    out = np.zeros((n, n))
    pick = np.argmin if mode == "min" else np.argmax
    blocked = np.inf if mode == "min" else -np.inf

    for start in range(0, len(source), chunk):
        i, j = source[start:start + chunk], target[start:start + chunk]
        sets = np.stack([i, j], axis=1)
        rows = np.arange(len(sets))
        omega = None

        while sets.shape[1] < max_order:
            inv, logdet, diag_inv = _set_statistics(R, sets)
            candidates = _omega_of_extensions(R, sets, inv, logdet, diag_inv)
            candidates[rows[:, None], sets] = blocked        # can't re-add a member
            best = pick(candidates, axis=1)
            omega = candidates[rows, best]
            sets = np.concatenate([sets, best[:, None]], axis=1)

        out[i, j] = out[j, i] = omega

    return out


MEASURES = {
    "pearson (FC)": pearson,
    "spearman": spearman,
    "partial correlation": partial_correlation,
    "mutual information": mutual_information,
    "phase locking (PLV)": phase_locking_value,
    "phase lag index (PLI)": phase_lag_index,
    "granger causality": granger_causality,
    "structural connectivity": structural_connectivity,
    # At order 3 the greedy search is a single exhaustive sweep: every third region is scored
    # and the extreme is kept. "synergy" is the most negative Omega(i,j,k) over k, "redundancy"
    # the most positive. Same 422 values, opposite ends.
    "o-information (synergy)": partial(o_information, max_order=3, mode="min"),
    "o-information (redundancy)": partial(o_information, max_order=3, mode="max"),
    "random (baseline)": random_baseline,
    "uniform (baseline)": uniform_baseline,
}

SYMMETRIC = {"pearson (FC)", "spearman", "partial correlation", "mutual information",
             "phase locking (PLV)", "phase lag index (PLI)", "structural connectivity",
             "o-information (synergy)", "o-information (redundancy)",
             "random (baseline)", "uniform (baseline)"}
