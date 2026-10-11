"""Relative sufficiency gate and minimum predicted-token selection.

No labels, generated text, recoverability scores, or model calls enter this API.
"""
import numpy as np

RATIOS = np.array([.3, .4, .5, .6, .7, .8, .9, 1.0])

def _validate(scores, ratios):
    scores = np.asarray(scores, dtype=float)
    ratios = np.asarray(ratios, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(ratios):
        raise ValueError('Scores must have shape [queries, ratios].')
    if not len(ratios) or not np.isfinite(ratios).all() or ratios[-1] != 1 or np.any(np.diff(ratios) <= 0) or ratios[0] <= 0:
        raise ValueError('Ratios must increase strictly and end in full retention (1).')
    if not np.isfinite(scores).all():
        raise ValueError('Sufficiency scores must be finite.')
    return scores, ratios

def admissible(scores, *, ratios=RATIOS, min_ratio=.5, slack=0.):
    scores, ratios = _validate(scores, ratios)
    if not np.isfinite(slack) or slack < 0 or not np.isfinite(min_ratio) or not 0 < min_ratio <= 1:
        raise ValueError('Require slack >= 0 and 0 < min_ratio <= 1.')
    allowed = (ratios[None, :] >= min_ratio) & (scores >= scores[:, -1:] - slack)
    allowed[:, -1] = True
    return allowed

def select_ratios(scores, costs, *, ratios=RATIOS, min_ratio=.5, slack=0.):
    scores, ratios = _validate(scores, ratios)
    costs = np.asarray(costs, dtype=float)
    if costs.shape != scores.shape or not np.isfinite(costs).all() or np.any(costs <= 0):
        raise ValueError('Costs must be finite, positive, and match the score matrix.')
    allowed = admissible(scores, ratios=ratios, min_ratio=min_ratio, slack=slack)
    # np.argmin breaks exact cost ties in increasing ratio order, as in V13.
    actions = np.argmin(np.where(allowed, costs, np.inf), axis=1)
    return {'actions': actions, 'ratios': ratios[actions], 'allowed': allowed}
