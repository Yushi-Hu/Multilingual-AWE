import numpy as np


def add_deltas(feat):
  feat = np.pad(feat, ((2, 2), (0, 0)), 'edge')
  d = feat[2:, :] - feat[:-2, :]
  dd = d[2:, :] - d[:-2, :]
  return np.concatenate((feat[2:-2, :], d[1:-1], dd), axis=1)


def stack(feat, rstack=1, lstack=0):
  deci_rate = rstack + lstack + 1
  n, d = feat.shape
  feat_stack = np.zeros((n, d * deci_rate))

  for r in range(-lstack, rstack + 1):
    start = d * (r + lstack)
    stop = d * (r + lstack + 1)
    feat_stack[:, start:stop] = np.roll(feat, -r, axis=0)

  return feat_stack[::deci_rate]
