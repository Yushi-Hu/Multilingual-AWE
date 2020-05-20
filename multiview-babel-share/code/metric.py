import scipy
import scipy.spatial
import numpy as np


def compute_precision(pairs):
  """Calculate precision"""

  return np.cumsum(pairs) / np.arange(1, len(pairs) + 1)


def compute_recall(pairs):
  """Calculate recall"""

  return np.cumsum(pairs) / np.sum(pairs)


def compute_ap(pairs):
  """Calculate average precision"""

  pos_indices = np.arange(1, np.sum(pairs) + 1)
  all_indices = np.arange(1, len(pairs) + 1)[pairs]

  return np.sum(pos_indices / all_indices) / np.sum(pairs)


def compute_prb(pairs):
  """Calculate precision-recall breakeven"""

  precision = compute_precision(pairs)

  # Multiple precisions can be at single recall point, take max
  for i in range(len(pairs) - 2, -1, -1):
    precision[i] = max(precision[i], precision[i + 1])

  recall = compute_recall(pairs)
  i = np.argmin(np.abs(recall - precision))

  return (recall[i] + precision[i]) / 2.


def acoustic_ap(embs, ids, k=None, low_mem=False):
  """Calculate acoustic average precision.

  Note:
    If you want an accuracy of +/-0.0005 (about 2^-11), the maximum
    size that the number can be is 1. Any larger than this and the
    distance between floating point numbers is greater than 0.0005.
  """

  n = len(embs)

  # Get pairwise distances
  dists = scipy.spatial.distance.pdist(embs, metric="cosine")
  if low_mem:
    dists /= 2.0
    dists = dists.astype(np.float16)
  else:
    dists = dists.astype(np.float32)
  del embs

  # Get sorted indices
  indices = dists.argsort()
  del dists

  # Get boolean pairs
  pairs = np.zeros(n * (n - 1) // 2, dtype=np.bool)
  i = 0
  for j in range(n):
    pairs[i:(i + n - j - 1)][ids[j] == ids[j + 1:]] = True
    i += n - j - 1
  del ids

  # Sort pairs by distance
  if k is None:
    pairs = pairs[indices]
  else:
    pairs = pairs[indices[:int(k)]]
  del indices

  return compute_ap(pairs)


def crossview_ap(embs1, ids1, embs2, ids2, k=None, low_mem=False):
  """Calculate crossview average precision.

  Note:
    If you want an accuracy of +/-0.0005 (about 2^-11), the maximum
    size that the number can be is 1. Any larger than this and the
    distance between floating point numbers is greater than 0.0005.
  """

  n, m = len(embs1), len(embs2)

  # Get pairwise distances
  dists = scipy.spatial.distance.cdist(embs1, embs2, metric="cosine")
  if low_mem:
    dists /= 2.0
    dists = dists.astype(np.float16)
  else:
    dists = dists.astype(np.float32)
  del embs1, embs2

  # Get sorted indices
  indices = dists.ravel().argsort()
  del dists

  # Get boolean pairs
  pairs = np.zeros((n, m), dtype=np.bool)
  for j in range(m):
    pairs[ids1 == ids2[j], j] = True
  del ids1, ids2

  # Sort pairs by distance
  if k is None:
    pairs = pairs.ravel()[indices]
  else:
    pairs = pairs.ravel()[indices[:int(k)]]
  del indices

  return compute_ap(pairs)
