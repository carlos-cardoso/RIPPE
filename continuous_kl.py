from __future__ import print_function, division, absolute_import
# https://mail.python.org/pipermail/scipy-user/2011-May/029521.html

import numpy as np

def KL_from_samples(x, y):
  """Compute the Kullback-Leibler divergence between two multivariate samples.

  Parameters
  ----------
  x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
  y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.

  Returns
  -------
  out : float
    The estimated Kullback-Leibler divergence D(P||Q).

  References
  ----------
  Pérez-Cruz, F. Kullback-Leibler divergence estimation of
continuous distributions IEEE International Symposium on Information
Theory, 2008.
  """
  from scipy.spatial import cKDTree as KDTree

  # Check the dimensions are consistent
  x = np.atleast_2d(x)
  y = np.atleast_2d(y)

  n, d = x.shape
  m, dy = y.shape

  assert(d == dy)


  # Build a KD tree representation of the samples and find the nearest neighbour
  # of each point in x.
  xtree = KDTree(x)
  ytree = KDTree(y)

  # Get the first two nearest neighbours for x, since the closest one is the
  # sample itself.
  r = xtree.query(x, k=2, eps=.01, p=2)[0][:, 1]
  s = ytree.query(x, k=1, eps=.01, p=2)[0]

  # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
  # on the first term of the right hand side.
  #return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))
  # or the negation can be avoided like:
  return np.log(s/r).sum() * d / n + np.log(m / (n - 1.))

def KL_from_distributions(x, y):
  """Compute the Kullback-Leibler divergence between two multivariate Normal samples.

  Parameters
  ----------
  x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
  y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.

  Returns
  -------
  out : float
    The estimated Kullback-Leibler divergence D(P||Q).

  References
  ----------
  https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/distributions/mvn_linear_operator.py
  """

  # Check the dimensions are consistent
  x = np.atleast_2d(x)
  y = np.atleast_2d(y)

  n, d = x.shape
  m, dy = y.shape
  assert(d == dy)

  x_mean, x_cov = np.mean(x, axis=0), np.cov(x, rowvar=False)
  y_mean, y_cov = np.mean(y, axis=0), np.cov(y, rowvar=False)
  #
  # x_scale, y_scale = np.linalg.cholesky(x_cov), np.linalg.cholesky(y_cov)
  #
  # y_scale_inv_x_scale = np.linalg.solve(y_scale, x_scale)
  #
  # kl_div = (np.linalg.slogdet(y_scale)[1]-np.linalg.slogdet(x_scale)[1] +
  #           0.5 * (-d + np.linalg.norm(y_scale_inv_x_scale, ord="fro")**2 +
  #                  np.linalg.norm(np.linalg.solve(y_scale, (y_mean-x_mean)[..., np.newaxis]), ord="fro")**2))
  return KL_from_distribution_params(x_mean, x_cov, y_mean, y_cov)

def KL_from_distribution_params(P_mean, P_cov, Q_mean, Q_cov):


  x_scale, y_scale = np.linalg.cholesky(P_cov), np.linalg.cholesky(Q_cov)

  y_scale_inv_x_scale = np.linalg.solve(y_scale, x_scale)

  kl_div = (np.linalg.slogdet(y_scale)[1]-np.linalg.slogdet(x_scale)[1] +
            0.5 * (-len(P_mean) + np.linalg.norm(y_scale_inv_x_scale, ord="fro")**2 +
                   np.linalg.norm(np.linalg.solve(y_scale, (Q_mean-P_mean)[..., np.newaxis]), ord="fro")**2))
  return kl_div


if __name__ == "__main__":

  P_mean = np.zeros(2, dtype=np.float)
  P_cov = np.eye(2, dtype=np.float)
  Q_mean = np.array([0.5, -0.5])
  Q_cov = np.array([[0.5, 0.1], [0.1, 0.3]])
  P_samples = np.random.multivariate_normal(P_mean, P_cov, size=100)
  Q_samples = np.random.multivariate_normal(Q_mean, Q_cov, size=10)
  print("estimate from samples P||Q: ", KL_from_samples(P_samples, Q_samples))
  print("estimate from distributions P||Q: ", KL_from_distributions(P_samples, Q_samples))
  print("estimate from distribution params P||Q: ", KL_from_distribution_params(P_mean, P_cov, Q_mean, Q_cov))
  print("estimate from P||Q GT: ~", 1.75)
  print("estimate from samples Q||P: ", KL_from_samples(Q_samples, P_samples))
  print("estimate from distributions Q||P: ", KL_from_distributions(Q_samples, P_samples))
  print("estimate from distribution params Q||P: ", KL_from_distribution_params(Q_mean, Q_cov, P_mean, P_cov))
  print("estimate from Q||P GT: ~", 0.65)