import numpy as np
from scipy import stats
import argparse


def compute_p_value(m1, s1, m2, s2, n):
    # t = (m1 - m2)/np.sqrt(s1 ** 2 + s2 ** 2)
    # dof = (s1 ** 2 + s2 **2) ** 2/(s1 ** 4 + s2 ** 4) * (n -1)
    # p_value = scipy.stats.t(dof).pdf(t)
    results = stats.ttest_ind_from_stats(m1, s1, n, m2, s2, n, equal_var=False)
    return results


if __name__ == '__main__':
    """
    We assume that both distributions have the same sample size
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1", type=float)
    parser.add_argument('--s1', type=float)
    parser.add_argument('--m2', type=float)
    parser.add_argument('--s2', type=float)
    parser.add_argument('--n', type=float, default=4., help='sample size')
    args = parser.parse_args()

    p_value = compute_p_value(args.m1, args.s1, args.m2, args.s2, args.n)
    print(p_value)

