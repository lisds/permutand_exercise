""" Permutation on mosquito data
"""

import numpy as np

rng = np.random.default_rng()

import pandas as pd

pd.set_option('mode.copy_on_write', True)

mosquitoes = pd.read_csv('data/mosquito_beer.csv')

after = mosquitoes.loc[mosquitoes['test'] == 'after',
                       ['group', 'activated']]

groups = after['group']
activated = after['activated']

means = activated.groupby(groups).mean()
actual_diff = np.diff(means)[0]

n_subjects = len(groups)

n_iters = 10_000
fake_diffs = np.zeros(n_iters)

for i in range(n_iters):
    shuffled = rng.permuted(groups)
    fake_means = activated.groupby(shuffled).mean()
    fake_diffs[i] = np.diff(fake_means)[0]

n_lte = np.count_nonzero(fake_diffs <= actual_diff)
p = n_lte / n_iters

print('p', p)

import scipy.stats as sps

res = sps.ttest_ind(activated[groups == 'beer'],
                    activated[groups == 'water'],
                    alternative='greater')


import statsmodels.formula.api as smf
sm_res = smf.ols('activated ~ group', data=after).fit().summary()
