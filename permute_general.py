#!/usr/bin/env python3
""" Permute beer script

Run directly from IPython with:

    run permute_general.py

When you want to move onto the more advanced tests:

    run permute_general.py
    test_random()  # Run the random test
    test_median()  # Run the median test
    test_alternative()  # Run the alternative test

Or - if you are brave, from the terminal command line:

    pytest permute_general
"""

import numpy as np

rng = np.random.default_rng()

import pandas as pd
pd.set_option('mode.copy_on_write', True)


# Load up example data frame
_mosquitoes = pd.read_csv('data/mosquito_beer.csv')
EG_DF = _mosquitoes[_mosquitoes['test'] == 'after']


def permutation_test(data,
                     group_col_name,
                     value_col_name,
                     summary_func='mean',
                     alternative='greater',
                     n_iters=10_000):
    # Your code here.
    return actual_diff, fake_diffs, n_alt / n_iters


def test_example():
    actual_stat, fake_diffs, p = permutation_test(EG_DF,
                                                  'group',
                                                  'activated')
    print('Actual diff', actual_stat)
    print('First 10 fake differences', fake_diffs[:10])
    print('p value', p)
    assert np.isclose(actual_stat, -4.433333333333334)
    assert p < 0.065
    assert p > 0.05
    assert np.round(np.mean(fake_diffs), 1) == 0


def test_random():
    groups = np.repeat(['one', 'two'], [100, 200])
    values = rng.normal(10, 2, size=len(groups))
    test_df = pd.DataFrame({'labels': groups,
                            'numbers': values})
    actual_stat, fake_diffs, p = permutation_test(test_df,
                                                  'labels',
                                                  'numbers')
    assert np.round(np.abs(actual_stat)) < 0.5
    assert p < 1
    assert p > 0.01  # One in a thousand times, this will fail.
    assert np.round(np.mean(fake_diffs), 1) == 0


def test_median():
    actual_stat, fake_diffs, p = permutation_test(EG_DF,
                                                  'group',
                                                  'activated',
                                                  'median')
    expected = np.diff(EG_DF.groupby('group')['activated'].median())[0]
    assert np.isclose(actual_stat, expected)
    assert p < 0.065
    assert p > 0.05
    assert np.round(np.mean(fake_diffs), 1) == 0


def test_alternative():
    actual_stat, fake_diffs, p = permutation_test(EG_DF,
                                                  'group',
                                                  'activated',
                                                  alternative='less')
    assert np.isclose(actual_stat, -4.433333333333334)
    assert p > 0.9
    assert np.round(np.mean(fake_diffs), 1) == 0


# Can you think of any other tests?
# For example, of the fake differences?
# What would you do to implement the 'two-sided' alternative?

# Run example test
test_example()
