"""
Split tests.

The property under test is the absence of a patient on both sides of a boundary, in the
held out split and in every cross validation fold alike. It cannot be observed in any
training curve or any metric, so it has to be asserted here instead.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from preparing.splitting import (
    DEFAULT_TEST_PATIENTS,
    RANDOM_SERIES,
    SELECTED_SERIES,
    SHARED_SUBJECT_RECORDS,
    format_split,
    group_kfold,
    group_records,
    holdout_split,
    patient_of,
    record_series,
    verify_folds,
    verify_split,
)

MITDB = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
         '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
         '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
         '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
         '222', '223', '228', '230', '231', '232', '233', '234']


# --- patient identity ----------------------------------------------------

def test_records_from_one_patient_share_an_identifier():
    """MIT-BIH documents 201 and 202 as coming from the same person."""
    assert patient_of('201') == patient_of('202')


def test_unrelated_records_keep_their_own_identifier():
    assert patient_of('100') == '100'
    assert patient_of('100') != patient_of('101')


def test_grouping_merges_only_the_documented_pair():
    groups = group_records(MITDB)
    assert len(groups) == len(MITDB) - len(SHARED_SUBJECT_RECORDS)
    assert sorted(groups[patient_of('201')]) == ['201', '202']


def test_series_membership_follows_the_record_number():
    assert record_series('100') == RANDOM_SERIES
    assert record_series('234') == SELECTED_SERIES


# --- held out pool -------------------------------------------------------

def test_the_test_pool_holds_the_requested_number_of_patients():
    split = holdout_split(MITDB, n_test_patients=5)
    assert len({patient_of(record) for record in split['test']}) == 5


def test_the_two_pools_partition_the_archive():
    split = holdout_split(MITDB)
    assert sorted(split['development'] + split['test']) == sorted(MITDB)


def test_no_patient_appears_in_both_pools():
    split = holdout_split(MITDB)
    development = {patient_of(record) for record in split['development']}
    test = {patient_of(record) for record in split['test']}
    assert not development & test


def test_a_patient_takes_both_of_his_records_with_him():
    for seed in range(30):
        split = holdout_split(MITDB, seed=seed)
        for part in ('development', 'test'):
            present = {'201', '202'} & set(split[part])
            assert present in (set(), {'201', '202'})


def test_the_draw_is_deterministic():
    assert holdout_split(MITDB, seed=7) == holdout_split(MITDB, seed=7)


def test_different_seeds_draw_different_patients():
    assert holdout_split(MITDB, seed=1) != holdout_split(MITDB, seed=2)


def test_the_draw_ignores_the_order_of_the_input():
    """A reordered record list is the same archive and must give the same split."""
    assert holdout_split(MITDB, seed=3) == holdout_split(list(reversed(MITDB)), seed=3)


def test_stratification_puts_both_families_in_the_test_pool():
    split = holdout_split(MITDB, n_test_patients=DEFAULT_TEST_PATIENTS, stratify=True)
    series = {record_series(record) for record in split['test']}
    assert series == {RANDOM_SERIES, SELECTED_SERIES}


def test_an_impossible_number_of_test_patients_is_rejected():
    with pytest.raises(ValueError, match='test patients'):
        holdout_split(MITDB, n_test_patients=0)
    with pytest.raises(ValueError, match='test patients'):
        holdout_split(MITDB, n_test_patients=len(MITDB))


def test_duplicate_records_are_rejected():
    with pytest.raises(ValueError, match='duplicates'):
        holdout_split(['100', '101', '102', '100'])


# --- cross validation over the development pool --------------------------

def test_every_fold_keeps_its_patients_apart():
    development = holdout_split(MITDB)['development']
    for train, val in group_kfold(development, n_splits=5):
        assert not ({patient_of(record) for record in train} &
                    {patient_of(record) for record in val})


def test_the_folds_use_every_record_for_validation_exactly_once():
    development = holdout_split(MITDB)['development']
    seen = [record for _, val in group_kfold(development, n_splits=5) for record in val]
    assert sorted(seen) == sorted(development)


def test_each_fold_covers_the_whole_pool():
    development = holdout_split(MITDB)['development']
    for train, val in group_kfold(development, n_splits=5):
        assert sorted(train + val) == sorted(development)


def test_the_folds_never_reach_into_the_held_out_pool():
    split = holdout_split(MITDB)
    verify_folds(group_kfold(split['development'], n_splits=5), test=split['test'])


def test_verify_folds_catches_a_held_out_patient():
    split = holdout_split(MITDB)
    leaked = split['test'][0]
    folds = [(split['development'][:-3] + [leaked], split['development'][-3:])]
    with pytest.raises(ValueError, match='held out'):
        verify_folds(folds, test=split['test'])


def test_the_folds_are_deterministic():
    development = holdout_split(MITDB)['development']
    assert group_kfold(development, 5) == group_kfold(development, 5)


def test_an_impossible_number_of_folds_is_rejected():
    development = holdout_split(MITDB)['development']
    with pytest.raises(ValueError, match='folds requested'):
        group_kfold(development, n_splits=1)
    with pytest.raises(ValueError, match='folds requested'):
        group_kfold(development, n_splits=200)


def test_more_folds_give_smaller_validation_parts():
    development = holdout_split(MITDB)['development']
    small = max(len(val) for _, val in group_kfold(development, 10))
    large = max(len(val) for _, val in group_kfold(development, 3))
    assert small < large


# --- verification --------------------------------------------------------

def test_verify_split_rejects_a_repeated_record():
    with pytest.raises(ValueError, match='appears in'):
        verify_split({'development': ['100', '101'], 'test': ['100']})


def test_verify_split_rejects_a_split_patient():
    with pytest.raises(ValueError, match='patient'):
        verify_split({'development': ['201'], 'test': ['202']})


def test_verify_split_rejects_an_empty_part():
    with pytest.raises(ValueError, match='empty'):
        verify_split({'development': ['100'], 'test': []})


# --- reporting -----------------------------------------------------------

def test_the_formatted_table_lists_both_pools_and_the_folds():
    split = holdout_split(MITDB)
    text = format_split(split, group_kfold(split['development'], 5))
    assert 'development' in text and 'test' in text and 'fold' in text
