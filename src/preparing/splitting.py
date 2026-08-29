"""
Assignment of patients to the development pool and to the held out test pool.

Two levels, not three. A fixed group of patients is set aside once and touched only when
a method is finished; everyone else forms a development pool that `group_kfold` divides
into training and validation folds during training. Cross validation over the pool uses
every patient for training in some fold and for validation in another, which is what
makes a small archive yield enough material, while the held out pool stays untouched by
model selection and remains the only set a reported number may come from.

The unit is the patient. Consecutive beats of one recording resemble each other far more
than beats of two people do, so a method tested on a recording it was fitted on is being
asked to recognise a waveform it has already seen. The resulting error is one sided and
silent: every metric improves and nothing warns.

Two records of the MIT-BIH Arrhythmia Database come from one person. The database
documentation notes that records 201 and 202 were taken from the same male subject, so
record identifiers are not patient identifiers. `SHARED_SUBJECT_RECORDS` collapses the
pair. Verify the entry against the current database documentation before the final run.

Records also fall into two families: 100 to 124 were drawn at random from four thousand
recordings, while 200 to 234 were selected for rare phenomena a random sample would have
missed. They do not share a distribution, so `stratify` keeps both present in the test
pool rather than leaving it accidentally easy or accidentally hard.
"""

from typing import Iterable, Optional, Sequence

import numpy as np

DEFAULT_TEST_PATIENTS = 5
DEFAULT_SEED = 0
DEFAULT_N_SPLITS = 5

# MIT-BIH: one subject contributed two records
SHARED_SUBJECT_RECORDS = (('201', '202'),)

RANDOM_SERIES = '1xx'
SELECTED_SERIES = '2xx'


def patient_of(record: str) -> str:
    """Patient identifier, equal to the record identifier except for documented pairs."""
    record = str(record)
    for group in SHARED_SUBJECT_RECORDS:
        if record in group:
            return group[0]
    return record


def record_series(record: str) -> str:
    """Family a record belongs to: the random sample or the deliberately selected one."""
    record = str(record)
    if not record[:1].isdigit():
        return RANDOM_SERIES
    return SELECTED_SERIES if int(record[0]) >= 2 else RANDOM_SERIES


def group_records(records: Iterable[str]) -> dict:
    """Records collected by patient."""
    groups: dict = {}
    for record in records:
        groups.setdefault(patient_of(record), []).append(str(record))
    return groups


def holdout_split(records: Iterable[str],
                  n_test_patients: int = DEFAULT_TEST_PATIENTS,
                  seed: int = DEFAULT_SEED,
                  stratify: bool = True) -> dict:
    """
    Sets aside `n_test_patients` patients and leaves the rest as the development pool.

    Drawing patients rather than records means a patient who contributed two recordings
    takes both with him. The draw is deterministic in `seed` and independent of the order
    the records are listed in, so the same archive always yields the same held out group.
    """
    records = [str(record) for record in records]
    if len(records) != len(set(records)):
        raise ValueError('the record list holds duplicates')

    groups = group_records(records)
    n_test_patients = int(n_test_patients)
    if not 1 <= n_test_patients < len(groups):
        raise ValueError(f'{n_test_patients} test patients requested out of '
                         f'{len(groups)} available')

    rng = np.random.default_rng(seed)
    keys = sorted(groups)

    if stratify:
        chosen: list = []
        by_series = {series: [key for key in keys
                              if record_series(groups[key][0]) == series]
                     for series in (RANDOM_SERIES, SELECTED_SERIES)}
        for index in range(n_test_patients):
            series = (RANDOM_SERIES, SELECTED_SERIES)[index % 2]
            pool = by_series[series] or by_series[
                SELECTED_SERIES if series == RANDOM_SERIES else RANDOM_SERIES]
            if not pool:
                break
            pick = pool.pop(int(rng.integers(len(pool))))
            chosen.append(pick)
    else:
        chosen = [keys[index] for index in
                  rng.choice(len(keys), size=n_test_patients, replace=False)]

    test = sorted(record for key in chosen for record in groups[key])
    development = sorted(record for key in keys if key not in set(chosen)
                         for record in groups[key])

    split = {'development': development, 'test': test}
    verify_split(split)
    return split


def group_kfold(records: Iterable[str], n_splits: int = DEFAULT_N_SPLITS) -> list:
    """
    Folds of the development pool, split by patient.

    Wraps `sklearn.model_selection.GroupKFold` with the patient as the group so that no
    recording of a validation patient appears in the training part of the same fold. The
    folds are deterministic and cover the pool exactly once as validation.
    """
    from sklearn.model_selection import GroupKFold

    records = sorted(str(record) for record in records)
    groups = [patient_of(record) for record in records]
    n_patients = len(set(groups))

    n_splits = int(n_splits)
    if not 2 <= n_splits <= n_patients:
        raise ValueError(f'{n_splits} folds requested over {n_patients} patients')

    index = np.arange(len(records))
    folds = []
    for train_index, val_index in GroupKFold(n_splits=n_splits).split(index, groups=groups):
        folds.append(([records[position] for position in train_index],
                      [records[position] for position in val_index]))
    return folds


def verify_split(split: dict, extra: Optional[Sequence[str]] = None) -> None:
    """
    Raises unless the parts are disjoint and no patient appears in two of them.

    Called at the end of every split rather than left to the caller, because the failure
    it guards against produces no symptom other than results that are too good.
    """
    seen_records: dict = {}
    seen_patients: dict = {}
    for name in split:
        if not split[name]:
            raise ValueError(f'the {name!r} part is empty')
        for record in split[name]:
            if record in seen_records:
                raise ValueError(
                    f'record {record} appears in {seen_records[record]!r} and {name!r}')
            seen_records[record] = name

            patient = patient_of(record)
            if seen_patients.setdefault(patient, name) != name:
                raise ValueError(
                    f'patient {patient} appears in {seen_patients[patient]!r} '
                    f'and {name!r} through record {record}')


def verify_folds(folds: Sequence, test: Optional[Iterable[str]] = None) -> None:
    """Raises unless every fold keeps its patients apart and clear of the test pool."""
    test_patients = {patient_of(record) for record in (test or [])}
    for number, (train, val) in enumerate(folds):
        verify_split({f'fold{number}_train': list(train), f'fold{number}_val': list(val)})
        crossing = ({patient_of(record) for record in train} |
                    {patient_of(record) for record in val}) & test_patients
        if crossing:
            raise ValueError(f'fold {number} uses held out patients: {sorted(crossing)}')


def format_split(split: dict, folds: Optional[Sequence] = None) -> str:
    """Table of the split, for the console and for the thesis appendix."""
    lines = ['%-14s %9s %10s %7s %7s' % ('czesc', 'rekordy', 'pacjenci', '1xx', '2xx'),
             '-' * 52]
    for name, records in split.items():
        series = [record_series(record) for record in records]
        lines.append('%-14s %9d %10d %7d %7d' % (
            name, len(records), len({patient_of(record) for record in records}),
            series.count(RANDOM_SERIES), series.count(SELECTED_SERIES)))
    if folds:
        lines.append('')
        lines.append('%-14s %9s %10s' % ('fold', 'train', 'val'))
        lines.append('-' * 52)
        for number, (train, val) in enumerate(folds):
            lines.append('%-14d %9d %10d' % (number, len(train), len(val)))
    return '\n'.join(lines)
