import numpy as np
import pandas as pd
import pandas.api.types

import kaggle_metric_utilities

import sklearn.metrics

from typing import Sequence, Union, Optional


class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, normalize: bool=True, weights_column_name: Optional[str]=None) -> float:
    '''
    Wrapper for https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    Accuracy classification score.

    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.

    Parameters
    ----------
    solution : 1d DataFrame. Ground truth (correct) labels.

    submission : 1d DataFrame. Predicted labels, as returned by a classifier.

    normalize : bool, default=True
        If False, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.

    weights_column_name: optional str, the name of the sample weights column in the solution file.

    Examples
    --------
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> y_pred = [0, 2, 1, 3]
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred["id"] = range(len(y_pred))
    >>> y_true = [0, 1, 2, 3]
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true["id"] = range(len(y_true))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name)
    0.5
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name, normalize=False)
    2.0
    '''
    # Skip sorting and equality checks for the row_id_column since that should already be handled
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    sample_weight = None
    if weights_column_name:
        if weights_column_name not in solution.columns:
            raise ValueError(f'The solution weights column {weights_column_name} is not found')
        sample_weight = solution.pop(weights_column_name).values
        if not pandas.api.types.is_numeric_dtype(sample_weight):
            raise ParticipantVisibleError('The solution weights are not numeric')

    if not((len(submission.columns) == 1) or (len(submission.columns) == len(solution.columns))):
        raise ParticipantVisibleError(f'Invalid number of submission columns. Found {len(submission.columns)}')


    solution = solution.values
    submission = submission.values

    score_result = kaggle_metric_utilities.safe_call_score(sklearn.metrics.accuracy_score, solution, submission, normalize=normalize, sample_weight=sample_weight)

    return float(score_result)