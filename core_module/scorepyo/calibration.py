""" Class for calibrators
"""

from abc import ABC, abstractmethod
from typing import Any, List

import cvxpy as cp
import numpy as np
import pandas as pd
from numpy.random import default_rng

from scorepyo.exceptions import NonIncreasingProbabilities, NonProbabilityValues


class Calibrator(ABC):
    """Base Class for calibrator.

    RiskScore models will assign a score to each sample. Since the scores are a simple sum of few integers,
    we can enumerate all the possible values for scores. At each score value, there are negative and positive samples.
    Perfectly calibrated probabilities on a dataset would be proportion of positives at each score.
    However, there is a need for preserving the increasing trend between score and probability.
    For that, we optimize the logloss which naturally leads to calibrated probabilities, under ordering constraints
    The optimization is based on the counting of positive and negative samples at each possible score value.
    Args:
        ABC (_type_): _description_
    """

    def __init__(self, **kwargs):  # pylint: disable=W0613
        ...

    def calibrate(
        self, df_score: pd.DataFrame, min_sum_point: int, max_sum_point: int, **kwargs
    ) -> pd.DataFrame:
        """Funtion that takes a Dataframe of scores and binary target, and compute the associated probabilities for each sum of points

        It will use the _calibrate function defined in the child classes to compute these probabilities.

        Args:
            df_score (pd.DataFrame): DataFrame of scores and binary target
            min_sum_point (int): minimum possible sum of points
            max_sum_point (int): maximum possible sum of points

        Returns:
            pd.DataFrame: DataFrame containing the probability assigned to each possible score( or sum of points)
        """

        # computing the proportion of positive samples for each possible score value
        # This corresponds to the natural probability that would be the most calibrated without the reordering constraints

        df_score_proba_association = df_score.groupby("score")["target"].mean()
        df_score_proba_association.columns = ["proba"]

        # computing the full range of possible scores from min to max
        full_index = np.arange(min_sum_point, max_sum_point + 1e-3)

        # Reindexing the dataframe with the missing scores from the data
        df_score_proba_association = df_score_proba_association.reindex(full_index)

        # Interpolating the natural probability linearly
        df_score_proba_association = df_score_proba_association.interpolate(
            method="linear"
        )

        df_reordering = pd.DataFrame(df_score_proba_association.copy())

        # Count cardinality of each score
        df_reordering["count"] = df_score.groupby("score")["target"].count()
        df_reordering["count"] = df_reordering["count"].fillna(0)
        df_reordering["target"] = df_reordering["target"].fillna(0)

        # Adjust probability to have an optimized logloss and calibration
        df_cvx = df_reordering.copy()

        df_cvx["positive_count"] = df_cvx["target"] * df_cvx["count"]
        df_cvx["negative_count"] = df_cvx["count"] - df_cvx["positive_count"]

        df_cvx["positive_proba"] = df_cvx["positive_count"] / df_cvx["count"].sum()
        df_cvx["negative_proba"] = df_cvx["negative_count"] / df_cvx["count"].sum()

        # Call _calibrate function of child class
        list_calibrated_proba = self._calibrate(df_cvx, **kwargs)

        _check_is_probability = all(
            [(0 <= p) and (p <= 1) for p in list_calibrated_proba]
        )

        if not _check_is_probability:
            raise NonProbabilityValues(
                f"Probability values outputed by calibrator are not between 0 and 1. \n Probability list : {list_calibrated_proba}."
            )
        _check_increasing = all(
            [
                list_calibrated_proba[i] <= list_calibrated_proba[i + 1]
                for i in range(len(list_calibrated_proba) - 1)
            ]
        )
        if not _check_increasing:
            raise NonIncreasingProbabilities(
                f"Probability values outputed by calibrator must be increasing. \n Probability list : {list_calibrated_proba}."
            )

        # Get the optimized value for the probabilities
        df_reordering["sorted_proba"] = list_calibrated_proba
        return df_reordering

    @abstractmethod
    def _calibrate(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> List[float]:
        raise NotImplementedError


class VanillaCalibrator(Calibrator):
    """Vanilla calibrator that simply optimizes the logloss under ordering constraints.

    Given a list of increasing scores, and an associated count of negative and positive samples,
    this calibrator computes the probabilities by optimizing the logloss on the whole dataset,
    and respecting the ordering of probabilities according to scores.

    This calibrator should be favored when calibrating on a large dataset.

    """

    def __init__(self, **kwargs):
        ...

    def _calibrate(self, df_cvx: pd.DataFrame, **_kwargs) -> List[float]:

        # Declare the list of probabilities to be set as variables
        list_proba = [cp.Variable(1) for _ in df_cvx.index]

        # Compute total size of samples to normalize the logloss
        total_count = df_cvx["count"].sum()

        constraints = []
        for p in list_proba:
            constraints.append(p >= 0)
            constraints.append(p <= 1)
        for i in range(1, len(list_proba)):
            # TODO : Put the threshold away and combine all similar scores into 1
            constraints.append(list_proba[i] - list_proba[i - 1] - 1e-3 >= 0)

        # # Compute number of positive and negative samples at each score value
        positive_sample_count = (
            df_cvx["target"].values * df_cvx["count"].values
        ).astype(int)
        negative_sample_count = df_cvx["count"].values - positive_sample_count
        list_expression = [
            -cp.log(p) * w_pos - cp.log(1 - p) * w_neg
            for p, w_pos, w_neg in zip(
                list_proba, positive_sample_count, negative_sample_count
            )
        ]
        objective = cp.Minimize(cp.sum(list_expression) / total_count)  # Objectiv

        problem = cp.Problem(objective, constraints)

        _ = problem.solve(verbose=False)

        return [p.value[0] for p in list_proba]


class BootstrappedCalibrator(Calibrator):
    """Bootstrapped calibrator that optimizes the logloss under ordering constraints on different bootstrapped sets.

    Given an original dataset, this calibrator bootstraps several times other dataset and finds probability that optimize the logloss on all datasets,
    still respecting the probability ordering by score.
    This BootstrappedCalibrator class has two modes, it can either:
    - optimize the average logloss across all bootstrapped datasets
    - optimize the worse logloss among dataset

    The latter will lead to worse logloss on the training dataset, but more robust logloss on the test set if it's similar to the distribution on the training dataset.
    The BootstrappedCalibrator should be favored when calibrating on a small dataset.

    """

    def __init__(self, nb_experiments: int = 20, method: str = "average", **_kwargs):
        """
        Args:
            nb_experiments (int, optional): Number of bootstrapped datasets used to optimize the logloss. Defaults to 20.
            method (str, optional): Indicator of average mode or worst_case mode. Defaults to "average".
        """
        self.nb_experiments = nb_experiments
        if method not in ("average", "worst_case"):
            raise NotImplementedError
        self.method = method

    def _calibrate(self, df_cvx: pd.DataFrame, **_kwargs) -> List[float]:
        list_proba_multinomial = list(df_cvx["positive_proba"].values)
        list_proba_multinomial += list(df_cvx["negative_proba"].values)

        rng = default_rng()
        nb_experiment = self.nb_experiments
        size_sample = df_cvx["count"].sum()

        # Bootstrapping here corresponds to samples drawn from a multinomial distribution
        bootstrap_samples = rng.multinomial(
            n=size_sample, pvals=list_proba_multinomial, size=nb_experiment
        )

        positive_count_samples = bootstrap_samples[:, : len(df_cvx)].T
        negative_count_samples = bootstrap_samples[:, len(df_cvx) :].T
        positive_col_bootstrap = [f"positive_count_{i}" for i in range(nb_experiment)]
        negative_col_bootstrap = [f"negative_count_{i}" for i in range(nb_experiment)]
        df_positive = pd.DataFrame(
            positive_count_samples, columns=positive_col_bootstrap, index=df_cvx.index
        )
        df_negative = pd.DataFrame(
            negative_count_samples, columns=negative_col_bootstrap, index=df_cvx.index
        )

        df_cvx = pd.concat([df_cvx, df_positive, df_negative], axis=1)

        # Declare the list of probabilities to be set as variables
        list_proba = [cp.Variable(1) for _ in df_cvx.index]

        # Compute total size of samples to normalize the logloss
        total_count = df_cvx["count"].sum()

        # Compute the logloss at each score in a list in order to sum it later
        # the logloss at each score is simple as all samples will have the same
        # probability p. for all positive samples, add -log(p), for all negative samples add -log(1-p)
        list_bootstrap_sample_objective = []
        for i in range(nb_experiment):
            positive_sample_count = df_cvx[f"positive_count_{i}"].values
            negative_sample_count = df_cvx[f"negative_count_{i}"].values
            list_expression = [
                -cp.log(p) * w_pos - cp.log(1 - p) * w_neg
                for p, w_pos, w_neg in zip(
                    list_proba, positive_sample_count, negative_sample_count
                )
            ]
            list_bootstrap_sample_objective.append(list_expression.copy())
        # objective = cp.Minimize(cp.sum(list_expression) / total_count)  # Objectiv

        constraints = []
        for p in list_proba:
            constraints.append(p >= 0)
            constraints.append(p <= 1)
        for i in range(1, len(list_proba)):
            # TODO : Put the threshold away and combine all similar scores into 1
            constraints.append(list_proba[i] - list_proba[i - 1] - 1e-3 >= 0)

        # If worst_case mode is chosen, optimize an auxiliary variable
        # that will correspond to the worst log loss among one of the bootstrapped datasets
        if self.method == "worst_case":
            worse_log_loss = cp.Variable(1)
            for single_bootstrap_logloss in list_bootstrap_sample_objective:
                constraints.append(
                    worse_log_loss >= cp.sum(single_bootstrap_logloss) / total_count
                )
            objective = cp.Minimize(worse_log_loss)

        # else if average mode, optimize the mean logloss across all bootstrapped datasets
        elif self.method == "average":
            objective = cp.Minimize(
                cp.sum(
                    [
                        cp.sum(single_bootstrap_logloss) / total_count
                        for single_bootstrap_logloss in list_bootstrap_sample_objective
                    ]
                )
            )

            objective = cp.Minimize(cp.sum(list_expression) / total_count)  # Objectiv

        else:
            raise Exception

        problem = cp.Problem(objective, constraints)

        _ = problem.solve(verbose=False)

        return [p.value[0] for p in list_proba]
