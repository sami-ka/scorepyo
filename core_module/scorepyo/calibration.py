from abc import ABC, abstractmethod

import cvxpy as cp
import numpy as np
import pandas as pd
from numpy.random import default_rng


class Calibrator(ABC):
    def __init__(self, **kwargs):
        ...

    def calibrate(self, df_score, min_sum_point, max_sum_point, **kwargs):
        # TODO check Pandera input
        # TODO check pandera input?
        df_score_proba_association = df_score.groupby("score")["target"].mean()
        df_score_proba_association.columns = ["proba"]
        # min_sum_point = np.sum(np.clip(best_scenario, a_max=0, a_min=None))
        # max_sum_point = np.sum(np.clip(best_scenario, a_min=0, a_max=None))
        full_index = np.arange(min_sum_point, max_sum_point + 1e-3)

        df_score_proba_association = df_score_proba_association.reindex(full_index)
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
        list_calibrated_proba = self._calibrate(df_cvx, **kwargs)
        # TODO check Pandera output
        # Get the optimized value for the probabilities
        df_reordering["sorted_proba"] = list_calibrated_proba
        return df_reordering

    @abstractmethod
    def _calibrate(self, df_score, **kwargs):
        raise NotImplementedError


class VanillaCalibrator(Calibrator):
    def __init__(self, **kwargs):
        ...

    def _calibrate(self, df_cvx, **kwargs):

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

        opt = problem.solve(verbose=False)

        return [p.value[0] for p in list_proba]


class BootstrappedCalibrator(Calibrator):
    def __init__(self, nb_experiments=20, method="average", **kwargs):
        self.nb_experiments = nb_experiments
        if method not in ("average", "worst_case"):
            raise NotImplementedError
        self.method = method

    def _calibrate(self, df_cvx, **kwargs):
        list_proba_multinomial = list(df_cvx["positive_proba"].values)
        list_proba_multinomial += list(df_cvx["negative_proba"].values)

        rng = default_rng()
        nb_experiment = self.nb_experiments
        size_sample = df_cvx["count"].sum()
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
        # df_cvx[positive_col_bootstrap] = positive_count_samples
        # df_cvx[negative_col_bootstrap] = negative_count_samples
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

        if self.method == "worst_case":
            worse_log_loss = cp.Variable(1)
            for single_bootstrap_logloss in list_bootstrap_sample_objective:
                constraints.append(
                    worse_log_loss >= cp.sum(single_bootstrap_logloss) / total_count
                )
            objective = cp.Minimize(worse_log_loss)
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

        opt = problem.solve(verbose=False)

        return [p.value[0] for p in list_proba]
