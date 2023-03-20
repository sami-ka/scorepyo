"""Classes for binary features rankers
"""

from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np
import pandas as pd
import pandera as pa
from fasterrisk.fasterrisk import RiskScoreOptimizer
from sklearn.linear_model import OrthogonalMatchingPursuit, lars_path, lasso_path


class Ranker(ABC):
    """Base class for binary features ranker

    Child classes must implement the _compute_ranking_features method.

    """

    def compute_ranking_features(
        self, df: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.Series:
        series_ranking = self._compute_ranking_features(df, *args, **kwargs)

        series_ranking_schema = pa.SeriesSchema(
            # dtype=pa.Int(),
            name="rank",
            checks=[
                pa.Check.isin(range(len(df))),
                pa.Check(
                    lambda x: int(x) == x, element_wise=True
                ),  # check that type is an int (any bit number int32, int64, etc.)
            ],
        )
        series_ranking_schema.validate(series_ranking)
        return series_ranking

    @abstractmethod
    def _compute_ranking_features(
        # self, df: pd.DataFrame,
        *args: Any,
        **kwargs: Any,
    ) -> pd.Series:
        raise NotImplementedError


class LogOddsDensity(Ranker):
    """Binary features ranker based on logodds contribution of binary feature,
    and reduced importance of density of the binary feature (number of samples with a positive value on the binary feature)



    Child class of Ranker.
    """

    def __init__(self, **kwargs):  # pylint: disable=W0613
        ...

    def _compute_ranking_features(
        self, df: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.Series:  # pylint disable=W0613

        """
        This method ranks the binary features based on the product of :
         - logodds contribution of the binary feature
         - reduced importance of density

        This prevents having a too large emphasis on a binary feature with a large density and low log odd contribution
        Args:
            df (pd.DataFrame): DataFrame with logodds and density information, with the binary feature name as an index

        Returns:
            pd.Series: Rank for each binary feature in a decreasing order of importance
        """

        dataframe_schema = pa.DataFrameSchema(
            {"log_odds": pa.Column(nullable=True), "density": pa.Column(nullable=True)},
            index=pa.Index(str, name="binary_feature"),
            strict=False,  # disable check of other columns in the dataframe
        )

        dataframe_schema.validate(df)

        df_ = df.copy()

        # The density importance is reduced by taking a power of 0.95
        # TODO Put the power in the argument of the class
        df_["abs_contribution"] = df_["log_odds"].abs() * df_["density"].fillna(
            0
        ).astype(int).pow(0.95)
        df_["rank"] = df_["abs_contribution"].rank(
            method="min", na_option="bottom", ascending=False
        )
        index_without_intercept = [c for c in df_.index if c != "intercept"]
        return df_.loc[index_without_intercept, "rank"].astype(int)


class DiverseLogOddsDensity(Ranker):
    """Binary features ranker based on logodds contribution of binary feature,
    and density of the binary feature (number of samples with a positive value on the binary feature)

    In order to diversify the ranking, a small number of binary features coming from the same origin will be at the top of the ranking


    Child class of Ranker.
    """

    def __init__(self, rank_diversity: int = 1, **kwargs):  # pylint: disable=W0613
        """
        Args:
            rank_diversity (int, optional): Indicates the number of additional binary features from the same origin to include in the ranking. Defaults to 1.
        """
        # TODO check parameters
        self.rank_diversity = rank_diversity

    def _compute_ranking_features(
        self, df: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.Series:
        """This method ranks the binary features based on the product of :
         - logodds contribution of the binary feature
         - importance of density
         - origin of binary features

         If several binary features are at the top of the ranking, only self.rank_diversity+1 will be kept

        Args:
            df (pd.DataFrame): DataFrame with logodds, density and origin information, with the binary feature name as an index

        Returns:
            pd.Series: Rank for each binary feature in a decreasing order of importance
        """
        dataframe_schema = pa.DataFrameSchema(
            {
                "log_odds": pa.Column(nullable=True),
                "density": pa.Column(nullable=True),
                "feature": pa.Column(nullable=True),
            },
            index=pa.Index(str, name="binary_feature"),
            strict=False,  # disable check of other columns in the dataframe
        )

        dataframe_schema.validate(df)

        df_ = df.copy()

        df_["log_odds_sum"] = df_["log_odds"].abs() * df_["density"].fillna(0)

        df_["absolute_log_odds"] = df_["log_odds"].abs()

        # keep first kth binary feature of a feature and mark it as priority
        idx_up_to_kth = (
            df_.sort_values(
                ["log_odds_sum", "absolute_log_odds"], ascending=[False, False]
            )
            .reset_index()
            .groupby("feature")["binary_feature"]
            .nth(n=list(range(self.rank_diversity + 1)))
            .values
        )

        idx_up_to_kth = [f for f in idx_up_to_kth if f != "intercept"]

        df_["prio"] = np.where(df_.index.isin(idx_up_to_kth), 1, 0)

        df_ = df_.sort_values(by=["prio", "log_odds_sum"], ascending=[False, False])
        df_["rank"] = list(range(len(df_)))
        df_["rank"] = df_["rank"] + 1

        index_without_intercept = [c for c in df_.index if c != "intercept"]
        return df_.loc[index_without_intercept, "rank"].astype(int)


class CumulativeMetric(Ranker):
    """Binary features ranker based on computing a metric on a growing number of binary features.

    This ranker initially sorts binary features depending on a specified ranker.
    Then it computes a classification metric by adding one log odd contribution of a binary feature at a time.
    The binary features are then ranked according to the incremental difference they made on the metric.

    Child class of Ranker
    """

    def __init__(self, metric, ranker: Ranker, **kwargs):  # pylint: disable=W0613
        """
        Args:
            metric (function): function taking 2 arguments: binary target and probabilities/scores
                this function should be maximized
            ranker (Ranker): original ranker
        """
        # TODO put a check on the metric function
        self.metric = metric
        self.ranker = ranker

    def _compute_ranking_features(
        self,
        df: pd.DataFrame,
        X_binarized: pd.DataFrame,
        y: pd.Series,
        *args: Any,
        **kwargs: Any,
    ) -> pd.Series:  # type: ignore
        """This ranker initially sorts binary features depending on a specified ranker.
        Then it computes a classification metric by adding one log odd contribution of a binary feature at a time.
        The binary features are then ranked according to the magnitude of the incremental difference they made on the metric.

        Args:
            df (pd.DataFrame): information dataframe, it should contain the log_odds contribution of binary feature
            X_binarized (pd.DataFrame): binary dataset
            y (pd.Series): binary target

        Returns:
            pd.Series: Rank for each binary feature in a decreasing order of importance
        """
        # TODO add check on intercept in index
        dataframe_schema = pa.DataFrameSchema(
            {
                "log_odds": pa.Column(nullable=True),
            },
            index=pa.Index(str, name="binary_feature"),
            strict=False,  # disable check of other columns in the dataframe
        )

        dataframe_schema.validate(df)

        df_ = df.copy()

        df_rank = self.ranker.compute_ranking_features(df_)

        df_ = df_.merge(
            df_rank, right_index=True, left_index=True, how="left", validate="1:1"
        )

        # Sort binary features according to a ranker
        df_ = df_.sort_values(by="rank", ascending=True)

        features_sorted = pd.Index([f for f in df_.index if f != "intercept"])

        # Compute log_odds vector of each binary feature
        log_odds_vector = df_.loc[features_sorted, "log_odds"].values

        log_odd_intercept = df_.loc["intercept", "log_odds"]

        log_odds_vector = log_odds_vector.reshape(-1, 1)

        # Add logodd intercept to all log_odds
        log_odds_vector = log_odds_vector + log_odd_intercept

        # Duplicate log_odds vector as many times as there are binary features
        log_odds_matrix = np.tile(log_odds_vector, reps=(1, len(features_sorted)))

        # Decreasingly put to 0 log odd the 2nd to nth binary feature, then 3rd to nth, etc.
        for i in range(len(features_sorted)):
            log_odds_matrix[i + 1 :, i] = 0

        # Compute cumulative logodds by introducing one binary feature on top of each other
        cumulative_log_odds = np.matmul(
            X_binarized[features_sorted].values, log_odds_matrix
        )

        def metric_function_1d(y_proba):
            return self.metric(y.astype(float).values, np.array(y_proba, dtype=float))

        # Compute metric on cumulative logodds
        cumulative_metric = np.apply_along_axis(
            metric_function_1d, axis=0, arr=cumulative_log_odds
        )

        # Compute the incremental difference between a metric and the previous value
        diff_cumulative_metric = [
            t - s for s, t in zip(cumulative_metric, cumulative_metric[1:])
        ]
        diff_cumulative_metric = [diff_cumulative_metric[0]] + diff_cumulative_metric

        # Sort the binary features based on the magnitude of metric difference in a decreasing fashion
        idx_sorted_features_metric_diff = np.argsort(np.array(diff_cumulative_metric))[
            ::-1
        ]

        ranked_idx = features_sorted[idx_sorted_features_metric_diff]
        ranked_idx = list(ranked_idx) + ["intercept"]

        df_ = df_.loc[ranked_idx, :]
        df_["rank"] = list(range(len(df_)))
        df_["rank"] = df_["rank"] + 1

        index_without_intercept = [c for c in df_.index if c != "intercept"]
        return df_.loc[index_without_intercept, "rank"].astype(int)


class BordaRank(Ranker):
    """Based on a list of Ranker, computes the Borda rank of each binary feature based on all rankers.

    Borda rank : https://en.wikipedia.org/wiki/Borda_count

    Child class of ranker
    """

    def __init__(self, list_ranker: List[Ranker], **kwargs):  # pylint: disable=W0613
        self.list_ranker = list_ranker.copy()

    def _compute_ranking_features(
        self, df: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.Series:
        """Based on a list of Ranker, computes the Borda rank of each binary feature based on all rankers.

        Borda rank : https://en.wikipedia.org/wiki/Borda_count

        Args:
            df (pd.DataFrame): DataFrame of logodds, density, features and binary features information needed for all rankers

        Returns:
            pd.Series: Rank for each binary feature in a decreasing order of importance
        """
        df_ = df.copy()

        df_list_ranker = pd.DataFrame(
            columns=[str(i) for i in range(len(self.list_ranker))], index=df_.index
        )
        for i, ranker in enumerate(self.list_ranker):
            df_list_ranker[str(i)] = ranker.compute_ranking_features(df_, **kwargs)

        df_ = df_.merge(
            df_list_ranker.sum(axis=1).to_frame(name="rank"),
            left_index=True,
            right_index=True,
            validate="1:1",
            how="left",
        )

        index_without_intercept = [c for c in df_.index if c != "intercept"]
        return df_.loc[index_without_intercept, "rank"].astype(int)


class LassoPathRank(Ranker):
    """Binary feature ranker based on lasso path.

    Lasso path stores the coefficient values of features along different values of the regularization parameters,
    of a Lasso regression.
    Based on this, the ranker selects the lasso path step where the number of binary features with non-zero coefficient
    is equal to the specified target.

    For more info on Lasso path : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lasso_path.html



    Child class of Ranker
    """

    def __init__(self, **kwargs):  # pylint: disable=W0613
        ...

    def _compute_ranking_features(
        self,
        df: pd.DataFrame,
        X_binarized: pd.DataFrame,
        y: pd.Series,
        nb_steps: int,
        **kwargs,
    ) -> pd.Series:
        """This functions returns the rank according to the lasso path
        Args:
            df (pd.DataFrame): Information dataframe on logodds, density and binary feature name
            X_binarized (pd.DataFrame): Binary features
            y (pd.Series): Binary target
            nb_steps (int): Number of non zero coefficient to identify the step in the lasso path

        Returns:
            pd.Series: Rank for each binary feature in a decreasing order of importance
        """
        dataframe_schema = pa.DataFrameSchema(
            {
                "log_odds": pa.Column(nullable=True),
                "density": pa.Column(nullable=True),
            },
            index=pa.Index(str, name="binary_feature"),
            strict=False,  # disable check of other columns in the dataframe
        )

        dataframe_schema.validate(df)

        df_ = df.copy()
        df_["log_odds_sum"] = df_["log_odds"].abs() * df_["density"].fillna(0)

        _, coefs_lasso, _ = lasso_path(X_binarized.values, y, verbose=False)
        coefs_lasso_updated = [coefs_lasso[:, 0]]
        for i in range(1, coefs_lasso.shape[1]):
            previous_column_zero_count = np.sum(np.where(coefs_lasso_updated[-1] == 0))
            current_column_zero_count = np.sum(np.where(coefs_lasso[:, i] == 0))
            if current_column_zero_count < previous_column_zero_count:
                coefs_lasso_updated.append(coefs_lasso[:, i])
        coefs_lasso_updated_ = np.array(coefs_lasso_updated).T
        lasso_selected_features = [
            c
            for v, c in zip(coefs_lasso_updated_[:, nb_steps], X_binarized.columns)
            if v != 0
        ]

        df_["lasso_selected_feature"] = np.where(
            df_.index.isin(lasso_selected_features), 1, 0
        )

        df_ = df_.sort_values(
            by=["lasso_selected_feature", "log_odds_sum"], ascending=[False, False]
        )
        df_["rank"] = list(range(len(df_)))
        df_["rank"] = df_["rank"] + 1

        index_without_intercept = [c for c in df_.index if c != "intercept"]
        return df_.loc[index_without_intercept, "rank"].astype(int)


class LarsPathRank(Ranker):
    """Binary feature ranker based on LARS lasso path.

    LARS lasso path is the coefficient values of features along different values of the regularization parameters,
    of a LARS Lasso regression.
    Based on this, the ranker selects the LARS lasso path step where the number of binary features with non-zero coefficient
    is equal to the specified target.

    For more info on LARS Lasso path : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lars_path.html#sklearn.linear_model.lars_path


    Child class of Ranker
    """

    def __init__(self, **kwargs):  # pylint: disable=W0613
        ...

    def _compute_ranking_features(
        self,
        df: pd.DataFrame,
        X_binarized: pd.DataFrame,
        y: pd.Series,
        nb_steps: int,
        **kwargs,
    ) -> pd.Series:
        """This function returns binary features rank based on LARS Lasso path

        Args:
            df (pd.DataFrame): Information dataframe on logodds, density and binary feature name
            X_binarized (pd.DataFrame): Binary features
            y (pd.Series): Binary target
            nb_steps (int): Number of non zero coefficient to identify the step in the LARS lasso path

        Returns:
            pd.Series: Rank for each binary feature in a decreasing order of importance
        """
        dataframe_schema = pa.DataFrameSchema(
            {
                "log_odds": pa.Column(nullable=True),
                "density": pa.Column(nullable=True),
            },
            index=pa.Index(str, name="binary_feature"),
            strict=False,  # disable check of other columns in the dataframe
        )

        dataframe_schema.validate(df)

        df_ = df.copy()
        df_["log_odds_sum"] = df_["log_odds"].abs() * df_["density"].fillna(0)

        _, _, coefs_lars = lars_path(
            X_binarized.astype(float).values,
            y.astype(float).values,
            method="lasso",
            verbose=False,
        )
        lars_selected_features = [
            c for v, c in zip(coefs_lars[:, nb_steps], X_binarized.columns) if v != 0
        ]

        df_["lars_selected_feature"] = np.where(
            df_.index.isin(lars_selected_features), 1, 0
        )

        df_ = df_.sort_values(
            by=["lars_selected_feature", "log_odds_sum"], ascending=[False, False]
        )
        df_["rank"] = list(range(len(df_)))
        df_["rank"] = df_["rank"] + 1

        index_without_intercept = [c for c in df_.index if c != "intercept"]
        return df_.loc[index_without_intercept, "rank"].astype(int)


class OMPRank(Ranker):
    """Binary feature ranker based on Orthogonal Matching Pursuit.

       From https://scikit-learn.org/stable/modules/linear_model.html#omp:
       " Orthogonal Matching Pursuit algorithm  approximates the fit of a linear model with constraints imposed on the number of non-zero coefficients (ie. the
    pseudo-norm).

       Based on this, the ranker selects the binary features selected by the OMP algorithm where the number of non zero coefficients is equal to the specified target.

       Child class of Ranker.
    """

    def __init__(self, **kwargs):  # pylint: disable=W0613
        ...

    def _compute_ranking_features(
        self,
        df: pd.DataFrame,
        X_binarized: pd.DataFrame,
        y: pd.Series,
        nb_steps: int,
        **kwargs,
    ) -> pd.Series:
        """This function returns binary features rank based on OMP

        Args:
            df (pd.DataFrame): Information dataframe on logodds, density and binary feature name
            X_binarized (pd.DataFrame): Binary features
            y (pd.Series): Binary target
            nb_steps (int): Number of non zero coefficient to identify the step in the LARS lasso path

        Returns:
            pd.Series: Rank for each binary feature in a decreasing order of importance
        """
        dataframe_schema = pa.DataFrameSchema(
            {
                "log_odds": pa.Column(nullable=True),
                "density": pa.Column(nullable=True),
            },
            index=pa.Index(str, name="binary_feature"),
            strict=False,  # disable check of other columns in the dataframe
        )

        dataframe_schema.validate(df)

        df_ = df.copy()
        df_["log_odds_sum"] = df_["log_odds"].abs() * df_["density"].fillna(0)

        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=nb_steps, normalize=False)
        omp.fit(X_binarized, y)

        omp_selected_features = [
            c for v, c in zip(omp.coef_, X_binarized.columns) if v != 0
        ]

        df_["omp_selected_features"] = np.where(
            df_.index.isin(omp_selected_features), 1, 0
        )

        df_ = df_.sort_values(
            by=["omp_selected_features", "log_odds_sum"], ascending=[False, False]
        )
        df_["rank"] = list(range(len(df_)))
        df_["rank"] = df_["rank"] + 1

        index_without_intercept = [c for c in df_.index if c != "intercept"]
        return df_.loc[index_without_intercept, "rank"].astype(int)


class FasterRiskRank(Ranker):
    """Binary features ranker based on FasterRisk, another risk score model library.

    FasterRisk has its own algorithm to select candidate binary features.
    Based on their algorithm, we take the top features based on number of appearances in candidate models.
    Then we sort by the product of density and lod odds of binary features

    For more information of FasterRisk package and usage: https://fasterrisk.readthedocs.io/en/latest/

    Child class of Ranker
    """

    def __init__(
        self,
        parent_size=10,
        child_size=None,
        max_attempts=50,
        num_ray=20,
        lineSearch_early_stop_tolerance=1e-3,
        min_point_value=-2,
        max_point_value=3,
        nb_max_features=4,
        **kwargs,
    ):  # pylint: disable=W0613
        self.parent_size = parent_size
        self.child_size = child_size
        self.max_attempts = max_attempts
        self.num_ray = num_ray
        self.lineSearch_early_stop_tolerance = lineSearch_early_stop_tolerance
        self.min_point_value = min_point_value
        self.max_point_value = max_point_value
        self.nb_max_features = nb_max_features

    def _compute_ranking_features(
        self,
        df: pd.DataFrame,
        X_binarized: pd.DataFrame,
        y: pd.Series,
        nb_steps: int,
        **kwargs,
    ) -> pd.Series:
        # pylint disable=W0613
        """This function returns binary features rank based on number of appearances in FasterRisk candidate risk score models

        Args:
            df (pd.DataFrame): Information dataframe on logodds, density and binary feature name
            X_binarized (pd.DataFrame): Binary features
            y (pd.Series): Binary target
            nb_steps (int): Number of non zero coefficient to identify the step in the LARS lasso path

        Returns:
            pd.Series: Rank for each binary feature in a decreasing order of importance
        """
        dataframe_schema = pa.DataFrameSchema(
            {
                "log_odds": pa.Column(nullable=True),
                "density": pa.Column(nullable=True),
            },
            index=pa.Index(str, name="binary_feature"),
            strict=False,  # disable check of other columns in the dataframe
        )

        dataframe_schema.validate(df)

        df_ = df.copy()
        df_["log_odds_sum"] = df_["log_odds"].abs() * df_["density"].fillna(0)

        RiskScoreOptimizer_m = RiskScoreOptimizer(
            X=X_binarized.values,
            y=np.where(y == 0, -1, 1),
            lb=self.min_point_value,
            ub=self.max_point_value,
            k=self.nb_max_features,
            parent_size=self.parent_size,
            child_size=self.child_size,
            maxAttempts=self.max_attempts,
            num_ray_search=self.num_ray,
            lineSearch_early_stop_tolerance=self.lineSearch_early_stop_tolerance,
        )

        RiskScoreOptimizer_m.sparseLogRegModel_object.get_sparse_sol_via_OMP(
            k=RiskScoreOptimizer_m.k,
            parent_size=RiskScoreOptimizer_m.parent_size,
            child_size=RiskScoreOptimizer_m.child_size,
        )

        (
            beta0,
            betas,
            ExpyXB,
        ) = RiskScoreOptimizer_m.sparseLogRegModel_object.get_beta0_betas_ExpyXB()
        RiskScoreOptimizer_m.sparseDiversePoolLogRegModel_object.warm_start_from_beta0_betas_ExpyXB(
            beta0=beta0, betas=betas, ExpyXB=ExpyXB
        )

        (
            _,
            sparseDiversePool_betas,
        ) = RiskScoreOptimizer_m.sparseDiversePoolLogRegModel_object.get_sparseDiversePool(
            gap_tolerance=RiskScoreOptimizer_m.sparseDiverseSet_gap_tolerance,
            select_top_m=RiskScoreOptimizer_m.sparseDiverseSet_select_top_m,
            maxAttempts=RiskScoreOptimizer_m.sparseDiverseSet_maxAttempts,
        )
        features_sorted_by_occurence = sorted(
            [
                (f, v)
                for f, v in zip(
                    X_binarized.columns,
                    np.sum((sparseDiversePool_betas != 0), axis=0),
                )
            ],
            key=lambda t: t[1],
        )
        fasterrisk_chosen_features = [
            f for f, _ in features_sorted_by_occurence[-(nb_steps):]
        ]

        df_["fasterrisk_chosen_features"] = np.where(
            df_.index.isin(fasterrisk_chosen_features), 1, 0
        )

        df_ = df_.sort_values(
            by=["fasterrisk_chosen_features", "log_odds_sum"], ascending=[False, False]
        )
        df_["rank"] = list(range(len(df_)))
        df_["rank"] = df_["rank"] + 1

        index_without_intercept = [c for c in df_.index if c != "intercept"]
        return df_.loc[index_without_intercept, "rank"].astype(int)
