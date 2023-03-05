"""ebm log-odds based
    
        ebm log-odds based + diversity
        
        ebm log-odds based 2nd + diversity

        cumulative roc auc

        cumulative log loss

        cumulative average precision

        cumulative average precision and roc auc

        borda ebm log-odds based ebm log-odds based+diversity

        lasso path

        lars

        omp

        fasterrisk

        fasterrisk mix

        # TODO : Output ranking instead of 0/1 selection ?
"""

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np
import pandas as pd
import pandera as pa
from fasterrisk.fasterrisk import RiskScoreClassifier, RiskScoreOptimizer
from sklearn.linear_model import OrthogonalMatchingPursuit, lars_path, lasso_path


class Ranker(ABC):
    def compute_ranking_features(self, df, **kwargs) -> pd.Series:
        # TODO : pandera checks for output
        return self._compute_ranking_features(df, **kwargs)

    @abstractmethod
    def _compute_ranking_features(self, df, **kwargs) -> pd.Series:
        raise NotImplemented


class LogOddsDensity(Ranker):
    def __init__(self, **kwargs):
        ...

    def _compute_ranking_features(self, df, **kwargs) -> pd.Series:
        dataframe_schema = pa.DataFrameSchema(
            {"log_odds": pa.Column(nullable=True), "density": pa.Column(nullable=True)},
            index=pa.Index(str, name="binary_feature"),
            strict=False,  # disable check of other columns in the dataframe
        )

        dataframe_schema.validate(df)

        df_ = df.copy()

        df_["abs_contribution"] = df_["log_odds"].abs() * df_["density"].fillna(
            0
        ).astype(int).pow(0.95)
        df_["rank"] = df_["abs_contribution"].rank(
            method="min", na_option="bottom", ascending=False
        )
        index_without_intercept = [c for c in df_.index if c != "intercept"]
        return df_.loc[index_without_intercept, "rank"]


class DiverseLogOddsDensity(Ranker):
    def __init__(self, rank_diversity=1, **kwargs):
        self.rank_diversity = rank_diversity

    def _compute_ranking_features(self, df, **kwargs) -> pd.Series:
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
        return df_.loc[index_without_intercept, "rank"]


class CumulativeMetric(Ranker):
    def __init__(self, metric, ranker, **kwargs):
        self.metric = metric
        self.ranker = ranker

    def _compute_ranking_features(self, df, X_binarized, y, **kwargs) -> pd.Series:
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

        cumulative_metric = np.apply_along_axis(
            metric_function_1d, axis=0, arr=cumulative_log_odds
        )
        diff_cumulative_metric = [
            t - s for s, t in zip(cumulative_metric, cumulative_metric[1:])
        ]
        diff_cumulative_metric = [diff_cumulative_metric[0]] + diff_cumulative_metric

        idx_sorted_features_metric_diff = np.argsort(np.array(diff_cumulative_metric))[
            ::-1
        ]

        ranked_idx = features_sorted[idx_sorted_features_metric_diff]
        ranked_idx = list(ranked_idx) + ["intercept"]

        df_ = df_.loc[ranked_idx, :]
        df_["rank"] = list(range(len(df_)))
        df_["rank"] = df_["rank"] + 1

        index_without_intercept = [c for c in df_.index if c != "intercept"]
        return df_.loc[index_without_intercept, "rank"]


class BordaRank(Ranker):
    def __init__(self, list_ranker, **kwargs):
        self.list_ranker = list_ranker.copy()

    def _compute_ranking_features(self, df, **kwargs) -> pd.Series:
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
        return df_.loc[index_without_intercept, "rank"]


class LassoPathRank(Ranker):
    def __init__(self, **kwargs):
        ...

    def _compute_ranking_features(
        self, df, X_binarized, y, nb_steps, **kwargs
    ) -> pd.Series:
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
        coefs_lasso_updated = np.array(coefs_lasso_updated).T
        lasso_selected_features = [
            c
            for v, c in zip(coefs_lasso_updated[:, nb_steps], X_binarized.columns)
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
        return df_.loc[index_without_intercept, "rank"]


class LarsPathRank(Ranker):
    def __init__(self, **kwargs):
        ...

    def _compute_ranking_features(
        self, df, X_binarized, y, nb_steps, **kwargs
    ) -> pd.Series:
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
        return df_.loc[index_without_intercept, "rank"]


class OMPRank(Ranker):
    def __init__(self, **kwargs):
        ...

    def _compute_ranking_features(
        self, df, X_binarized, y, nb_steps, **kwargs
    ) -> pd.Series:
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
        return df_.loc[index_without_intercept, "rank"]


class FasterRiskRank(Ranker):
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
    ):
        self.parent_size = parent_size
        self.child_size = child_size
        self.max_attempts = max_attempts
        self.num_ray = num_ray
        self.lineSearch_early_stop_tolerance = lineSearch_early_stop_tolerance
        self.min_point_value = min_point_value
        self.max_point_value = max_point_value
        self.nb_max_features = nb_max_features

    def _compute_ranking_features(
        self, df, X_binarized, y, nb_steps, **kwargs
    ) -> pd.Series:
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
            sparseDiversePool_beta0,
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
        return df_.loc[index_without_intercept, "rank"]
