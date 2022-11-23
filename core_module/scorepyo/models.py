"""TODO Add docstring Basescorecard+Optuna scorecard
TODO Add typing
TODO Comment code
TODO Improve naming and code clarity
TODO add Neurips risk scorer
TODO add class constrained OptunaScorecard

Returns:
    _type_: _description_
"""

from abc import abstractmethod
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss


class _BaseScoreCard:
    def __init__(
        self, nb_max_features, min_point_value=-2, max_point_value=3, df_info=None
    ):
        # TODO test int and positif
        self.nb_max_features = nb_max_features

        # TODO test value
        self.min_point_value = min_point_value
        self.max_point_value = max_point_value

        if df_info is not None:
            # TODO pandera check that it contains the columns binary feature et original feature
            self._df_info = df_info.copy()

        self.computation_card = None
        self.score_card = None

    @staticmethod
    def _predict_proba_score(score, intercept):
        score_associated_proba = 1 / (1 + np.exp(-(score + intercept)))
        return score_associated_proba

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X, proba_threshold=0.5):
        if (self.computation_card is None) or (self.score_card is None):
            print("ScoreCard model has not been fitted yet")
        proba = self.predict_proba(X)

        return proba[:, 1] >= proba_threshold

    def predict_proba(self, X):
        # TODO have the same behaviour as predict proba from sklearn
        df_score_table = self.score_card.copy().T.set_index("SCORE")
        list_features = self.computation_card["binary_feature"].values
        # TODO check that X only has selected featureso r just that it contains them?
        X_selected_features = X[list_features]

        points = self.computation_card["point"].values
        X_total_points = np.matmul(X_selected_features.values, points)
        proba = df_score_table.loc[X_total_points, "_RISK_FLOAT"].values.reshape(-1, 1)

        return np.concatenate([1 - proba, proba], axis=1)

    def summary(self):
        if (self.computation_card is None) or (self.score_card is None):
            print("ScoreCard model has not been fitted yet")
        print("======================")
        print("| FEATURE-POINT CARD |")
        print("======================")
        print(
            self.computation_card[["Description", "point"]]
            .sort_values(by="point")
            .to_markdown()
        )
        print()
        print("=======================================")
        print("=======================================")
        print()
        print("======================")
        print("|     SCORE CARD     |")
        print("======================")
        print(self.score_card.loc[["SCORE", "RISK"], :].to_markdown(headers="firstrow"))


class OptunaScoreCard(_BaseScoreCard):
    def __init__(
        self,
        nb_max_features,
        min_point_value=-2,
        max_point_value=3,
        df_info=None,
        optuna_optimize_params=None,
    ):
        super().__init__(nb_max_features, min_point_value, max_point_value, df_info)
        self.optuna_optimize_params = dict()
        if optuna_optimize_params is not None:
            self.optuna_optimize_params = optuna_optimize_params
        else:
            self.optuna_optimize_params["n_trials"] = 100
            self.optuna_optimize_params["timeout"] = 90

    def score_logloss_objective(self, trial, X, y):
        # TODO : put option for no duplicate features
        # change way of doing the sampling by creating all combinations of binary features
        # https://stackoverflow.com/questions/73388133/is-there-a-way-for-optuna-suggest-categoricalto-return-multiple-choices-from-l
        param_grid_point = {
            f"feature_{i}_score": trial.suggest_int(
                f"feature_{i}_score", self.min_point_value, self.max_point_value
            )
            for i in range(self.nb_max_features)
        }

        param_grid_choice_feature = {
            f"feature_{i}_choice": trial.suggest_categorical(
                f"feature_{i}_choice", X.columns
            )
            for i in range(self.nb_max_features)
        }

        score_intercept = trial.suggest_float(
            "intercept",
            -10 + self.min_point_value * self.nb_max_features,
            10 + self.max_point_value * self.nb_max_features,
            step=0.5,
        )

        selected_features = [
            param_grid_choice_feature[f"feature_{i}_choice"]
            for i in range(self.nb_max_features)
        ]

        score_vector = [
            param_grid_point[f"feature_{i}_score"] for i in range(self.nb_max_features)
        ]

        score_train = np.matmul(X[selected_features].values, np.transpose(score_vector))
        score_train_associated_probabilities = self._predict_proba_score(
            score_train, score_intercept
        )
        logloss_train = log_loss(y, score_train_associated_probabilities)

        return logloss_train

    def fit(self, X, y):
        # Setting the logging level WARNING, the INFO logs are suppressed.
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", study_name="Score search")
        optuna_objective = lambda trial: self.score_logloss_objective(
            trial,
            X,
            y,
        )
        study.optimize(optuna_objective, **self.optuna_optimize_params)
        selected_scorepyo_features_point = [
            study.best_params[f"feature_{i}_score"]
            for i in range(self.nb_max_features)
            if study.best_params[f"feature_{i}_score"] != 0
        ]
        selected_scorepyo_features = [
            study.best_params[f"feature_{i}_choice"]
            for i in range(self.nb_max_features)
            if study.best_params[f"feature_{i}_score"] != 0
        ]
        self._intercept = study.best_params["intercept"]

        self.computation_card = pd.DataFrame(index=selected_scorepyo_features)
        self.computation_card["point"] = selected_scorepyo_features_point
        if self._df_info is not None:
            self.computation_card = self.computation_card.merge(
                self._df_info, left_index=True, right_on=["binary_feature"]
            )
            # self.computation_card = self.computation_card.drop("binary_feature", axis=1)
        else:
            self.computation_card["feature"] = self.computation_card.index.copy()
        self.computation_card = self.computation_card.set_index("feature")
        self.computation_card["Description"] = self.computation_card["binary_feature"]

        min_range_score = sum(
            np.clip(i, a_min=None, a_max=0)
            for i in self.computation_card["point"].values
        )
        max_range_score = sum(
            np.clip(i, a_min=0, a_max=None)
            for i in self.computation_card["point"].values
        )
        possible_scores = list(range(min_range_score, max_range_score + 1))
        possible_risks = [
            self._predict_proba_score(s, self._intercept) for s in possible_scores
        ]
        possible_risks_pct = [f"{r:.2%}" for r in possible_risks]
        self.score_card = pd.DataFrame(
            index=["SCORE", "RISK", "_RISK_FLOAT"],
            data=[possible_scores, possible_risks_pct, possible_risks],
        )
