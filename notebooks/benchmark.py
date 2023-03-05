import time

import numpy as np
import pandas as pd
from fasterrisk.fasterrisk import RiskScoreClassifier, RiskScoreOptimizer
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from scorepyo.models import EBMRiskScoreNew


def read_heart_data():
    columns = [
        "age",
        "sex",
        "chest pain type",
        "resting blood pressure",
        "serum cholestoral in mg/dl",
        "fasting blood sugar > 120 mg/dl",
        "resting electrocardiographic results",
        "maximum heart rate achieved",
        "exercise induced angina",
        "oldpeak = ST depression induced by exercise relative to rest",
        "the slope of the peak exercise ST segment",
        "number of major vessels (0-3) colored by flourosopy",
        "thal",
        "target",
    ]
    data = pd.read_csv(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat",
        header=None,
        sep=" ",
    )
    data.columns = columns

    categorical_columns = [
        "sex",
        "chest pain type",
        "fasting blood sugar > 120 mg/dl",
        "resting electrocardiographic results",
        "thal",
        "exercise induced angina",
    ]

    for c in categorical_columns:
        data[c] = data[c].astype("category")
    else:
        data[c] = data[c].astype(float)

    data["target"] = np.where(data["target"] == 2, 1, 0)

    features = [c for c in columns if c != "target"]
    X = data[features]

    y = data["target"]

    return X, y, categorical_columns


def read_spam_data():
    data = pd.read_csv(
        "https://raw.githubusercontent.com/ustunb/risk-slim/master/examples/data/spambase_data.csv",
        sep=",",
    )
    features = list(data.columns[1:])

    categorical_columns = []
    X = data[features]
    y = data["Spam"]

    return X, y, categorical_columns


def read_adult_data():
    data = pd.read_csv(
        "https://raw.githubusercontent.com/ustunb/risk-slim/master/examples/data/adult_data.csv",
        sep=",",
    )

    features = list(data.columns[1:])

    categorical_columns = features

    for c in categorical_columns:
        data[c] = data[c].astype("category")
    else:
        data[c] = data[c].astype(float)

    features = list(data.columns[1:])
    X = data[features]

    y = data["Over50K"]

    return X, y, categorical_columns


def read_bank_data():
    data = pd.read_csv(
        "https://raw.githubusercontent.com/ustunb/risk-slim/master/examples/data/bank_data.csv",
        sep=",",
    )

    features = list(data.columns[1:])

    categorical_columns = features

    for c in categorical_columns:
        data[c] = data[c].astype("category")
    else:
        data[c] = data[c].astype(float)

    features = list(data.columns[1:])
    X = data[features]

    y = data["sign_up"]

    return X, y, categorical_columns


def read_mammo_data():
    data = pd.read_csv(
        "https://raw.githubusercontent.com/ustunb/risk-slim/master/examples/data/mammo_data.csv",
        sep=",",
    )

    features = list(data.columns[1:])

    categorical_columns = features

    for c in categorical_columns:
        data[c] = data[c].astype("category")
    else:
        data[c] = data[c].astype(float)

    features = list(data.columns[1:])
    X = data[features]

    y = data["Malignant"]

    return X, y, categorical_columns


def read_mushroom_data():
    data = pd.read_csv(
        "https://raw.githubusercontent.com/ustunb/risk-slim/master/examples/data/mushroom_data.csv",
        sep=",",
    )

    features = list(data.columns[1:])

    categorical_columns = features

    for c in categorical_columns:
        data[c] = data[c].astype("category")
    else:
        data[c] = data[c].astype(float)

    features = list(data.columns[1:])
    X = data[features]

    y = data["poisonous"]

    return X, y, categorical_columns


def read_data(dataset):
    if dataset == "heart":
        return read_heart_data()
    elif dataset == "spam":
        return read_spam_data()
    elif dataset == "adult":
        return read_adult_data()
    elif dataset == "bank":
        return read_bank_data()
    elif dataset == "mammo":
        return read_mammo_data()
    elif dataset == "mushroom":
        return read_mushroom_data()
    else:
        raise ValueError


def train_faster_risk(
    X,
    y,
    min_point_value,
    max_point_value,
    nb_max_features,
    parent_size=10,
    child_size=None,
    max_attempts=50,
    num_ray=20,
    lineSearch_early_stop_tolerance=1e-3,
):

    RiskScoreOptimizer_m = RiskScoreOptimizer(
        X=X.values,
        y=y,
        lb=min_point_value,
        ub=max_point_value,
        k=nb_max_features,
        parent_size=parent_size,
        child_size=child_size,
        maxAttempts=max_attempts,
        num_ray_search=num_ray,
        lineSearch_early_stop_tolerance=lineSearch_early_stop_tolerance,
    )

    RiskScoreOptimizer_m.optimize()

    (
        multipliers,
        sparseDiversePool_beta0_integer,
        sparseDiversePool_betas_integer,
    ) = RiskScoreOptimizer_m.get_models()

    model_index = 0  # first model
    multiplier = multipliers[model_index]
    intercept = sparseDiversePool_beta0_integer[model_index]
    coefficients = sparseDiversePool_betas_integer[model_index]

    RiskScoreClassifier_m = RiskScoreClassifier(multiplier, intercept, coefficients)

    return RiskScoreClassifier_m


def compare_random_seed_fixed(
    X,
    y,
    categorical_features,
    nb_additional_features=4,
    random_state=0,
    use_calib=False,
    random_state_calib=0,
    sorting_method="new",
    nb_binaries_by_features=3,
    optimization_method="worse",
):
    X_train_orig, X_test, y_train_orig, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    min_point_value = -2
    max_point_value = 3
    nb_max_features = 4

    ebm_model = EBMRiskScoreNew(
        min_point_value=min_point_value,
        max_point_value=max_point_value,
        nb_max_features=nb_max_features,
        nb_additional_features=nb_additional_features,
        max_number_binaries_by_features=nb_binaries_by_features,
        # optimization_metric="roc_auc",
    )

    if use_calib:
        X_train, X_calib, y_train, y_calib = train_test_split(
            X_train_orig, y_train_orig, test_size=0.4, random_state=random_state_calib
        )
    else:
        X_train = X_train_orig.copy()
        y_train = y_train_orig.copy()
        X_calib = X_train
        y_calib = y_train
        # # TODO Delete
        # X_calib = X_test.copy()
        # y_calib = y_test.copy()

    ebm_model.fit(
        X_train,
        y_train,
        X_calib=X_calib,
        y_calib=y_calib,
        categorical_features=categorical_features,
        sorting_method=sorting_method,
        ranker=sorting_method,
        optimization_method=optimization_method,
    )

    binarizer = ebm_model._binarizer

    X_train_binarized = binarizer.transform(X_train)
    X_calib_binarized = binarizer.transform(X_calib)
    X_test_binarized = binarizer.transform(X_test)

    X_train_orig_binarized = binarizer.transform(X_train_orig)
    y_train_fasterrisk = np.where(y_train_orig == 0, -1, 1)

    start_time = time.time()
    faster_risk_model = train_faster_risk(
        X_train_orig_binarized,
        y_train_fasterrisk,
        min_point_value,
        max_point_value,
        nb_max_features,
    )
    fit_time_faster_risk = time.time() - start_time

    y_proba_faster_risk_train_orig = faster_risk_model.predict_prob(
        X_train_orig_binarized.values
    )
    y_proba_faster_risk_train = faster_risk_model.predict_prob(X_train_binarized.values)
    y_proba_faster_risk_calib = faster_risk_model.predict_prob(X_calib_binarized.values)
    y_proba_faster_risk_test = faster_risk_model.predict_prob(X_test_binarized.values)

    y_proba_ebm_train_orig = ebm_model.predict_proba(X_train_orig)[:, 1].reshape(-1, 1)
    y_proba_ebm_train = ebm_model.predict_proba(X_train)[:, 1].reshape(-1, 1)
    y_proba_ebm_calib = ebm_model.predict_proba(X_calib)[:, 1].reshape(-1, 1)
    y_proba_ebm_test = ebm_model.predict_proba(X_test)[:, 1].reshape(-1, 1)

    proba_faster_risk = [
        y_proba_faster_risk_train_orig,
        y_proba_faster_risk_train,
        y_proba_faster_risk_calib,
        y_proba_faster_risk_test,
    ]
    proba_ebm = [
        y_proba_ebm_train_orig,
        y_proba_ebm_train,
        y_proba_ebm_calib,
        y_proba_ebm_test,
    ]

    targets = [y_train_orig, y_train, y_calib, y_test]

    metrics = [log_loss, average_precision_score, roc_auc_score]

    data_metrics = []
    for list_y_probas in [proba_faster_risk, proba_ebm]:
        for y_proba, y_true in zip(list_y_probas, targets):
            for metric in metrics:
                data_metrics.append(metric(y_true.astype(int), y_proba))

    data_metrics.append(fit_time_faster_risk)
    data_metrics.append(ebm_model._model_fit_time)
    return data_metrics


def run(
    n_random_splits,
    list_datasets=("heart", "spam", "adult", "bank", "mammo", "mushroom"),
    range_nb_additional_features=None,
    sorting_methods=("vanilla", "new"),
    optimization_methods=("vanilla", "worse", "mean"),
    range_nb_binaries_by_features=(3,),
    test_calibration=False,
):
    pd.set_option("mode.chained_assignment", None)
    column_names = [
        model + "_" + datasets + "_" + metric
        for model in ["faster_risk", "ebm"]
        for datasets in ["train_orig", "train", "calib", "test"]
        for metric in ["log_loss", "average_precision", "roc_auc"]
    ]

    column_names.extend(
        [
            "faster_risk_fit_time",
            "ebm_risk_score_fit_time",
            "random_seed",
            "use_calibration",
            "dataset",
            "nb_additional_features",
            "sorting_method",
            "nb_binaries_by_feature",
            "optimization_method",
        ]
    )

    results = []
    range_nb_additional_features = (
        range_nb_additional_features
        if range_nb_additional_features is not None
        else [4]
    )
    i = 0
    range_calibration = [True, False] if test_calibration else [False]
    for dataset_name in list_datasets:
        X, y, categorical_features = read_data(dataset=dataset_name)
        for random_seed in np.random.randint(0, 1000, size=n_random_splits):
            for use_calibration in range_calibration:

                for nb_additional_features in range_nb_additional_features:
                    for sorting_method in sorting_methods:
                        for nb_binaries_by_features in range_nb_binaries_by_features:
                            for optimization_method in optimization_methods:
                                i += 1
                                # try:
                                results_tmp = compare_random_seed_fixed(
                                    X,
                                    y=y,
                                    nb_additional_features=nb_additional_features,
                                    categorical_features=categorical_features,
                                    random_state=random_seed,
                                    random_state_calib=random_seed,
                                    use_calib=use_calibration,
                                    sorting_method=sorting_method,
                                    nb_binaries_by_features=nb_binaries_by_features,
                                    optimization_method=optimization_method,
                                )

                                results_tmp.extend(
                                    [
                                        random_seed,
                                        use_calibration,
                                        dataset_name,
                                        nb_additional_features,
                                        sorting_method.__class__,
                                        nb_binaries_by_features,
                                        optimization_method,
                                    ]
                                )
                                # except:
                                #     results_tmp = [None for _ in results[0]]

                                # print("\t", results_tmp[9:12], results_tmp[-11:])
                                if i % 20 == 0:
                                    print(i)
                                results.append(results_tmp.copy())

    df_results = pd.DataFrame(data=results, columns=column_names)
    return df_results


def run_fasterrisk_benchmark(
    X,
    y,
    min_point_value,
    max_point_value,
    nb_max_features,
    parent_size_range=(10,),
    child_size_range=(None,),
    max_attempts_range=(50,),
    num_ray_range=(20,),
    lineSearch_early_stop_tolerance_range=(1e-3,),
):
    list_models = []
    for parent_size in parent_size_range:
        for child_size in child_size_range:
            for max_attempts in max_attempts_range:
                for num_ray in num_ray_range:
                    for (
                        lineSearch_early_stop_tolerance
                    ) in lineSearch_early_stop_tolerance_range:
                        st = time.time()
                        model_ = train_faster_risk(
                            X,
                            y,
                            min_point_value,
                            max_point_value,
                            nb_max_features,
                            parent_size=parent_size,
                            child_size=child_size,
                            max_attempts=max_attempts,
                            num_ray=num_ray,
                            lineSearch_early_stop_tolerance=lineSearch_early_stop_tolerance,
                        )
                        end = time.time()
                        result_ = [
                            model_,
                            end - st,
                            parent_size,
                            child_size,
                            max_attempts,
                            num_ray,
                            lineSearch_early_stop_tolerance,
                        ]
                        list_models.append(result_)

    return list_models
