from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier

from smallfields.models.mlp import MLP, MLPWrapper, train_mlp


def build_classifier(
    model_name: str,
    num_classes: int = None,
    njobs: int = 1,
    rf_n_estimators: int = 100,
    rf_max_depth=None,
    rf_min_samples_split: int = 2,
    rf_max_features="sqrt",
    xgb_n_estimators: int = 100,
    xgb_max_depth: int = 6,
    xgb_learning_rate: float = 0.1,
):
    """Return an uninitialised sklearn-compatible classifier.

    Each classifier is configured with parameters that match the source scripts.
    The returned object has a ``.fit(X, y)`` method but has not been trained yet.

    Parameters
    ----------
    model_name : str
        One of: 'LogisticRegression', 'RandomForest', 'SVM', 'XGBoost'.
        MLP is handled separately by ``fit_classifier`` (uses PyTorch).
    num_classes : int
        Required for XGBoost (``num_class`` in ``multi:softprob`` objective).
    njobs : int
        Parallelism for applicable models (RF, LR).
    rf_n_estimators, rf_max_depth, rf_min_samples_split, rf_max_features
        RandomForest hyperparameters. Defaults match multirun toy scripts.
        For ``austria_s1s2_rf_smallfields.py`` use:
          n_estimators=200, max_depth=50, min_samples_split=5, max_features=None
    xgb_n_estimators, xgb_max_depth, xgb_learning_rate
        XGBoost hyperparameters. Defaults match multirun toy scripts.

    Returns
    -------
    Untrained estimator with a ``.fit()`` method.
    """
    name = model_name.lower()

    if name == "logisticregression":
        # Multinomial softmax with very weak regularisation (C=1e4 ≈ no reg).
        # class_weight='balanced' compensates for class imbalance in the
        # area-weighted split (smaller classes get fewer training pixels).
        return LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            C=1e4,          # very large C = minimal L2 regularisation
            max_iter=100000, # high limit; lbfgs may need many iterations for 7 classes
            n_jobs=njobs,
            random_state=42,
            class_weight="balanced",
        )

    elif name == "randomforest":
        # RF with balanced class weights (handled internally by sklearn).
        # max_features='sqrt' is sklearn's default for classifiers.
        # Set max_features=None to use all features (matches austria_s1s2_rf_smallfields.py).
        return RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,           # None = grow until pure leaves
            min_samples_split=rf_min_samples_split,
            max_features=rf_max_features,      # 'sqrt' default; None = all features
            min_samples_leaf=1,
            n_jobs=njobs,
            random_state=42,
            class_weight="balanced",
        )

    elif name == "svm":
        # Linear SVM with balanced class weights.
        # probability=True enables predict_proba (required for some downstream uses).
        # Note: SVM is very slow on large pixel datasets; only practical on toy/downsampled data.
        return SVC(
            kernel="linear",
            C=1.0,
            class_weight="balanced",
            random_state=42,
            probability=True,
        )

    elif name == "xgboost":
        if num_classes is None:
            raise ValueError("num_classes must be provided for XGBoost.")
        # multi:softprob outputs per-class probabilities; argmax gives the label.
        # Labels must be 0-indexed (enforced by zero_index_labels=True in loading).
        return XGBClassifier(
            n_estimators=xgb_n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            objective="multi:softprob",
            num_class=num_classes,    # must equal the number of distinct label values
            n_jobs=njobs,
            random_state=42,
        )

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            "Use one of: LogisticRegression, RandomForest, SVM, XGBoost, MLP."
        )


def fit_classifier(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    valid_classes: set,
    num_classes: int = None,
    njobs: int = 1,
    mlp_kwargs: dict = None,
    device=None,
    rf_n_estimators: int = 100,
    rf_max_depth=None,
    rf_min_samples_split: int = 2,
    rf_max_features="sqrt",
    xgb_n_estimators: int = 100,
    xgb_max_depth: int = 6,
    xgb_learning_rate: float = 0.1,
    svm_param_grid: dict = None,
    svm_cv_folds: int = 10,
):
    """Train and return a fitted classifier.

    Dispatches to the appropriate training path:
    - MLP: trains via ``train_mlp`` (PyTorch), wraps in ``MLPWrapper``.
    - SVM + svm_param_grid: uses ``GridSearchCV`` to find best hyperparams.
    - All others: builds the estimator via ``build_classifier`` then calls ``.fit()``.

    Parameters
    ----------
    model_name : str
    X_train, y_train : np.ndarray
        Training features and labels.
    X_val, y_val : np.ndarray
        Validation features and labels (used by MLP for early stopping).
    valid_classes : set
        Set of class label values that appear in the data.
        Used to infer ``num_classes`` when not provided explicitly.
    num_classes : int, optional
        Override the inferred class count (rarely needed).
    njobs : int
        Parallelism for RF, LR, SVM GridSearchCV.
    mlp_kwargs : dict, optional
        Extra kwargs forwarded to ``train_mlp`` (hidden_sizes, dropout_rate,
        learning_rate, batch_size, num_epochs, patience, use_class_weights).
    device : torch.device, optional
        PyTorch device for MLP training.
    rf_n_estimators, rf_max_depth, rf_min_samples_split, rf_max_features
        RandomForest hyperparameters passed to ``build_classifier``.
    xgb_n_estimators, xgb_max_depth, xgb_learning_rate
        XGBoost hyperparameters passed to ``build_classifier``.
    svm_param_grid : dict, optional
        If provided with model='SVM', ``GridSearchCV`` searches over this grid.
        Example: {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]}
    svm_cv_folds : int
        Number of CV folds for SVM ``GridSearchCV`` (default 10).

    Returns
    -------
    Fitted model with a ``.predict(X)`` method.
    """
    if num_classes is None:
        # Infer from the number of distinct valid class labels
        num_classes = len(valid_classes)

    name = model_name.lower()

    if name == "mlp":
        if mlp_kwargs is None:
            mlp_kwargs = {}
        input_size = X_train.shape[1]
        # MLP uses 0-indexed labels internally; label_shift=1 converts
        # 1-indexed source labels to 0-indexed during training.
        # max(valid_classes) gives the number of output neurons needed
        # (equals num_classes for 1-indexed labels).
        mlp_model = train_mlp(
            X_train, y_train, X_val, y_val,
            num_classes=max(valid_classes),
            input_size=input_size,
            label_shift=1,     # subtract 1 from y_train/y_val inside train_mlp
            device=device,
            **mlp_kwargs,
        )
        # MLPWrapper adds label_offset=1 back during predict() to restore
        # the original 1-indexed label space
        return MLPWrapper(mlp_model, label_offset=1, device=device)

    elif name == "svm" and svm_param_grid is not None:
        # Tune SVM hyperparameters via cross-validation on the training set.
        # The base SVC has no fixed C/gamma so GridSearchCV explores freely.
        base_svm = SVC(random_state=42, probability=True)
        gs = GridSearchCV(
            base_svm, svm_param_grid,
            cv=svm_cv_folds, n_jobs=njobs, scoring="accuracy", verbose=1,
        )
        gs.fit(X_train, y_train)
        return gs.best_estimator_  # already fitted on the full training set

    else:
        # Standard sklearn path: build estimator and call .fit()
        clf = build_classifier(
            model_name,
            num_classes=num_classes,
            njobs=njobs,
            rf_n_estimators=rf_n_estimators,
            rf_max_depth=rf_max_depth,
            rf_min_samples_split=rf_min_samples_split,
            rf_max_features=rf_max_features,
            xgb_n_estimators=xgb_n_estimators,
            xgb_max_depth=xgb_max_depth,
            xgb_learning_rate=xgb_learning_rate,
        )
        clf.fit(X_train, y_train)
        return clf
