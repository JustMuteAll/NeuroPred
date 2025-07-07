import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.decomposition import PCA
from typing import Optional, List, Union


def _pearson_r(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    noise_ceiling: Optional[np.ndarray] = None
) -> Union[float, np.ndarray]:
    """
    Compute Pearson correlation coefficient, with optional noise ceiling correction.

    Args:
      y_true:   np.ndarray of shape (T,) or (N, T)
      y_pred:   np.ndarray of same shape as y_true
      noise_ceiling: 
        - None: no correction.
        - scalar: correct all r by dividing by this scalar.
        - 1D array of length N (when y_true is 2D): element-wise correction.

    Returns:
      float (if 1D inputs) or 1D np.ndarray of length N (if 2D inputs).
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes must match, got {y_true.shape} vs {y_pred.shape}")

    # 1D case
    if y_true.ndim == 1:
        r = np.corrcoef(y_true, y_pred)[0, 1]

    # 2D case: row-wise
    elif y_true.ndim == 2:
        yt = y_true - y_true.mean(axis=0, keepdims=True)
        yp = y_pred - y_pred.mean(axis=0, keepdims=True)
        num   = np.sum(yt * yp, axis=0)
        denom = np.linalg.norm(yt, axis=0) * np.linalg.norm(yp, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = num / denom
    else:
        raise ValueError(f"Input arrays must be 1D or 2D, got ndim={y_true.ndim}")

    # --- Noise ceiling correction ---
    if noise_ceiling is not None:
        nc = np.asarray(noise_ceiling)
        # scalar case
        if nc.ndim == 0:
            r = r / float(nc)
        # vector case: must match r's shape
        else:
            if r.ndim == 0:
                raise ValueError(
                    f"noise_ceiling must be scalar when y_true is 1D, got array of shape {nc.shape}"
                )
            if nc.shape != r.shape:
                raise ValueError(
                    f"noise_ceiling shape {nc.shape} does not match number of rows {r.shape}"
                )
            r = r / nc

    return r


class Encoder:
    def __init__(
        self,
        method: str = 'PLS',
        pls_components: Optional[List[int]] = None,
        ridge_alphas: Optional[List[float]] = None,
        pca_comps: Optional[int] = None,
        cv_folds: int = 5,
        n_jobs: int = -1,
        random_state: int = 42
    ):
        self.method = method.upper()
        self.pls_components = pls_components if (pls_components is not None and len(pls_components) > 0) else [5, 10, 20, 50]
        self.ridge_alphas  = ridge_alphas  if (ridge_alphas  is not None and len(ridge_alphas) > 0) else np.logspace(-5, 5, 11)
        self.pca_comps     = pca_comps
        self.cv_folds      = cv_folds
        self.n_jobs        = n_jobs
        self.random_state  = random_state

        # PCA attributes (set in fit)
        self.pca_: Optional[PCA] = None
        self.pca_components_: Optional[np.ndarray] = None

        # Fitted model attributes
        self.best_params_ = {}
        self.model_       = None
        self.coef_        = None
        self.intercept_   = None
        self.cv_pred_     = None
        self.cv_r_        = None

    def _make_cv(self):
        return KFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray, noise_ceiling=None) -> None:
        """
        1) Optional PCA on X
        2) Hyperparameter search (PLS or Ridge)
        3) Cross-val predict & score
        4) Final fit on full data
        """
        # --- PCA Step ---
        if self.pca_comps is not None:
            self.pca_ = PCA(n_components=self.pca_comps, random_state=self.random_state)
            X = self.pca_.fit_transform(X)
            self.pca_components_ = self.pca_.components_

        # --- Model & Search Setup ---
        if self.method == 'PLS':
            base = PLSRegression()
            param_grid = {'n_components': self.pls_components}
            search = GridSearchCV(
                estimator=base,
                param_grid=param_grid,
                cv=self._make_cv(),
                scoring=lambda est, X_, y_: _pearson_r(
                    y_, est.predict(X_)).mean(),
                n_jobs=self.n_jobs
            )
        elif self.method == 'RIDGE':
            search = RidgeCV(
                alphas=self.ridge_alphas,
                cv=self._make_cv(),
                scoring=None,
                store_cv_results=False
            )
        else:
            raise ValueError(f"Unsupported method: {self.method}")

        # --- Hyperparameter Search ---
        search.fit(X, y)
        if self.method == 'PLS':
            best_est = search.best_estimator_
            self.best_params_ = search.best_params_
        else:
            best_est = search
            self.best_params_ = {'alpha': float(search.alpha_)}

        # --- Cross-val Predict and Score ---
        self.cv_pred_ = cross_val_predict(
            best_est, X, y,
            cv=self._make_cv(),
            n_jobs=self.n_jobs
        )
        self.cv_r_ = _pearson_r(y, self.cv_pred_, noise_ceiling)

        # --- Final Fit on Full Data ---
        self.model_ = best_est
        self.model_.fit(X, y)
        self.coef_      = getattr(self.model_, 'coef_', None)
        self.intercept_ = getattr(self.model_, 'intercept_', None)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted. Call .fit() first.")
        if self.pca_ is not None:
            X = self.pca_.transform(X)
        return self.model_.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray, noise_ceiling=None) -> float:
        y_pred = self.predict(X)
        return _pearson_r(y, y_pred, noise_ceiling)
