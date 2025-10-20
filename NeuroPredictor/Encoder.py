import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
from typing import Optional, Union
import warnings
import test


def _prepare_noise_ceiling(
    noise_ceiling: Optional[Union[float, np.ndarray]],
    r: Union[float, np.ndarray]
) -> Optional[Union[float, np.ndarray]]:
    """
    Standardize and validate noise ceiling parameter shape.
    """
    if noise_ceiling is None:
        return None
    nc = np.asarray(noise_ceiling)
    if nc.ndim == 0:
        return float(nc)
    if isinstance(r, np.ndarray) and nc.shape == r.shape:
        return nc
    raise ValueError(f"noise_ceiling shape {nc.shape} does not match r shape {r.shape}")

def _pearson_r(y_true, y_pred, noise_ceiling=None, return_nan_if_const=False, eps=1e-12):
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes must match")
    if y_true.ndim == 1:
        # 1D 情况下直接调用 np.corrcoef 可能仍遇到常数列→NaN，这里保持原样
        r = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        yt = y_true - np.nanmean(y_true, axis=0, keepdims=True)
        yp = y_pred - np.nanmean(y_pred, axis=0, keepdims=True)

        num = np.nansum(yt * yp, axis=0)
        den = np.sqrt(np.nansum(yt**2, axis=0) * np.nansum(yp**2, axis=0))

        # 分母安全化
        if return_nan_if_const:
            r = np.divide(num, den, out=np.full_like(den, np.nan, dtype=float), where=den > eps)
        else:
            r = np.divide(num, den, out=np.zeros_like(den, dtype=float), where=den > eps)

    nc = _prepare_noise_ceiling(noise_ceiling, r)
    if nc is not None:
        # 噪声上限也做安全除法
        nc_safe = np.where(np.abs(nc) > eps, nc, np.nan if return_nan_if_const else 1.0)
        r = r / nc_safe
    return r

def _explained_variance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    noise_ceiling: Optional[Union[float, np.ndarray]] = None
) -> Union[float, np.ndarray]:
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes must match")
    if y_true.ndim == 1:
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    else:
        ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
        ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
        r2 = 1 - ss_res / ss_tot
        r2 = np.where(ss_tot == 0, 0.0, r2)
    nc = _prepare_noise_ceiling(noise_ceiling, r2)
    return r2 / (nc ** 2) if nc is not None else r2


class Encoder:
    def __init__(
        self,
        method: str = 'PLS',
        scoring: str = 'pearson',
        components: Optional[list] = None,
        alphas: Optional[list] = None,
        pca_components: Optional[int] = None,
        cv_splits: int = 5,
        eval_method: str = 'whole',  # 'cv' or 'whole'
        random_state: int = 42,
        n_jobs: int = -1,
        max_iter: int = 2000,
        per_target: bool = False
    ):
        self.method = method.upper()
        self.scoring = scoring.lower()
        if self.scoring not in ['pearson', 'explained_variance']:
            raise ValueError("scoring must be 'pearson' or 'explained_variance'")
        if self.method == 'PLS':
            self.components = components if (components is not None and len(components) > 0) else list(range(1, 31, 2))
            self.alphas = None
        else:
            self.alphas = alphas if (alphas is not None and len(alphas) > 0) else np.logspace(-2, 6, 9).tolist()
            self.components = None
        self.pca_components = pca_components
        self.cv_splits = cv_splits
        self.cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        self.eval_method = eval_method
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.per_target = per_target

        # placeholders
        self.pca_: Optional[PCA] = None
        self.model_ = None
        self.best_params_ = {}
        self.cv_pred_ = None
        self.coef_ = None
        self.intercept_ = None
        self.best_per_target_ = None
        self.best_scores_ = None

    def _scorer(self):
        if self.scoring == 'pearson':
            return _pearson_r
        elif self.scoring in ['explained_variance_custom', 'explained_variance']:
            return _explained_variance
        else:
            # For string scoring methods, return the string itself
            return self.scoring
    
    def _get_cv_scoring(self):
        """Return the appropriate scoring parameter for GridSearchCV."""
        scoring_func = self._scorer()
        
        # If it's a callable (custom function), wrap it in lambda
        if callable(scoring_func):
            return lambda est, X_, y_: scoring_func(y_, est.predict(X_)).mean()
        # If it's a string, return as-is for sklearn's built-in scoring
        else:
            return scoring_func
    
    def _hyperparam_selection_whole_dataset(self, X: np.ndarray, y: np.ndarray, noise_ceiling=None):
        """
        Improved: Select hyperparameters by evaluating with CV only,
        do NOT fit on whole dataset inside loop. Final model is fit once at the end.
        """
        scoring_func = self._scorer()
        cv = self.cv
        best_score = -np.inf
        best_params = None
        best_param_value = 0  # hold the best hyperparameter value
        
        if self.method == 'PLS':
            print(f"Testing {len(self.components)} PLS components (evaluate with CV)...")
            for n_comp in self.components:
                n_comp_clamped = min(n_comp, min(X.shape[0], X.shape[1]) - 1)
                pls_model = PLSRegression(n_components=n_comp_clamped, max_iter=self.max_iter)
                cv_pred = cross_val_predict(pls_model, X, y, cv=cv, n_jobs=self.n_jobs)
                score = scoring_func(y, cv_pred, noise_ceiling)
                score_mean = score.mean() if hasattr(score, 'mean') else score
                # print(f"PLS n_components={n_comp_clamped} score={score_mean:.4f}")
                if score_mean > best_score:
                    best_score = score_mean
                    best_params = {'n_components': n_comp_clamped}
                    best_param_value = n_comp_clamped

            # Build final best model and fit on whole data
            best_model = PLSRegression(n_components=best_param_value, max_iter=self.max_iter)
            best_model.fit(X, y)
        
        elif self.method == 'RIDGE':
            print(f"Testing {len(self.alphas)} Ridge alphas (evaluate with CV)...")
            for alpha in self.alphas:
                ridge_model = Ridge(alpha=alpha, fit_intercept=True)
                cv_pred = cross_val_predict(ridge_model, X, y, cv=cv, n_jobs=self.n_jobs)
                score = scoring_func(y, cv_pred, noise_ceiling)
                score_mean = score.mean() if hasattr(score, 'mean') else score
                # print(f"Ridge alpha={alpha} score={score_mean:.4f}")
                if score_mean > best_score:
                    best_score = score_mean
                    best_params = {'alpha': alpha}
                    best_param_value = alpha

            best_model = Ridge(alpha=best_param_value, fit_intercept=True)
            best_model.fit(X, y)

        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
        return best_model, best_params, best_score

    def _evaluate_with_cv_average(self, model, X: np.ndarray, y: np.ndarray, noise_ceiling=None):
        """
        NEW: Evaluate model using CV average (like GridSearchCV does).
        """
        scoring_func = self._scorer()
        cv = self.cv
        scores = np.zeros(y.shape[1])
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Clone and fit model on training fold
            if self.method == 'PLS':
                fold_model = PLSRegression(n_components=model.n_components, max_iter=self.max_iter)
            elif self.method == 'RIDGE':
                fold_model = Ridge(alpha=model.alpha, fit_intercept=True)
            else:
                raise ValueError(f"Unsupported method: {self.method}")
            
            fold_model.fit(X_train, y_train)
            y_pred = fold_model.predict(X_val)
            
            fold_score = scoring_func(y_val, y_pred, noise_ceiling)
            fold_score = np.atleast_1d(fold_score)
            scores += fold_score
        scores /= self.cv.get_n_splits()
        return scores

    def fit(
        self, X: np.ndarray, y: np.ndarray, noise_ceiling=None
    ) -> None:
        """
        1) Optional PCA on X
        2) Hyperparameter search (PLS or Ridge, with optional per-target)
        3) Cross-val predict & score
        4) Final fit on full data
        """
        # Optional PCA
        if self.pca_components is not None:
            self.pca_ = PCA(n_components=self.pca_components, random_state=self.random_state)
            X = self.pca_.fit_transform(X)
            self.pca_components_ = self.pca_.components_

        # Get scoring functions
        scoring_func = _pearson_r if self.scoring == 'pearson' else _explained_variance
        cv_scoring = self._get_cv_scoring()

        # Per-target branch
        if self.per_target and y.ndim == 2:
            if self.method == 'PLS':
                return self._fit_pls_per_target(X, y, noise_ceiling)
            else:
                return self._fit_ridge_per_target(X, y, noise_ceiling)

        # Hyperparameter selection
        if self.eval_method == 'whole':
            best_est, self.best_params_, score = self._hyperparam_selection_whole_dataset(
                X, y, noise_ceiling
            )
        elif self.eval_method == 'cv':
            if self.method == 'PLS':
                search = GridSearchCV(
                    PLSRegression(max_iter=self.max_iter),
                    {'n_components': self.components},
                    cv=self.cv,
                    scoring=cv_scoring,
                    n_jobs=self.n_jobs
                )
            else:
                search = GridSearchCV(
                    Ridge(),
                    {'alpha': self.alphas},
                    cv=self.cv,
                    scoring=cv_scoring,
                    n_jobs=self.n_jobs
                )
            search.fit(X, y)
            best_est = search.best_estimator_
            self.best_params_ = search.best_params_
         
        # Evaluation
        self.cv_pred_ = cross_val_predict(
                best_est, X, y, cv=self.cv, n_jobs=self.n_jobs
            )
        if self.eval_method == 'cv':
            self.best_scores_ = self._evaluate_with_cv_average(
                best_est, X, y, noise_ceiling
            )
        elif self.eval_method == 'whole':
            self.best_scores_ = scoring_func(y, self.cv_pred_, noise_ceiling)

        # Final full-data fit
        self.model_ = best_est
        self.model_.fit(X, y)
        self.coef_ = getattr(self.model_, 'coef_', None)
        self.intercept_ = getattr(self.model_, 'intercept_', None)
    
    # Nested CV, confirming no data leaking
    def fit_nested_cv(self, X: np.ndarray, y: np.ndarray, noise_ceiling=None):
        # Optional PCA
        if self.pca_components is not None:
            self.pca_ = PCA(n_components=self.pca_components, random_state=self.random_state)
            X = self.pca_.fit_transform(X)
            self.pca_components_ = self.pca_.components_

        # Get scoring functions
        scoring_func = _pearson_r if self.scoring == 'pearson' else _explained_variance
        cv_scoring = self._get_cv_scoring()

        outer_splits, inner_splits = self.cv_splits, self.cv_splits
        n_samples = X.shape[0]
        pred_y = np.zeros_like(y, dtype=float)

        outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=self.random_state)
        for train_idx, test_idx in outer_cv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=self.random_state)
            # Hyperparameter selection
            if self.eval_method == 'whole':
                best_est, _, score = self._hyperparam_selection_whole_dataset(
                    X_train, y_train, noise_ceiling
                )
            elif self.eval_method == 'cv':
                if self.method == 'PLS':
                    search = GridSearchCV(
                        PLSRegression(max_iter=self.max_iter),
                        {'n_components': self.components},
                        cv=inner_cv,
                        scoring=cv_scoring,
                        n_jobs=self.n_jobs
                    )
                else:
                    search = GridSearchCV(
                        Ridge(),
                        {'alpha': self.alphas},
                        cv=inner_cv,
                        scoring=cv_scoring,
                        n_jobs=self.n_jobs
                    )
                search.fit(X_train, y_train)
                best_est = search.best_estimator_
            best_est.fit(X_train, y_train)
            pred_y[test_idx] = best_est.predict(X_test)
        
        pred_score = scoring_func(y, pred_y, noise_ceiling)
        return pred_y, pred_score

    def _fit_ridge_per_target(
        self, X: np.ndarray, y: np.ndarray, noise_ceiling=None
    ):
        best_scores, best_alphas = None, None
        
        for alpha in self.alphas:
            model = Ridge(alpha=alpha)
            if self.eval_method == 'whole':
                preds = cross_val_predict(
                    model, X, y, cv=self.cv, n_jobs=self.n_jobs
                )
                scores = (_pearson_r if self.scoring == 'pearson' else _explained_variance)(
                    y, preds, noise_ceiling
                )
            elif self.eval_method == 'cv':
                scores = self._evaluate_with_cv_average(model, X, y, noise_ceiling)
            scores = np.atleast_1d(scores)
            if best_scores is None:
                best_scores, best_alphas = scores.copy(), np.full(scores.shape, alpha)
            else:
                mask = scores > best_scores
                best_scores[mask] = scores[mask]
                best_alphas[mask] = alpha
                
        self.best_per_target_, self.best_scores_ = best_alphas, best_scores
        coefs, inters = [], []
        for i, alpha in enumerate(best_alphas):
            m = Ridge(alpha=alpha).fit(X, y[:, i])
            coefs.append(m.coef_)
            inters.append(m.intercept_)
        self.coef_ = np.vstack(coefs)
        self.intercept_ = np.array(inters)
        mor = MultiOutputRegressor(Ridge(), n_jobs=self.n_jobs)
        mor.estimators_ = [Ridge(alpha=a) for a in best_alphas]
        self.model_ = mor

    def _fit_pls_per_target(
        self, X: np.ndarray, y: np.ndarray, noise_ceiling=None
    ):
        best_scores, best_comps = None, None
        for comp in self.components:
            model = PLSRegression(n_components=comp, max_iter=self.max_iter)
            if self.eval_method == 'whole':
                preds = cross_val_predict(
                    model, X, y, cv=self.cv, n_jobs=self.n_jobs
                )
                scores = (_pearson_r if self.scoring == 'pearson' else _explained_variance)(
                    y, preds, noise_ceiling
                )
            elif self.eval_method == 'cv':
                scores = self._evaluate_with_cv_average(model, X, y, noise_ceiling)
            scores = np.atleast_1d(scores)
            if best_scores is None:
                best_scores, best_comps = scores.copy(), np.full(scores.shape, comp)
            else:
                mask = scores > best_scores
                best_scores[mask] = scores[mask]
                best_comps[mask] = comp
        self.best_per_target_, self.best_scores_ = best_comps, best_scores
        coefs, inters = [], []
        for i, comp in enumerate(best_comps):
            m = PLSRegression(n_components=comp, max_iter=self.max_iter).fit(X, y[:, i])
            coefs.append(m.coef_.flatten())
            inters.append(m._y_mean)
        self.coef_ = np.vstack(coefs)
        self.intercept_ = np.array(inters)
        mor = MultiOutputRegressor(PLSRegression(), n_jobs=self.n_jobs)
        mor.estimators_ = [PLSRegression(n_components=c, max_iter=self.max_iter) for c in best_comps]
        self.model_ = mor

    def predict(
        self, X: np.ndarray
    ) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
        if self.pca_ is not None:
            X = self.pca_.transform(X)
        return self.model_.predict(X)

    def score(
        self, X: np.ndarray, y: np.ndarray, noise_ceiling=None
    ) -> Union[float, np.ndarray]:
        y_pred = self.predict(X)
        fn = _pearson_r if self.scoring == 'pearson' else _explained_variance
        return fn(y, y_pred, noise_ceiling)

    def get_results(self) -> dict:
        return {
            'best_params': self.best_params_,
            'best_per_target': self.best_per_target_,
            'best_scores': self.best_scores_,
            'scoring': self.scoring,
            'method': self.method
        }
