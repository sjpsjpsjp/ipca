"""
IPCA: Instrumented Principal Components Analysis
Estimation class for Kelly, Pruitt, and Su (2019 JFE)

version 2.0.1

copyright Seth Pruitt (2020-2026)
"""

import copy
import warnings
from datetime import datetime
from timeit import default_timer as timer
from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg as sla
from sklearn.linear_model import lasso_path
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed



# noinspection PyPep8Naming
class ipca(object):
    def __init__(self, RZ=None, return_column=0, X=None, W=None, Nts=None, add_constant=True):
        """
        [Inputs]
        RZ : df(TotalObs x L0+1) : returns and characteristics in a df with a 2-level MultiIndex index where Date
            is level 0 and AssetID is level 1. In a balanced dataset, TotalObs would equal the number of unique
            AssetIDs times the number of unique Dates, T; however, data is generally unbalanced and therefore the
            number of AssetIDs present varies over Dates. We split RZ into R df(TotalObs x 1) and Z
            df(TotalObs x L0). The code assumes that Z and R have been properly timed outside of IPCA, such that
            the rows of R with Date=d correspond to the rows of Z with Date=d according to the IPCA model in mind.
            E.g. the conditional APT of Kelly, Pruitt, Su (JFE 2019) says that the characteristics known at
            Date=d-1 determine the exposures associated with the returns realized at Date=d; hence, here we should
            have shifted the characteristics in Z relative to the returns in R. L0 is number of characteristics,
            and the unique CharNames are the 1-level column index of df Z.
            RZ is optional because we might use this code by passing in only the cross-sectional second moment
            df W, described below. E.g. when using the code for bootstrapped tests.

        return_column : int or str : which column of RZ, if given, is the return; all others are assumed to be
            chars. Default is 0: the first column. An int can be supplied to determine which column, or a
            column name str.

        X : df(L x T) : Managed portfolio returns in a df with an index of CharNames (plus perhaps constant) and
            columns of the unique Dates. X is optional because we might pass in R and Z directly and create X.
            See add_constant input description about dimension L.

        W : df(TL x L) : Cross-sectional second moments of characteristics, for the T unique Dates.
            W is optional because we might pass in Z directly and create W.
            See add_constant input description about dimension L.

        Nts : srs(T) : number of available assets for each Date.
            Nts is optional because we can construct it from Z and R, if those are passed. Otherwise it is only
            useful if we are passing X and W, but not Z and R.

        add_constant : bool : True (default) adds a constant to the calculation of X, W and it becomes the last
            char listed in their indices. If true, then L = L0+1; if false, then L = L0.

        [Transformed Inputs]
        X and W and Nts : df(L x T) and df(TL x L) and srs(T) : if Z, R input, these are created.
        """
        self.RZ, self.X, self.W, self.Nts = RZ, X, W, Nts
        self.has_RZ = RZ is not None
        self.has_X = X is not None
        self.has_W = W is not None

        if self.has_X and self.has_W and not self.has_RZ and self.Nts is None:
            raise ValueError('RZ are not passed; X,W were passed with no Nts')

        if self.has_RZ:
            if isinstance(return_column, int):
                if return_column >= RZ.shape[1]:
                    raise ValueError('return_column int is greater than the number of columns in RZ')
                self.R = RZ.iloc[:, return_column].to_frame()
            elif isinstance(return_column, (str, tuple)):
                self.R = RZ.loc[:, return_column].to_frame()
            else:
                raise ValueError('Did not know how to pick return column')
            self.Z = RZ.drop(columns=self.R.columns)

        if not self.has_X or not self.has_W:
            charlist = list(self.Z.columns) + (['Constant'] if add_constant else [])
            datelist = self.Z.index.get_level_values(0).unique()

            if not self.has_X:
                self.X = pd.DataFrame(index=charlist, columns=datelist, data=np.nan)
            if not self.has_W:
                self.W = pd.DataFrame(
                    data=0.,
                    index=pd.MultiIndex.from_product((datelist, charlist), names=['date', 'char']),
                    columns=charlist)
            self.Nts = pd.Series(index=datelist, data=np.nan)

            def _xw_for_date(t):
                Zt = self.Z.loc[t, :].values
                if add_constant:
                    Zt = np.concatenate((Zt, np.ones((Zt.shape[0], 1))), axis=1)
                Nt = Zt.shape[0]
                Xt = Zt.T.dot(self.R.loc[t]) / Nt if not self.has_X else None
                Wt = Zt.T.dot(Zt) / Nt         if not self.has_W else None
                return Nt, Xt, Wt

            per_date = Parallel(n_jobs=-1, prefer='threads')(
                delayed(_xw_for_date)(t) for t in datelist)

            for t, (Nt, Xt, Wt) in zip(datelist, per_date):
                self.Nts[t] = Nt
                if not self.has_X:
                    self.X[t] = Xt
                if not self.has_W:
                    self.W.loc[t] = Wt

        self.Chars = self.X.index
        self.Dates = self.X.columns
        self.L = self.X.shape[0]
        self.add_constant = add_constant

        # Pre-build float64 ndarray versions for fast inner-loop access
        # _X  : (L, T)      managed portfolio returns
        # _W  : (L, L, T)   cross-sectional second moments
        # _Nts: (T,)         number of assets per period
        T = len(self.Dates)
        self._X = self.X.values.astype(np.float64)
        self._Nts_arr = self.Nts.values.astype(np.float64)
        self._W = np.empty((self.L, self.L, T), dtype=np.float64)
        for ct, t in enumerate(self.Dates):
            self._W[:, :, ct] = self.W.loc[t].values

    # ------------------------------------------------------------------
    # Public estimation entry point
    # ------------------------------------------------------------------

    def fit(self, K=1, OOS=False, gFac=None,
            normalization_choice='PCA_positivemean', normalization_choice_specs=None,
            OOS_window='recursive', OOS_window_specs=60,
            factor_mean='constant', R_fit=True, Beta_fit=False,
            dispIters=False, minTol=1e-4, maxIters=5000,
            F_names=None, G_names=None, R2_bench='zero', dispItersInt=100,
            MacroData=None, target_variance=None, regularization=None, alpha=None,
            ridge_df=5,
            pass2_intercept=True,
            min_combo_periods=None, min_train_periods=None,
            kappa_max=1e8, const_tol=1e-6, tan_target_vol=1.0):
        """
        [Inputs]

        K : int : number of latent factors. We define KM = K + M; KM must be at least 1.

        OOS : bool : False (default) runs in-sample estimation; True runs out-of-sample using the OOS_* options.

        gFac : df(M x T) : pre-specified factors. Optional.
            At most one row may be constant (near-zero relative standard deviation).
            A constant row must have value 1; any other constant value raises a
            ValueError.  If a constant row is detected, it is moved to the last
            column position in Factor and Gamma output and a message is printed;
            its Gamma loading (GammaAlpha) is used for ArbPtf.

        normalization_choice : str :
            'PCA_positivemean' (default) - Gamma unitary, orthogonal factors with non-negative means.
            'Identity' - selected characteristics have unit loadings on one factor each;
                         requires normalization_choice_specs.

        normalization_choice_specs : list of str : used when normalization_choice='Identity'; K characteristic
            names that anchor each factor.

        OOS_window : str : 'recursive' (default) expanding window; 'rolling' fixed-length window.

        OOS_window_specs : int or str or pd.Timestamp : minimum window length (recursive) or window
            length (rolling). Default 60.
            If a timestamp (or a string parseable as one) is passed, it is interpreted as the first
            OOS date: the code finds its position in self.Dates and uses that index as the int value.
            If the timestamp is not present exactly, the nearest date on or after it is used and a
            warning is issued.  If no date on or after the timestamp exists, the default (60) is
            used and a warning is issued.

        factor_mean : str :
            'constant'   (default) - factor mean estimated as training-sample mean.
            'VAR1'                 - VAR(1) model for time-varying conditional means.
            'macro'                - macro-data-only factor mean; no VAR(1) component, no
                                     combination step.
                IS:  single full-sample regression of factors on macro data, evaluated at
                     all T in-sample dates.
                OOS: expanding- or rolling-window regression evaluated at each new date.
            'forecombo'            - forecast combination of VAR(1) and macro predictors.
                IS:  full-sample VAR(1) forecasts and macro forecasts are combined via full-sample
                     OLS regression (weights constant across time).
                OOS: expanding-window OLS combination of VAR(1) and macro forecasts; equal-weight
                     average used as fallback until min_combo_periods OOS observations accumulate.

        R_fit : bool : compute individual-return fitted values (requires RZ input).

        Beta_fit : bool : return individual-return betas (requires RZ input).

        dispIters : bool : print convergence progress.

        dispItersInt : int : print every N iterations.

        minTol : scalar : convergence tolerance for ALS.

        maxIters : int : maximum ALS iterations.

        F_names : list : K names for the latent factors.

        G_names : list : M names for the pre-specified factors.

        R2_bench : str : benchmark used in the denominator of all R2 calculations.
            'zero'        (default) denominator is sum of squared actuals; equivalent to
                          benchmarking against a forecast of zero.
            'mean'        denominator is sum of squared deviations from the unit-specific
                          (characteristic-specific for X, permno-level for R) historical mean.
                          IS uses the full-sample unit mean; OOS uses the expanding-window
                          unit mean up to (but not including) each period.
            'pooled_mean' denominator is sum of squared deviations from the grand mean pooled
                          across all units.  IS uses the full-sample grand mean; OOS uses the
                          expanding-window grand mean.  A weaker baseline than 'mean': R2 values
                          will be lower than under 'mean' but higher than under 'zero'.

        MacroData : df(T x P) : macroeconomic predictors used when factor_mean='macro' or 'forecombo'.
            These should be timed like the characteristics in RZ -- in normal usage, they should be lagged.
            That is, if the returns are known only by date d, the row d characteristic columns of RZ *AND*
            the row d MacroData should actually have been known at date d-1. Rows must correspond
            positionally to self.Dates.

        target_variance : float or int or None : controls PCA pre-processing of MacroData
            (used when factor_mean='macro' or 'forecombo', with any regularization setting).
            float in (0, 1] — retain the minimum number of principal components that explain
                              at least this fraction of total variance.
            int >= 1          — retain exactly this many principal components.
            None              — no PCA; regression runs on the raw macro predictors.

        regularization : str or None : regression method for macro-to-factor mapping.
            Applies when factor_mean='macro' or 'forecombo'.
            None   -> OLS (default).
            'ridge'-> Ridge regression.
            'lasso'-> LASSO; alpha controls the active-predictor count.
            '3prf' -> Three-Pass Regression Filter (Kelly & Pruitt 2015).
                      Each IPCA factor k is used as a scalar proxy; three passes
                      are run independently for each k.  target_variance may be
                      used for PCA pre-processing before Pass 1.  pass2_intercept
                      controls whether a cross-sectional intercept is included in
                      Pass 2.  alpha is unused for '3prf'.
            None, 'ridge', and 'lasso' can be combined with target_variance for
            PCA pre-processing before the regression.

        alpha : float or int or None :
            ridge  — regularisation penalty strength (positive float). Ignored when
                     ridge_df is not None (ridge_df takes precedence).
            lasso  — target number of active macro predictors (positive int).
                     The full LASSO regularisation path is computed and the
                     least-penalised solution with at most alpha non-zero
                     coefficients is selected independently for each factor.
            3prf   — unused; pass2_intercept controls Pass 2 behaviour.

        ridge_df : float or None : target effective degrees of freedom for ridge regression.
            Applies when regularization='ridge' (with or without PCA pre-processing).
            Effective df is defined as  df(λ) = Σ σ_i² / (σ_i² + λ)  where σ_i are the
            singular values of the (standardised) macro panel.  df ranges from P (λ=0, OLS)
            down to 0 (full shrinkage).  ridge_df=5 (default) targets the equivalent of
            roughly 5 active directions.  When ridge_df is not None it overrides alpha for
            ridge; when ridge_df=None alpha is used directly as the penalty λ.

        pass2_intercept : bool : (default True) applies only when regularization='3prf'.
            True  — include a cross-sectional intercept in Pass 2, as in Kelly &
                    Pruitt (2015).  The intercept absorbs cross-sectional mean
                    differences but is discarded before Pass 3.
            False — no intercept in Pass 2; the filtered factor is a pure
                    projection of the cross-section onto the loading vector b_k.

        min_combo_periods : int or None : minimum number of past OOS observations required before the
            expanding-window OLS combination weights are estimated in OOS forecombo mode.
            None (default) uses max(3, KM+2). Equal-weight average is used as fallback before this threshold.

        min_train_periods : int or None : minimum training-window size required before OOS estimation
            begins. None (default) auto-computes a floor based on factor_mean and MacroData using the
            rule 3 x (number of free parameters in the most demanding regression). Applies to all
            factor_mean options. Raise OOS_window_specs or pass min_train_periods explicitly to
            override the auto-computed floor.

        kappa_max : float : maximum acceptable condition number for the K×K Factor-step lhs matrix
            GammaF.T @ W_t @ GammaF in the ALS. Default 1e8.

        const_tol : float : relative standard-deviation threshold (std / |mean|) used to detect a
            constant row in gFac. A row is considered constant when rel_std < const_tol AND its mean
            is within const_tol of 1.0. At most one such row is permitted; more raises a ValueError.
            Default 1e-6.

        tan_target_vol : float : target portfolio volatility for TanPtf weight scaling. Default 1.0
            (unit-volatility tangency portfolio).

        [Outputs] dict with the following keys

        xfits : dict : 'Fits_Total' df(L x T), 'Fits_Pred' df(L x T), 'R2_Total', 'R2_Pred'.

        Gamma : df(L x KM) [IS] or df(TL x KM) [OOS] : loading matrix. OOS version is indexed by Date
            on the first MultiIndex level so Gamma.loc[t] gives the loading used at period t.

        Factor : df(KM x T) : factor realizations (IS: full-sample; OOS: one-step-ahead).

        Lambda : dict with 'estimate' (constant: df(KM x 1); VAR1/forecombo: df(KM x T)) and
            'VAR1' (df(KM x KM+1) of VAR coefficients, or None when factor_mean='constant').

        LambdaM : df(KM x T) or None : macro-only factor predictions when factor_mean='macro' or
            'forecombo'; None for other factor_mean options. For 'macro', LambdaM equals Lambda.

        rfits : dict analogous to xfits, or None if R_fit=False or no RZ supplied.

        fittedBeta : df(TotalObs x KM) or None.

        TanPtf : pd.Series : tangency portfolio returns over the estimation sample.
            Uses variable factors only: K latent + M_nz non-constant observed (K_tan = K + M_nz).
            IS:  full-sample tangency weights applied to each period's factor realization.
            OOS: training-window weights applied to each OOS factor realization.
            Scaled to tan_target_vol. Returns np.nan for periods with insufficient observations.

        ArbPtf : pd.Series or None : arbitrage portfolio returns.
            Populated only when a constant gFac row (value=1) is detected.
            Computes GammaAlpha' @ solve(W_t, X_t) at each period, equivalent to the
            return of the GammaAlpha-weighted GLS managed portfolio.
            IS:  full-sample GammaAlpha with each period's W_t, X_t.
            OOS: training-window GammaAlpha with OOS-period W_t, X_t.
            None when no constant gFac row is present.

        numerical : dict of convergence statistics.
        """
        fitstart = datetime.now()
        B = None  # VAR1 coefficient matrix; defined conditionally below

        # ------------------------------------------------------------------
        # Factor structure setup
        # ------------------------------------------------------------------
        if gFac is None or (gFac is not None and gFac.shape[0] == 0):
            gFac_arr = None
            M = 0
            M_nz = 0
            has_const = False
            K_tan = K
            const_col = None
        else:
            M = gFac.shape[0]
            # Validate gFac: detect constant row (must equal 1), enforce at most one,
            # and reorder so the constant row is last. See _validate_gfac docstring.
            gFac, gfac_new_order, M_nz, has_const, _const_row_name = \
                self._validate_gfac(gFac, const_tol)
            if G_names is None:
                G_names = [str(idx) for idx in gFac.index]
            else:
                G_names = [G_names[i] for i in gfac_new_order]
            gFac_arr = gFac.reindex(columns=self.Dates).values.astype(np.float64)  # (M, T)
            K_tan     = K + M_nz    # variable factors used for TanPtf
            const_col = K + M_nz   # column index of constant factor in Gamma/Factor (if has_const)

        KM = K + M
        if KM == 0:
            raise ValueError('K+M=0: there must be at least one factor')

        if F_names is None and K > 0:
            F_names = list(range(K))
        if G_names is None and M > 0:
            G_names = list(gFac.index)

        if K == KM:
            Factor_names = list(F_names)
        elif M == KM:
            Factor_names = list(G_names)
        else:
            Factor_names = list(F_names) + list(G_names)

        self.F_names, self.G_names, self.Factor_names = F_names, G_names, Factor_names

        if normalization_choice == 'Identity':
            normalization_choice_specs = self._find_sublist(normalization_choice_specs)
            if np.min(normalization_choice_specs) == -1:
                raise ValueError(
                    'normalization_choice_specs contains a characteristic not found in self.Chars')

        IS_or_OOS = 'OOS' if OOS else 'IS'

        # Validate factor_mean and MacroData (common to IS and OOS)
        if factor_mean not in ('constant', 'VAR1', 'macro', 'forecombo'):
            raise ValueError("factor_mean must be 'constant', 'VAR1', 'macro', or 'forecombo'")
        if factor_mean in ('macro', 'forecombo') and MacroData is None:
            raise ValueError(
                "MacroData must be provided for factor_mean='%s'" % factor_mean)

        # LambdaM is populated only for factor_mean='macro' or 'forecombo'; None otherwise
        LambdaM = None

        # Sentinels — assigned inside IS or OOS branch; declared here so
        # static analysis sees them as always bound before the return dict.
        fittedX = fittedR = fittedBeta = None
        Gamma = Factor = TanPtf = ArbPtf = None
        lasso_selected = None
        numerical_stats: dict = {}

        # ------------------------------------------------------------------
        # In-sample estimation
        # ------------------------------------------------------------------
        if IS_or_OOS == 'IS':
            date_ints_full = np.arange(len(self.Dates))
            Gamma0, Factor0 = self._svd_initial(K=K, M=M, gFac_arr=gFac_arr,
                                                date_ints=date_ints_full)
            tol, iters = float('inf'), 0
            timerstart = timer()
            while iters < maxIters and tol > minTol:
                iters += 1
                Gamma1, Factor1 = self._linear_als_estimation(
                    Gamma0=Gamma0, K=K, M=M, KM=KM,
                    normalization_choice=normalization_choice,
                    normalization_choice_specs=normalization_choice_specs,
                    gFac_arr=gFac_arr, date_ints=date_ints_full,
                    kappa_max=kappa_max)
                tol = max(np.max(np.abs(Gamma1 - Gamma0)), np.max(np.abs(Factor1 - Factor0)))
                if dispIters and iters % dispItersInt == 0:
                    print('iters %i: tol = %0.8f' % (iters, tol))
                Gamma0, Factor0 = Gamma1, Factor1

            Gamma_arr, Factor_arr = Gamma0, Factor0
            numerical_stats = {'tol': tol, 'minTol': minTol, 'iters': iters, 'maxIters': maxIters,
                               'time': timer() - timerstart}
            print('ipca.fit IS finished: %i seconds, %i iterations'
                  % (numerical_stats['time'], numerical_stats['iters']))

            Rdo, Betado, fittedX, fittedR, fittedBeta = self._setup_fits(R_fit, Beta_fit)

            Gamma = pd.DataFrame(data=Gamma_arr, index=self.Chars, columns=Factor_names)
            GV = Gamma_arr  # (L, KM) ndarray alias

            # --- build Lambda ---
            if factor_mean == 'constant':
                fac_mean = np.mean(Factor_arr, axis=1)  # (KM,)
                B = np.hstack((np.zeros((KM, KM)), fac_mean.reshape(-1, 1))).T  # (KM+1, KM)
                Lambda = pd.DataFrame(data=fac_mean, index=Factor_names)  # single-col df
                lamt_const = fac_mean  # (KM,) used in loop below

            elif factor_mean == 'VAR1':
                B = self._VARB(X=Factor_arr)
                Lambda = pd.DataFrame(data=np.nan, index=Factor_names, columns=self.Dates)
                Lambda.iloc[:, 0] = np.mean(Factor_arr, axis=1)

            elif factor_mean == 'macro':
                # B stays None — no VAR component
                Lambda = pd.DataFrame(data=np.nan, index=Factor_names, columns=self.Dates)

            elif factor_mean == 'forecombo':
                B = self._VARB(X=Factor_arr)
                Lambda = pd.DataFrame(data=np.nan, index=Factor_names, columns=self.Dates)
                Lambda.iloc[:, 0] = np.mean(Factor_arr, axis=1)

            Factor = pd.DataFrame(data=Factor_arr, index=Factor_names, columns=self.Dates)
            FV = Factor_arr  # (KM, T) ndarray alias
            T = len(self.Dates)
            # Sentinels — overwritten by their respective factor_mean branches below;
            # declared here so static analysis sees them as always bound.
            lamt_const = np.zeros(KM)       # overwritten when factor_mean == 'constant'
            LambdaV    = np.zeros((KM, T))  # overwritten for VAR1 / macro / forecombo

            # lasso_selected: populated for regularization='lasso' (IS); None otherwise.
            lasso_selected = None

            # For macro IS: single batch call — fit on all T, evaluate at all T
            if factor_mean == 'macro':
                LambdaM_arr, _, _is_sel = self._dispatch_macro_predict(
                    FV,
                    MacroData.iloc[:T],   # training rows
                    MacroData.iloc[:T],   # test rows = same T dates (IS)
                    regularization, target_variance, alpha,
                    pass2_intercept,
                    return_selected=True,
                    ridge_df=ridge_df)  # always returns 3-tuple when return_selected=True
                Lambda  = pd.DataFrame(LambdaM_arr, index=Factor_names, columns=self.Dates)
                LambdaM = Lambda.copy()
                if _is_sel is not None:
                    lasso_selected = _is_sel  # (selected_mask, predictor_names) for IS

            # For forecombo IS: fill VAR1 forecasts (per-t), batch macro forecasts, then combine
            elif factor_mean == 'forecombo':
                # (a) VAR1 forecasts — must be per-t (uses Factor[t-1])
                for ti in range(1, T):
                    Lambda.iloc[:, ti] = B.T.dot(
                        np.hstack((FV[:, ti - 1], 1)).reshape(-1, 1)).ravel()

                # (b) Macro forecasts — single regression fit, all T evaluations
                #     MacroData_train and MacroData_test are the same T rows for IS
                LambdaM_arr, _, _is_sel = self._dispatch_macro_predict(
                    FV,
                    MacroData.iloc[:T],   # training rows
                    MacroData.iloc[:T],   # test rows = same (IS: predict in-sample)
                    regularization, target_variance, alpha,
                    pass2_intercept,
                    return_selected=True,
                    ridge_df=ridge_df)
                LambdaM = pd.DataFrame(LambdaM_arr, index=Factor_names, columns=self.Dates)
                if _is_sel is not None:
                    lasso_selected = _is_sel  # (selected_mask, predictor_names) for IS

                # (c) Combine VAR1 and macro (overwrites Lambda cols 1..T-1)
                #     _calculate_combined_predictions returns (KM, T-1)
                Lambda.iloc[:, 1:] = self._calculate_combined_predictions(
                    FV,
                    Lambda.iloc[:, 1:].values.T,    # (T-1, KM) VAR1 forecasts
                    LambdaM.iloc[:, 1:].values.T)   # (T-1, KM) macro forecasts

            # --- vectorised X fits (avoids per-t W.loc lookups) ---
            # Fits_Total[i,t] = W[i,:,t] @ GV @ F[:,t]
            fits_total_arr = np.einsum('ijt,jk,kt->it', self._W, GV, FV)   # (L, T)
            fittedX['Fits_Total'] = pd.DataFrame(fits_total_arr,
                                                 index=self.Chars, columns=self.Dates)

            if factor_mean == 'constant':
                # same lamt for all t
                fits_pred_arr = np.einsum('ijt,jk,k->it', self._W, GV, lamt_const)  # (L, T)
            elif factor_mean in ('VAR1', 'forecombo', 'macro'):
                # fill LambdaV ndarray for vectorised einsum
                if factor_mean == 'VAR1':
                    LambdaV = np.zeros((KM, T))
                    LambdaV[:, 0] = Lambda.iloc[:, 0].values
                    for ti in range(1, T):
                        lamt = B.T.dot(
                            np.hstack((FV[:, ti - 1], 1)).reshape(-1, 1)).ravel()
                        LambdaV[:, ti] = lamt
                        Lambda.iloc[:, ti] = lamt
                else:
                    LambdaV = Lambda.values  # already fully populated (forecombo or macro)
                fits_pred_arr = np.einsum('ijt,jk,kt->it', self._W, GV, LambdaV)  # (L, T)

            fittedX['Fits_Pred'] = pd.DataFrame(fits_pred_arr,
                                                index=self.Chars, columns=self.Dates)

            # --- R fits (parallelised across dates; iterations are independent) ---
            if Rdo or Betado:
                # Resolve to a single (KM, T) array so the nested function closes
                # over one always-bound name (both sentinels initialised above).
                _lam_arr = (np.tile(lamt_const[:, None], T)
                            if factor_mean == 'constant' else LambdaV)  # (KM, T)

                def _r_fits_for_date(ti, t):
                    Betat  = self._compute_beta(t, Gamma_arr)          # (N_t, KM)
                    ft     = Betat.dot(FV[:, ti]).reshape(-1, 1) if Rdo else None
                    fp     = Betat.dot(_lam_arr[:, ti]).reshape(-1, 1) if Rdo else None
                    return Betat, ft, fp

                per_t = Parallel(n_jobs=-1, prefer='threads')(
                    delayed(_r_fits_for_date)(ti, t) for ti, t in enumerate(self.Dates))

                for (ti, t), (Betat, ft, fp) in zip(enumerate(self.Dates), per_t):
                    if Rdo:
                        fittedR['Fits_Total'].loc[t] = ft
                        fittedR['Fits_Pred'].loc[t]  = fp
                        if Betado:
                            fittedBeta.loc[t] = Betat
                    elif Betado:
                        fittedBeta.loc[t] = Betat

            # --- R2s ---
            benchR2_X = self._make_bench_is(R2_bench, for_R=False)
            fittedX['R2_Total'], fittedX['R2_Pred'] = self._R2_calc(
                self.X, fittedX['Fits_Total'], fittedX['Fits_Pred'], benchR2_X)
            if Rdo:
                benchR2_R = self._make_bench_is(R2_bench, for_R=True)
                fittedR['R2_Total'], fittedR['R2_Pred'] = self._R2_calc(
                    self.R, fittedR['Fits_Total'], fittedR['Fits_Pred'], benchR2_R)

            # --- TanPtf IS ---
            # S_tan is always the full-sample covariance (IS estimator).
            # For factor_mean='constant' the weights are static (mu = full-sample mean).
            # For VAR1/macro/forecombo the weights are time-varying: mu_t = LambdaV[:,t],
            # computed via a single matrix multiply (no per-t loop).
            if K_tan > 0:
                F_tan = Factor_arr[:K_tan, :]           # (K_tan, T)
                iota  = np.ones(K_tan)
                try:
                    if factor_mean == 'constant':
                        # Static weights — S from raw factors (constant mean, so
                        # cov(F) == cov(F - mean) up to ddof; no time-varying correction needed)
                        S_tan   = np.atleast_2d(np.cov(F_tan))   # (K_tan, K_tan)
                        mu_tan  = F_tan.mean(axis=1)    # (K_tan,)
                        Sinv_mu = np.linalg.solve(S_tan, mu_tan)
                        denom   = iota @ Sinv_mu
                        if abs(denom) > 1e-12:
                            tw = Sinv_mu / denom
                            if tw @ mu_tan < 0:
                                tw = -tw
                            pv = np.sqrt(tw @ S_tan @ tw)
                            if pv > 1e-12:
                                tw *= tan_target_vol / pv
                            TanPtf = pd.Series(F_tan.T @ tw, index=self.Dates)
                        else:
                            TanPtf = pd.Series(np.nan, index=self.Dates)
                    else:
                        # Time-varying weights: mu_t = LambdaV[:K_tan, t]
                        # S is estimated from prediction residuals, not raw factors:
                        # conditional covariance = Cov(F - LambdaV), not Cov(F).
                        resid_tan    = F_tan - LambdaV[:K_tan, :]         # (K_tan, T)
                        S_tan        = np.atleast_2d(np.cov(resid_tan))   # (K_tan, K_tan)
                        # Vectorised: precompute S^{-1}, then broadcast across T.
                        Sinv         = np.linalg.inv(S_tan)               # (K_tan, K_tan)
                        Mu_all       = LambdaV[:K_tan, :]                 # (K_tan, T)
                        Sinv_mu_all  = Sinv @ Mu_all                      # (K_tan, T)
                        denom_all    = iota @ Sinv_mu_all                 # (T,)
                        valid        = np.abs(denom_all) > 1e-12          # (T,)
                        tw_all       = np.where(
                            valid, Sinv_mu_all / np.where(valid, denom_all, 1.0), np.nan)
                        # sign: flip columns where tw' mu < 0
                        dots = (tw_all * Mu_all).sum(axis=0)              # (T,)
                        tw_all = np.where(dots[np.newaxis, :] < 0, -tw_all, tw_all)
                        # scale to target_vol
                        port_var = (tw_all * (S_tan @ tw_all)).sum(axis=0)
                        port_vol = np.sqrt(np.maximum(port_var, 0.0))
                        scale    = np.where(port_vol > 1e-12,
                                            tan_target_vol / port_vol, 1.0)
                        tw_all  *= scale[np.newaxis, :]
                        tan_rets = (F_tan * tw_all).sum(axis=0)           # (T,)
                        tan_rets = np.where(valid, tan_rets, np.nan)
                        TanPtf   = pd.Series(tan_rets, index=self.Dates)
                except np.linalg.LinAlgError:
                    TanPtf = pd.Series(np.nan, index=self.Dates)
            else:
                TanPtf = pd.Series(np.nan, index=self.Dates)

            # --- ArbPtf IS ---
            # Full-sample GammaAlpha applied to each period's W_t, X_t.
            if has_const:
                GammaAlpha_IS = Gamma_arr[:, const_col]     # (L,)
                ArbPtf = pd.Series(
                    [float(GammaAlpha_IS @ np.linalg.solve(
                        self._W[:, :, ti], self._X[:, ti]))
                     for ti in range(T)],
                    index=self.Dates)
            else:
                ArbPtf = None

        # ------------------------------------------------------------------
        # Out-of-sample estimation
        # ------------------------------------------------------------------
        elif IS_or_OOS == 'OOS':

            # --- Resolve timestamp OOS_window_specs to an int ---
            _OWS_DEFAULT = 60
            if isinstance(OOS_window_specs, (str, pd.Timestamp, datetime)):
                try:
                    ts = pd.Timestamp(OOS_window_specs)
                except Exception:
                    warnings.warn(
                        "OOS_window_specs %r could not be parsed as a timestamp; "
                        "falling back to default (%i)." % (OOS_window_specs, _OWS_DEFAULT),
                        UserWarning, stacklevel=2)
                    OOS_window_specs = _OWS_DEFAULT
                else:
                    idx = self.Dates.searchsorted(ts)           # first index >= ts
                    if idx >= len(self.Dates):
                        warnings.warn(
                            "OOS_window_specs timestamp %s is after all available dates; "
                            "falling back to default (%i)." % (ts.date(), _OWS_DEFAULT),
                            UserWarning, stacklevel=2)
                        OOS_window_specs = _OWS_DEFAULT
                    elif self.Dates[idx] == ts:
                        OOS_window_specs = int(idx)             # exact match — silent
                    else:
                        snapped = self.Dates[idx]
                        warnings.warn(
                            "OOS_window_specs timestamp %s not found in Dates; snapped to "
                            "nearest date on or after: %s (index %i)."
                            % (ts.date(), snapped.date(), idx),
                            UserWarning, stacklevel=2)
                        OOS_window_specs = int(idx)

            T_total = len(self.Dates)

            # Enforce minimum training window (auto-computed or user-supplied)
            _min_train = (int(min_train_periods) if min_train_periods is not None
                          else self._compute_min_train_periods(
                              KM, factor_mean, MacroData, target_variance))

            if OOS_window_specs is None:
                # Expanding window: start from the minimum required training window
                OOS_window_specs = _min_train
            else:
                if OOS_window_specs >= T_total:
                    raise ValueError(
                        'OOS_window_specs (%i) must be less than the total number of dates (%i)'
                        % (OOS_window_specs, T_total))
                if OOS_window_specs < _min_train:
                    raise ValueError(
                        "OOS_window_specs (%i) is below the minimum training window (%i) "
                        "required for factor_mean='%s'. Increase OOS_window_specs or pass "
                        "min_train_periods explicitly to override."
                        % (OOS_window_specs, _min_train, factor_mean))

            date_ints_init = np.arange(OOS_window_specs)
            Gamma0, Factor0 = self._svd_initial(K=K, M=M, gFac_arr=gFac_arr,
                                                date_ints=date_ints_init)
            Rdo, Betado, fittedX, fittedR, fittedBeta = self._setup_fits(R_fit, Beta_fit)

            Lambda = pd.DataFrame(data=np.nan, index=Factor_names, columns=self.Dates)
            Factor = Lambda.copy()
            Gamma = pd.DataFrame(
                data=np.nan, columns=Factor_names,
                index=pd.MultiIndex.from_product([self.Dates, self.Chars]))

            # Benchmark storage
            benchX = None if R2_bench == 'zero' else pd.DataFrame(
                data=0., columns=self.Dates, index=self.Chars)
            benchR = None
            if Rdo and R2_bench in ('mean', 'pooled_mean'):
                benchR = pd.DataFrame(data=0., index=self.R.index, columns=self.R.columns)

            # Pre-compute running sums for R2 mean benchmarks (avoids O(T²) recomputation)
            if R2_bench == 'mean':
                # Characteristic-specific: cumulative sum per row (axis=1) of _X
                X_cumsum = np.cumsum(self._X, axis=1)        # (L, T) per-char running sum
                X_cumcnt = np.arange(1, T_total + 1)         # (T,)   count of periods
            elif R2_bench == 'pooled_mean':
                # Grand (pooled) mean: single scalar per period
                X_cumsum = np.cumsum(self._X.sum(axis=0))    # (T,)   total sum across chars
                X_cumcnt = np.arange(1, T_total + 1) * self.L  # (T,) total element count
            if R2_bench in ('mean', 'pooled_mean') and Rdo:
                R_dates_arr = self.R.index.get_level_values(0)
                R_vals_arr = self.R.values.ravel()
                R_sum_by_date = np.array([
                    R_vals_arr[R_dates_arr == t].sum() for t in self.Dates])
                R_cnt_by_date = np.array([
                    (R_dates_arr == t).sum() for t in self.Dates])
                R_cumsum_arr = np.cumsum(R_sum_by_date)
                R_cumcnt_arr = np.cumsum(R_cnt_by_date)

            numerical_stats = {
                'minTol': minTol, 'maxIters': maxIters,
                'tol':   pd.DataFrame(data=np.nan, columns=self.Dates, index=[0]),
                'iters': pd.DataFrame(data=np.nan, columns=self.Dates, index=[0]),
                'time':  pd.DataFrame(data=np.nan, columns=self.Dates, index=[0])}

            TanPtf = pd.Series(np.nan, index=self.Dates)
            ArbPtf = pd.Series(np.nan, index=self.Dates) if has_const else None

            # Initialise output storage for macro and forecombo OOS modes
            lasso_sel_list = []   # list of (t, selected_info) tuples for lasso selection tracking
            if factor_mean == 'macro':
                LambdaM = pd.DataFrame(data=np.nan, index=Factor_names, columns=self.Dates)
            elif factor_mean == 'forecombo':
                oos_var_hist = []   # list of (KM,) VAR1 forecasts made for past OOS periods
                oos_mac_hist = []   # list of (KM,) macro forecasts made for past OOS periods
                oos_fac_hist = []   # list of (KM,) realized factors for past OOS periods
                LambdaM = pd.DataFrame(data=np.nan, index=Factor_names, columns=self.Dates)
                _min_combo = (max(3, KM + 2) if min_combo_periods is None
                              else int(min_combo_periods))

            ct = 0
            OOS_dates = self.Dates[OOS_window_specs:]
            for t in OOS_dates:
                t_idx = self.Dates.get_loc(t)  # integer position of t in self.Dates
                tol, iters = float('inf'), 0

                # Date integer indices for the training window (excludes t itself)
                if OOS_window == 'rolling':
                    date_ints = np.arange(t_idx - OOS_window_specs, t_idx)
                else:  # recursive (expanding)
                    date_ints = np.arange(t_idx)

                # Reset Factor0 to the current window shape before each ALS run.
                # Factor0 is used only for the convergence check; Gamma0 is the
                # actual warm start.  Without this reset, the recursive window
                # grows Factor0 by one column each period, causing a shape
                # mismatch on the first ALS iteration of every period after the first.
                Factor0 = np.full((KM, len(date_ints)), np.inf)

                timerstart = timer()
                while iters < maxIters and tol > minTol:
                    iters += 1
                    Gamma1, Factor1 = self._linear_als_estimation(
                        Gamma0=Gamma0, K=K, M=M, KM=KM,
                        normalization_choice=normalization_choice,
                        normalization_choice_specs=normalization_choice_specs,
                        gFac_arr=gFac_arr, date_ints=date_ints,
                        kappa_max=kappa_max)
                    tol = max(np.max(np.abs(Gamma1 - Gamma0)), np.max(np.abs(Factor1 - Factor0)))
                    if dispIters and iters % dispItersInt == 0:
                        print('iters {}: tol = {:.8f}'.format(iters, tol))
                    Gamma0, Factor0 = Gamma1, Factor1

                numerical_stats['tol'][t] = tol
                numerical_stats['iters'][t] = iters
                numerical_stats['time'][t] = timer() - timerstart
                Gamma.loc[t] = Gamma0

                # OOS factor realization at t:
                # Use Gamma estimated from training data (through t-1) with W, X at t
                Wt = self._W[:, :, t_idx]
                Xt = self._X[:, t_idx]
                if M == 0:
                    F_t = self._conditioned_solve(
                        Gamma0.T @ Wt @ Gamma0,
                        Gamma0.T @ Xt,
                        kappa_max=kappa_max)
                else:
                    GammaF0 = Gamma0[:, :K]
                    GammaG0 = Gamma0[:, K:]
                    F_t_lat = self._conditioned_solve(
                        GammaF0.T @ Wt @ GammaF0,
                        GammaF0.T @ (Xt - Wt @ GammaG0 @ gFac_arr[:, t_idx]),
                        kappa_max=kappa_max)
                    F_t = np.concatenate([F_t_lat, gFac_arr[:, t_idx]])

                Factor[t] = F_t
                fittedX['Fits_Total'][t] = (Wt @ Gamma0 @ F_t).reshape(-1, 1)

                # ArbPtf: GammaAlpha' @ W_t^{-1} @ X_t (only when constant gFac detected)
                if has_const:
                    ArbPtf[t] = float(
                        Gamma0[:, const_col] @ np.linalg.solve(Wt, Xt))

                # Factor mean forecast (Lambda at t) — known before t is observed
                # mu_train_tan: training-window fitted values for K_tan factors,
                # used to form the residual covariance for TanPtf (None for constant).
                mu_train_tan = None

                if factor_mean == 'constant':
                    lamt = Factor0.mean(axis=1)  # (KM,) training-sample mean
                    Lambda[t] = lamt
                    B = np.hstack((np.zeros((KM, KM)), lamt.reshape(-1, 1))).T
                    # mu_train_tan stays None: raw cov(F) is correct for constant mean

                elif factor_mean == 'VAR1':
                    B = self._VARB(X=Factor0)
                    lamt = B.T.dot(np.hstack((Factor0[:, -1], 1)).reshape(-1, 1)).ravel()
                    Lambda[t] = lamt
                    # VAR1 training fitted: B.T @ [F_{t-1}; 1] for t=1..T_tr-1,
                    # unconditional mean for t=0. Pure matrix multiply — no re-fitting.
                    if K_tan > 0:
                        T_tr = Factor0.shape[1]
                        Ftil = np.vstack([Factor0, np.ones((1, T_tr))])  # (KM+1, T_tr)
                        _mu_tr = np.empty_like(Factor0)
                        _mu_tr[:, 0]  = Factor0.mean(axis=1)
                        _mu_tr[:, 1:] = B.T @ Ftil[:, :-1]              # (KM, T_tr-1)
                        mu_train_tan  = _mu_tr[:K_tan, :]

                elif factor_mean == 'forecombo':
                    # (a) VAR1 forecast using training-window factors
                    B = self._VARB(X=Factor0)
                    lamt_var = B.T.dot(np.hstack((Factor0[:, -1], 1)).reshape(-1, 1)).ravel()

                    # (b) Macro forecast — training window aligned with Factor0
                    if OOS_window == 'rolling':
                        macro_train = MacroData.iloc[t_idx - OOS_window_specs : t_idx]
                    else:  # recursive
                        macro_train = MacroData.iloc[:t_idx]
                    lamt_mac, mac_train_fitted, _sel = self._dispatch_macro_predict(
                        Factor0, macro_train, MacroData.loc[t],
                        regularization, target_variance, alpha,
                        pass2_intercept,
                        return_train_fitted=True,
                        return_selected=True,
                        ridge_df=ridge_df)  # (KM,), (KM, T_train), selected_info|None
                    LambdaM[t] = lamt_mac
                    if _sel is not None:
                        lasso_sel_list.append((t, _sel))

                    # (c) OLS combination — expanding window over past OOS history
                    n_hist = len(oos_fac_hist)
                    if n_hist >= _min_combo:
                        lamt = self._combine_forecasts_oos(
                            np.array(oos_fac_hist).T,   # (KM, n_hist)
                            np.array(oos_var_hist).T,   # (KM, n_hist)
                            np.array(oos_mac_hist).T,   # (KM, n_hist)
                            lamt_var, lamt_mac)
                    else:
                        # equal-weight fallback during burn-in
                        lamt = 0.5 * lamt_var + 0.5 * lamt_mac

                    Lambda[t] = lamt

                    # Update histories after forecast is made and F_t is observed
                    oos_var_hist.append(lamt_var.copy())
                    oos_mac_hist.append(lamt_mac.copy())
                    oos_fac_hist.append(F_t.copy())

                    # Combined training fitted values for TanPtf residual covariance:
                    # VAR1 training fitted (matrix multiply, no re-fitting)
                    if K_tan > 0:
                        T_tr = Factor0.shape[1]
                        Ftil = np.vstack([Factor0, np.ones((1, T_tr))])  # (KM+1, T_tr)
                        var_train = np.empty_like(Factor0)
                        var_train[:, 0]  = Factor0.mean(axis=1)
                        var_train[:, 1:] = B.T @ Ftil[:, :-1]           # (KM, T_tr-1)
                        # OLS combination over training window
                        _combo_train = np.zeros_like(Factor0)
                        for _k in range(KM):
                            _Xc = np.column_stack([
                                np.ones(T_tr), var_train[_k, :], mac_train_fitted[_k, :]])
                            _wc = np.linalg.lstsq(_Xc, Factor0[_k, :], rcond=None)[0]
                            _combo_train[_k, :] = _Xc @ _wc
                        mu_train_tan = _combo_train[:K_tan, :]

                elif factor_mean == 'macro':
                    # Macro-only: no VAR(1), no combination step
                    if OOS_window == 'rolling':
                        macro_train = MacroData.iloc[t_idx - OOS_window_specs : t_idx]
                    else:  # recursive
                        macro_train = MacroData.iloc[:t_idx]
                    lamt, mac_train_fitted, _sel = self._dispatch_macro_predict(
                        Factor0, macro_train, MacroData.loc[t],
                        regularization, target_variance, alpha,
                        pass2_intercept,
                        return_train_fitted=True,
                        return_selected=True,
                        ridge_df=ridge_df)  # (KM,), (KM, T_train), selected_info|None
                    if _sel is not None:
                        lasso_sel_list.append((t, _sel))
                    Lambda[t]  = lamt
                    LambdaM[t] = lamt
                    # B stays None — no VAR component
                    if K_tan > 0:
                        mu_train_tan = mac_train_fitted[:K_tan, :]

                fittedX['Fits_Pred'][t] = (Wt @ Gamma0 @ lamt).reshape(-1, 1)

                # TanPtf: tangency portfolio using lamt as the expected-return vector
                # so that time-varying factor predictions (VAR1, macro, forecombo) drive
                # the weights rather than the static training-window mean.
                # S is estimated from prediction residuals (mu_train_tan) so that the
                # conditional covariance Cov(F - mu) is used, not Cov(F).
                if K_tan > 0:
                    TanPtf[t] = self._tangency_ptf(
                        Factor0[:K_tan, :], F_t[:K_tan], tan_target_vol,
                        mu=lamt[:K_tan], mu_train=mu_train_tan)

                if R2_bench == 'mean':
                    # per-characteristic mean of _X up to (not including) t
                    benchX[t] = X_cumsum[:, t_idx - 1] / X_cumcnt[t_idx - 1]
                elif R2_bench == 'pooled_mean':
                    # grand mean of _X up to (not including) t
                    benchX[t] = X_cumsum[t_idx - 1] / X_cumcnt[t_idx - 1]

                if Rdo or Betado:
                    Betat = self._compute_beta(t, Gamma0)  # (N_t, KM)
                    if Rdo:
                        fittedR['Fits_Total'].loc[t] = Betat.dot(F_t).reshape(-1, 1)
                        fittedR['Fits_Pred'].loc[t] = Betat.dot(lamt).reshape(-1, 1)
                        if R2_bench in ('mean', 'pooled_mean'):
                            # mean of R up to (not including) t
                            benchR.loc[t] = R_cumsum_arr[t_idx - 1] / R_cumcnt_arr[t_idx - 1]
                        if Betado:
                            fittedBeta.loc[t] = Betat
                    elif Betado:
                        fittedBeta.loc[t] = Betat

                # Extend Factor0 for the next iteration's tolerance check.
                # Rolling: slide window (drop oldest column, append placeholder zero).
                # Recursive: append placeholder zero column.
                if OOS_window == 'rolling':
                    Factor0 = np.concatenate(
                        (Factor0[:, 1:], np.zeros((Factor0.shape[0], 1))), axis=1)
                else:
                    Factor0 = np.concatenate(
                        (Factor0, np.zeros((Factor0.shape[0], 1))), axis=1)

                ct += 1
                if dispIters and ct % 12 == 0:
                    print('%s done: %i iters, %.2f sec'
                          % (t, numerical_stats['iters'][t].values[0],
                             numerical_stats['time'][t].values[0]))

            # Build lasso_selected DataFrame from per-period accumulation.
            # Shape: (KM, T_oos) with columns=OOS_dates, index=Factor_names.
            # Each column is the selected_mask (P, KM) bool array for that period,
            # stored as a MultiIndex DataFrame: index=(Predictor, Factor), columns=Dates.
            if lasso_sel_list:
                sel_dates  = [item[0] for item in lasso_sel_list]
                sel_masks  = [item[1][0] for item in lasso_sel_list]   # list of (P, KM) bool
                pnames     = lasso_sel_list[0][1][1]                   # predictor names (P,)
                # Stack into (n_oos, P, KM) then reshape to MultiIndex DataFrame
                mask_arr = np.stack(sel_masks, axis=0)                 # (n_oos, P, KM)
                midx = pd.MultiIndex.from_product([pnames, Factor_names],
                                                  names=['Predictor', 'Factor'])
                # Reshape to (P*KM, n_oos) for DataFrame construction
                lasso_selected = pd.DataFrame(
                    mask_arr.reshape(len(sel_dates), -1).T,            # (P*KM, n_oos)
                    index=midx,
                    columns=sel_dates)
            else:
                lasso_selected = None

            # OOS R2s (computed only over OOS periods)
            oos_slice = self.Dates[OOS_window_specs:]
            fittedX['R2_Total'], fittedX['R2_Pred'] = self._R2_calc(
                self.X.loc[:, oos_slice],
                fittedX['Fits_Total'].loc[:, oos_slice],
                fittedX['Fits_Pred'].loc[:, oos_slice],
                benchX.loc[:, oos_slice] if benchX is not None else None)
            if Rdo:
                fittedR['R2_Total'], fittedR['R2_Pred'] = self._R2_calc(
                    self.R.loc[oos_slice],
                    fittedR['Fits_Total'].loc[oos_slice],
                    fittedR['Fits_Pred'].loc[oos_slice],
                    benchR.loc[oos_slice] if benchR is not None else None)

        # ------------------------------------------------------------------
        # Assemble output
        # ------------------------------------------------------------------
        Lambda_dict = {
            'estimate': Lambda,
            'VAR1': (pd.DataFrame(data=B.T,
                                  index=Factor_names,
                                  columns=list(Factor_names) + ['cons'])
                     if B is not None else None)}

        fitend = datetime.now()
        numerical_stats['fit_start_time'] = fitstart
        numerical_stats['fit_end_time'] = fitend
        return {'xfits':          fittedX,
                'Gamma':          Gamma,
                'Factor':         Factor,
                'Lambda':         Lambda_dict,
                'LambdaM':        LambdaM,
                'TanPtf':         TanPtf,
                'ArbPtf':         ArbPtf,
                'rfits':          fittedR,
                'fittedBeta':     fittedBeta,
                'lasso_selected': lasso_selected,
                'numerical':      numerical_stats}

    # ------------------------------------------------------------------
    # Linear-system helpers
    # ------------------------------------------------------------------

    def _conditioned_solve(self, A, b, kappa_max=None):
        """
        Solve A x = b after optionally applying adaptive ridge to A.

        When kappa_max is given, adds the smallest ridge λ ≥ 0 such that
        cond(A + λI) ≤ kappa_max, matching the conditioning used inside
        _num_als_estimation.  When kappa_max is None the system is solved
        as-is via sla.lstsq (robust to rank-deficiency but no explicit
        conditioning).

        Parameters
        ----------
        A         : (n, n) array
        b         : (n,) or (n, m) array
        kappa_max : float or None

        Returns
        -------
        x : same shape as b
        """
        A = np.array(A)
        b = np.array(b)
        if kappa_max is not None:
            eigs  = np.linalg.eigvalsh(A)
            l_rid = max(0.0, (eigs[-1] - kappa_max * eigs[0]) / (kappa_max - 1))
            if l_rid > 0.0:
                A = A + l_rid * np.eye(A.shape[0])
            return np.linalg.solve(A, b)
        return sla.lstsq(A, b, check_finite=False)[0]

    # ------------------------------------------------------------------
    # Portfolio helpers
    # ------------------------------------------------------------------

    def _tangency_ptf(self, F_train, F_next, target_vol, mu=None, mu_train=None):
        """
        Tangency portfolio return at the next period.

        Matches tanptfnext.m: excess returns with rf=0, weights normalised to
        sum to 1, sign convention ensures positive expected return, then scaled
        to target_vol.

        F_train    : (K_tan, T_train) ndarray — training-sample factor returns
        F_next     : (K_tan,)         ndarray — factor realization at the new period
        target_vol : float            — scale portfolio weights to this volatility
        mu         : (K_tan,) ndarray or None — expected factor return used to form
                     tangency weights. When None (default, factor_mean='constant'),
                     falls back to the training-sample mean F_train.mean(axis=1).
                     Pass the factor-mean forecast (lamt[:K_tan]) for VAR1/macro so
                     that time-varying predictions drive the portfolio weights.
        mu_train   : (K_tan, T_train) ndarray or None — training-window conditional
                     mean forecasts (fitted values from the factor-mean model). When
                     provided, S is estimated from the prediction residuals
                     F_train - mu_train rather than from raw F_train. Pass for
                     VAR1/macro/forecombo; leave None for factor_mean='constant'.

        Returns a scalar float, or np.nan when K_tan == 0, fewer than K_tan+1
        training observations are available, or the covariance matrix is singular.
        """
        K_tan, T_train = F_train.shape
        if K_tan == 0 or T_train <= K_tan:
            return np.nan
        F_for_S = F_train if mu_train is None else F_train - mu_train
        S    = np.atleast_2d(np.cov(F_for_S))  # (K_tan, K_tan), ddof=1 matches MATLAB cov()
        mu   = F_train.mean(axis=1) if mu is None else mu   # (K_tan,)
        iota = np.ones(K_tan)
        try:
            Sinv_mu = np.linalg.solve(S, mu)
        except np.linalg.LinAlgError:
            return np.nan
        denom = iota @ Sinv_mu
        if abs(denom) < 1e-12:
            return np.nan
        tw = Sinv_mu / denom                    # weights sum to 1
        if tw @ mu < 0:
            tw = -tw                            # positive expected-return convention
        port_vol = np.sqrt(tw @ S @ tw)
        if port_vol > 1e-12:
            tw *= target_vol / port_vol
        return float(F_next @ tw)

    # ------------------------------------------------------------------
    # gFac validation
    # ------------------------------------------------------------------

    def _validate_gfac(self, gFac, const_tol):
        """
        Validate and partition gFac rows into non-constant and constant groups.

        Rules
        -----
        - A row is near-constant if std / |mean| < const_tol (relative std).
        - A near-constant row must have mean within const_tol of 1.0; otherwise
          a ValueError is raised (constant gFac rows must equal 1).
        - At most one near-constant row is permitted; more raises a ValueError.
        - If a near-constant row is found, it is moved to the last position in
          the returned DataFrame and a message is printed (once, before estimation).

        Parameters
        ----------
        gFac      : df(M x T)
        const_tol : float — relative std threshold

        Returns
        -------
        gFac_out     : df(M x T)  — possibly reordered so constant row is last
        new_order    : list(M)    — original row indices in the new order (use to
                                    reorder a user-supplied G_names list)
        M_nz         : int        — number of non-constant rows
        has_const    : bool
        const_row_name : str or None
        """
        M = gFac.shape[0]
        const_indices = []

        for m in range(M):
            row      = gFac.iloc[m]
            row_mean = float(row.mean())
            if abs(row_mean) < 1e-12:
                continue                     # mean≈0 cannot be constant-1
            rel_std = float(row.std()) / abs(row_mean)
            if rel_std < const_tol:
                if abs(row_mean - 1.0) > const_tol:
                    raise ValueError(
                        "gFac row '{}' appears constant (rel_std={:.2e}) but its "
                        "mean ({:.6f}) is not 1. A constant gFac row must equal 1."
                        .format(gFac.index[m], rel_std, row_mean))
                const_indices.append(m)

        if len(const_indices) > 1:
            raise ValueError(
                "At most one gFac row may be constant; found {:d}: {}"
                .format(len(const_indices),
                        [str(gFac.index[i]) for i in const_indices]))

        if len(const_indices) == 1:
            cidx           = const_indices[0]
            const_row_name = str(gFac.index[cidx])
            other          = [i for i in range(M) if i != cidx]
            new_order      = other + [cidx]
            gFac_out       = gFac.iloc[new_order].copy()
            print("ipca: gFac row '{}' detected as constant (value=1) and placed "
                  "last in Factor/Gamma output. GammaAlpha corresponds to this "
                  "factor; ArbPtf will be computed.".format(const_row_name))
            return gFac_out, new_order, M - 1, True, const_row_name
        else:
            new_order = list(range(M))
            return gFac, new_order, M, False, None

    # ------------------------------------------------------------------
    # Core ALS routines
    # ------------------------------------------------------------------

    def _linear_als_estimation(self, Gamma0, K, M, KM, normalization_choice,
                                normalization_choice_specs, gFac_arr, date_ints,
                                kappa_max=1e8):
        """
        One ALS iteration: given Gamma0, estimate Factor then Gamma then normalise.

        Parameters
        ----------
        Gamma0    : (L, KM) ndarray
        gFac_arr  : (M, T) ndarray or None
        date_ints : 1-D integer array of time indices into self._W, self._X, self._Nts_arr
        kappa_max : float — maximum acceptable condition number for the K×K Factor-step
                    lhs matrix GammaF.T @ W_t @ GammaF.  For each period where the
                    condition number exceeds kappa_max (e.g. because W_t is rank-deficient
                    and GammaF intersects its null space), the minimum ridge penalty l is
                    added: l = max(0, (σ_max − kappa_max·σ_min) / (kappa_max − 1)).
                    Default 1e8.  Must be > 1.

        Returns
        -------
        Gamma1  : (L, KM) ndarray
        Factor1 : (KM, len(date_ints)) ndarray
        """
        W_sub  = self._W[:, :, date_ints]      # (L, L, T_sub)
        X_sub  = self._X[:, date_ints]         # (L, T_sub)
        N_sub  = self._Nts_arr[date_ints]      # (T_sub,)
        T_sub  = len(date_ints)

        if K == KM:
            GammaF = Gamma0
            GammaG = None
        elif M == KM:
            GammaF = None
            GammaG = Gamma0
        else:
            GammaF = Gamma0[:, :K]
            GammaG = Gamma0[:, K:]

        # ------ Step 1: estimate latent factors (vectorised + adaptive ridge) ------
        if K > 0:
            # RHS: GammaF.T @ (X[:,t] - W[:,:,t] @ GammaG @ gFac[:,t])  for all t
            if M > 0:
                gFac_sub = gFac_arr[:, date_ints]              # (M, T_sub)
                GG_fac   = GammaG @ gFac_sub                   # (L, T_sub)
                WGG      = np.einsum('ijt,jt->it',
                                     W_sub, GG_fac)            # (L, T_sub)
                rhs_all  = GammaF.T @ (X_sub - WGG)           # (K, T_sub)
            else:
                rhs_all  = GammaF.T @ X_sub                    # (K, T_sub)

            # LHS: GammaF.T @ W[:,:,t] @ GammaF  for all t
            WG      = np.einsum('ijt,jk->ikt', W_sub, GammaF)  # (L, K, T_sub)
            lhs_all = np.einsum('li,ljt->ijt', GammaF, WG)     # (K, K, T_sub)

            # Reshape to leading batch dimension for LAPACK broadcasting
            lhs_T = lhs_all.transpose(2, 0, 1)                 # (T_sub, K, K)
            rhs_T = rhs_all.T[:, :, None]                      # (T_sub, K, 1)

            # Adaptive ridge: minimum l per period s.t. cond(lhs + l*I) <= kappa_max
            # l = max(0,  (sigma_max - kappa_max * sigma_min) / (kappa_max - 1))
            eigs      = np.linalg.eigvalsh(lhs_T)              # (T_sub, K), ascending
            sigma_min = eigs[:, 0]                              # (T_sub,)
            sigma_max = eigs[:, -1]                             # (T_sub,)
            l_ridge   = np.maximum(
                0.0,
                (sigma_max - kappa_max * sigma_min) / (kappa_max - 1)
            )                                                   # (T_sub,)
            lhs_T = lhs_T + l_ridge[:, None, None] * np.eye(K) # (T_sub, K, K)

            # Batched solve — well-conditioned by construction
            FactorF = np.linalg.solve(lhs_T, rhs_T)[..., 0].T  # (K, T_sub)
        else:
            FactorF = None

        # Assemble full Factor array for the Gamma step
        if K == KM:
            Factor = FactorF
        elif M == KM:
            Factor = gFac_arr[:, date_ints]
        else:
            Factor = np.concatenate((FactorF, gFac_arr[:, date_ints]), axis=0)

        # ------ Step 2: estimate Gamma (vectorised) ------
        # numer[i, k] = sum_t N_t * X[i,t] * F[k,t]
        numer = np.einsum('it,kt,t->ik', X_sub, Factor, N_sub).ravel()           # (L*KM,)
        # denom[i,k,j,l] = sum_t N_t * W[i,j,t] * F[k,t] * F[l,t]
        denom = np.einsum('ijt,kt,lt,t->ikjl', W_sub, Factor, Factor, N_sub)     # (L,K,L,K)
        denom = denom.reshape(self.L * KM, self.L * KM)

        Gamma1 = np.reshape(
            sla.lstsq(denom, numer, check_finite=False, overwrite_a=True, overwrite_b=True)[0],
            (self.L, KM))

        # ------ Step 3: normalisation ------
        if K > 0:
            Gamma1, Factor1 = self._normalization_choice(
                Gamma=Gamma1, Factor=Factor,
                K=K, M=M, KM=KM,
                normalization_choice=normalization_choice,
                normalization_choice_specs=normalization_choice_specs)
        else:
            Factor1 = Factor.copy()

        return Gamma1, Factor1

    def _svd_initial(self, K, M, gFac_arr=None, date_ints=None):
        """
        SVD-based initial conditions for the ALS algorithm.

        Parameters
        ----------
        date_ints : 1-D integer array into self._X, or None (uses all dates)
        """
        if date_ints is None:
            date_ints = np.arange(len(self.Dates))

        Gamma, Factor = None, None

        if K > 0:
            U, s, VT = sla.svd(self._X[:, date_ints], full_matrices=False)
            Gamma = U[:, :K]
            Factor = np.diag(s[:K]).dot(VT[:K, :])
        if M > 0 and K > 0:
            Gamma = np.concatenate((Gamma, np.zeros((self.L, M))), axis=1)
            Factor = np.concatenate((Factor, gFac_arr[:, date_ints]), axis=0)
        elif M > 0 and K == 0:
            Gamma = np.zeros((self.L, M))
            Factor = gFac_arr[:, date_ints]

        return Gamma, Factor

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def _normalization_choice(self, Gamma, Factor, K, M, KM,
                              normalization_choice, normalization_choice_specs):
        if M == KM:
            raise ValueError('_normalization_choice called with no latent factors (M == KM)')

        if normalization_choice == 'PCA_positivemean':
            if K == KM:
                GammaF, FactorF = Gamma, Factor
            else:
                GammaF, GammaG = Gamma[:, :K], Gamma[:, K:]
                FactorF, FactorG = Factor[:K, :], Factor[K:, :]

            R1 = sla.cholesky(GammaF.T.dot(GammaF))
            R2, _, _ = sla.svd(R1.dot(FactorF).dot(FactorF.T).dot(R1.T))
            GammaF  = sla.lstsq(R1.T, GammaF.T, check_finite=False)[0].T.dot(R2)
            FactorF = sla.lstsq(R2,   R1.dot(FactorF), check_finite=False)[0]
            # sign convention: positive factor means
            sign_conv = np.sign(np.mean(FactorF, axis=1)).reshape(-1, 1)
            sign_conv[sign_conv == 0] = 1.0
            FactorF *= sign_conv
            GammaF  *= sign_conv.T
            if M > 0:
                GammaG  = (np.eye(self.L) - GammaF.dot(GammaF.T)).dot(GammaG)
                FactorF = FactorF + (GammaF.T.dot(GammaG)).dot(FactorG)
                # re-apply sign convention after orthogonalisation
                sign_conv = np.sign(np.mean(FactorF, axis=1)).reshape(-1, 1)
                sign_conv[sign_conv == 0] = 1.0
                FactorF *= sign_conv
                GammaF  *= sign_conv.T

        elif normalization_choice == 'Identity':
            if K == KM:
                GammaF, FactorF = Gamma, Factor
            else:
                GammaF, GammaG = Gamma[:, :K], Gamma[:, K:]
                FactorF, FactorG = Factor[:K, :], Factor[K:, :]

            R = GammaF[normalization_choice_specs, :]
            GammaF  = sla.lstsq(R.T, GammaF.T, check_finite=False)[0].T
            FactorF = R.dot(FactorF)
            if M > 0:
                GammaG  = (np.eye(self.L) - GammaF.dot(GammaF.T)).dot(GammaG)
                FactorF = FactorF + (GammaF.T.dot(GammaG)).dot(FactorG)
        else:
            raise ValueError("Unknown normalization_choice: '%s'" % normalization_choice)

        if M > 0:
            Gamma  = np.concatenate((GammaF, GammaG), axis=1)
            Factor = np.concatenate((FactorF, FactorG), axis=0)
        else:
            Gamma  = GammaF.copy()
            Factor = FactorF.copy()

        return Gamma, Factor

    # ------------------------------------------------------------------
    # VAR(1) helper
    # ------------------------------------------------------------------

    def _VARB(self, X):
        """
        Estimate VAR(1) coefficients (with constant) for the K x T factor matrix X.

        Regresses X[:, 1:] on [X[:, :-1]; 1].  Returns B of shape (KM+1, KM) such that
        the one-step-ahead forecast is  B.T @ [f_{t-1}; 1].
        """
        Xtil = np.concatenate((X, np.ones((1, X.shape[1]))), axis=0)  # (KM+1, T)
        B = sla.lstsq(Xtil[:, :-1].T, X[:, 1:].T)[0]                 # (KM+1, KM)
        return B

    # ------------------------------------------------------------------
    # R2 helpers
    # ------------------------------------------------------------------

    def _R2_calc(self, reals, fits_total, fits_pred, benchR2=None):
        """Compute pooled R2 for total and predictive fits."""
        if benchR2 is None:
            denom = (reals ** 2).values.sum()
        else:
            denom = ((reals - benchR2) ** 2).values.sum()
        R2_Total = 1.0 - ((reals - fits_total) ** 2).values.sum() / denom
        R2_Pred  = 1.0 - ((reals - fits_pred)  ** 2).values.sum() / denom
        return R2_Total, R2_Pred

    def _make_bench_is(self, R2_bench, for_R):
        """Build the benchmark DataFrame for in-sample R2 calculation."""
        if R2_bench == 'zero':
            return None
        elif R2_bench == 'mean':
            if for_R:
                return pd.DataFrame(
                    data=np.ones((self.R.shape[0], 1)) * self.R.mean().values,
                    index=self.R.index, columns=self.R.columns)
            else:
                return pd.DataFrame(
                    data=np.tile(self.X.mean(axis=1).values.reshape(-1, 1), (1, self.X.shape[1])),
                    index=self.X.index, columns=self.X.columns)
        elif R2_bench == 'pooled_mean':
            if for_R:
                return pd.DataFrame(
                    data=self.R.values.mean(),
                    index=self.R.index, columns=self.R.columns)
            else:
                return pd.DataFrame(
                    data=self.X.values.mean(),
                    index=self.X.index, columns=self.X.columns)
        else:
            raise ValueError("R2_bench must be 'zero', 'mean', or 'pooled_mean'")

    def R2_of_fits(self, results=None, date_range=None, R2_bench='zero',
                   recursive=False, R2name=None, inplace=True):
        """
        Compute R2 over a specified date range, optionally adding results to a fit dict.

        Parameters
        ----------
        results    : dict returned by ipca.fit()
        date_range : datetime-like index of dates over which to evaluate
        R2_bench   : str, default 'zero' — benchmark used in the R2 denominator.
                     Uses the same vocabulary as the R2_bench parameter of fit():
                     'zero'        benchmark = 0 (denominator is sum of squared actuals)
                     'mean'        unit-specific means; for X, the mean per characteristic;
                                   for R, the mean per permno
                     'pooled_mean' grand mean pooled across all units
                     When recursive=False (default), means are computed over date_range
                     only.  When recursive=True, means are expanding-window estimates
                     (each period's benchmark uses only data prior to that period),
                     matching the OOS convention inside fit().
        recursive  : bool, default False
                     False — benchmark means are computed over the full date_range
                             (appropriate for post-hoc evaluation over a fixed window)
                     True  — benchmark means are expanding-window estimates up to
                             (but not including) each period, matching fit() OOS behavior
        R2name     : str suffix for the R2 key names added to results
                     (default: 'YYYYMMDD-YYYYMMDD' from date_range endpoints)
        inplace    : bool, default True; if True modify results in place and return None
        """
        if results is None:
            raise ValueError('ipca.R2_of_fits: a results dict must be passed')
        if date_range is None:
            raise ValueError('ipca.R2_of_fits: a date_range must be passed')
        if R2_bench not in ('zero', 'mean', 'pooled_mean'):
            raise ValueError("R2_bench must be 'zero', 'mean', or 'pooled_mean'; "
                             "got %r" % R2_bench)

        # --- benchmarks ---
        if R2_bench == 'zero':
            benchX = pd.DataFrame(0., index=self.X.index, columns=self.X.columns)
            benchR = pd.DataFrame(0., index=self.R.index, columns=self.R.columns) \
                     if self.has_RZ else None

        elif R2_bench == 'pooled_mean':
            if not recursive:
                # grand mean over date_range only
                benchX = pd.DataFrame(np.mean(self.X.T.loc[date_range].values),
                                       index=self.X.index, columns=self.X.columns)
                if self.has_RZ:
                    benchR = pd.DataFrame(np.mean(self.R.loc[date_range].values),
                                           index=self.R.index, columns=self.R.columns)
            else:
                # expanding-window grand mean (matches fit() 'pooled_mean' OOS)
                benchX = pd.DataFrame(0., index=self.X.index, columns=self.X.columns)
                X_cs = np.cumsum(self._X.sum(axis=0))
                X_cn = np.arange(1, len(self.Dates) + 1) * self.L
                for i, t in enumerate(self.X.columns[1:], start=1):
                    benchX[t] = X_cs[i - 1] / X_cn[i - 1]
                if self.has_RZ:
                    benchR = pd.DataFrame(0., index=self.R.index, columns=self.R.columns)
                    for t in self.X.columns[1:]:
                        benchR.loc[t] = self.R.loc[
                            self.R.index.get_level_values(0) < t].values.mean()

        else:  # R2_bench == 'mean'
            if not recursive:
                # unit-specific means over date_range only (matches fit() 'mean' IS)
                x_mean = self.X.T.loc[date_range].mean(axis=0).values.reshape(-1, 1)
                benchX = pd.DataFrame(np.tile(x_mean, (1, self.X.shape[1])),
                                       index=self.X.index, columns=self.X.columns)
                if self.has_RZ:
                    # per-permno mean over date_range; transform broadcasts it back
                    benchR = self.R.loc[date_range].groupby(level=1).transform('mean')
            else:
                # expanding-window unit means (matches fit() 'mean' OOS)
                benchX = self.X.T.shift(1).expanding().mean().T
                if self.has_RZ:
                    benchR = self.R.groupby(level=1).shift(1).expanding().mean()

        # --- R2 calculations ---
        Xr   = self.X.T.loc[date_range]
        bXr  = benchX.T.loc[date_range]
        r2_x_t = (1 - ((Xr - results['xfits']['Fits_Total'].T.loc[date_range]) ** 2).values.sum()
                     / ((Xr - bXr) ** 2).values.sum())
        r2_x_p = (1 - ((Xr - results['xfits']['Fits_Pred'].T.loc[date_range]) ** 2).values.sum()
                     / ((Xr - bXr) ** 2).values.sum())

        r2_r_t = r2_r_p = None
        if self.has_RZ and results['rfits'] is not None:
            Rr  = self.R.loc[date_range]
            bRr = benchR.loc[date_range]
            r2_r_t = (1 - ((Rr - results['rfits']['Fits_Total'].loc[date_range]) ** 2).values.sum()
                         / ((Rr - bRr) ** 2).values.sum())
            r2_r_p = (1 - ((Rr - results['rfits']['Fits_Pred'].loc[date_range]) ** 2).values.sum()
                         / ((Rr - bRr) ** 2).values.sum())

        if R2name is None:
            R2name = date_range[0].strftime('%Y%m%d') + '-' + date_range[-1].strftime('%Y%m%d')

        if inplace:
            results['xfits']['R2_Total_' + R2name] = r2_x_t
            results['xfits']['R2_Pred_'  + R2name] = r2_x_p
            if results['rfits'] is not None and r2_r_t is not None:
                results['rfits']['R2_Total_' + R2name] = r2_r_t
                results['rfits']['R2_Pred_'  + R2name] = r2_r_p
            return None
        else:
            newresults = copy.deepcopy(results)
            newresults['xfits']['R2_Total_' + R2name] = r2_x_t
            newresults['xfits']['R2_Pred_'  + R2name] = r2_x_p
            if newresults['rfits'] is not None and r2_r_t is not None:
                newresults['rfits']['R2_Total_' + R2name] = r2_r_t
                newresults['rfits']['R2_Pred_'  + R2name] = r2_r_p
            return newresults

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_fits(self, R_fit, Beta_fit):
        Rdo = False
        if R_fit and not self.has_RZ:
            print('ipca.fit: R_fit=True but no RZ supplied; rfits will be None')
        elif R_fit:
            Rdo = True

        Betado = False
        if Beta_fit and not self.has_RZ:
            print('ipca.fit: Beta_fit=True but no RZ supplied; fittedBeta will be None')
        elif Beta_fit:
            Betado = True

        fittedX = {'Fits_Total': pd.DataFrame(np.nan, index=self.X.index, columns=self.Dates),
                   'Fits_Pred':  pd.DataFrame(np.nan, index=self.X.index, columns=self.Dates)}
        fittedR     = ({'Fits_Total': pd.DataFrame(np.nan, index=self.R.index, columns=self.R.columns),
                        'Fits_Pred':  pd.DataFrame(np.nan, index=self.R.index, columns=self.R.columns)}
                       if Rdo else None)
        fittedBeta  = (pd.DataFrame(np.nan, index=self.R.index, columns=self.Factor_names)
                       if Betado else None)

        return Rdo, Betado, fittedX, fittedR, fittedBeta

    def _compute_beta(self, t, Gamma_arr):
        """
        Compute the (N_t x KM) beta matrix for date t.
        Gamma_arr is the (L, KM) ndarray of loadings.
        """
        if self.add_constant:
            return self.Z.loc[t].values.dot(Gamma_arr[:-1, :]) + Gamma_arr[-1, :]
        else:
            return self.Z.loc[t].values.dot(Gamma_arr)

    def _find_sublist(self, sub):
        """Map characteristic names in sub to integer row indices in self.Chars."""
        chars_list = list(self.Chars)
        pos = []
        for subj in sub:
            try:
                pos.append(chars_list.index(subj))
            except ValueError:
                pos.append(-1)
                print('_find_sublist: %s not found in self.Chars' % subj)
        return pos

    # ------------------------------------------------------------------
    # Minimum training window helper
    # ------------------------------------------------------------------

    def _compute_min_train_periods(self, KM, factor_mean, MacroData, target_variance,
                                   multiplier=2):
        """
        Auto-compute the minimum training-window size for OOS estimation.

        Uses the rule: multiplier × (number of free parameters in the most demanding
        regression in the pipeline).

        Parameters
        ----------
        KM              : int   total number of factors (latent + pre-specified)
        factor_mean     : str   one of 'constant', 'VAR1', 'macro', 'forecombo'
        MacroData       : df or None
        target_variance : float or None  if given, PCA is used and P_eff = n_comp
        multiplier      : int   obs-per-parameter multiplier (default 3)

        Returns
        -------
        int : minimum required OOS_window_specs
        """
        # VAR(1) floor: KM lags + intercept per factor equation
        floor_var = multiplier * (KM + 1)

        if factor_mean == 'constant':
            return max(KM + 2, multiplier)   # sample mean; trivially small

        if factor_mean == 'VAR1':
            return floor_var

        # 'macro' or 'forecombo': binding constraint may be the macro regression
        if MacroData is not None:
            if target_variance is not None:
                # Run PCA on full MacroData to determine effective predictor count
                Vt_n, _, _ = self._pca_fit(MacroData, target_variance)
                P_eff = Vt_n.shape[0]           # n_comp after dimension reduction
            else:
                P_eff = MacroData.shape[1]       # raw predictor count
            floor_mac = multiplier * (P_eff + 1)
            return max(floor_var, floor_mac)

        return floor_var  # fallback (MacroData unexpectedly None)

    # ------------------------------------------------------------------
    # Factor prediction helpers for factor_mean='forecombo'
    # ------------------------------------------------------------------

    def _predict_factors_with_forecombo_uncon(self, Factor, MacroData_train, MacroData_test,
                                              return_train_fitted=False):
        """
        Predict latent factors via OLS regression of factors on standardised macro data.

        Factor              : (KM, T_train) ndarray of factor realisations over the training window
        MacroData_train     : df(T_train x P)              macro data aligned with Factor columns
        MacroData_test      : df(T_test x P) or Series(P,) macro data at prediction date(s)
        return_train_fitted : bool — when True return (test_pred, train_fitted) tuple where
                              train_fitted is (KM, T_train); otherwise return test_pred only.
        Returns             : (KM, T_test) ndarray if MacroData_test is a DataFrame
                              (KM,) ndarray       if MacroData_test is a Series
        """
        X_train = MacroData_train.values.astype(np.float64)

        mean = X_train.mean(axis=0)
        std  = np.where(X_train.std(axis=0) > 1e-12, X_train.std(axis=0), 1.0)
        X_train_s = (X_train - mean) / std

        X_aug = np.column_stack([np.ones(X_train.shape[0]), X_train_s])
        beta  = np.linalg.lstsq(X_aug, Factor.T, rcond=None)[0]  # (1+P, KM)

        if isinstance(MacroData_test, pd.Series):
            x_t   = (MacroData_test.values.astype(np.float64) - mean) / std
            x_aug = np.hstack([1.0, x_t])
            test_pred = (x_aug @ beta).ravel()                    # (KM,)
        else:
            X_test   = MacroData_test.values.astype(np.float64)
            X_test_s = (X_test - mean) / std
            X_test_aug = np.column_stack([np.ones(len(X_test)), X_test_s])
            test_pred = (X_test_aug @ beta).T                     # (KM, T_test)

        if return_train_fitted:
            return test_pred, (X_aug @ beta).T                    # (KM, T_train)
        return test_pred

    def _predict_factors_with_forecombo_ridge(self, Factor, MacroData_train, MacroData_test, alpha,
                                              return_train_fitted=False, ridge_df=None):
        """
        Predict latent factors via Ridge regression on standardised macro data.

        Factor              : (KM, T_train) ndarray
        MacroData_train     : df(T_train x P)
        MacroData_test      : df(T_test x P) or Series(P,)
        alpha               : Ridge penalty strength. Ignored when ridge_df is not None.
        ridge_df            : target effective degrees of freedom; when provided, overrides alpha.
        return_train_fitted : bool — when True return (test_pred, train_fitted) tuple.
        Returns             : (KM, T_test) or (KM,)
        """
        X_train = MacroData_train.values.astype(np.float64)

        mean = X_train.mean(axis=0)
        std  = np.where(X_train.std(axis=0) > 1e-12, X_train.std(axis=0), 1.0)
        X_train_s = (X_train - mean) / std

        if ridge_df is not None:
            alpha = self._ridge_lambda_from_df(X_train_s, ridge_df)

        KM   = Factor.shape[0]
        beta = np.zeros((X_train_s.shape[1] + 1, KM))
        for k in range(KM):
            beta[:, k] = self._ridge_regression(X_train_s, Factor[k, :], lambda_=alpha)

        X_train_aug = np.column_stack([np.ones(X_train.shape[0]), X_train_s])

        if isinstance(MacroData_test, pd.Series):
            x_t   = (MacroData_test.values.astype(np.float64) - mean) / std
            x_aug = np.hstack([1.0, x_t])
            test_pred = (x_aug @ beta).ravel()                    # (KM,)
        else:
            X_test   = MacroData_test.values.astype(np.float64)
            X_test_s = (X_test - mean) / std
            X_test_aug = np.column_stack([np.ones(len(X_test)), X_test_s])
            test_pred = (X_test_aug @ beta).T                     # (KM, T_test)

        if return_train_fitted:
            return test_pred, (X_train_aug @ beta).T              # (KM, T_train)
        return test_pred

    def _predict_factors_with_forecombo_lasso(self, Factor, MacroData_train, MacroData_test, alpha,
                                              return_train_fitted=False, return_selected=False):
        """
        Predict latent factors via LASSO regression on standardised macro data.

        The full regularisation path is computed for each factor and the
        least-penalised solution with at most alpha non-zero coefficients is
        selected, giving direct control over the number of active predictors.

        Factor              : (KM, T_train) ndarray
        MacroData_train     : df(T_train x P)
        MacroData_test      : df(T_test x P) or Series(P,)
        alpha               : int — target number of active macro predictors.
                              The least-penalised path solution with at most alpha
                              non-zero coefficients is used; if the path never
                              reaches alpha active predictors the densest solution
                              is used instead.
        return_train_fitted : bool — when True include train_fitted (KM, T_train) in return.
        return_selected     : bool — when True include selected_mask (P, KM) bool ndarray
                              indicating which predictors have non-zero coefficients for
                              each factor.  Combine with return_train_fitted freely; see
                              return-value description below.
        Returns
        -------
        test_pred                                   when both flags False
        (test_pred, train_fitted)                   when return_train_fitted only
        (test_pred, selected_mask)                  when return_selected only
        (test_pred, train_fitted, selected_mask)    when both flags True
        """
        X_train = MacroData_train.values.astype(np.float64)

        mean = X_train.mean(axis=0)
        std  = np.where(X_train.std(axis=0) > 1e-12, X_train.std(axis=0), 1.0)
        X_train_s = (X_train - mean) / std

        KM          = Factor.shape[0]
        P           = X_train_s.shape[1]
        coef_matrix = np.zeros((P, KM))
        intercepts  = np.zeros(KM)

        # Parallelise across factors: each lasso_path call is independent and
        # uses sklearn's LARS implementation (Cython/numpy), which releases the
        # GIL.  The threading backend therefore achieves true parallelism with
        # zero pickling overhead.  Meaningful when KM >= 3 and P is large
        # (wide macro panel); overhead dominates for KM <= 2 or narrow panels.
        def _lasso_for_k(k):
            y      = Factor[k, :]
            y_mean = float(y.mean())
            y_c    = y - y_mean                       # centre; lasso_path has no fit_intercept
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                coefs_path = lasso_path(X_train_s, y_c)[1]
            # alphas_path is decreasing → n_nonzero increases along the path
            n_nonzero = (coefs_path != 0).sum(axis=0)
            valid = np.where(n_nonzero <= int(alpha))[0]
            idx   = valid[-1] if len(valid) > 0 else coefs_path.shape[1] - 1
            return coefs_path[:, idx], y_mean         # (P,), scalar

        per_k = Parallel(n_jobs=-1, prefer='threads')(
            delayed(_lasso_for_k)(k) for k in range(KM))

        for k, (coef_k, intercept_k) in enumerate(per_k):
            coef_matrix[:, k] = coef_k
            intercepts[k]     = intercept_k

        if isinstance(MacroData_test, pd.Series):
            x_t = (MacroData_test.values.astype(np.float64) - mean) / std
            test_pred = coef_matrix.T @ x_t + intercepts          # (KM,)
        else:
            X_test   = MacroData_test.values.astype(np.float64)
            X_test_s = (X_test - mean) / std
            test_pred = coef_matrix.T @ X_test_s.T + intercepts[:, None]  # (KM, T_test)

        train_fitted   = (coef_matrix.T @ X_train_s.T + intercepts[:, None]  # (KM, T_train)
                          if return_train_fitted else None)
        selected_mask  = (coef_matrix != 0) if return_selected else None      # (P, KM) bool

        if return_train_fitted and return_selected:
            return test_pred, train_fitted, selected_mask
        if return_train_fitted:
            return test_pred, train_fitted
        if return_selected:
            return test_pred, selected_mask
        return test_pred

    def _predict_factors_with_forecombo_pca(self, Factor, MacroData_train, MacroData_test,
                                            target_variance, return_train_fitted=False):
        """PCA pre-processing then OLS prediction."""
        Vt_n, pca_mean, pca_std = self._pca_fit(MacroData_train, target_variance)
        train_red = self._pca_transform(MacroData_train, Vt_n, pca_mean, pca_std)
        test_red  = self._pca_transform(MacroData_test,  Vt_n, pca_mean, pca_std)
        return self._predict_factors_with_forecombo_uncon(
            Factor, train_red, test_red, return_train_fitted=return_train_fitted)

    def _predict_factors_with_forecombo_pca_ridge(self, Factor, MacroData_train, MacroData_test,
                                                  target_variance, alpha,
                                                  return_train_fitted=False, ridge_df=None):
        """PCA pre-processing then Ridge prediction."""
        Vt_n, pca_mean, pca_std = self._pca_fit(MacroData_train, target_variance)
        train_red = self._pca_transform(MacroData_train, Vt_n, pca_mean, pca_std)
        test_red  = self._pca_transform(MacroData_test,  Vt_n, pca_mean, pca_std)
        return self._predict_factors_with_forecombo_ridge(
            Factor, train_red, test_red, alpha,
            return_train_fitted=return_train_fitted, ridge_df=ridge_df)

    def _predict_factors_with_forecombo_pca_lasso(self, Factor, MacroData_train, MacroData_test,
                                                  target_variance, alpha,
                                                  return_train_fitted=False,
                                                  return_selected=False):
        """PCA pre-processing then LASSO prediction.

        When return_selected=True the selected_mask rows correspond to PCs, not
        original predictors.  PC labels ('PC_1', 'PC_2', …) are assigned by the
        caller (_dispatch_macro_predict) which knows n_comp after fitting.
        """
        Vt_n, pca_mean, pca_std = self._pca_fit(MacroData_train, target_variance)
        train_red = self._pca_transform(MacroData_train, Vt_n, pca_mean, pca_std)
        test_red  = self._pca_transform(MacroData_test,  Vt_n, pca_mean, pca_std)
        return self._predict_factors_with_forecombo_lasso(
            Factor, train_red, test_red, alpha,
            return_train_fitted=return_train_fitted,
            return_selected=return_selected)

    def _predict_factors_with_forecombo_3prf(self, Factor, MacroData_train, MacroData_test,
                                             pass2_intercept=True, return_train_fitted=False):
        """
        Predict latent factors via the Three-Pass Regression Filter (Kelly & Pruitt 2015).

        Each IPCA factor k is used as a scalar proxy z_k.  Three passes are run
        independently per k and fully vectorised across k using matrix operations:

            Pass 1 — loading vector (with intercept, via FWL):
                b_k = X_s_train' z_c_k / (z_c_k' z_c_k)        (P,)
                where X_s_train is the standardised macro panel (T_train x P)
                and z_c_k is the centred proxy (z_k - mean(z_k)).

            Pass 2 — filtered factor:
                Without intercept: f̂_k(t) = x_s(t)' b_k / (b_k' b_k)
                With intercept:    f̂_k(t) is the slope from OLS of x_s(t) (P x 1)
                                   on [1, b_k] (P x 2); intercept is discarded.
                The 2x2 Gram matrix [[P, sum(b_k)], [sum(b_k), b_k'b_k]] is the same
                for all t, so it is computed once per k.

            Pass 3 — calibration:
                OLS of z_k on [1, f̂_k] over the training window → (c_k, π_k).
                Forecast: λ_k = c_k + π_k * f̂_k(test).

        Passes 1 and 2 are batched across all k via matrix multiplications.
        Pass 3 uses a small loop (KM iterations of a 2-parameter lstsq).

        Predictors are standardised using training-window mean and std.
        The target (Factor) is NOT centred.

        Factor              : (KM, T_train) ndarray of factor realisations
        MacroData_train     : df(T_train x P)
        MacroData_test      : df(T_test x P) or Series(P,)
        pass2_intercept     : bool — include cross-sectional intercept in Pass 2
        return_train_fitted : bool — when True return (test_pred, train_fitted) tuple.
        Returns             : (KM, T_test) ndarray or (KM,) if MacroData_test is a Series
        """
        KM = Factor.shape[0]

        # --- Standardise predictors on training statistics ---
        X_train  = MacroData_train.values.astype(np.float64)     # (T_train x P)
        T_train, P = X_train.shape
        mean_x   = X_train.mean(axis=0)                           # (P,)
        std_x    = np.where(X_train.std(axis=0) > 1e-12,
                            X_train.std(axis=0), 1.0)             # (P,)
        X_s      = (X_train - mean_x) / std_x                    # (T_train x P)

        is_series = isinstance(MacroData_test, pd.Series)
        X_test_raw = (MacroData_test.values.astype(np.float64).reshape(1, -1)
                      if is_series
                      else MacroData_test.values.astype(np.float64))  # (T_test x P)
        T_test    = X_test_raw.shape[0]
        X_test_s  = (X_test_raw - mean_x) / std_x                # (T_test x P)

        # --- Pass 1: loading matrix B (P x KM) ---
        # Centre the proxy so that Pass 1 estimates the slope from the
        # intercept regression  x_s = a + c*z  (FWL: regress on centred z).
        # X_s is already zero-mean over the training window.
        Z     = Factor.T                                           # (T_train x KM)
        Z_c   = Z - Z.mean(axis=0)                                # centred proxy
        z_c_sq = (Z_c ** 2).sum(axis=0)                           # (KM,)
        B     = X_s.T @ Z_c / z_c_sq                              # (P x KM)

        # --- Pass 2: filtered factors (T x KM) for both train and test ---
        b_sq  = (B ** 2).sum(axis=0)                              # (KM,)

        def _pass2(X_std):
            """Apply Pass 2 to a (T x P) standardised array; returns (T x KM)."""
            XB = X_std @ B                                        # (T x KM)
            if not pass2_intercept:
                return XB / b_sq                                  # (T x KM)
            # With intercept: per-k 2x2 Gram solve (b_sum same for train and test)
            b_sum  = B.sum(axis=0)                                # (KM,)
            xsum   = X_std.sum(axis=1)                            # (T,) sum over P
            out    = np.empty((X_std.shape[0], KM))
            for k in range(KM):
                A_k = np.array([[P,         b_sum[k]],
                                [b_sum[k],  b_sq[k]]])            # (2 x 2)
                rhs = np.column_stack([xsum, XB[:, k]])           # (T x 2)
                out[:, k] = np.linalg.solve(A_k, rhs.T)[1]       # slope row
            return out                                            # (T x KM)

        F_hat_train = _pass2(X_s)                                 # (T_train x KM)
        F_hat_test  = _pass2(X_test_s)                            # (T_test  x KM)

        # --- Pass 3: OLS of z_k on [1, f̂_k] over training window ---
        intercepts = np.zeros(KM)
        slopes     = np.zeros(KM)
        for k in range(KM):
            X_p3          = np.column_stack(
                [np.ones(T_train), F_hat_train[:, k]])            # (T_train x 2)
            coef          = np.linalg.lstsq(X_p3, Z[:, k], rcond=None)[0]
            intercepts[k] = coef[0]
            slopes[k]     = coef[1]

        # --- Forecast ---
        pred = intercepts + slopes * F_hat_test                   # (T_test x KM)

        if is_series:
            test_pred = pred[0]                                   # (KM,)
        else:
            test_pred = pred.T                                    # (KM x T_test)

        if return_train_fitted:
            train_fitted = (intercepts + slopes * F_hat_train).T  # (KM, T_train)
            return test_pred, train_fitted
        return test_pred

    def _dispatch_macro_predict(self, Factor, MacroData_train, MacroData_test,
                                regularization, target_variance, alpha,
                                pass2_intercept=True, return_train_fitted=False,
                                return_selected=False, ridge_df=None) -> Any:
        """
        Dispatch macro-to-factor prediction to the appropriate forecombo helper.

        Factor              : (KM, T_train) ndarray
        MacroData_train     : df(T_train x P)  — caller supplies the correct training slice
        MacroData_test      : df(T_test x P) or Series(P,) — caller supplies the correct test slice
        pass2_intercept     : bool — passed through to 3PRF only (ignored otherwise)
        return_train_fitted : bool — when True include train_fitted (KM, T_train) in return.
        return_selected     : bool — when True include selected_info in return.
                              selected_info is a (selected_mask, predictor_names) tuple for
                              regularization='lasso' (raw or PCA), or None for other methods.
                              For raw lasso, predictor_names = MacroData_train.columns.
                              For PCA+lasso, predictor_names = ['PC_1', 'PC_2', ...] (PC-space
                              selection; PC_k corresponds to the k-th retained component).

        Returns (depending on flags)
        ----------------------------
        test_pred                                      both flags False
        (test_pred, train_fitted)                      return_train_fitted only
        (test_pred, selected_info)                     return_selected only
        (test_pred, train_fitted, selected_info)       both flags True
        """
        is_lasso = (regularization == 'lasso')

        # --- dispatch to the appropriate helper ---
        if regularization == '3prf':
            if target_variance is not None:
                Vt_n, pca_mean, pca_std = self._pca_fit(MacroData_train, target_variance)
                train_red = self._pca_transform(MacroData_train, Vt_n, pca_mean, pca_std)
                test_red  = self._pca_transform(MacroData_test,  Vt_n, pca_mean, pca_std)
                result = self._predict_factors_with_forecombo_3prf(
                    Factor, train_red, test_red, pass2_intercept,
                    return_train_fitted=return_train_fitted)
            else:
                result = self._predict_factors_with_forecombo_3prf(
                    Factor, MacroData_train, MacroData_test, pass2_intercept,
                    return_train_fitted=return_train_fitted)
        elif target_variance is not None:
            if regularization == 'ridge':
                result = self._predict_factors_with_forecombo_pca_ridge(
                    Factor, MacroData_train, MacroData_test, target_variance, alpha,
                    return_train_fitted=return_train_fitted, ridge_df=ridge_df)
            elif is_lasso:
                result = self._predict_factors_with_forecombo_pca_lasso(
                    Factor, MacroData_train, MacroData_test, target_variance, alpha,
                    return_train_fitted=return_train_fitted,
                    return_selected=return_selected)
            else:
                result = self._predict_factors_with_forecombo_pca(
                    Factor, MacroData_train, MacroData_test, target_variance,
                    return_train_fitted=return_train_fitted)
        elif is_lasso:
            result = self._predict_factors_with_forecombo_lasso(
                Factor, MacroData_train, MacroData_test, alpha,
                return_train_fitted=return_train_fitted,
                return_selected=return_selected)
        elif regularization == 'ridge':
            result = self._predict_factors_with_forecombo_ridge(
                Factor, MacroData_train, MacroData_test, alpha,
                return_train_fitted=return_train_fitted, ridge_df=ridge_df)
        else:
            result = self._predict_factors_with_forecombo_uncon(
                Factor, MacroData_train, MacroData_test,
                return_train_fitted=return_train_fitted)

        # --- handle return_selected ---
        if not return_selected:
            return result

        # When return_selected=True we always return a consistent 3-tuple:
        #   (test_pred, train_fitted, selected_info)
        # train_fitted is None when return_train_fitted=False.
        # selected_info is (selected_mask, predictor_names) for lasso, else None.
        train_fitted  = None
        selected_info = None

        if is_lasso:
            # Lasso helpers include selected_mask as the last element.
            if return_train_fitted:
                test_pred, train_fitted, selected_mask = result
            else:
                test_pred, selected_mask = result
            # Predictor names: PC labels when PCA was applied, else raw macro names.
            if target_variance is not None:
                n_comp = selected_mask.shape[0]
                pnames = [f'PC_{i + 1}' for i in range(n_comp)]
            else:
                pnames = list(MacroData_train.columns)
            selected_info = (selected_mask, pnames)
        else:
            # Non-lasso paths: unpack test_pred (and train_fitted if requested).
            if return_train_fitted:
                test_pred, train_fitted = result
            else:
                test_pred = result

        return test_pred, train_fitted, selected_info

    # ------------------------------------------------------------------
    # Combining predictions (forecombo mode)
    # ------------------------------------------------------------------

    def _calculate_combined_predictions(self, Factor, lagged_factors, macro_to_factors):
        """
        Combine two sets of KM-vector forecasts in-sample via full-sample OLS weights.

        Parameters
        ----------
        Factor           : (KM, T) ndarray of realised factors (full IS sample)
        lagged_factors   : (T_pred, KM) array of VAR1 predictions  (T_pred = T-1)
        macro_to_factors : (T_pred, KM) array of macro predictions

        Notes
        -----
        Column 0 of Factor is the unconditional mean (not a prediction target).
        Targets are Factor[:, 1:] (T_pred = T-1 periods matching the prediction rows).
        Returns (KM, T_pred) combined in-sample predictions.
        """
        KM     = lagged_factors.shape[1]
        T_pred = lagged_factors.shape[0]
        targets  = Factor[:, 1:]   # (KM, T_pred)
        combined = np.zeros((KM, T_pred))
        for k in range(KM):
            X = np.column_stack([
                np.ones(T_pred),
                lagged_factors[:, k],
                macro_to_factors[:, k]])           # (T_pred, 3)
            w = np.linalg.lstsq(X, targets[k, :], rcond=None)[0]
            combined[k, :] = X @ w
        return combined                            # (KM, T_pred)

    def _combine_forecasts_oos(self, fac_hist, var_hist, mac_hist, lamt_var, lamt_mac):
        """
        Expanding-window OLS combination of VAR1 and macro forecasts (OOS forecombo).

        Fits per-factor: F_k[s] = w0 + w1*lamt_var_k[s] + w2*lamt_mac_k[s]
        on the accumulated OOS history, then evaluates at the current-period forecasts.

        Parameters
        ----------
        fac_hist : (KM, n_hist) realised OOS factors from past OOS periods
        var_hist : (KM, n_hist) VAR1 forecasts made for those past periods
        mac_hist : (KM, n_hist) macro forecasts made for those past periods
        lamt_var : (KM,) current VAR1 forecast
        lamt_mac : (KM,) current macro forecast
        Returns  : (KM,) combined forecast
        """
        KM, n_hist = fac_hist.shape
        combined = np.zeros(KM)
        for k in range(KM):
            X_tr = np.column_stack([np.ones(n_hist), var_hist[k], mac_hist[k]])  # (n_hist, 3)
            w    = np.linalg.lstsq(X_tr, fac_hist[k], rcond=None)[0]
            combined[k] = np.dot([1.0, lamt_var[k], lamt_mac[k]], w)
        return combined                            # (KM,)

    def _calculate_combined_predictions_dma(self, Factor, Lambda, LambdaM,
                                            forgetting_factor=1.0,
                                            var_initial_weight=0.8,
                                            macro_initial_weight=0.2,
                                            var_converging_weight=0.8,
                                            macro_converging_weight=0.2):
        """
        Dynamic Model Averaging (DMA) combination of VAR1 and macro predictions.
        Alternative to OLS combination for forecombo mode (not wired into fit() by default).

        Factor  : (KM, T_oos) ndarray of realised OOS factors
        Lambda  : df(KM x T) VAR1 predictions (full date range)
        LambdaM : df(KM x T) macro predictions (full date range)

        Returns a copy of Lambda with the relevant columns replaced by DMA-weighted combinations.
        """
        KM = Factor.shape[0]
        LambdaCombined = Lambda.copy()
        total_cols = Lambda.shape[1]
        nF = Factor.shape[1]
        start_col = total_cols - (nF + 1)

        w_var   = np.full(KM, var_initial_weight)
        w_macro = np.full(KM, macro_initial_weight)

        for i in range(nF + 1):
            col = start_col + i
            if i > 0:
                actual       = Factor[:, i - 1]
                forecast_var = Lambda.iloc[:, col - 1].values
                forecast_mac = LambdaM.iloc[:, col - 1].values
                err_var  = actual - forecast_var
                err_mac  = actual - forecast_mac
                for r in range(KM):
                    L_var   = np.exp(-0.5 * err_var[r] ** 2)
                    L_mac   = np.exp(-0.5 * err_mac[r] ** 2)
                    nv = w_var[r] * L_var
                    nm = w_macro[r] * L_mac
                    d  = nv + nm
                    wv_new = nv / d
                    wm_new = nm / d
                    wv_s = forgetting_factor * wv_new + (1 - forgetting_factor) * var_converging_weight
                    wm_s = forgetting_factor * wm_new + (1 - forgetting_factor) * macro_converging_weight
                    ws = wv_s + wm_s
                    w_var[r]   = wv_s / ws
                    w_macro[r] = wm_s / ws

            combined = (w_var   * Lambda.iloc[:, col].values
                        + w_macro * LambdaM.iloc[:, col].values)
            LambdaCombined.iloc[:, col] = combined

        return LambdaCombined

    # ------------------------------------------------------------------
    # PCA helpers for forecombo mode
    # ------------------------------------------------------------------

    def _pca_fit(self, data_train, target_variance):
        """
        Fit PCA on training macro data.

        Standardises data_train, then determines the number of principal components
        to retain based on target_variance.

        Parameters
        ----------
        data_train      : df(T_train x P) training macro data
        target_variance : float in (0, 1] — retain minimum components explaining at
                          least this fraction of total variance.
                          int >= 1         — retain exactly this many components.

        Returns
        -------
        Vt_n   : (n_comp, P) ndarray  — PCA direction matrix (rows = components)
        mean   : (P,) ndarray         — training-sample mean (for standardisation)
        std    : (P,) ndarray         — training-sample std  (zero-variance protected)
        """
        vals = data_train.values.astype(np.float64)
        mean = vals.mean(axis=0)
        std  = np.where(vals.std(axis=0) > 1e-12, vals.std(axis=0), 1.0)
        vals_s = (vals - mean) / std

        _, S, Vt = np.linalg.svd(vals_s, full_matrices=False)

        if isinstance(target_variance, (int, np.integer)):
            n_comp = min(int(target_variance), Vt.shape[0])
        else:
            var_ratio = (S ** 2) / max((len(vals) - 1), 1)
            cum_var   = np.cumsum(var_ratio) / var_ratio.sum()
            n_comp    = int(np.searchsorted(cum_var, target_variance)) + 1

        return Vt[:n_comp], mean, std   # (n_comp, P), (P,), (P,)

    def _pca_transform(self, data, Vt_n, mean, std):
        """
        Apply a fitted PCA transform (standardise then project) to new data.

        Parameters
        ----------
        data  : df(T x P) or Series(P,)
        Vt_n  : (n_comp, P) from _pca_fit — PCA direction matrix
        mean  : (P,) from _pca_fit
        std   : (P,) from _pca_fit

        Returns
        -------
        DataFrame of shape (T, n_comp) if data is a DataFrame,
        Series   of length n_comp      if data is a Series.
        """
        if isinstance(data, pd.Series):
            vals  = data.values.astype(np.float64)
            score = (vals - mean) / std @ Vt_n.T        # (n_comp,)
            return pd.Series(score, name=data.name)
        else:
            vals   = data.values.astype(np.float64)
            scores = (vals - mean) / std @ Vt_n.T       # (T, n_comp)
            return pd.DataFrame(scores, index=data.index)

    # ------------------------------------------------------------------
    # Ridge helper
    # ------------------------------------------------------------------

    def _ridge_lambda_from_df(self, X_s, target_df):
        """
        Find the ridge penalty λ such that the effective degrees of freedom equals target_df.

        Effective df: df(λ) = Σ_i σ_i² / (σ_i² + λ)
        where σ_i are the singular values of X_s (standardised design matrix, no intercept).
        df is monotone decreasing in λ, ranging from P (λ=0) to 0 (λ→∞).
        Solved via bisection (Brent's method).

        X_s        : (T, P) standardised design matrix
        target_df  : desired effective df (float in (0, P])
        Returns    : λ (float)
        """
        sv = np.linalg.svd(X_s, compute_uv=False)   # (min(T,P),) singular values
        sv2 = sv ** 2

        def eff_df(lam):
            return float(np.sum(sv2 / (sv2 + lam)))

        df_max = eff_df(0.0)   # ≈ min(T, P); may be < P if T < P

        # Clamp target so it's achievable
        target = float(np.clip(target_df, 1e-6, df_max))
        if target >= df_max - 1e-10:
            return 0.0  # no penalty needed

        # Bracket: find upper bound where df < target
        lam_hi = 1.0
        while eff_df(lam_hi) > target:
            lam_hi *= 10.0

        # Bisection (df is strictly monotone decreasing, so simple bisect suffices)
        lam_lo = 0.0
        for _ in range(60):  # 60 iterations → error < 2^{-60} * lam_hi
            lam_mid = 0.5 * (lam_lo + lam_hi)
            if eff_df(lam_mid) > target:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid
            if lam_hi - lam_lo < 1e-10 * (1.0 + lam_hi):
                break
        return 0.5 * (lam_lo + lam_hi)

    def _ridge_regression(self, X, y, lambda_):
        """
        Ridge regression closed-form solution with intercept (intercept not penalised).

        X : (n, p) standardised design matrix (no intercept column)
        y : (n,) target vector
        Returns beta : (p+1,) -- intercept first, then coefficients
        """
        X_aug  = np.column_stack([np.ones(X.shape[0]), X])  # (n, p+1)
        n_feat = X_aug.shape[1]
        I      = np.eye(n_feat)
        I[0, 0] = 0.0  # do not penalise intercept
        # Use solve (more stable than explicit inverse)
        beta = np.linalg.solve(X_aug.T @ X_aug + lambda_ * I, X_aug.T @ y)
        return beta
