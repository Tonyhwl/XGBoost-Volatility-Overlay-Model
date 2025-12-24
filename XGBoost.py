"""
===== Import Libraries ===== 
"""

import numpy as np, pandas as pd 
import yfinance as yf  
import xgboost as xgb  
from sklearn.preprocessing import StandardScaler  # feature scaling (no bias from large-magnitude features)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit  # auto hyperparameter tuning and time-series cross-validation 
from sklearn.metrics import make_scorer  # define  performance score for optimising hyperparameters 
import matplotlib.pyplot as plt  # for plotting graphs
from typing import Iterable, Tuple


"""
===== Main Parameters ===== 
"""

TICKERS = {
    "Bitcoin": {
        "name": "BTC-USD",  # Yahoo Finance ticker for Bitcoin in USD
        "trading_days": 365,  # cryptos trade every day
        "start_date": '2015-01-01',  # start date for historical data
        "train_end": '2020-12-31',  # end date for training data
        "cost_bps": 0.0002,  # fees in basis points (0.02% = 2 bps)
        "slippage_bps": 0.0002,  # slippage in basis points (2 bps)
        "spread_bps": 0.0002,  # round-trip spread (2 bps)
        "target_daily_vol": 0.015,  # target daily volatility (1.5%)
        "size_floor": 0.0,  # minimum position size
        "size_cap": 1.0,   # maximum position size (no leverage)
        "max_trade_fraction": 0.10,  # max fraction of portfolio to trade per day (10%)
        "horizon": 10,  # days for multi-day return target 
        "impact_k": 0.0005,  # square-root impact coefficient
        "ml_multiplier": 0.10,  # multiplier for model-based position sizing
        "deadband": 0.25,  # deadband for model predictions
        "drawdown_threshold": 0.25,  # drawdown threshold for crash protection
        "recovery_threshold": 0.10,  # recovery threshold for crash protection
        "ramp_down_factor": 0.80,  # ramp-down factor for high-volatility assets
        "min_pred_std": 1e-4,  # minimum prediction std to avoid degenerate scaling
        "min_trade_size": 0.005,  # minimum trade size to avoid tiny trades
        "grid_n_iter": 100,
        "turnover_penalty_weight": 0.1,
    }
}


""" 
===== XGBoost Hyperparameter Grid =====
"""

param_grid = {
    'learning_rate': [0.001, 0.005, 0.01, 0.05],  
    # controls how much each new tree changes the model 
    # smaller values learn more slowly but reduce overfitting

    'max_depth': [3, 5, 6, 7],  
    # limits how complex each decision tree can be
    # deeper trees can capture more patterns but may overfit

    'n_estimators': [500, 1000, 1500, 2000],  
    # number of trees added to model
    # more trees -> improve performance but increase training time + overfitting

    'min_child_weight': [1, 3, 5, 10],  
    # minimum data required to create a new split in a tree
    # larger values make the model less sensitive to noise

    'subsample': [0.5, 0.7, 0.8, 1.0],  
    # fraction of features used per tree
    # lower values add randomness and reduce overfitting

    'colsample_bytree': [0.6, 0.8, 1.0],  
    # fraction of features used per tree
    # prevents relying too heavily on a single feature

    'gamma' : [0, 0.1, 0.5, 1.0],  
    # minimum improvement required to make a split
    # higher values make the model less sensitive to noise

    'reg_lambda': [0, 0.5, 1, 10],  
    # penalises large weights in trees to reduce overfitting and complexity

    'reg_alpha': [0, 1, 2, 10]    
    # encourages sparsity in feature weights to reduce overfitting
}

initial_capital = 1000000   # initial capital of $1,000,000


"""
===== Helper Functions =====
- Realised Volatility
- Relative Strength Index (RSI)
- Pearson correlation
- Purged Time-Series Cross-Validation
- Cost-Adjusted Scorer
- Cumulative Returns and Drawdown
- Algorithm metrics (position sizing, turnover, costs, returns)
"""

# calculate realised volatility over a given window (in days)
def compute_realised_volatility(daily_returns, window_days, trading_days):

    # .rolling(window_days) builds a moving window of the last 30 days for each date
    # .std(ddof=0) computes the population standard deviation of the last 30 days for each date. 'ddof=1' is sample standard dev
    rolling_std = daily_returns.rolling(window_days).std(ddof=0)

    # converting volatility measured over a day into volatility measured over a year using square root of time rule
    # this rule assumes that the returns are independent and identically distributed (i.i.d)
    realised_vol_annualised = rolling_std * np.sqrt(trading_days)

    return realised_vol_annualised

def compute_rsi(data, window=14):
    """
    # RSI used to detect overbought/oversold and short-term momentum
    # measures recent average gains vs losses over a window (normally 14 days)
    """

    delta = data.diff()  # calculate price differences
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()  # positive gains
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()  # negative losses
    rs = gain / loss  # Relative Strength

    # avoid division by zero
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))  # RSI calculation

    # fill NaNs with neutral 50
    return rsi.fillna(50.0)

def pearson_corr(a, b):
    """
    # calculate Pearson correlation between two arrays 
    # measures how well model predictions align with future returns
    """

    a = np.asarray(a, dtype=float)  # convert inputs to numpy arrays 
    b = np.asarray(b, dtype=float)
    mask = (~np.isnan(a)) & (~np.isnan(b))  # boolean mask (true if both a,b are non-NaN)
    a = a[mask]
    b = b[mask]

    if a.size == 0:  # no valid paired data, return 0
        return 0.0
    
    sa = a.std(ddof=0)
    sb = b.std(ddof=0)

    if sa == 0 or sb == 0:   # zero std deviation, return 0
        return 0.0
    
    return ((a - a.mean()) * (b - b.mean())).mean() / (sa * sb)  # calculate Pearson correlation


def get_purged_splits(X, n_splits=7, purge=1, embargo=1, min_train_size: int = 30):
    """
    # uses TimeSeriesSplit to generate folds 
    # removes training indices that overlap the purge/embargo window around each test block.
    """
    n_samples = X.shape[0]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for train_index, test_index in tscv.split(X):
        test_start = test_index[0]
        test_end = test_index[-1]
        purge_start = max(0, test_start - purge)
        embargo_end = min(n_samples - 1, test_end + embargo)
        filtered_train = np.array([i for i in train_index if (i < purge_start) or (i > embargo_end)])
        # skip tiny/empty training splits to avoid unstable folds
        if len(filtered_train) < min_train_size or len(test_index) == 0:
            continue
        splits.append((filtered_train, test_index))
    return splits


def cost_adjusted_scorer(
    feature_cols: Iterable[str],
    trading_days: int,
    target_daily_vol: float,
    size_floor: float,
    size_cap: float,
    ml_multiplier: float,
    deadband: float,
    min_pred_std: float,
    turnover_penalty_weight: float,
):
    """
    # custom scorer that adjusts for transaction costs based on turnover
    """

    def scorer(estimator, X, y_true):
        try:
            if not isinstance(X, pd.DataFrame):
                # defensive: if shapes mismatch try to proceed but warn
                if hasattr(X, 'shape') and X.shape[1] != len(list(feature_cols)):
                    print('Warning: scorer received X with unexpected number of columns')
                X_df = pd.DataFrame(X, columns=list(feature_cols))
            else:
                X_df = X.copy()

            y_pred = estimator.predict(X)
            y_pred = np.asarray(y_pred)

            corr = pearson_corr(np.asarray(y_true), y_pred)
            corr_component = float(corr) * np.sqrt(trading_days)

            pred_std = np.nanstd(y_pred)
            pred_std = pred_std if (pred_std and not np.isnan(pred_std)) else min_pred_std
            pred_std = max(pred_std, min_pred_std)
            normalized = np.clip(y_pred / (pred_std + 1e-9), -3.0, 3.0)
            normalized_db = np.where(np.abs(normalized) < deadband, 0.0, normalized)
            ml_factor_local = 1.0 + (ml_multiplier * normalized_db)

            rv30 = X_df['rv_30'].values
            rv30_safe = np.where(rv30 <= 0, np.nan, rv30)
            base_pos_local = (target_daily_vol / (rv30_safe / np.sqrt(trading_days)))
            base_pos_local = np.nan_to_num(base_pos_local, nan=size_floor)
            base_pos_local = np.clip(base_pos_local, size_floor, size_cap)

            position_ml_local = np.clip(base_pos_local * ml_factor_local, size_floor, size_cap)
            lagged = np.concatenate(([size_floor], position_ml_local[:-1]))
            turnover = np.abs(position_ml_local - lagged)
            annual_turnover = np.nanmean(turnover) * trading_days

            score = corr_component - (turnover_penalty_weight * annual_turnover)
            return float(score)
        except Exception:
            return 0.0

    return scorer


def compute_cumulative_and_drawdown(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    cumulative = (1 + series).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return cumulative, drawdown


def plot_cumulative(dates, *series_pairs, title: str, filename: str, show: bool = False):
    plt.figure(figsize=(12, 6))
    for vals, label, color, lw in series_pairs:
        plt.plot(dates, vals, label=label, color=color, linewidth=(lw if lw is not None else 1))
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (Initial = 1.0)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    if show:
        plt.show()


def plot_drawdown(dates, *draw_pairs, title: str, filename: str, show: bool = False):
    plt.figure(figsize=(12, 6))
    for vals, label, color in draw_pairs:
        plt.plot(dates, vals * 100, label=label, color=color, alpha=0.7)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    if show:
        plt.show()


"""
Performance metric helper functions
- Sharpe Ratio
- Omega Ratio
- Sortino Ratio
- CAGR
- Max Drawdown  
- Annualised Volatility
- Information Ratio
- Win Rate
- Average Turnover
- Cumulative Return and Percentage
"""

def sharpe_daily(returns: pd.Series, trading_days: int) -> float:
    r = returns.dropna()
    if len(r) == 0 or r.std(ddof=0) == 0:
        return np.nan
    return (r.mean() / r.std(ddof=0)) * np.sqrt(trading_days)

def sortino_ratio(returns: pd.Series, trading_days: int) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    downside = r[r < 0]
    downside_std = downside.std(ddof=0) if len(downside) > 0 else np.nan
    if not downside_std or np.isnan(downside_std) or downside_std == 0:
        return np.nan
    return (r.mean() / downside_std) * np.sqrt(trading_days)

def omega_ratio(returns: pd.Series, threshold: float = 0):
    gains = returns[returns > threshold]
    losses = returns[returns < threshold]
    return gains.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan

def cagr(initial_value, final_value, years):
    if years <= 0:
        return np.nan
    return (final_value / initial_value) ** (1 / years) - 1

def max_drawdown(drawdown_series: pd.Series) -> float:
    return drawdown_series.min()

def annualised_vol(returns: pd.Series, trading_days: int) -> float:
    r = returns.dropna()
    return r.std(ddof=0) * np.sqrt(trading_days) if len(r) > 0 else np.nan

def win_rate_pct(returns: pd.Series) -> float:
    r = returns.dropna()
    return (r > 0).sum() / max(1, len(r))

def average_turnover(turnover_series: pd.Series) -> float:
    return turnover_series.mean() if turnover_series is not None else 0.0

def cumulative_return_and_pct(returns: pd.Series) -> tuple:
    r = returns.fillna(0)
    if len(r) == 0:
        return 1.0, 0.0
    equity = (1 + r).cumprod().iloc[-1]
    return equity, (equity - 1.0) * 100.0


"""
Calculate lagged position, turnover, transaction costs and strategy returns.
- `spread_bps`: round-trip spread cost applied per unit turnover
- `slippage_bps`: base slippage per turnover (existing parameter)
- `impact_k`: square-root market impact coefficient
- the dynamic slippage added is `impact_k * sqrt(turnover)` which captures larger slippage for bigger trades

All cost terms are applied proportionally to turnover
We use a realistic fixed value for spread, slippage and impact without the need for order book simulation
"""

def calculate_algo_metrics(
    df,  # dataframe with position and returns columns
    position_col,  # fraction of portfolio
    returns_col,  # daily returns
    cost_bps,  # transaction fee rate per turnover
    slippage_bps=0.0,  # base slippage per turnover
    spread_bps=0.0,  # round-trip spread cost per turnover
    impact_k=0.0,  # square-root impact coefficient
    fill_value=None,  # initial position for lagged calculation (None -> use size_floor if provided)
    max_trade_fraction=1.0,  # max fraction of portfolio to trade per day
    min_trade_size=0.0,  # ignore trades smaller than this fraction of portfolio
):

    # previous day's target (what would be executed without execution limits)
    # default fill_value: if None, use 0.0 (no initial exposure) to avoid accidental full exposure
    if fill_value is None:
        fill_value = 0.0
    prev_target = df[position_col].shift(1).fillna(fill_value)

    # desired change in position (based on today's target minus yesterday's executed)
    desired_change = df[position_col] - prev_target

    # limit the executed change per day to `max_trade_fraction` to simulate limited liquidity / pace-in
    abs_change = desired_change.abs()
    capped_vals = abs_change.clip(upper=max_trade_fraction)
    capped_change = np.sign(desired_change) * capped_vals
    # ignore tiny trades below `min_trade_size` to reduce churn
    if min_trade_size and min_trade_size > 0:
        capped_change = capped_change.mask(capped_vals < min_trade_size, 0.0)

    # executed (lagged) position after applying daily trade cap
    lagged_pos = prev_target + capped_change

    # turnover is absolute executed change
    turnover = (lagged_pos - prev_target).abs().fillna(0)

    # dynamic slippage model: base slippage + square-root impact based on turnover
    # using sqrt(turnover) keeps units sensible and increases slippage for larger trades
    dynamic_slippage = slippage_bps + (impact_k * np.sqrt(turnover))

    # spread cost (round-trip) is applied per unit turnover as well
    # total cost per unit turnover combines explicit fees, slippage and spread
    total_cost_per_unit = cost_bps + dynamic_slippage + spread_bps

    # total cost for the day (in fraction of portfolio) is cost per unit * turnover
    cost = total_cost_per_unit * turnover

    # Net Returns: (Asset Return * Executed Position) - Cost
    algo_ret = (df[returns_col] * lagged_pos) - cost

    return lagged_pos, turnover, cost, algo_ret


"""
===== Main Function =====
- download data
- prepare features
- train XGBoost model
- size positions
- evaluate performance
- plot results
"""

def run_strategy(
    ticker,
    trading_days,
    start_date,
    train_end,
    cost_bps,
    target_daily_vol,
    size_floor,
    size_cap,
    max_trade_fraction,
    ml_multiplier=0.05,
    deadband=0.25,
    horizon=1,
    slippage_bps=0.0002,
    spread_bps=0.0001,
    impact_k=0.01,
    grid_n_iter=100,
    min_pred_std=1e-4,
    min_trade_size=0.0,
    ramp_down_factor=0.65,
    drawdown_threshold=0.15,
    recovery_threshold=0.05,
    turnover_penalty_weight=0.05,
    show_plots: bool = True,
):

    # ========== Data Download ==========
    # yf.download() returns dataframe indexed by trading dates
    # we only want the close column from 'Open, High, Low, Close, Adj Close, Volume'
    # Realised Volatility is calculated from daily close-to-close returns
    price_data = yf.download(
        ticker,
        start=start_date,
        auto_adjust=True, # 'auto_adjust = True' adjusts the prices for splits and dividends
        progress=False
    )[["Close"]].dropna()

    # yahoo has holes with empty data (holidays, partial days etc.)
    # 'px.dropna()' removes any rows with empty data
    price_data = price_data.dropna()
    # computes simple daily returns: (today's closing price - yesterday's closing price) / yesterday's closing price
    daily_returns = price_data['Close'].pct_change()  


    # ========== Data Preparation =========
    # analysis_df is our main dataframe for features and labels (like an Excel spreadsheet)
    analysis_df = pd.DataFrame(index=price_data.index)

    # closing price
    analysis_df["close"] = price_data["Close"]  

    # --- Exponential Moving Average (EMA) ---
    analysis_df['ema_5'] = analysis_df['close'].ewm(span=5, adjust=False).mean()
    analysis_df['ema_20'] = analysis_df['close'].ewm(span=20, adjust=False).mean()
    analysis_df['ema_50'] = analysis_df['close'].ewm(span=50, adjust=False).mean()
    analysis_df['ema_100'] = analysis_df['close'].ewm(span=100, adjust=False).mean()

    # --- Moving Average Convergence Divergence (MACD) ---
    analysis_df['macd'] = analysis_df['close'].ewm(span=12).mean() - analysis_df['close'].ewm(span=26).mean()
    # compute MACD Signal Line (9-day EMA of MACD)
    analysis_df['macd_signal'] = analysis_df['macd'].ewm(span=9).mean()
    
    # --- Trend Signal ---
    # 1 if short-term EMA > long-term EMA, else 0
    analysis_df['trend_signal'] = (analysis_df['ema_20'] > analysis_df['ema_50']).astype(int)

    # --- Realised Volatility (RV) ---
    analysis_df["daily_returns"] = daily_returns.fillna(0)
    analysis_df["rv_30"] = compute_realised_volatility(daily_returns, 30, trading_days)
    analysis_df["rv_30_daily"] = analysis_df["rv_30"] / np.sqrt(trading_days)
    analysis_df[f'rv_{horizon}'] = compute_realised_volatility(daily_returns, horizon, trading_days)

    # --- Volatility Jump ---
    analysis_df['vol_jump'] = analysis_df[f'rv_{horizon}'] / analysis_df["rv_30"]

    # create a regression target: multi-day return over `horizon` days
    # use log-returns for numerical stability: multi-day = exp(sum(log1p(returns))) - 1
    log_ret = np.log1p(analysis_df['daily_returns'].replace(-1, np.nan)).fillna(0.0)
    analysis_df[f'target_return_{horizon}d'] = (
        np.expm1(log_ret.rolling(window=horizon).sum()).shift(-horizon)
    )
    analysis_df['target_return'] = analysis_df[f'target_return_{horizon}d']

    # --- Position ML Feature ---
    analysis_df['position_ml_feature'] = np.exp(-0.1 * analysis_df['rv_30'])

    # --- Exponential Weighted Moving Average (EWMA) ---
    # smoothing factor (lambda) controls how much weight is placed on recent data.
    # higher lambda (closer to 1) places more weight on recent data, making the model more responsive.
    lambda_ = 0.85
    # calculates a moving average where recent data points are given more weight than older ones.
    analysis_df["ewma_vol"] = daily_returns.ewm(alpha=1 - lambda_).std() * np.sqrt(trading_days)
     # best for capturing the most up-to-date trends, like volatility, in financial time series.

    # ---- Relative Strength Index (RSI) ----
    analysis_df["rsi_14"] = compute_rsi(analysis_df["close"], window=14)
    analysis_df["rsi_5"] = compute_rsi(analysis_df["close"], window=5)
    
    # ---- Rate of Change (ROC) ----
    analysis_df['roc_10'] = analysis_df['close'].pct_change(10)  # 10-day ROC


    # ========== Feature Engineering ==========
    feature_cols = ["rv_30", "rv_30_daily", "ewma_vol", "position_ml_feature", "rsi_14","trend_signal", "macd_signal", "rsi_5", "roc_10"]

    # shift features once by the `horizon` to ensure strict precedence to target
    analysis_df[feature_cols] = analysis_df[feature_cols].shift(horizon)

    # =========== ML Labels ============
    # drop any rows with NaN values in feature columns or the target (do this BEFORE splitting)
    analysis_df = analysis_df.dropna(subset=feature_cols + ['target_return'])

    # convert annualised 30d realised volatility to daily volatility (guard zeros)
    rv30_daily_safe = analysis_df["rv_30_daily"].replace(0, np.nan)
    analysis_df["position_iv"] = (target_daily_vol / rv30_daily_safe).clip(size_floor, size_cap).fillna(size_floor)
    # smooth inverse-vol benchmark to reduce churn and make it realistic
    analysis_df["position_iv"] = analysis_df["position_iv"].ewm(span=200, adjust=False).mean()    


    """
    ===== Model Training =====
    - split data into training and testing sets based on date
    - scale features using StandardScaler
    - use Purged Time-Series Cross-Validation to avoid leakage
    - define Sharpe-based scorer for hyperparameter tuning
    ️- train XGBoost regression model to predict next-day returns
    ️- predict next-day returns for both training and testing sets
    ️- store predictions in analysis_df
    """

    # split the data into training and testing sets based on the date
    train = analysis_df.loc[analysis_df.index <= pd.to_datetime(train_end)]
    test = analysis_df.loc[analysis_df.index > pd.to_datetime(train_end)]

    # drop rows with missing target (caused by horizon shift)
    train = train.dropna(subset=['target_return'])
    test = test.dropna(subset=['target_return'])
    
    """" 
    - separate features and target variable
    - features are the engineered columns
    - target variable is the multi-day return
    - we will train the model to predict this target
    - using only past data to predict future returns
    - this avoids look-ahead bias
    """

    x_train = train[feature_cols]
    x_test = test[feature_cols]

    # use regression target (next-day return) for training
    y_train = train['target_return']
    y_test = test['target_return']


    """
    What is Purged Time-Series CV?
 
    """

    # generate purged splits relative to the training set used for grid search
    cv_method = get_purged_splits(train, n_splits=5, purge=horizon, embargo=horizon, min_train_size=30)

    # prepare scoring
    scoring_used = cost_adjusted_scorer(
            feature_cols=feature_cols,
            trading_days=trading_days,
            target_daily_vol=target_daily_vol,
            size_floor=size_floor,
            size_cap=size_cap,
            ml_multiplier=ml_multiplier,
            deadband=deadband,
            min_pred_std=min_pred_std,
            turnover_penalty_weight=turnover_penalty_weight,
        )

    """
    ===== XGBoost Regressor =====
    - use RandomizedSearchCV to find best hyperparameters based on Sharpe score
    - fit the model on training data
    """

    # initialize XGBoost regressor
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        n_jobs=-1,
        random_state=42
    )

    # create a pipeline that first scales features then fits the XGBoost model
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', xgb_model)])

    # map param grid to pipeline parameter names 
    param_grid_pipeline = {f"model__{k}": v for k, v in param_grid.items()}

    n_iter_search = grid_n_iter
    n_iter_search = grid_n_iter

    
    # set up randomised search over hyperparameters
    grid_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid_pipeline,
        scoring=scoring_used,
        cv=cv_method,
        verbose=0,
        n_iter=n_iter_search,
        n_jobs=-1,
        random_state=42,
    )

    # run randomised search to find best hyperparameters (fit on raw X; pipeline handles scaling)
    grid_search.fit(x_train, y_train)
    print(f"Best parameters found: {grid_search.best_params_}")
    
    # get the best model from the grid search
    best_pipeline = grid_search.best_estimator_

    # predict returns for both training and testing sets
    y_pred_test = best_pipeline.predict(x_test)
    y_pred_train = best_pipeline.predict(x_train)
    
    # store predictions in analysis_df
    analysis_df.loc[test.index, 'pred_return'] = y_pred_test
    analysis_df.loc[train.index, 'pred_return'] = y_pred_train
    analysis_df['pred_return'] = analysis_df['pred_return'].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # ----- Feature importances  -----
    try:
        model_for_fi = best_pipeline.named_steps.get('model') if hasattr(best_pipeline, 'named_steps') else None
        if model_for_fi is not None and hasattr(model_for_fi, 'feature_importances_'):
            fi_vals = model_for_fi.feature_importances_
            fi_df = pd.DataFrame({'feature': feature_cols, 'importance': fi_vals})

            # normalize to percent and sort
            total = fi_df['importance'].sum()
            fi_df['importance_pct'] = (fi_df['importance'] / total * 100) if total > 0 else 0.0
            fi_df = fi_df.sort_values('importance_pct', ascending=False).reset_index(drop=True)
            fi_filename = f"feature_importances_{ticker.replace('/', '_').replace(' ', '_')}.csv"
            fi_df.to_csv(fi_filename, index=False)

            # print feature importances
            print('\nFeature importances (%):')
            print(fi_df[['feature', 'importance_pct']].to_string(index=False, formatters={'importance_pct': lambda x: f"{x:.2f}%"}))
        else:
            print('Feature importances not available for the fitted model')
    except Exception as _fi_err:
        print('Error computing feature importances:', _fi_err)



    """
    ===== Position Sizing =====
    - Base position sizing using inverse of rolling volatility
    - Adjust position size based on normalized model predictions
    - Smooth positions using EMA
    - Calculate preliminary returns for ML strategy
    - Implement crash protection based on drawdown and volatility jump signals
    """

    # base position sizing using inverse of realised volatility
    # avoid division by zero by replacing 0 with a tiny number before division
    rv_safe = (analysis_df['rv_30_daily'].abs().replace(0, np.nan)).clip(lower=1e-9)
    base_pos = (target_daily_vol / rv_safe).clip(size_floor, size_cap)
    base_pos = base_pos.replace([np.inf, -np.inf], size_cap).fillna(size_floor)

    # normalize predictions using training prediction std to get stable scaling
    pred_train_std = np.nanstd(y_pred_train) if 'y_pred_train' in locals() else np.nanstd(analysis_df.loc[train.index, 'pred_return'].dropna())

    # guard against collapsed prediction std (floor) and clip normalized range
    pred_train_std = pred_train_std if (pred_train_std and not np.isnan(pred_train_std)) else min_pred_std
    pred_train_std = max(pred_train_std, min_pred_std)
    analysis_df['normalized_pred'] = (analysis_df['pred_return'] / (pred_train_std + 1e-9)).fillna(0.0)
    # clip to avoid extreme ml_factors coming from degenerate predictions
    analysis_df['normalized_pred'] = analysis_df['normalized_pred'].clip(-3.0, 3.0)

    # apply deadband to avoid trading on tiny noisy signals and lower multiplier to reduce churn
    analysis_df['normalized_pred_db'] = analysis_df['normalized_pred'].where(analysis_df['normalized_pred'].abs() >= deadband, 0.0)

    # create factor from predicted returns: increase/decrease base position by factor of normalized prediction
    ml_factor = 1.0 + (ml_multiplier * analysis_df['normalized_pred_db'])

    # adjust positions and smooth with a stronger EMA to reduce reactivity and turnover
    analysis_df['position_ml'] = (base_pos * ml_factor)
    analysis_df['position_ml'] = analysis_df['position_ml'].ewm(span=200, adjust=False).mean()
    analysis_df['position_ml'] = analysis_df['position_ml'].fillna(1.0)
    analysis_df['position_ml'] = analysis_df['position_ml'].clip(size_floor, size_cap)

    # define Preliminary Returns (to generate the drawdown signal)
    analysis_df["ret_bh"] = analysis_df["daily_returns"]  # Buy & Hold benchmark

    # calculate preliminary metrics for ML strategy
    # include spread and impact parameters for a more realistic execution model
    analysis_df["position_ml_lag_pre"], analysis_df["turnover_ml_pre"], analysis_df["cost_ml_pre"], analysis_df["ret_ml"] \
        = calculate_algo_metrics(
            analysis_df,
            "position_ml",
            "daily_returns",
            cost_bps,
            slippage_bps=slippage_bps,
            spread_bps=spread_bps,
            impact_k=impact_k,
            fill_value=size_floor,
            max_trade_fraction=max_trade_fraction,
            min_trade_size=min_trade_size,
        )

    # calculate metrics for IV strategy
    analysis_df["position_iv_lag"], analysis_df["turnover_iv"], analysis_df["cost_iv"], analysis_df["ret_iv"] \
        = calculate_algo_metrics(
            analysis_df,
            "position_iv",
            "daily_returns",
            cost_bps,
            slippage_bps=slippage_bps,
            spread_bps=spread_bps,
            impact_k=impact_k,
            fill_value=size_floor,
            max_trade_fraction=max_trade_fraction,
            min_trade_size=min_trade_size,
        )


    # ======= Reducing Position Size ========

    # calculate drawdown metrics
    for strategy in ['ret_bh', 'ret_iv', 'ret_ml']:
        test_returns = analysis_df.loc[test.index, strategy].fillna(0)
        test_cumulative = (1 + test_returns).cumprod()
        # store only the test period results back into the DataFrame
        analysis_df.loc[test.index, f'{strategy}_cumulative_test'] = test_cumulative

        # calculate running max and drawdown using the test period only
        test_running_max = test_cumulative.cummax()
        test_drawdown = (test_cumulative - test_running_max) / test_running_max

        analysis_df.loc[test.index, f'{strategy}_drawdown_test'] = test_drawdown

        analysis_df.loc[test.index, f'{strategy}_cumulative'] = analysis_df.loc[
            test.index, f'{strategy}_cumulative_test']
        analysis_df.loc[test.index, f'{strategy}_drawdown'] = analysis_df.loc[test.index, f'{strategy}_drawdown_test']

    # final_ml_position starts as the smoothed ML position 
    final_ml_position = analysis_df['position_ml'].copy()

    # -------- Ramp-Down Factor --------
    """
    # for high-volatility assets we reduce exposure by `ramp_down_factor`
    # when strategy drawdown exceeds `drawdown_threshold` and resume when drawdown recovers above `recovery_threshold`
    """
    dd_series = analysis_df.loc[test.index, 'ret_ml_drawdown_test'].fillna(0.0)
    locked = pd.Series(False, index=dd_series.index)
    is_locked = False
    for idx in dd_series.index:
        if (not is_locked) and (dd_series.loc[idx] < -drawdown_threshold):
            is_locked = True
        elif is_locked and (dd_series.loc[idx] >= -recovery_threshold):
            is_locked = False
        locked.loc[idx] = is_locked

    # apply ramp-down to final positions during locked periods 
    locked_idx = locked[locked].index
    if len(locked_idx) > 0:
        final_ml_position.loc[locked_idx] = final_ml_position.loc[locked_idx] * ramp_down_factor

    # enforce bounds
    final_ml_position = final_ml_position.clip(size_floor, size_cap) 

    # assign final ML positions back
    analysis_df.loc[test.index, "position_ml"] = final_ml_position

    # recalculate ML returns with final positions
    analysis_df["position_ml_lag"], analysis_df["turnover_ml"], analysis_df["cost_ml"], analysis_df["ret_ml"] = \
        calculate_algo_metrics(
            analysis_df,
            "position_ml",
            "daily_returns",
            cost_bps,
            slippage_bps=slippage_bps,
            spread_bps=spread_bps,
            impact_k=impact_k,
            fill_value=size_floor,
            max_trade_fraction=max_trade_fraction,
            min_trade_size=min_trade_size,
        )

    # recompute ML cumulative and drawdown on the test period
    test_returns_ml = analysis_df.loc[test.index, 'ret_ml'].fillna(0)
    test_cumulative_ml = (1 + test_returns_ml).cumprod()
    analysis_df.loc[test.index, 'ret_ml_cumulative_test'] = test_cumulative_ml
    test_running_max_ml = test_cumulative_ml.cummax()
    test_drawdown_ml = (test_cumulative_ml - test_running_max_ml) / test_running_max_ml
    analysis_df.loc[test.index, 'ret_ml_drawdown_test'] = test_drawdown_ml
    analysis_df.loc[test.index, 'ret_ml_cumulative'] = test_cumulative_ml
    analysis_df.loc[test.index, 'ret_ml_drawdown'] = test_drawdown_ml


    """
    ===== Performance Evaluation =====
    - Calculate key metrics
    - Print backtest summary
    - Plot cumulative returns and drawdowns
    """

    # Annualised Transaction Costs
    cost_ml_test = analysis_df.loc[test.index, "cost_ml"].fillna(0)
    cost_iv_test = analysis_df.loc[test.index, "cost_iv"].fillna(0)
    annual_cost_ml = cost_ml_test.mean() * trading_days
    annual_cost_iv = cost_iv_test.mean() * trading_days

    # ----- BACKTEST SUMMARY :) -----

    print(f"\n=== Backtest Summary for {ticker} (Test period) ===")

    # filter the returns series to only include the test period
    ret_bh_test = analysis_df.loc[test.index, "ret_bh"].dropna()
    ret_iv_test = analysis_df.loc[test.index, "ret_iv"].dropna()
    ret_ml_test = analysis_df.loc[test.index, "ret_ml"].dropna()

    strategies = [
        ("Buy & Hold", ret_bh_test),
        ("Inv-Vol", ret_iv_test),
        ("ML Vol", ret_ml_test),
    ]

    results = []

    for name, series in strategies:
        equity_curve = (1 + series).cumprod().iloc[-1]
        years = len(series) / trading_days
        final_dollar_value = initial_capital * equity_curve

        if name == "Buy & Hold":
            avg_turnover = 0.0
            annual_cost = 0.0
            max_dd = max_drawdown(analysis_df.loc[test.index, "ret_bh_drawdown"])
        elif name == "Inv-Vol":
            avg_turnover = average_turnover(analysis_df.loc[test.index, "turnover_iv"])
            annual_cost = annual_cost_iv
            max_dd = max_drawdown(analysis_df.loc[test.index, "ret_iv_drawdown"])
        elif name == "ML Vol":
            avg_turnover = average_turnover(analysis_df.loc[test.index, "turnover_ml"])
            annual_cost = annual_cost_ml
            max_dd = max_drawdown(analysis_df.loc[test.index, "ret_ml_drawdown"])

        sharpe = sharpe_daily(series, trading_days)
        annual_vol = annualised_vol(series, trading_days)
        sortino = sortino_ratio(series, trading_days)
        omega = omega_ratio(series)
        cagr_value = cagr(1.0, equity_curve, years)
        win_rate_val = win_rate_pct(series)
        avg_win = series[series > 0].mean() if (series[series > 0].size > 0) else 0.0
        avg_loss = abs(series[series < 0].mean()) if (series[series < 0].size > 0) else 0.0
        annual_turnover = avg_turnover * trading_days
        _, cumulative_return_pct = cumulative_return_and_pct(series)

        results.append({
            'ticker': ticker,
            'strategy': name,
            'sharpe': sharpe,
            'sortino': sortino,
            'cagr': cagr_value,
            'ann_vol': annual_vol,
            'max_dd': max_dd,
            'win_rate': win_rate_val,
            'annual_turnover': annual_turnover,
            'annual_cost': annual_cost,
            'final_portfolio': final_dollar_value,
            'cum_return_pct': cumulative_return_pct,
            'omega': omega,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
        })

    # present results
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        primary_cols = ['strategy', 'sharpe', 'sortino', 'cagr', 'ann_vol', 'max_dd', 'win_rate', 'annual_turnover', 'annual_cost', 'final_portfolio', 'cum_return_pct']
        secondary_cols = ['strategy', 'omega', 'avg_win', 'avg_loss']

        primary_df = results_df[primary_cols].copy()
        secondary_df = results_df[secondary_cols].copy()

        # formatting
        primary_metrics = {
            'sharpe': lambda x: f"{x:.3f}" if pd.notnull(x) else 'nan',
            'sortino': lambda x: f"{x:.3f}" if pd.notnull(x) else 'nan',
            'cagr': lambda x: f"{x:.2%}" if pd.notnull(x) else 'nan',
            'ann_vol': lambda x: f"{x:.2%}" if pd.notnull(x) else 'nan',
            'max_dd': lambda x: f"{x:.2%}" if pd.notnull(x) else 'nan',
            'win_rate': lambda x: f"{x:.2%}" if pd.notnull(x) else 'nan',
            'annual_turnover': lambda x: f"{x:.2%}" if pd.notnull(x) else 'nan',
            'annual_cost': lambda x: f"{x:.2%}" if pd.notnull(x) else 'nan',
            'final_portfolio': lambda x: f"${x:,.0f}",
            'cum_return_pct': lambda x: f"{x:.2f}%",
        }

        secondary_metrics = {
            'info_ratio': lambda x: f"{x:.3f}" if pd.notnull(x) else 'nan',
            'omega': lambda x: f"{x:.3f}" if pd.notnull(x) else 'nan',
            'avg_win': lambda x: f"{x:.4f}" if pd.notnull(x) else 'nan',
            'avg_loss': lambda x: f"{x:.4f}" if pd.notnull(x) else 'nan',
        }

        print('\n=== Primary Metrics ===')
        print(primary_df.to_string(index=False, formatters=primary_metrics))

        print('\n=== Secondary Metrics ===')
        print(secondary_df.to_string(index=False, formatters=secondary_metrics))

    """ 
    ===== Plotting Results ===== 
    - Cumulative Returns Plot
    - Drawdown Plot
    """

    # save the dataframe 
    analysis_df.loc[test.index, ['ret_bh', 'ret_iv', 'ret_ml']].to_csv('backtest_returns.csv')

    cumulative_bh_test = (1 + ret_bh_test).cumprod()
    cumulative_iv_test = (1 + ret_iv_test).cumprod()
    cumulative_ml_test = (1 + ret_ml_test).cumprod()

    # --- Cumulative Returns Plot ---
    plot_cumulative(
        cumulative_bh_test.index,
        (cumulative_bh_test.values, 'Buy & Hold', 'blue', None),
        (cumulative_iv_test.values, 'Inverse Volatility', 'green', None),
        (cumulative_ml_test.values, 'XGBoost Volatility', 'red', 2),
        title=f"Cumulative Returns for {ticker} (Test Period: {test.index[0].strftime('%Y-%m-%d')} to {test.index[-1].strftime('%Y-%m-%d')})",
        filename='cumulative_returns_final.png',
        show=show_plots,
    )

    # --- Drawdown Plot ---
    drawdown_bh_test = analysis_df.loc[test.index, "ret_bh_drawdown_test"]
    drawdown_iv_test = analysis_df.loc[test.index, "ret_iv_drawdown_test"]
    drawdown_ml_test = analysis_df.loc[test.index, "ret_ml_drawdown_test"]

    plot_drawdown(
        drawdown_bh_test.index,
        (drawdown_bh_test.values, 'Buy & Hold', 'blue'),
        (drawdown_iv_test.values, 'Inverse Volatility', 'green'),
        (drawdown_ml_test.values, 'XGBoost Volatility', 'red'),
        title=f"Drawdown Comparison for {ticker} (Test Period)",
        filename='drawdown_plot_final.png',
        show=show_plots,
    )

    # return a DataFrame summarising results for this ticker
    try:
        return pd.DataFrame(results)
    except Exception:
        return pd.DataFrame(results)


"""
===== Run Strategy =====
"""

all_summaries = []


def run_strategy_from_cfg(cfg: dict):

    # accept `name` as ticker and pass allowed keys to run_strategy.
    cfg_copy = dict(cfg)  
    if 'name' in cfg_copy:
        cfg_copy['ticker'] = cfg_copy.pop('name')

    allowed = {
        'ticker', 'trading_days', 'start_date', 'train_end',
        'cost_bps', 'target_daily_vol', 'size_floor', 'size_cap',
        'max_trade_fraction', 'slippage_bps', 'spread_bps', 'impact_k',
        'ml_multiplier', 'deadband', 'horizon', 'grid_n_iter',
        'min_pred_std', 'min_trade_size', 'ramp_down_factor',
        'drawdown_threshold', 'recovery_threshold', 'turnover_penalty_weight'
    }

    keyword_arguments = {k: v for k, v in cfg_copy.items() if k in allowed}
    return run_strategy(**keyword_arguments)


for label, cfg in TICKERS.items():
    print(f"\nRunning backtest for {label} ({cfg.get('name')})...")
    try:
        df_summary = run_strategy_from_cfg(cfg)
        if df_summary is not None and not df_summary.empty:
            all_summaries.append(df_summary)
    except Exception as e:
        print(f"Error running {label}: {e}")








