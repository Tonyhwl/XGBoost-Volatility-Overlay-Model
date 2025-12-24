# XGBoost-Volatility-Overlay-Model

## Strategy Summary
- Trains an XGBoost model on multi-horizon realised volatility, EWMA, momentum and RSI features  
- Predicts short- to medium-horizon return pressure to scale position size  
- Applies volatility targeting, deadband filtering, drawdown-based risk controls and turnover limits  
- Rebalances daily with strict execution and risk constraints

---

## Why Bitcoin?
- Highly liquid and traded 24/7
- Significant volatility provides opportunities for ML models
- 6+ years of stable historical data available from Yahoo Finance
- Main cryptocurrency with extensive market interest

---

## Why XGBoost?
- Powerful and efficient gradient-boosting library
- Handles tabular data well (like financial time series)
- Offers flexibility with hyperparameter tuning
Since we have 6+ years of historical data for Bitcoin, XGBoost is my chosen model for this strategy.

---

## Backtesting & Results
**Out-of-sample period:** 2021–2025  
**Asset:** Bitcoin (BTC-USD)

| Strategy | Sharpe | CAGR | Max Drawdown | Annual Vol | Turnover |
|--------|--------|------|--------------|------------|----------|
| Buy & Hold | 0.68 | 25.8% | −76.6% | 58.7% | 0% |
| Inverse-Vol | 0.82 | 23.7% | −46.0% | 32.2% | ~0% |
| ML Vol Overlay | 0.85 | 28.7% | −50.4% | 38.6% | ~11% |

---

## Execution & Cost Assumptions
All backtests incorporate realistic execution modelling:
- Transaction fees, bid-ask spread and slippage (approximations)
- Square-root market impact calibrated to BTC liquidity  
- Maximum daily trade size and minimum trade thresholds  
- No leverage

---

## Validation & Robustness
- Strict time-ordered training and out-of-sample testing  
- No look-ahead bias  
- Conservative turnover constraints  
- Sensitivity checks across cost, impact and deadband parameters

---

## Additional details


