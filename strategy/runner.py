import numpy as np

def run_single_day_optimization(
    close_pivot,           # DataFrame (index: timestamps, columns: symbols)
    alpha_pivot,           # DataFrame (same shape)
    is_limit_up_pivot,     # DataFrame bool
    is_limit_down_pivot,   # DataFrame bool
    is_st_pivot,           # DataFrame bool
    initial_weights,       # np.array [n]
    initial_portfolio_value,  # float
    max_trade_pct,         # float or np.array [n]
    optimizer,             # PortfolioOptimizer 实例
    day_timestamps=None    # list-like of timestamps (单日)
):
    if day_timestamps is None:
        day_timestamps = close_pivot.index.sort_values()

    n_symbols = len(initial_weights)
    portfolio_value = float(initial_portfolio_value)
    w0 = np.asarray(initial_weights, dtype=float).copy()

    buyhold_value = float(initial_portfolio_value)
    buyhold_weights = w0.copy()

    intraday_portfolio_values = [portfolio_value]
    intraday_weights = [w0.copy()]

    cumulative_trades = np.zeros(n_symbols, dtype=float)
    buyhold_trades = np.zeros(n_symbols, dtype=float)

    if len(day_timestamps) < 2:
        return {
            "final_portfolio_value": portfolio_value,
            "final_weights": w0.copy(),
            "buyhold_final_value": buyhold_value,
            "buyhold_final_weights": buyhold_weights.copy(),
            "daily_cumulative_trades": cumulative_trades,
            "buyhold_daily_trades": buyhold_trades,
            "intraday_portfolio_values": intraday_portfolio_values,
            "intraday_weights": intraday_weights,
        }

    for i in range(len(day_timestamps) - 1):
        current_ts = day_timestamps[i]
        next_ts = day_timestamps[i + 1]

        alpha = alpha_pivot.loc[current_ts].values
        # True=可交易
        tradable_mask = ~(is_limit_up_pivot.loc[current_ts].values |
                          is_limit_down_pivot.loc[current_ts].values |
                          is_st_pivot.loc[current_ts].values)

        current_prices = close_pivot.loc[current_ts].values
        next_prices = close_pivot.loc[next_ts].values
        returns = (next_prices - current_prices) / current_prices

        try:
            w_opt = optimizer.optimize(
                w0=w0,
                alpha=alpha,
                max_trade_pct=max_trade_pct,
                initial_weights=initial_weights,
                cumulative_trades=cumulative_trades,
                optimizable_mask=tradable_mask,
                current_portfolio_value=portfolio_value,
                initial_portfolio_value=initial_portfolio_value,
            )

            # 同优化器一致的交易比例定义
            denom = np.maximum(initial_weights * initial_portfolio_value, 1e-8)
            trade_ratio = np.abs(w_opt - w0) * portfolio_value / denom
            cumulative_trades += trade_ratio

            # 用优化权重赚取下个时刻收益
            portfolio_value *= (1.0 + float(np.dot(w_opt, returns)))

            # 买入持有曲线
            buyhold_value *= (1.0 + float(np.dot(buyhold_weights, returns)))
            buyhold_weights = buyhold_weights * (1.0 + returns)
            buyhold_weights /= buyhold_weights.sum()

            if i > 0:
                buyhold_trades += np.abs(buyhold_weights - intraday_weights[-1])

            # 价格变动后更新当前权重（持有 w_opt 经过收益后的再归一）
            w0 = w_opt * (1.0 + returns)
            w0 /= w0.sum()

        except ValueError as e:
            # 失败时：保持权重，继续买入持有的曲线更新
            buyhold_value *= (1.0 + float(np.dot(buyhold_weights, returns)))
            buyhold_weights = buyhold_weights * (1.0 + returns)
            buyhold_weights /= buyhold_weights.sum()

        intraday_portfolio_values.append(portfolio_value)
        intraday_weights.append(w0.copy())

    return {
        "final_portfolio_value": portfolio_value,
        "final_weights": w0.copy(),
        "buyhold_final_value": buyhold_value,
        "buyhold_final_weights": buyhold_weights.copy(),
        "daily_cumulative_trades": cumulative_trades,
        "buyhold_daily_trades": buyhold_trades,
        "intraday_portfolio_values": intraday_portfolio_values,
        "intraday_weights": intraday_weights,
    }