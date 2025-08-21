import numpy as np

def run_single_day_optimization(
    twap_pivot,           # DataFrame (index: timestamps, columns: symbols)
    alpha_pivot,           # DataFrame (same shape)
    hold_mask_pivot,     # DataFrame bool
    zero_mask_pivot,   # DataFrame bool
    initial_weights,       # np.array [n]
    max_trade_pivot,         # float or np.array [n]
    optimizer,             # PortfolioOptimizer 实例
    cost_func = None,               # 交易成本函数
    lambda_ = 0.01,                 # 交易成本系数
    max_turnover_ratio = 1.0,       # 最大交易比例
    initial_portfolio_value = 1,  # float
    solver=None,
    verbose=False,
    day_timestamps=None    # list-like of timestamps (单日)
):
    if day_timestamps is None:
        day_timestamps = twap_pivot.index.sort_values()

    buyhold_values = intraday_portfolio_values = [float(initial_portfolio_value)]
    buyhold_weights = intraday_weights = [np.asarray(initial_weights, dtype=float).copy()]
    cumulative_trade_pcts = [np.zeros(len(initial_weights), dtype=float)]

    if len(day_timestamps) < 2:
        return {
            "final_portfolio_value": portfolio_value,
            "final_weights": w0.copy(),
            "buyhold_final_value": buyhold_values,
            "buyhold_final_weights": buyhold_weights.copy(),
            "daily_cumulative_trade_pcts": cumulative_trade_pcts,
            "buyhold_daily_trades": buyhold_trades,
            "intraday_portfolio_values": intraday_portfolio_values,
            "intraday_weights": intraday_weights,
        }

    for i in range(1, len(day_timestamps)):
        prev_ts = day_timestamps[i - 1]
        current_ts = day_timestamps[i]

        prev_prices = twap_pivot.loc[prev_ts].values
        current_prices = twap_pivot.loc[current_ts].values
        returns = (current_prices - prev_prices) / prev_prices
        
        updated_intraday_weight = intraday_weights[-1] * (1.0 + returns)
        updated_intraday_weight /= updated_intraday_weight.sum()
        intraday_weights.append(updated_intraday_weight.copy())
        intraday_portfolio_values.append(intraday_portfolio_values[-1] * (1.0 + float(np.dot(updated_intraday_weight, returns))))

        updated_buyhold_weight = buyhold_weights[-1] * (1.0 + returns)
        updated_buyhold_weight /= updated_buyhold_weight.sum()
        buyhold_weights.append(updated_buyhold_weight.copy())
        buyhold_values.append(buyhold_values[-1] * (1.0 + float(np.dot(updated_buyhold_weight, returns))))
        
        alpha = alpha_pivot.loc[current_ts].values
        zero_mask = zero_mask_pivot.loc[current_ts].values
        hold_mask = hold_mask_pivot.loc[current_ts].values
        max_trade_pct = max_trade_pivot.loc[current_ts].values / intraday_portfolio_values[-1] # 当前时刻最大交易比例

        try:
            w_opt = optimizer.optimize(
                w0=intraday_weights[-1],
                alpha=alpha,
                max_trade_pct=max_trade_pct,
                hold_mask=hold_mask,
                zero_mask=zero_mask,
                cost_func=cost_func,
                lambda_=lambda_,
                initial_weights=buyhold_weights[-1],
                cumulative_trade_pct=cumulative_trade_pcts[-1],
                max_turnover_ratio=max_turnover_ratio,
                current_portfolio_value=intraday_portfolio_values[-1],
                initial_portfolio_value=buyhold_values[-1],
                solver=solver,
                verbose=verbose,
            )

    


    return {
        "final_portfolio_value": portfolio_value,
        "final_weights": w0.copy(),
        "buyhold_final_value": buyhold_values,
        "buyhold_final_weights": buyhold_weights.copy(),
        "daily_cumulative_trade_pcts": cumulative_trade_pcts,
        "buyhold_daily_trades": buyhold_trades,
        "intraday_portfolio_values": intraday_portfolio_values,
        "intraday_weights": intraday_weights,
    }