import numpy as np

def run_single_day_optimization(
    twap_pivot,           # DataFrame (index: timestamps, columns: symbols)
    alpha_pivot,           # DataFrame (same shape)
    hold_mask_pivot,     # DataFrame bool
    zero_mask_pivot,   # DataFrame bool
    initial_weights,       # np.array [n]
    max_trade_pivot,         # float or np.array [n]
    optimizer,             # PortfolioOptimizer 实例
    slippage_1=0.0,           # 滑点1，单位为bp (基点) 日内交易的滑点
    slippage_2=0.0,           # 滑点2，单位为bp (基点) 最终(反向交易)的滑点
    cost_func = None,               # 交易成本函数
    lambda_ = 0.01,                 # 交易成本系数
    max_turnover_ratio = 1.0,       # 最大交易比例
    initial_portfolio_value = 1,  # float
    solver=None,
    verbose=False,
    day_timestamps=None,    # list-like of timestamps (单日)
):
    if day_timestamps is None:
        day_timestamps = twap_pivot.index.sort_values()

    buyhold_values = [float(initial_portfolio_value)]
    intraday_portfolio_values = [float(initial_portfolio_value)]
    buyhold_weights = [np.asarray(initial_weights, dtype=float).copy()]
    intraday_portfolio_weights = [np.asarray(initial_weights, dtype=float).copy()]
    cumulative_trade_pcts = [np.zeros(len(initial_weights), dtype=float)]
    total_trading_frictions = [0.0]

    for i in range(1, len(day_timestamps)):
        prev_ts = day_timestamps[i - 1]
        current_ts = day_timestamps[i]

        prev_prices = twap_pivot.loc[prev_ts].values
        current_prices = twap_pivot.loc[current_ts].values
        returns = (current_prices - prev_prices) / prev_prices
        
        buyhold_values.append(buyhold_values[-1] * (1.0 + float(np.dot(buyhold_weights[-1], returns))))
        updated_buyhold_weight = buyhold_weights[-1] * (1.0 + returns)
        updated_buyhold_weight /= updated_buyhold_weight.sum()
        buyhold_weights.append(updated_buyhold_weight.copy())
        
        intraday_portfolio_values.append(intraday_portfolio_values[-1] * (1.0 + float(np.dot(intraday_portfolio_weights[-1], returns))))
        updated_intraday_weight = intraday_portfolio_weights[-1] * (1.0 + returns)
        updated_intraday_weight /= updated_intraday_weight.sum()
        intraday_portfolio_weights.append(updated_intraday_weight.copy())
        
        alpha = alpha_pivot.loc[current_ts].values
        zero_mask = zero_mask_pivot.loc[current_ts].values
        hold_mask = hold_mask_pivot.loc[current_ts].values
        max_trade_pct = max_trade_pivot.loc[current_ts].values / intraday_portfolio_values[-1] # 当前时刻最大交易比例

        if i != len(day_timestamps) - 1:
            try:
                w_opt = optimizer.optimize(
                    w0=intraday_portfolio_weights[-1],
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
                
            except Exception as e:
                print(f"优化失败在时间戳 {current_ts}: {e}")
                w_opt = updated_intraday_weight.copy() # 优化失败时保持当前权重不变
        else:
            w_opt = buyhold_weights[-1].copy() # 最后一个时刻将持仓恢复至(最新的)初始权重

        # 计算交易量
        trade_weights = w_opt - intraday_portfolio_weights[-1]
        
        # 计算滑点成本：基于交易金额的比例损失
        slippage = slippage_1 if i != len(day_timestamps) - 1 else slippage_2
        slippage_factor = slippage / 10000.0  # 将bp转换为小数
        trade_amounts = np.abs(trade_weights) * intraday_portfolio_values[-1]
        slippage_cost = np.sum(trade_amounts) * slippage_factor
        
        # 计算交易费用
        trade_cost = lambda_ * np.sum(trade_amounts)
        
        # 总交易磨损
        total_trading_friction = slippage_cost + trade_cost
        total_trading_frictions.append(total_trading_friction)
        
        # 更新权重（使用优化后的权重）
        intraday_portfolio_weights[-1] = w_opt.copy()
        
        # 从组合价值中扣除交易磨损
        intraday_portfolio_values[-1] = intraday_portfolio_values[-1] - total_trading_frictions[-1]
        
        # 计算新的累计交易比例
        denom = np.maximum(buyhold_weights[-1] * buyhold_values[-1], 1e-8)
        new_trade_pct = trade_amounts / denom
        cumulative_trade_pcts.append(cumulative_trade_pcts[-1] + new_trade_pct)
        
    return {
            "buyhold_values": buyhold_values,
            "intraday_portfolio_values": intraday_portfolio_values,
            "buyhold_weights": buyhold_weights,
            "intraday_portfolio_weights": intraday_portfolio_weights,
            "cumulative_trade_pcts": cumulative_trade_pcts,
            "total_trading_frictions": total_trading_frictions,
        }