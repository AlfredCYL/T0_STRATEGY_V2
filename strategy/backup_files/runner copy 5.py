import numpy as np

def run_single_day_optimization(
    close_pivot,           # DataFrame (index: timestamps, columns: symbols)
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
        day_timestamps = close_pivot.index.sort_values()

    buyhold_values = [float(initial_portfolio_value)]
    intraday_portfolio_values = [float(initial_portfolio_value)]
    buyhold_weights = [np.asarray(initial_weights, dtype=float).copy()]
    intraday_portfolio_weights = [np.asarray(initial_weights, dtype=float).copy()]
    cumulative_trade_pcts = [np.zeros(len(initial_weights), dtype=float)]
    total_trading_frictions = [0.0]
    
    # 在t时刻执行t-1时刻发出的交易指令
    pending_trade_weights = np.zeros(len(initial_weights), dtype=float)

    for i in range(1, len(day_timestamps)):
        prev_ts = day_timestamps[i - 1]
        current_ts = day_timestamps[i]

        prev_close = close_pivot.loc[prev_ts].values
        current_close = close_pivot.loc[current_ts].values
        current_twap = twap_pivot.loc[current_ts].values
        
        # 计算不同部分的收益率
        close_returns = (current_close - prev_close) / prev_close  # 未交易部分收益率
        
        # 更新buyhold组合
        buyhold_values.append(buyhold_values[-1] * (1.0 + float(np.dot(buyhold_weights[-1], close_returns))))
        updated_buyhold_weight = buyhold_weights[-1] * (1.0 + close_returns)
        updated_buyhold_weight /= updated_buyhold_weight.sum()
        buyhold_weights.append(updated_buyhold_weight.copy())
        
        # 更新intraday组合，考虑t-1时刻发出的交易指令在t时刻执行
        if i > 1:
            prev_weights = intraday_portfolio_weights[-1]
            prev_portfolio_value = intraday_portfolio_values[-1]
            
            # 向量化处理：分离买入和卖出指令
            sell_mask = pending_trade_weights < 0
            buy_mask = pending_trade_weights > 0
            hold_mask = pending_trade_weights == 0
            
            # 第一步：执行卖出指令（向量化）
            sell_weights = np.abs(pending_trade_weights * sell_mask)  # 卖出权重
            # 实际能卖出的权重（不能超过现有持仓）
            actual_sell_weights = np.minimum(sell_weights, prev_weights)
            sell_ratios = np.divide(actual_sell_weights, prev_weights, 
                                  out=np.zeros_like(prev_weights), where=prev_weights>0)
            
            # 卖出部分收益：close t-1 -> twap t
            sell_returns = (current_twap - prev_close) / prev_close
            # 持有部分收益：close t-1 -> close t
            hold_returns = close_returns
            
            # 卖出获得的现金权重
            cash_from_sales = actual_sell_weights * (1.0 + sell_returns)
            total_cash_available = np.sum(cash_from_sales)
            
            # 执行卖出后的中间权重
            remaining_weights = prev_weights - actual_sell_weights  # 剩余持仓
            intermediate_weights = remaining_weights * (1.0 + hold_returns)  # 按close价格更新
            
            # 第二步：执行买入指令（向量化）
            buy_demands = pending_trade_weights * buy_mask  # 买入需求
            total_buy_demand = np.sum(buy_demands)
            
            # 计算买入执行比例
            execution_ratio = min(1.0, total_cash_available / total_buy_demand) if total_buy_demand > 0 else 1.0
            
            # 实际执行的买入权重
            actual_buy_weights = buy_demands * execution_ratio
            
            # 买入部分收益计算
            buy_returns = (current_close - current_twap) / current_twap
            # 按twap价格买入后的权重调整
            buy_weights_at_twap = actual_buy_weights * prev_close / current_twap
            # 买入权重的最终价值
            buy_weights_final = buy_weights_at_twap * (1.0 + buy_returns)
            
            # 最终权重 = 中间权重 + 买入权重
            new_weights = intermediate_weights + buy_weights_final
            
            # 标准化权重
            new_weights /= new_weights.sum()
            intraday_portfolio_weights.append(new_weights.copy())
            
            # 向量化计算组合收益率
            # 卖出部分的加权收益
            sell_weighted_returns = prev_weights * (sell_ratios * sell_returns + (1 - sell_ratios) * hold_returns)
            # 原有持仓的收益（非卖出部分）
            hold_weighted_returns = prev_weights * hold_returns * (1 - sell_mask.astype(float))
            # 买入部分的收益
            buy_weighted_returns = buy_weights_at_twap * buy_returns
            
            # 总收益率
            total_value_change = np.sum(sell_weighted_returns * sell_mask.astype(float) + 
                                      hold_weighted_returns * hold_mask.astype(float) + 
                                      buy_weighted_returns)
            
            intraday_portfolio_values.append(intraday_portfolio_values[-1] * (1.0 + total_value_change))
            
            # 向量化计算交易磨损
            actual_trade_amounts = actual_sell_weights + actual_buy_weights
            
            if np.sum(actual_trade_amounts) > 0:
                slippage = slippage_1 if i != len(day_timestamps) - 1 else slippage_2
                slippage_factor = slippage / 10000.0
                trade_amounts_twap = actual_trade_amounts * intraday_portfolio_values[-2] * current_twap / prev_close
                slippage_cost = np.sum(trade_amounts_twap) * slippage_factor
                
                # 计算交易费用
                trade_cost = lambda_ * np.sum(trade_amounts_twap)
                
                # 总交易磨损
                total_trading_friction = slippage_cost + trade_cost
                total_trading_frictions.append(total_trading_friction)
                
                # 从组合价值中扣除交易磨损
                intraday_portfolio_values[-1] = intraday_portfolio_values[-1] - total_trading_friction
                
                # 计算累计交易比例（向量化）
                denom = np.maximum(buyhold_weights[-1] * buyhold_values[-1], 1e-8)
                new_trade_pct = trade_amounts_twap / denom
                cumulative_trade_pcts.append(cumulative_trade_pcts[-1] + new_trade_pct)
            else:
                total_trading_frictions.append(0.0)
                cumulative_trade_pcts.append(cumulative_trade_pcts[-1])
                
        else:
            # 第一个时间步，直接使用close收益率
            intraday_portfolio_values.append(intraday_portfolio_values[-1] * (1.0 + float(np.dot(intraday_portfolio_weights[-1], close_returns))))
            updated_intraday_weight = intraday_portfolio_weights[-1] * (1.0 + close_returns)
            updated_intraday_weight /= updated_intraday_weight.sum()
            intraday_portfolio_weights.append(updated_intraday_weight.copy())
            total_trading_frictions.append(0.0)
            cumulative_trade_pcts.append(cumulative_trade_pcts[-1])
        
        # 在t时刻发出交易指令，将在t+1时刻执行
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
                if verbose:
                    print(f"优化失败在时间戳 {current_ts}: {e}")
                w_opt = intraday_portfolio_weights[-1].copy() # 优化失败时保持当前权重不变
        else:
            w_opt = buyhold_weights[-1].copy() # 最后一个时刻将持仓恢复至(最新的)初始权重

        # 计算t时刻发出的交易指令，将在t+1时刻执行
        pending_trade_weights = w_opt - intraday_portfolio_weights[-1]
        
    # 执行最后一个时间戳的交易指令
    if np.any(pending_trade_weights != 0):
        final_ts = day_timestamps[-1]
        final_close = close_pivot.loc[final_ts].values
        final_twap = twap_pivot.loc[final_ts].values
        
        prev_weights = intraday_portfolio_weights[-1]
        prev_portfolio_value = intraday_portfolio_values[-1]
        
        # 使用与日内交易相同的逻辑执行最终交易
        # [执行买卖逻辑 - 与第52-147行相同的向量化处理]
        
        # 计算最终的交易成本（使用slippage_2）
        actual_trade_amounts = np.abs(pending_trade_weights)
        if np.sum(actual_trade_amounts) > 0:
            slippage_factor = slippage_2 / 10000.0
            trade_amounts_twap = actual_trade_amounts * prev_portfolio_value
            slippage_cost = np.sum(trade_amounts_twap) * slippage_factor
            trade_cost = lambda_ * np.sum(trade_amounts_twap)
            total_trading_friction = slippage_cost + trade_cost
            
            # 更新最终的组合价值和权重
            intraday_portfolio_values[-1] -= total_trading_friction
            intraday_portfolio_weights[-1] = w_opt.copy()
            
            total_trading_frictions.append(total_trading_friction)
            
            # 更新累计交易比例
            denom = np.maximum(buyhold_weights[-1] * buyhold_values[-1], 1e-8)
            new_trade_pct = trade_amounts_twap / denom
            cumulative_trade_pcts[-1] = cumulative_trade_pcts[-1] + new_trade_pct

    return {
            "buyhold_values": buyhold_values,
            "intraday_portfolio_values": intraday_portfolio_values,
            "buyhold_weights": buyhold_weights,
            "intraday_portfolio_weights": intraday_portfolio_weights,
            "cumulative_trade_pcts": cumulative_trade_pcts,
            "total_trading_frictions": total_trading_frictions,
        }