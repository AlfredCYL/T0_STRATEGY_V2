import numpy as np

TOLERANCE = 1e-15  # 浮点数比较容差
POSITION_PRECISION = 1e-6 # 仓位比较容差

def safe_divide(numerator, denominator, default_value=0.0):
    """安全除法，避免除零错误"""
    denominator = np.asarray(denominator, dtype=np.float64)
    numerator = np.asarray(numerator, dtype=np.float64)
    
    # 使用 np.divide 的 where 参数来处理除零情况
    return np.divide(numerator, denominator, 
                    out=np.full_like(numerator, default_value, dtype=np.float64),
                    where=np.abs(denominator) > TOLERANCE)

def safe_normalize_weights(weights, min_weight=TOLERANCE):
    """安全的权重归一化，避免精度损失"""
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 0.0)  # 确保非负
    
    total = np.sum(weights)
    if total < min_weight:
        # 如果总权重太小，返回均匀分布
        return np.ones(len(weights), dtype=np.float64) / len(weights)
    
    normalized = weights / total
    
    # 确保归一化后的权重和为1（处理舍入误差）
    normalized[-1] = 1.0 - np.sum(normalized[:-1])
    
    return normalized

def is_close_to_zero(value, tolerance=TOLERANCE):
    """安全的零值判断"""
    return np.abs(value) < tolerance

def clamp_to_bounds_with_tolerance(values, lower_bound=0.0, upper_bound=1.0, tolerance=TOLERANCE):
    """将数值限制在指定范围内，对由于浮点精度导致的轻微越界进行修正"""
    values = np.asarray(values, dtype=np.float64)
    
    # 处理上界越界：如果值在 (upper_bound, upper_bound + tolerance] 范围内，修正为upper_bound
    values = np.where((values > upper_bound) & (values <= upper_bound + tolerance),upper_bound,values)
    
    # 处理下界越界：如果值在 [lower_bound - tolerance, lower_bound) 范围内，修正为lower_bound
    values = np.where((values < lower_bound) & (values >= lower_bound - tolerance),lower_bound,values)
    return values

class PortfolioState:
    """组合状态管理类"""
    def __init__(self, initial_weights, initial_portfolio_value):
        self.buyhold_values = [float(initial_portfolio_value)]
        self.intraday_portfolio_values = [float(initial_portfolio_value)]
        self.buyhold_weights = [np.asarray(initial_weights, dtype=float).copy()]
        self.intraday_portfolio_weights = [np.asarray(initial_weights, dtype=float).copy()]
        self.cumulative_trade_pcts = [np.zeros(len(initial_weights), dtype=float)]
        self.total_trading_frictions = [float(0)]
        self.pending_trade_weights = np.zeros(len(initial_weights), dtype=float)
    
    def to_dict(self):
        return {
            "buyhold_values": self.buyhold_values,
            "intraday_portfolio_values": self.intraday_portfolio_values,
            "buyhold_weights": self.buyhold_weights,
            "intraday_portfolio_weights": self.intraday_portfolio_weights,
            "cumulative_trade_pcts": self.cumulative_trade_pcts,
            "total_trading_frictions": self.total_trading_frictions,
        }

def update_buyhold_portfolio(state, close_returns):
    """更新buyhold组合"""
    # 更新价值
    state.buyhold_values.append(
        state.buyhold_values[-1] * (1.0 + np.dot(state.buyhold_weights[-1], close_returns))
    )
    
    # 更新权重 - 使用安全归一化
    updated_weight = state.buyhold_weights[-1] * (1.0 + close_returns)
    updated_weight = safe_normalize_weights(updated_weight)
    state.buyhold_weights.append(updated_weight.copy())

def execute_trades(state, pending_trade_weights, prev_close, current_close, current_twap):
    """执行交易指令"""
    prev_weights = state.intraday_portfolio_weights[-1]
    
    # 分离买入和卖出指令 - 使用安全的零值判断
    sell_mask = pending_trade_weights < -TOLERANCE
    buy_mask = pending_trade_weights > TOLERANCE
    hold_mask = is_close_to_zero(pending_trade_weights)
    
    # 计算收益率 - 使用安全除法
    sell_returns = safe_divide(current_twap - prev_close, prev_close)
    hold_returns = safe_divide(current_close - prev_close, prev_close)
    buy_returns = safe_divide(current_close - current_twap, current_twap)
    
    # 执行卖出
    sell_weights = np.abs(pending_trade_weights * sell_mask.astype(float))
    actual_sell_weights = np.minimum(sell_weights, prev_weights)
    sell_ratios = safe_divide(actual_sell_weights, prev_weights)
    
    # 计算可用现金
    cash_from_sales = actual_sell_weights * (1.0 + sell_returns)
    total_cash_available = np.sum(cash_from_sales)
    
    # 执行买入
    buy_demands = pending_trade_weights * buy_mask
    total_buy_demand = np.sum(buy_demands)
    execution_ratio = min(1.0, total_cash_available / total_buy_demand) if total_buy_demand > 0 else 1.0
    actual_buy_weights = buy_demands * execution_ratio
    
    # 计算最终权重
    remaining_weights = prev_weights - actual_sell_weights
    intermediate_weights = remaining_weights * (1.0 + hold_returns)
    buy_weights_at_twap = actual_buy_weights * safe_divide(prev_close, current_twap, 1.0)
    buy_weights_final = buy_weights_at_twap * (1.0 + buy_returns)
    new_weights = intermediate_weights + buy_weights_final
    new_weights = safe_normalize_weights(new_weights)  # 使用安全的归一化
    
    # 计算组合收益率
    sell_weighted_returns = prev_weights * (sell_ratios * sell_returns + (1 - sell_ratios) * hold_returns)
    hold_weighted_returns = prev_weights * hold_returns * (1 - sell_mask.astype(float))
    buy_weighted_returns = buy_weights_at_twap * buy_returns
    
    total_value_change = np.sum(
        sell_weighted_returns * sell_mask.astype(float) + 
        hold_weighted_returns * hold_mask.astype(float) + 
        buy_weighted_returns
    )
    
    # 更新状态
    state.intraday_portfolio_weights.append(new_weights.copy())
    state.intraday_portfolio_values.append(
        state.intraday_portfolio_values[-1] * (1.0 + total_value_change)
    )
    
    return actual_sell_weights + actual_buy_weights

def calculate_trading_costs(trade_amounts, portfolio_value, current_twap, prev_close, 
                          slippage, lambda_, is_final=False):
    """计算交易成本"""
    if np.sum(np.abs(trade_amounts)) < TOLERANCE:
        return 0.0, np.zeros_like(trade_amounts)
    
    if is_final:
        trade_amounts_twap = trade_amounts * portfolio_value
    else:
        # 使用安全除法
        price_ratio = safe_divide(current_twap, prev_close, 1.0)
        trade_amounts_twap = trade_amounts * portfolio_value * price_ratio
    
    slippage_cost = np.sum(trade_amounts_twap) * slippage
    trade_cost = lambda_ * np.sum(trade_amounts_twap)
    
    return slippage_cost + trade_cost, trade_amounts_twap

def update_trading_metrics(state, trade_amounts_twap, total_trading_friction, current_twap, current_close):
    """更新交易相关指标"""
    state.total_trading_frictions.append(total_trading_friction)
    state.intraday_portfolio_values[-1] -= total_trading_friction
    
    # 更新累计交易比例 - 将TWAP调整至close价格
    # 使用安全除法进行价格调整
    price_adjustment = safe_divide(current_close, current_twap, 1.0)
    trade_amounts_close = trade_amounts_twap * price_adjustment
    
    # 计算交易比例，使用更高精度
    denominator = state.buyhold_weights[-1] * state.buyhold_values[-1]
    new_trade_pct = safe_divide(trade_amounts_close, denominator)
    updated_trade_pct = state.cumulative_trade_pcts[-1] + new_trade_pct
    
    # 处理精度问题
    updated_trade_pct = clamp_to_bounds_with_tolerance(updated_trade_pct, lower_bound=0.0, upper_bound=1.0, tolerance=POSITION_PRECISION)
    state.cumulative_trade_pcts.append(updated_trade_pct)

def optimize_portfolio(optimizer, state, alpha, max_trade_pct, hold_mask, zero_mask, 
                      cost_func, lambda_, max_turnover_ratio, max_average_turnover, solver, verbose, 
                      current_ts, is_final=False):
    """执行组合优化"""
    if is_final:
        return state.buyhold_weights[-1].copy()
    
    try:
        w_opt = optimizer.optimize(
            w0=state.intraday_portfolio_weights[-1],
            alpha=alpha,
            max_trade_pct=max_trade_pct,
            hold_mask=hold_mask,
            zero_mask=zero_mask,
            cost_func=cost_func,
            lambda_=lambda_,
            initial_weights=state.buyhold_weights[-1],
            cumulative_trade_pct=state.cumulative_trade_pcts[-1],
            max_turnover_ratio=max_turnover_ratio,
            current_portfolio_value=state.intraday_portfolio_values[-1],
            initial_portfolio_value=state.buyhold_values[-1],
            max_average_turnover=max_average_turnover,
            solver=solver,
            verbose=verbose,
        )
        return w_opt
    except Exception as e:
        print(f"优化失败在时间戳 {current_ts}: {e}")
        return state.intraday_portfolio_weights[-1].copy()

def execute_trade_and_calculate_costs(state, prev_close, current_close, current_twap, 
                                    slippage, lambda_, is_final_trade=False):
    """执行交易并计算成本的通用函数"""
    # 执行交易指令
    trade_amounts = execute_trades(
        state, state.pending_trade_weights, prev_close, current_close, current_twap
    )
    
    # 计算交易成本
    if np.sum(trade_amounts) > 0:
        total_trading_friction, trade_amounts_twap = calculate_trading_costs(
            trade_amounts, state.intraday_portfolio_values[-2], 
            current_twap, prev_close, slippage, lambda_, is_final_trade
        )
        update_trading_metrics(state, trade_amounts_twap, total_trading_friction, current_twap, current_close)
    else:
        state.total_trading_frictions.append(0.0)
        state.cumulative_trade_pcts.append(state.cumulative_trade_pcts[-1])

def run_single_day_optimization(
    close_pivot, twap_pivot, alpha_pivot, hold_mask_pivot, zero_mask_pivot,
    initial_weights, pred_value_pivot, optimizer,
    slippage_1=0.0, slippage_2=0.0, cost_func=None, lambda_=0.01, max_trade_pred_value_pct=0.05,
    max_turnover_ratio=1.0, initial_portfolio_value=1, max_average_turnover=None,
    solver=None, verbose=False, day_timestamps=None,
):
    """单日组合优化主函数"""
    if day_timestamps is None:
        day_timestamps = close_pivot.index.sort_values()
    
    # 初始化状态
    state = PortfolioState(initial_weights, initial_portfolio_value)
    
    # 主循环 - 从 i=0 开始
    for i in range(len(day_timestamps)):
        current_ts = day_timestamps[i]
        
        # 从 i=1 开始才处理价格更新和交易执行
        if i >= 1:
            prev_ts = day_timestamps[i - 1]
            
            # 获取价格数据
            prev_close = close_pivot.loc[prev_ts].values
            current_close = close_pivot.loc[current_ts].values
            current_twap = twap_pivot.loc[current_ts].values
            close_returns = (current_close - prev_close) / prev_close
            
            # 更新buyhold组合
            update_buyhold_portfolio(state, close_returns)
            
            # 执行前一时刻的交易指令
            slippage = slippage_1 if i != len(day_timestamps) - 1 else slippage_2
            execute_trade_and_calculate_costs(state, prev_close, current_close, current_twap, 
                                            slippage, lambda_)
        
        # 生成新的交易指令 - 从 i=0 开始就要生成
        alpha = alpha_pivot.loc[current_ts].values
        zero_mask = zero_mask_pivot.loc[current_ts].values
        hold_mask = hold_mask_pivot.loc[current_ts].values
        max_trade_pct = pred_value_pivot.loc[current_ts].values / state.intraday_portfolio_values[-1] * max_trade_pred_value_pct
        
        # 处理max_average_turnover参数
        if max_average_turnover is None:
            current_max_average_turnover = None
        elif np.isscalar(max_average_turnover):
            current_max_average_turnover = max_average_turnover
        else:
            current_max_average_turnover = max_average_turnover.loc[current_ts]
        
        is_final = i >= len(day_timestamps) - 2
        w_opt = optimize_portfolio(
            optimizer, state, alpha, max_trade_pct, hold_mask, zero_mask,
            cost_func, lambda_, max_turnover_ratio, current_max_average_turnover, solver, verbose,
            current_ts, is_final
        )
        
        state.pending_trade_weights = w_opt - state.intraday_portfolio_weights[-1]
    return state.to_dict()