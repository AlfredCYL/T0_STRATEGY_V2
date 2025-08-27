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

def execute_trades(prev_value, prev_weights, pending_trade_weights, prev_close, current_close, current_twap, lambda_):
    """执行交易指令"""
    # 分离买入和卖出指令 - 使用安全的零值判断
    sell_mask = pending_trade_weights < -TOLERANCE
    buy_mask = pending_trade_weights > TOLERANCE
    hold_mask = is_close_to_zero(pending_trade_weights)
    
    # 执行卖出t-1时刻
    sell_weights = np.abs(pending_trade_weights * sell_mask) 
    sell_ratios = safe_divide(sell_weights, prev_weights)
    
    # 计算卖出获得现金(t-1->t)
    cash_from_sales = sell_weights * safe_divide(current_twap, prev_close)
    sell_trading_friction = np.sum(cash_from_sales) * lambda_
    total_cash_available = np.sum(cash_from_sales) - sell_trading_friction
    
    # 执行买入t-1时刻
    buy_weights = np.abs(pending_trade_weights * buy_mask)
    expected_buy_cash_needed = np.sum(buy_weights * safe_divide(current_twap, prev_close)) 
    execution_ratio = min(1.0, total_cash_available / expected_buy_cash_needed) if expected_buy_cash_needed > 0 else 1.0
    buy_weights *= execution_ratio / (1 + lambda_)
    
    # 计算买入所用现金(t-1->t)
    cash_for_buying = buy_weights * safe_divide(current_twap, prev_close)
    buy_trading_friction = np.sum(cash_for_buying) * lambda_
    total_cash_consumed = np.sum(cash_for_buying) + buy_trading_friction

    # 计算剩余现金权重
    remaining_cash_weight = total_cash_available - total_cash_consumed
    
    # 统计买卖金额
    total_trade_value = (np.abs(buy_weights) + np.abs(sell_weights)) * safe_divide(current_twap, prev_close) * prev_value
    total_trading_friction = (sell_trading_friction + buy_trading_friction) * prev_value
    
    # 计算最终权重
    hold_weights = prev_weights * hold_mask * safe_divide(current_close, prev_close)
    sell_weights = prev_weights * sell_mask * (1 - sell_ratios) * safe_divide(current_close, prev_close) 
    buy_weights = prev_weights * buy_mask * safe_divide(current_close, prev_close) + buy_weights * safe_divide(current_close, current_twap)
    new_weights = hold_weights + sell_weights + buy_weights

    # 计算组合市价
    new_value = (np.sum(new_weights) + remaining_cash_weight) * prev_value
    
    # 归一化权重
    new_weights = safe_normalize_weights(new_weights)  # 使用安全的归一化
    new_weights = new_weights * (1 - remaining_cash_weight)
    
    return new_weights.copy(), new_value, total_trade_value, total_trading_friction

def update_trading_metrics(state, new_weights, new_value, trade_amounts_twap, total_trading_friction, current_close, current_twap):
    """更新交易相关指标"""
    # 更新状态
    state.intraday_portfolio_weights.append(new_weights.copy())
    state.intraday_portfolio_values.append(new_value)
    state.total_trading_frictions.append(total_trading_friction)

    # 更新累计交易比例 - 将TWAP调整至close价格 # 使用安全除法进行价格调整
    price_adjustment = safe_divide(current_close, current_twap, 1.0)
    trade_amounts_close = trade_amounts_twap * price_adjustment
    
    denominator = state.buyhold_weights[-1] * state.buyhold_values[-1]
    new_trade_pct = safe_divide(trade_amounts_close, denominator)
    updated_trade_pct = state.cumulative_trade_pcts[-1] + np.abs(new_trade_pct)
    
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

def execute_trade_and_calculate_costs(state, prev_close, current_close, current_twap, lambda_):
    """执行交易并记录的通用函数"""
    # 执行交易指令
    new_weights, new_value, trade_amounts_twap, total_trading_friction = execute_trades(state.intraday_portfolio_values[-1], state.intraday_portfolio_weights[-1], state.pending_trade_weights, prev_close, current_close, current_twap, lambda_)
    
    # 更新交易记录
    update_trading_metrics(state, new_weights, new_value, trade_amounts_twap, total_trading_friction, current_close, current_twap)

def run_single_day_optimization(
    close_pivot, twap_pivot, alpha_pivot, hold_mask_pivot, zero_mask_pivot,
    initial_weights, pred_value_pivot, optimizer,
    slippage_1=0.0, slippage_2=0.0, cost_func=None, trading_fee=0.01, max_trade_pred_value_pct=0.05,
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
        
        slippage = slippage_1 if i != len(day_timestamps) - 1 else slippage_2
        lambda_ = slippage + trading_fee

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
            execute_trade_and_calculate_costs(state, prev_close, current_close, current_twap, lambda_)
        
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