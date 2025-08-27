import cvxpy as cp
import numpy as np

class PortfolioOptimizer:
    def __init__(self):
        pass

    def optimize(
        self,
        w0,
        alpha,
        max_trade_pct,
        hold_mask=None,
        zero_mask=None,
        cost_func=None,
        lambda_=0.01,
        initial_weights=None,
        cumulative_trade_pct=None,
        max_turnover_ratio=1.0,
        current_portfolio_value=1.0,
        initial_portfolio_value=1.0,
        max_average_turnover=None,
        solver=None,
        verbose=False,
    ):
        """
        :param w0: 当前权重 (np.array, shape [n])
        :param alpha: 预测收益 (np.array, shape [n])
        :param max_trade_pct: 每资产最大交易比例 (标量或 np.array [n])
        :param hold_mask: True=保持原权重 (np.bool_ array [n])
        :param zero_mask: True=强制权重为0 (np.bool_ array [n])
        :param cost_func: 交易成本函数，默认 L1(dw), 双边交易
        :param lambda_: 成本系数
        :param initial_weights: 当日初始持仓的最新权重 (np.array [n])
        :param cumulative_trade_pct: 当日累计交易占初始持仓比例 (np.array [n])
        :param max_turnover_ratio: 当日累计交易比例上限（相对初始持仓）
        :param current_portfolio_value: 当前组合市值
        :param initial_portfolio_value: 当日初始组合市值
        :param max_average_turnover: 平均换手率上限 (标量)
        :return: 优化后的权重 (np.array [n])
        """
        w0 = np.asarray(w0, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        n = len(w0)

        if initial_weights is None:
            initial_weights = w0.copy()
        else:
            initial_weights = np.asarray(initial_weights, dtype=float)

        if cumulative_trade_pct is None:
            cumulative_trade_pct = np.zeros(n, dtype=float)
        else:
            cumulative_trade_pct = np.asarray(cumulative_trade_pct, dtype=float)

        if hold_mask is None:
            hold_mask = np.zeros(n, dtype=bool)
        else:
            hold_mask = np.asarray(hold_mask, dtype=bool)

        if zero_mask is None:
            zero_mask = np.zeros(n, dtype=bool)
        else:
            zero_mask = np.asarray(zero_mask, dtype=bool)

        # 检查hold_mask和zero_mask是否有重叠
        if np.any(hold_mask & zero_mask):
            raise ValueError("hold_mask和zero_mask不能有重叠")

        # 可优化mask = 既不在hold_mask也不在zero_mask中的股票
        optimizable_mask = ~(hold_mask | zero_mask)

        if np.isscalar(max_trade_pct):
            max_trade_pct = np.full(n, float(max_trade_pct))
        else:
            max_trade_pct = np.asarray(max_trade_pct, dtype=float)

        # 分母保护，避免初始权重为0导致除0
        denom = np.maximum(initial_weights * float(initial_portfolio_value), 1e-8)

        # 变量与表达式
        w = cp.Variable(n)
        dw = w - w0

        if cost_func is None:
            cost = cp.norm1(dw, p=1)  # L1 交易成本
        else:
            cost = cost_func(dw)

        # 交易比例（相对"初始组合价值中的该资产持仓金额"）
        trade_ratio = cp.abs(dw) * float(current_portfolio_value) / denom

        # 目标：最大化 预期收益 - 成本
        objective = cp.Maximize(w @ alpha - lambda_ * cost)

        constraints = [
            cp.sum(w) == 1.0,
            w >= 0.0,
            w <= 1.0,
            cp.abs(dw) <= max_trade_pct,
            trade_ratio + cumulative_trade_pct <= max_turnover_ratio,
        ]
        
        # 添加平均换手率约束: mean(trade_ratio) <= max_average_turnover
        if max_average_turnover is not None:
            constraints.append(cp.sum(cp.abs(trade_ratio)) <= n * max_average_turnover)

        # 处理保持权重的股票
        if np.any(hold_mask):
            constraints.append(w[hold_mask] == w0[hold_mask])
        
        # 处理强制归零的股票
        if np.any(zero_mask):
            constraints.append(w[zero_mask] == 0.0)

        problem = cp.Problem(objective, constraints)

        chosen_solver = solver or cp.ECOS
        problem.solve(solver=chosen_solver, warm_start=True, verbose=verbose)

        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise ValueError(f"优化失败, 状态: {problem.status}")

        return np.asarray(w.value, dtype=float)