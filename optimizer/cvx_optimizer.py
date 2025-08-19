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
        initial_weights=None,
        cumulative_trades=None,
        optimizable_mask=None,
        cost_func=None,
        lambda_cost=0.01,
        max_turnover_ratio=1.0,
        current_portfolio_value=1.0,
        initial_portfolio_value=1.0,
        solver=None,
        verbose=False,
    ):
        """
        :param w0: 当前权重 (np.array, shape [n])
        :param alpha: 预测收益 (np.array, shape [n])
        :param max_trade_pct: 每资产最大交易比例 (标量或 np.array [n])
        :param initial_weights: 当日初始持仓权重 (np.array [n])
        :param cumulative_trades: 当日累计交易占初始持仓比例 (np.array [n])
        :param optimizable_mask: True=可交易, False=锁定 (np.bool_ array [n])
        :param cost_func: 交易成本函数，默认 L1(dw)
        :param lambda_cost: 成本系数
        :param max_turnover_ratio: 当日累计交易比例上限（相对初始持仓）
        :param current_portfolio_value: 当前组合市值
        :param initial_portfolio_value: 当日初始组合市值
        :return: 优化后的权重 (np.array [n])
        """
        w0 = np.asarray(w0, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        n = len(w0)

        if initial_weights is None:
            initial_weights = w0.copy()
        else:
            initial_weights = np.asarray(initial_weights, dtype=float)

        if cumulative_trades is None:
            cumulative_trades = np.zeros(n, dtype=float)
        else:
            cumulative_trades = np.asarray(cumulative_trades, dtype=float)

        if optimizable_mask is None:
            optimizable_mask = np.ones(n, dtype=bool)
        else:
            optimizable_mask = np.asarray(optimizable_mask, dtype=bool)

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
            cost = cp.norm1(dw)  # L1 交易成本
        else:
            cost = cost_func(dw)

        # 交易比例（相对“初始组合价值中的该资产持仓金额”）
        trade_ratio = cp.abs(dw) * float(current_portfolio_value) / denom

        # 目标：最大化 预期收益 - 成本
        objective = cp.Maximize(w @ alpha - lambda_cost * cost)

        constraints = [
            cp.sum(w) == 1.0,
            w >= 0.0,
            w <= 1.0,
            cp.abs(dw) <= max_trade_pct,
            trade_ratio + cumulative_trades <= max_turnover_ratio,
        ]

        # 锁定不可交易资产逐元素相等
        if not np.all(optimizable_mask):
            constraints.append(w[~optimizable_mask] == w0[~optimizable_mask])

        problem = cp.Problem(objective, constraints)

        chosen_solver = solver or cp.ECOS
        problem.solve(solver=chosen_solver, warm_start=True, verbose=verbose)

        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise ValueError(f"优化失败, 状态: {problem.status}")

        return np.asarray(w.value, dtype=float)