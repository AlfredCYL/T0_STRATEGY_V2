import numpy as np
import pandas as pd
from optimizer.cvx_optimizer import PortfolioOptimizer
from strategy.runner import run_single_day_optimization

def test_smoke():
    # 2 分钟 x 3 标的
    idx = pd.to_datetime(["2023-01-02 09:30:00", "2023-01-02 09:31:00"])
    cols = ["A", "B", "C"]
    close = pd.DataFrame([[100, 100, 100], [101, 99, 100]], index=idx, columns=cols)
    alpha = pd.DataFrame([[0.01, -0.01, 0.0], [0.0, 0.0, 0.0]], index=idx, columns=cols)
    f = pd.DataFrame(False, index=idx, columns=cols)

    init_w = np.array([1/3, 1/3, 1/3], dtype=float)
    init_pv = 1_000_000.0
    max_trade = 1/25

    opt = PortfolioOptimizer()
    res = run_single_day_optimization(
        close_pivot=close,
        alpha_pivot=alpha,
        is_limit_up_pivot=f,
        is_limit_down_pivot=f,
        is_st_pivot=f,
        initial_weights=init_w,
        initial_portfolio_value=init_pv,
        max_trade_pct=max_trade,
        optimizer=opt,
        day_timestamps=idx
    )
    assert "final_portfolio_value" in res and res["final_portfolio_value"] > 0