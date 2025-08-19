import numpy as np
import pandas as pd
from .runner import run_single_day_optimization

def run_backtest(
    close_pivot, alpha_pivot, is_limit_up_pivot, is_limit_down_pivot, is_st_pivot,
    optimizer, symbols, dates, initial_weights, initial_portfolio_value, max_trade_pct
):
    """
    :param close_pivot: DataFrame [timestamp x symbol]
    :param alpha_pivot: DataFrame [timestamp x symbol]
    :param is_limit_up_pivot: DataFrame bool
    :param is_limit_down_pivot: DataFrame bool
    :param is_st_pivot: DataFrame bool
    :param symbols: list[str]
    :param dates: list[pd.Timestamp.date] 回测交易日
    :param initial_weights: np.array [n]
    :param initial_portfolio_value: float
    :param max_trade_pct: float or np.array [n]
    """
    w = np.asarray(initial_weights, dtype=float).copy()
    pv = float(initial_portfolio_value)

    daily_results = []
    for d in dates:
        day_mask = close_pivot.index.date == d
        if not day_mask.any():
            continue
        day_close = close_pivot.loc[day_mask, symbols]
        day_alpha = alpha_pivot.loc[day_mask, symbols]
        day_up = is_limit_up_pivot.loc[day_mask, symbols]
        day_down = is_limit_down_pivot.loc[day_mask, symbols]
        day_st = is_st_pivot.loc[day_mask, symbols]

        timestamps = day_close.index.sort_values()

        res = run_single_day_optimization(
            close_pivot=day_close,
            alpha_pivot=day_alpha,
            is_limit_up_pivot=day_up,
            is_limit_down_pivot=day_down,
            is_st_pivot=day_st,
            initial_weights=w,
            initial_portfolio_value=pv,
            max_trade_pct=max_trade_pct,
            optimizer=optimizer,
            day_timestamps=timestamps,
        )
        w = res["final_weights"]
        pv = res["final_portfolio_value"]
        daily_results.append((d, res))

    return daily_results, w, pv