import numpy as np
import pandas as pd
from optimizer.cvx_optimizer import PortfolioOptimizer
from strategy.data import build_intraday_pivots
from strategy.backtest import run_backtest

def main():
    # 假设此处读取你在 notebook 中构造的 df（也可从csv加载）
    # df = pd.read_csv("your_intraday_data.csv", parse_dates=["date"])
    # 这里示例：用户已在外部准备好 DataFrame df
    raise SystemExit("请将此脚本替换为实际数据加载代码，并把 df 传入 build_intraday_pivots(df)")

    # pivots = build_intraday_pivots(df)
    # close_pivot = pivots["close_pivot"]
    # alpha_pivot = pivots["alpha_pivot"]  # 实盘需替换成你的alpha
    # is_limit_up_pivot = pivots["is_limit_up_pivot"]
    # is_limit_down_pivot = pivots["is_limit_down_pivot"]
    # is_st_pivot = pivots["is_st_pivot"]
    # symbols = pivots["symbols"]

    # dates = sorted(pd.Series(close_pivot.index.date).unique())
    # initial_weights = np.full(len(symbols), 1.0 / len(symbols))
    # initial_portfolio_value = 1_000_000.0
    # max_trade_pct = 1.0 / 25.0

    # optimizer = PortfolioOptimizer()
    # daily_results, final_w, final_pv = run_backtest(
    #     close_pivot, alpha_pivot, is_limit_up_pivot, is_limit_down_pivot, is_st_pivot,
    #     optimizer, symbols, dates, initial_weights, initial_portfolio_value, max_trade_pct
    # )

    # print("Final PV:", final_pv)
    # print("Final Weights:", np.round(final_w, 4))

if __name__ == "__main__":
    main()