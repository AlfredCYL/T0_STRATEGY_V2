import pandas as pd

def build_intraday_pivots(df, symbols=None):
    """
    期望 df 至少包含列:
      ['date', 'time', 'symbol', 'close', 'is_limit_up', 'is_limit_down', 'is_st']
    你现有生成数据已满足（列名略有不同，注意映射）。
    """
    # 合成时间戳
    ts = pd.to_datetime(df["date"]) + df["time"]
    df = df.copy()
    df["timestamp"] = ts

    if symbols is None:
        symbols = sorted(df["symbol"].unique().tolist())

    def pivot_bool(col):
        return df.pivot(index="timestamp", columns="symbol", values=col).reindex(columns=symbols).fillna(False).astype(bool)

    def pivot_float(col):
        return df.pivot(index="timestamp", columns="symbol", values=col).reindex(columns=symbols).astype(float)

    close_pivot = pivot_float("close")
    # alpha 需要你准备；这里给出占位（全零），或外部传入后覆盖
    alpha_pivot = close_pivot.copy()
    alpha_pivot.loc[:, :] = 0.0

    is_limit_up_pivot = pivot_bool("is_limit_up")
    is_limit_down_pivot = pivot_bool("is_limit_down")
    is_st_pivot = pivot_bool("is_st")

    return dict(
        close_pivot=close_pivot,
        alpha_pivot=alpha_pivot,
        is_limit_up_pivot=is_limit_up_pivot,
        is_limit_down_pivot=is_limit_down_pivot,
        is_st_pivot=is_st_pivot,
        symbols=symbols,
    )