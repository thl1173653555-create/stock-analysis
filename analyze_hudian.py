import json
from pathlib import Path

import akshare as ak
import pandas as pd
import vectorbt as vbt


SYMBOL = "002463"  # 沪电股份
START_DATE = "20210811"
FEE_AND_SLIPPAGE = 0.0008  # ponytail: 基线成本模型；实盘需按券商费率、印花税和最低佣金替换。
CACHE_PATH = Path(__file__).with_name("002463_daily_qfq.csv")


def percentage(value: float) -> str:
    return f"{value:.2%}"


def main() -> None:
    try:
        # Sina-backed endpoint: the Eastmoney endpoint was unavailable during this run.
        data = ak.stock_zh_a_daily(symbol="sz002463", adjust="qfq").rename(
            columns={"date": "日期", "close": "收盘"}
        )
        data["日期"] = pd.to_datetime(data["日期"])
        data = data[data["日期"] >= pd.Timestamp(START_DATE)]
        data["涨跌幅"] = data["收盘"].pct_change().mul(100)
        data.to_csv(CACHE_PATH, index=False, encoding="utf-8-sig")
    except Exception:
        if not CACHE_PATH.exists():
            raise
        data = pd.read_csv(CACHE_PATH)
    data["日期"] = pd.to_datetime(data["日期"])
    data = data.sort_values("日期").set_index("日期")
    close = data["收盘"].astype(float)

    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    raw_entries = (ma20 > ma60) & (ma20.shift(1) <= ma60.shift(1))
    raw_exits = (ma20 < ma60) & (ma20.shift(1) >= ma60.shift(1))
    # Signals use the close, so execute on the following trading day to avoid look-ahead bias.
    entries = raw_entries.shift(1, fill_value=False)
    exits = raw_exits.shift(1, fill_value=False)

    strategy = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        init_cash=100_000,
        fees=FEE_AND_SLIPPAGE,
        freq="1D",
    )
    holding = vbt.Portfolio.from_holding(
        close, init_cash=100_000, fees=FEE_AND_SLIPPAGE, freq="1D"
    )

    def stats(portfolio: vbt.Portfolio) -> dict[str, str | int]:
        total_return = float(portfolio.total_return())
        years = len(close) / 252
        annual_return = (1 + total_return) ** (1 / years) - 1
        return {
            "total_return": percentage(total_return),
            "annualized_return": percentage(annual_return),
            "max_drawdown": percentage(float(portfolio.max_drawdown())),
            "trades": int(portfolio.trades.count()),
        }

    last_cross_dates = data.index[(raw_entries | raw_exits).fillna(False)]
    summary = {
        "name": "沪电股份",
        "symbol": SYMBOL,
        "data_range": f"{data.index[0].date()} 至 {data.index[-1].date()}",
        "trading_days": len(data),
        "latest": {
            "date": str(data.index[-1].date()),
            "close": round(float(close.iloc[-1]), 2),
            "daily_change": f"{float(data['涨跌幅'].iloc[-1]):.2f}%",
            "ma20": round(float(ma20.iloc[-1]), 2),
            "ma60": round(float(ma60.iloc[-1]), 2),
            "ma_state": "MA20 在 MA60 上方" if ma20.iloc[-1] > ma60.iloc[-1] else "MA20 在 MA60 下方",
            "last_cross": str(last_cross_dates[-1].date()) if len(last_cross_dates) else None,
        },
        "ma20_60_strategy": stats(strategy),
        "buy_and_hold": stats(holding),
        "assumptions": {
            "price": "AKShare 新浪源前复权日线",
            "strategy": "MA20 上穿 MA60 买入，下穿卖出；信号在下一交易日收盘执行",
            "cost": "每笔综合费率 0.08%，未单独模拟最低佣金、印花税和涨跌停/停牌约束",
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
