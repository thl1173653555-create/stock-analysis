"""A 股与场内 ETF 的自选研究和本地持仓看板。"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from stock_lab import (
    DataUnavailableError,
    DataValidationError,
    _ma_signal_tables,
    aggregate_scan,
    annual_returns,
    append_trade,
    apply_stop_loss_exits,
    asset_type,
    backtest,
    backtest_portfolio,
    benchmark_stats,
    cache_updated_at,
    close_from_prices,
    current_ma_state,
    dataframe_to_csv,
    fmt_perf,
    holdings_from_trades,
    holdings_valuation,
    label,
    load_benchmark,
    load_config,
    load_holdings,
    load_names,
    load_prices,
    load_signal_alerts,
    load_trades,
    load_update_log,
    load_watchlist,
    normalize_symbol,
    portfolio_metrics,
    portfolio_summary,
    save_holdings,
    save_watchlist,
    scan_params,
    summarize,
    trade_summary,
    trades_table,
    validate_strategy_inputs,
    walk_forward_scan,
    walk_forward_stability,
    walk_forward_summary,
)

st.set_page_config(page_title="股票研究与持仓看板", page_icon="📈", layout="wide")


def _records_names(records: list[dict[str, object]]) -> dict[str, str]:
    names = load_names(record["code"] for record in records)
    names.update({str(record["code"]): str(record.get("name", "")).strip() for record in records if record.get("name")})
    return names


def _labels(records: list[dict[str, object]]) -> dict[str, str]:
    names = _records_names(records)
    return {label(str(record["code"]), names): str(record["code"]) for record in records}


def _format_percent(value: float) -> str:
    return "-" if pd.isna(value) else f"{value:.2%}"


@st.cache_data(show_spinner="读取本地日线数据…")
def get_prices(
    codes: tuple[str, ...],
    refresh: bool,
    refresh_version: int,
    start_date: str,
    retries: int,
    workers: int,
):
    return load_prices(codes, refresh=refresh, start_date=start_date, retries=retries, workers=workers)


st.title("📈 股票研究与持仓看板")
st.caption("日线研究与本地记录工具；不接入券商账户，不构成投资建议。")

config = load_config()

if "refresh_version" not in st.session_state:
    st.session_state.refresh_version = 0
if "refresh_requested" not in st.session_state:
    st.session_state.refresh_requested = False

watchlist = load_watchlist()
holdings = load_holdings()

with st.sidebar:
    st.header("自选与参数")
    with st.form("add_watchlist", clear_on_submit=True):
        add_code = st.text_input("新增代码", placeholder="例如 002463 或 588000")
        add_name = st.text_input("名称（可选）")
        add_submit = st.form_submit_button("加入自选")
    if add_submit:
        try:
            code = normalize_symbol(add_code)
            if code not in {item["code"] for item in watchlist}:
                watchlist = save_watchlist([*watchlist, {"code": code, "name": add_name}])
                st.success(f"已加入 {code}（{asset_type(code)}）。")
                st.rerun()
            st.info(f"{code} 已在自选中。")
        except DataValidationError as exc:
            st.error(str(exc))

    all_labels = _labels(watchlist)
    selected_labels = st.multiselect("本次研究标的", list(all_labels), default=list(all_labels))
    research_codes = [all_labels[item] for item in selected_labels]

    if watchlist:
        manage_label = st.selectbox("管理自选", list(all_labels), key="manage_watchlist")
        move_left, move_middle, move_right = st.columns(3)
        with move_left:
            move_up = st.button("上移", use_container_width=True)
        with move_middle:
            move_down = st.button("下移", use_container_width=True)
        with move_right:
            delete = st.button("删除", type="secondary", use_container_width=True)
        if move_up or move_down or delete:
            code = all_labels[manage_label]
            index = next(index for index, item in enumerate(watchlist) if item["code"] == code)
            if delete:
                watchlist.pop(index)
            elif move_up and index:
                watchlist[index - 1], watchlist[index] = watchlist[index], watchlist[index - 1]
            elif move_down and index < len(watchlist) - 1:
                watchlist[index + 1], watchlist[index] = watchlist[index], watchlist[index + 1]
            save_watchlist(watchlist)
            st.rerun()

    st.divider()
    if st.button("更新日线数据", type="primary", use_container_width=True):
        st.session_state.refresh_requested = True
        st.session_state.refresh_version += 1
        st.rerun()
    st.caption("默认读取本地缓存；点击按钮才会强制更新全部日线。缓存缺失或覆盖不到起始日期时会自动补拉。")

    fast_default = max(1, min(120, int(config["fast"])))
    fast = st.slider("MA 快线", min_value=1, max_value=120, value=fast_default, key="fast")
    slow_default = max(2, min(250, int(config["slow"])))
    slow_min = fast + 1
    if "slow" not in st.session_state or st.session_state.slow < slow_min:
        st.session_state.slow = max(slow_min, slow_default)
    slow = st.slider("MA 慢线", min_value=slow_min, max_value=250, key="slow")

    stock_commission = st.number_input(
        "A股佣金（%）",
        min_value=0.0,
        max_value=1.0,
        value=max(0.0, min(1.0, float(config["stock_commission"]) * 100)),
        step=0.01,
    ) / 100
    stock_stamp_duty = st.number_input(
        "A股卖出印花税（%）",
        min_value=0.0,
        max_value=1.0,
        value=max(0.0, min(1.0, float(config["stock_stamp_duty"]) * 100)),
        step=0.01,
    ) / 100
    etf_commission = st.number_input(
        "ETF佣金（%）",
        min_value=0.0,
        max_value=1.0,
        value=max(0.0, min(1.0, float(config["etf_commission"]) * 100)),
        step=0.01,
    ) / 100
    init_cash = st.number_input(
        "回测初始资金",
        min_value=1000.0,
        value=max(1000.0, float(config["init_cash"])),
        step=10000.0,
        format="%.0f",
    )
    stop_loss = st.number_input(
        "止损比例（%，0=关闭）",
        min_value=0.0,
        max_value=50.0,
        value=max(0.0, min(50.0, float(config["stop_loss"]) * 100)),
        step=0.5,
    ) / 100
    benchmark_code = st.text_input("基准指数代码（留空关闭）", value=str(config["benchmark"]))
    trades_n = st.slider("每个标的显示最近交易数", 1, 30, 5)
    show_scan = st.checkbox("显示样本外参数扫描", value=False)
    show_walk_forward = st.checkbox("显示滚动窗口稳健性扫描（较慢）", value=False)

tracked_codes = list(dict.fromkeys([item["code"] for item in watchlist] + [item["code"] for item in holdings]))

if tracked_codes:
    refresh_requested = bool(st.session_state.refresh_requested)
    try:
        prices, load_errors = get_prices(
            tuple(tracked_codes),
            refresh_requested,
            int(st.session_state.refresh_version),
            str(config["start_date"]),
            int(config["update_retries"]),
            int(config["update_workers"]),
        )
    except DataUnavailableError as exc:
        st.error(str(exc))
        st.stop()
    finally:
        if refresh_requested:
            st.session_state.refresh_requested = False
else:
    prices, load_errors = {}, {}

for code, error in load_errors.items():
    st.warning(f"{code}：{error}")
for code, price in prices.items():
    if warning := price.attrs.get("warning"):
        st.warning(warning)

all_names = _records_names([*watchlist, *holdings])
all_types = {code: asset_type(code) for code in prices}
available_research = [code for code in research_codes if code in prices]

if prices:
    latest_dates = [price["日期"].iloc[-1].date() for price in prices.values()]
    st.caption(f"本次展示数据最晚截至：{max(latest_dates)}。")

with st.expander("📡 数据更新记录"):
    if tracked_codes:
        meta_rows = [{"代码": code, "最后成功更新": cache_updated_at(code) or "-"} for code in tracked_codes]
        st.dataframe(pd.DataFrame(meta_rows), width="stretch", hide_index=True)
    updates = load_update_log()
    if updates:
        st.dataframe(pd.DataFrame(updates).head(20), width="stretch", hide_index=True)
    else:
        st.caption("暂无更新记录；点击侧栏“更新日线数据”后生成。")

st.header("💼 本地持仓")
holding_columns = ["代码", "名称", "数量", "成本价", "备注"]
holding_editor = pd.DataFrame(
    [
        {
            "代码": record["code"],
            "名称": record["name"],
            "数量": record["quantity"],
            "成本价": record["avg_cost"],
            "备注": record["note"],
        }
        for record in holdings
    ],
    columns=holding_columns,
)
with st.form("holdings_form"):
    edited_holdings = st.data_editor(
        holding_editor,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        column_config={
            "代码": st.column_config.TextColumn(help="6位A股或场内ETF代码"),
            "数量": st.column_config.NumberColumn(min_value=0.0, format="%.4f"),
            "成本价": st.column_config.NumberColumn(min_value=0.0, format="%.4f"),
        },
    )
    save_positions = st.form_submit_button("保存持仓")
if save_positions:
    try:
        saved = save_holdings(
            {
                "code": row["代码"],
                "name": row["名称"],
                "quantity": row["数量"],
                "avg_cost": row["成本价"],
                "note": row["备注"],
            }
            for _, row in edited_holdings.iterrows()
        )
        st.success(f"已保存 {len(saved)} 条持仓记录。")
        st.rerun()
    except DataValidationError as exc:
        st.error(str(exc))

st.markdown("#### ➕ 记录成交")
with st.form("add_trade", clear_on_submit=True):
    trade_cols = st.columns([1, 1, 1, 1, 1, 1])
    with trade_cols[0]:
        trade_code = st.text_input("代码", placeholder="002463")
    with trade_cols[1]:
        trade_side = st.selectbox("方向", ["买", "卖"])
    with trade_cols[2]:
        trade_price = st.number_input("价格", min_value=0.0, value=0.0, format="%.4f")
    with trade_cols[3]:
        trade_quantity = st.number_input("数量", min_value=0.0, value=0.0, format="%.4f")
    with trade_cols[4]:
        trade_fee = st.number_input("费用", min_value=0.0, value=0.0, format="%.4f")
    with trade_cols[5]:
        trade_date = st.text_input("日期", placeholder="YYYY-MM-DD")
    trade_note = st.text_input("备注（可选）")
    trade_submit = st.form_submit_button("记录成交")
if trade_submit:
    try:
        code = normalize_symbol(trade_code)
        append_trade(
            {
                "date": trade_date,
                "code": code,
                "name": "",
                "side": trade_side,
                "price": trade_price,
                "quantity": trade_quantity,
                "fee": trade_fee,
                "note": trade_note,
            }
        )
        st.success(f"已记录 {code} {trade_side} {trade_quantity} 股。")
        st.rerun()
    except DataValidationError as exc:
        st.error(str(exc))

with st.expander("由成交流水生成持仓"):
    if st.button("用流水覆盖持仓"):
        try:
            trades_journal = load_trades()
            derived = holdings_from_trades(trades_journal)
            save_holdings(derived)
            st.success(f"已由 {len(trades_journal)} 条流水生成 {len(derived)} 条持仓。")
            st.rerun()
        except DataValidationError as exc:
            st.error(f"无法由成交流水生成持仓：{exc}")
    st.caption("按日期顺序和加权平均成本结转；买入费用计入成本，卖出费用不影响剩余成本。")

valuation = holdings_valuation(
    holdings, prices, fast, slow, all_names, stock_commission, stock_stamp_duty, etf_commission
)
if valuation.empty:
    st.info("在上表录入代码、数量和成本价后，这里会显示持仓估值与 MA 状态。")
else:
    metric_left, metric_middle, metric_right = st.columns(3)
    cost_total = (valuation["数量"] * valuation["成本价"]).sum()
    value_total = valuation["市值"].sum()
    pnl_total = valuation["税后盈亏"].sum()
    metric_left.metric("持仓成本", f"{cost_total:,.2f}")
    metric_middle.metric("最新市值", f"{value_total:,.2f}")
    metric_right.metric("税后盈亏", f"{pnl_total:,.2f}", f"{pnl_total / cost_total:.2%}" if cost_total else None)
    st.caption("税后盈亏已扣除卖出佣金/印花税估算；同一代码多条记录代表不同买入批次。")
    st.dataframe(
        valuation.style.format(
            {
                "数量": "{:,.4f}",
                "成本价": "{:,.3f}",
                "最新收盘": "{:,.3f}",
                "市值": "{:,.2f}",
                "浮动盈亏": "{:,.2f}",
                "卖出费用": "{:,.2f}",
                "税后盈亏": "{:,.2f}",
                "收益率": "{:.2%}",
                "税后收益率": "{:.2%}",
                "组合占比": "{:.2%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "导出持仓估值 CSV",
        dataframe_to_csv(valuation),
        file_name="holdings_valuation.csv",
        mime="text/csv",
    )

# 成交流水展示（有流水时即使没有当前持仓也可见）
trades_journal = load_trades()
if trades_journal:
    st.markdown("#### 📒 成交流水")
    journal_df = pd.DataFrame(trades_journal).sort_values(["date", "code"]).reset_index(drop=True)
    st.dataframe(journal_df, width="stretch", hide_index=True)
    st.markdown("#### 💰 成交流水汇总（已实现盈亏）")
    try:
        realized = trade_summary(trades_journal)
        st.dataframe(
            realized.style.format(
                {
                    "买入金额": "{:,.2f}",
                    "卖出金额": "{:,.2f}",
                    "费用": "{:,.2f}",
                    "已实现盈亏": "{:,.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
    except DataValidationError as exc:
        st.warning(f"已实现盈亏暂不可统计：{exc}")

if len(available_research) > 1:
    close_for_overview = close_from_prices(prices, available_research)
    metrics, correlation = portfolio_metrics(close_for_overview)
    st.header("📊 组合概览")
    overview = st.columns(4)
    overview[0].metric("组合累计收益", _format_percent(metrics["组合累计收益"]))
    overview[1].metric("组合年化收益", _format_percent(metrics["组合年化收益"]))
    overview[2].metric("组合最大回撤", _format_percent(metrics["组合最大回撤"]))
    overview[3].metric("组合夏普", "-" if pd.isna(metrics["组合夏普"]) else f"{metrics['组合夏普']:.2f}")
    st.plotly_chart(
        px.imshow(
            correlation,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            aspect="auto",
            labels={"color": "相关性"},
            title="持仓日收益相关性矩阵",
        ),
        width="stretch",
    )
    st.caption("基于所选标的研究日线收盘价的日收益相关性；等权组合。")

if not available_research:
    st.info("请在侧栏至少选择一个有可用日线的自选标的，查看研究结果。")
    st.stop()

try:
    close = close_from_prices(prices, available_research)
    validate_strategy_inputs(close, fast, slow)
except (DataValidationError, DataUnavailableError) as exc:
    st.error(str(exc))
    st.stop()

research_names = {code: all_names.get(code, code) for code in close.columns}
research_types = {code: all_types[code] for code in close.columns}
portfolio = backtest(
    close,
    fast,
    slow,
    research_types,
    stock_commission,
    stock_stamp_duty,
    etf_commission,
    init_cash,
    stop_loss,
)
summary = summarize(
    portfolio,
    close,
    research_types,
    stock_commission,
    stock_stamp_duty,
    etf_commission,
)
combo_portfolio = backtest_portfolio(
    close,
    fast,
    slow,
    research_types,
    stock_commission,
    stock_stamp_duty,
    etf_commission,
    init_cash,
    stop_loss=stop_loss,
)
combo_summary = portfolio_summary(combo_portfolio, init_cash)

st.header("🔔 当前 MA 信号")
st.dataframe(
    current_ma_state(close, fast, slow, research_names, research_types),
    width="stretch",
)
st.caption(
    "MA 交叉在信号出现后的下一交易日收盘执行；“信号执行日”为最近一次金叉/死叉对应的下一交易日，"
    "“已执行”表示该日早于或等于最新交易日，与回测成交日期一致。"
)
with st.expander("📨 信号提醒记录"):
    alerts = load_signal_alerts()
    if alerts:
        st.dataframe(pd.DataFrame(alerts).head(20), width="stretch", hide_index=True)
        st.caption("由 daily_report.py 在收盘更新时对比信号快照生成；看板只读展示。")
    else:
        st.caption("暂无提醒；运行 daily_report.py 后，MA 状态变化会记录在这里。")

st.header(f"📊 回测绩效（MA{fast}/{slow}）")
st.dataframe(fmt_perf(summary, research_names), width="stretch")
st.caption(
    f"成本：A股买入佣金 {stock_commission:.2%}，卖出佣金 {stock_commission:.2%} + 印花税 {stock_stamp_duty:.2%}；"
    f"ETF 双边佣金 {etf_commission:.2%}。买入持有收益已按同费率扣除双边费用。"
    f"止损：{stop_loss:.1%}（0 表示关闭，跌破买入价后下一交易日收盘卖出）。"
    "未模拟最低佣金、涨跌停、停牌与 T+1 成交约束。"
)
st.download_button(
    "导出回测绩效 CSV",
    dataframe_to_csv(summary.reset_index()),
    file_name=f"ma_{fast}_{slow}_performance.csv",
    mime="text/csv",
)

st.header("💰 组合回测（单账户共享资金）")
combo_columns = st.columns(4)
combo_columns[0].metric("组合累计收益", _format_percent(combo_summary["组合累计收益"]))
combo_columns[1].metric("组合年化收益", _format_percent(combo_summary["组合年化收益"]))
combo_columns[2].metric("组合最大回撤", _format_percent(combo_summary["组合最大回撤"]))
combo_columns[3].metric("组合夏普", "-" if pd.isna(combo_summary["组合夏普"]) else f"{combo_summary['组合夏普']:.2f}")
combo_metrics = st.columns(3)
combo_metrics[0].metric("组合交易次数", int(combo_summary["组合交易次数"]))
combo_metrics[1].metric("组合胜率", _format_percent(combo_summary["组合胜率"]))
combo_metrics[2].metric("平均持仓天数", "-" if pd.isna(combo_summary["组合平均持仓天数"]) else f"{combo_summary['组合平均持仓天数']:.0f} 天")
combo_values = combo_portfolio.value()
if isinstance(combo_values, pd.DataFrame):
    combo_values = combo_values.iloc[:, 0]
combo_norm = combo_values / float(init_cash)
combo_chart = px.line(combo_norm, labels={"value": "净值", "index": "日期"}, title="共享资金组合净值（起点 = 1.0）")
benchmark_frame = None
if benchmark_code:
    try:
        benchmark_frame = load_benchmark(benchmark_code, start_date=str(config["start_date"]))
        bench_series = (
            benchmark_frame.set_index("日期")["收盘"].astype(float)
            .reindex(combo_norm.index)
            .ffill()
            .dropna()
        )
        if len(bench_series):
            bench_norm = bench_series / bench_series.iloc[0]
            combo_chart.add_scatter(x=bench_norm.index, y=bench_norm, mode="lines", name=f"基准{benchmark_code}")
    except (DataUnavailableError, DataValidationError) as exc:
        st.warning(f"基准 {benchmark_code} 加载失败：{exc}")
st.plotly_chart(combo_chart, width="stretch")
if benchmark_frame is not None:
    bench_metrics = benchmark_stats(close, benchmark_frame, f"基准{benchmark_code}")
    bench_cols = st.columns(4)
    bench_cols[0].metric("基准累计收益", _format_percent(bench_metrics[f"基准{benchmark_code}累计收益"]))
    bench_cols[1].metric("基准年化收益", _format_percent(bench_metrics[f"基准{benchmark_code}年化收益"]))
    bench_cols[2].metric("基准最大回撤", _format_percent(bench_metrics[f"基准{benchmark_code}最大回撤"]))
    bench_cols[3].metric("基准夏普", "-" if pd.isna(bench_metrics[f"基准{benchmark_code}夏普"]) else f"{bench_metrics[f'基准{benchmark_code}夏普']:.2f}")
st.caption(
    "同一账户只有一份初始资金，所有标的共用现金池；同日多个买入信号按等权切分当时可用现金。"
    "若某标的信号出现时已无现金，则放弃该次买入。"
)
annual_combo = annual_returns(combo_values)
st.dataframe(annual_combo.style.format("{:.2%}"), width="stretch")
st.download_button(
    "导出组合分年度收益 CSV",
    dataframe_to_csv(annual_combo.reset_index()),
    file_name="ma_combo_annual_returns.csv",
    mime="text/csv",
)

st.header("🕯️ K线与实际成交信号")
chart_labels = {label(code, research_names): code for code in close.columns}
selected_label = st.selectbox("选择标的", list(chart_labels))
selected_code = chart_labels[selected_label]
series = close[selected_code]
ohlc = prices[selected_code].set_index("日期")
sig_entries, sig_exits = _ma_signal_tables(close, fast, slow)
merged_exits = apply_stop_loss_exits(close, sig_entries, sig_exits, stop_loss)
entry_execution = sig_entries[selected_code]
exit_execution = sig_exits[selected_code]
stop_only = merged_exits[selected_code] & ~sig_exits[selected_code]

kline = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.72, 0.28],
    vertical_spacing=0.03,
)
kline.add_trace(
    go.Candlestick(
        x=ohlc.index,
        open=ohlc["开盘"],
        high=ohlc["最高"],
        low=ohlc["最低"],
        close=ohlc["收盘"],
        name="K线",
        increasing_line_color="red",
        decreasing_line_color="green",
        hovertemplate=(
            "%{x|%Y-%m-%d}<br>开盘：%{open:.2f}<br>最高：%{high:.2f}"
            "<br>最低：%{low:.2f}<br>收盘：%{close:.2f}<extra></extra>"
        ),
    ),
    row=1,
    col=1,
)
kline.add_trace(
    go.Scatter(x=series.index, y=series.rolling(fast).mean(), name=f"MA{fast}", line=dict(width=1.3, color="orange")),
    row=1,
    col=1,
)
kline.add_trace(
    go.Scatter(x=series.index, y=series.rolling(slow).mean(), name=f"MA{slow}", line=dict(width=1.3, color="royalblue")),
    row=1,
    col=1,
)
kline.add_trace(
    go.Scatter(
        x=series.index[entry_execution],
        y=series[entry_execution],
        mode="markers",
        name="买入执行",
        marker=dict(symbol="triangle-up", size=13, color="red", line=dict(width=1, color="black")),
    ),
    row=1,
    col=1,
)
kline.add_trace(
    go.Scatter(
        x=series.index[exit_execution],
        y=series[exit_execution],
        mode="markers",
        name="卖出执行",
        marker=dict(symbol="triangle-down", size=13, color="green", line=dict(width=1, color="black")),
    ),
    row=1,
    col=1,
)
if stop_loss > 0:
    kline.add_trace(
        go.Scatter(
            x=series.index[stop_only],
            y=series[stop_only],
            mode="markers",
            name="止损卖出",
            marker=dict(symbol="x", size=11, color="purple", line=dict(width=1, color="black")),
        ),
        row=1,
        col=1,
    )
volume = prices[selected_code]["成交量"] if "成交量" in prices[selected_code].columns else None
if volume is not None:
    volume_colors = ["red" if close_val >= open_val else "green" for close_val, open_val in zip(ohlc["收盘"], ohlc["开盘"])]
    kline.add_trace(
        go.Bar(x=ohlc.index, y=volume, name="成交量", marker_color=volume_colors, opacity=0.5),
        row=2,
        col=1,
    )
kline.update_layout(
    height=720,
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", y=1.02),
    margin=dict(l=0, r=0, t=30, b=0),
)
kline.update_xaxes(rangeslider_visible=False)
st.plotly_chart(kline, width="stretch")

st.header("🧾 交易明细")
trades = trades_table(portfolio, trades_n, research_names)
st.dataframe(
    trades.style.format({"买入价": "{:.3f}", "卖出价": "{:.3f}", "收益率": "{:.2%}", "盈亏": "{:,.2f}"}),
    width="stretch",
    hide_index=True,
)
st.download_button(
    "导出交易明细 CSV",
    dataframe_to_csv(trades),
    file_name=f"ma_{fast}_{slow}_trades.csv",
    mime="text/csv",
)

if show_scan:
    st.header("🔎 样本外参数扫描")
    scan = None
    with st.spinner("正在扫描 MA 参数组合…"):
        try:
            scan = scan_params(
                close,
                [int(item) for item in config["scan_fasts"]],
                [int(item) for item in config["scan_slows"]],
                research_types,
                stock_commission,
                stock_stamp_duty,
                etf_commission,
                init_cash,
                stop_loss,
            )
        except DataValidationError as exc:
            st.warning(f"当前数据不足以执行参数扫描：{exc}")
    if scan is not None:
        aggregate = aggregate_scan(scan)
        st.dataframe(
            aggregate.style.format(
                {
                    "训练期中位收益": "{:.2%}",
                    "验证期中位收益": "{:.2%}",
                    "验证期平均收益": "{:.2%}",
                    "全样本中位收益": "{:.2%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        heatmap = aggregate.pivot(index="fast", columns="slow", values="验证期中位收益")
        st.plotly_chart(
            px.imshow(
                heatmap,
                text_auto=".0%",
                color_continuous_scale="RdYlGn",
                aspect="auto",
                labels={"x": "MA慢线", "y": "MA快线", "color": "验证期中位收益"},
                title="多标的样本外验证期中位收益",
            ),
            width="stretch",
        )
        st.caption(
            "前70%日线为训练期，后30%为验证期；按多个标的的验证期中位收益排序，"
            "不将单一股票的峰值当作最佳组合。该排序只是研究排序，不构成参数择优建议。"
        )
        drill_options = [f"MA{row['fast']}/{row['slow']}" for _, row in aggregate.head(20).iterrows()]
        drill_label = st.selectbox("下钻查看参数组合净值", drill_options)
        drill_index = drill_options.index(drill_label)
        drill_row = aggregate.iloc[drill_index]
        drill_fast, drill_slow = int(drill_row["fast"]), int(drill_row["slow"])
        with st.spinner("正在回测选中参数组合…"):
            drill_portfolio = backtest(
                close,
                drill_fast,
                drill_slow,
                research_types,
                stock_commission,
                stock_stamp_duty,
                etf_commission,
                init_cash,
                stop_loss,
            )
        drill_values = drill_portfolio.value()
        drill_fig = px.line(
            drill_values,
            labels={"value": "净值", "index": "日期", "column": "标的"},
            title=f"MA{drill_fast}/{drill_slow} 各标的净值曲线（起点={init_cash:,.0f}）",
        )
        st.plotly_chart(drill_fig, width="stretch")
        st.download_button(
            "导出参数扫描 CSV",
            dataframe_to_csv(aggregate),
            file_name="ma_out_of_sample_scan.csv",
            mime="text/csv",
        )

if show_walk_forward:
    st.header("🚶 滚动窗口稳健性扫描（walk-forward）")
    with st.spinner("正在滚动窗口训练/验证参数组合（较慢）…"):
        try:
            wf = walk_forward_scan(
                close,
                [int(item) for item in config["scan_fasts"]],
                [int(item) for item in config["scan_slows"]],
                research_types,
                stock_commission,
                stock_stamp_duty,
                etf_commission,
                init_cash,
                train_days=504,
                step_days=126,
                stop_loss=stop_loss,
            )
        except DataValidationError as exc:
            wf = None
            st.warning(f"当前数据不足以执行滚动窗口扫描：{exc}")
    if wf is not None:
        wf_summary = walk_forward_summary(wf)
        st.dataframe(
            wf_summary.style.format(
                {
                    "训练期中位收益": "{:.2%}",
                    "验证期中位收益": "{:.2%}",
                    "验证期平均收益": "{:.2%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.dataframe(
            walk_forward_stability(wf).style.format({"选中次数": "{:d}"}),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "每个窗口用前约2年训练、随后约半年验证；参数按训练期多标的收益中位数选择，"
            "验证期不参与选参。选中次数反映参数在不同市场阶段的稳健性，仅作研究参考。"
        )
        st.download_button(
            "导出滚动窗口扫描 CSV",
            dataframe_to_csv(wf),
            file_name="ma_walk_forward.csv",
            mime="text/csv",
        )
