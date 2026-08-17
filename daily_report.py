"""收盘后自动刷新日线并生成回测报告。

用法：
    .venv\\Scripts\\python.exe daily_report.py [--scan]

可配合系统计划任务在每个交易日收盘后运行；产物写入 reports/ 与 data/。
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime

import stock_lab as lab


def main() -> None:
    parser = argparse.ArgumentParser(description="每日收盘后刷新并生成报告")
    parser.add_argument("--scan", action="store_true", help="额外执行参数扫描")
    parser.add_argument("--walk-forward", action="store_true", help="额外执行滚动窗口扫描（较慢）")
    args = parser.parse_args()

    config = lab.load_config()
    symbols = config["symbols"]
    fast = int(config["fast"])
    slow = int(config["slow"])
    start_date = str(config["start_date"])
    init_cash = float(config["init_cash"])
    stop_loss = float(config["stop_loss"])
    stock_commission = float(config["stock_commission"])
    stock_stamp_duty = float(config["stock_stamp_duty"])
    etf_commission = float(config["etf_commission"])
    notify_url = str(config["notify_url"])
    benchmark_code = str(config["benchmark"])

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 刷新日线：{', '.join(symbols)}")
    prices, errors = lab.load_prices(
        symbols,
        refresh=True,
        start_date=start_date,
        retries=int(config["update_retries"]),
        workers=int(config["update_workers"]),
    )
    for code, error in errors.items():
        if code in prices:
            print(f"  警告：{code} 更新失败，使用缓存：{error}")
        else:
            print(f"  警告：{code} 更新失败，跳过：{error}")
    for code in prices:
        updated = lab.cache_updated_at(code)
        print(f"  {code} 缓存最后更新：{updated or '未知'}")

    close = lab.close_from_prices(prices, symbols)
    lab.validate_strategy_inputs(close, fast, slow)
    names = lab.load_names(close.columns)
    types = {code: lab.asset_type(code) for code in close.columns}
    portfolio = lab.backtest(
        close,
        fast,
        slow,
        types,
        stock_commission,
        stock_stamp_duty,
        etf_commission,
        init_cash,
        stop_loss,
    )
    summary = lab.summarize(
        portfolio,
        close,
        types,
        stock_commission,
        stock_stamp_duty,
        etf_commission,
    )
    combo_portfolio = lab.backtest_portfolio(
        close,
        fast,
        slow,
        types,
        stock_commission,
        stock_stamp_duty,
        etf_commission,
        init_cash,
        stop_loss=stop_loss,
    )
    combo_summary = lab.portfolio_summary(combo_portfolio, init_cash)
    scan = (
        lab.scan_params(
            close,
            config["scan_fasts"],
            config["scan_slows"],
            types,
            stock_commission,
            stock_stamp_duty,
            etf_commission,
            init_cash,
        )
        if args.scan
        else None
    )

    benchmark_frame = None
    if benchmark_code:
        try:
            benchmark_frame = lab.load_benchmark(benchmark_code, start_date=start_date)
            print(lab.benchmark_stats(close, benchmark_frame, f"基准{benchmark_code}").to_string())
        except (lab.DataValidationError, lab.DataUnavailableError) as exc:
            print(f"  警告：基准加载失败：{exc}")

    lab.export_report(
        summary,
        scan,
        close,
        portfolio,
        fast,
        slow,
        names,
        init_cash,
        portfolio_summary=combo_summary,
    )
    lab.plot_equity(close, portfolio, lab.REPORT_DIR / "equity_curve.png", init_cash, benchmark_frame)
    print(lab.fmt_perf(summary, names).to_string())
    print("\n共享资金组合绩效：")
    print(combo_summary.to_string())
    print("\n组合分年度收益：")
    print(lab.annual_returns(combo_portfolio.value()).to_string(float_format=lambda value: f"{value:.2%}"))

    if args.walk_forward:
        wf = lab.walk_forward_scan(
            close,
            config["scan_fasts"],
            config["scan_slows"],
            types,
            stock_commission,
            stock_stamp_duty,
            etf_commission,
            init_cash,
            stop_loss=stop_loss,
        )
        wf.to_csv(lab.REPORT_DIR / "walk_forward.csv", index=False, encoding="utf-8-sig")
        print(lab.walk_forward_summary(wf).to_string(index=False))
        print(lab.walk_forward_stability(wf).to_string(index=False))

    # 信号变化检测：首次运行只初始化快照，之后有新金叉/死叉才提醒。
    first_run = not lab.SIGNAL_SNAPSHOT_FILE.exists()
    changes = lab.detect_signal_changes(close, fast, slow)
    if first_run:
        print("首次运行：初始化 MA 信号快照，不发送提醒。")
    elif changes:
        alerts = lab.record_signal_alerts(changes)
        for alert in alerts:
            print(f"信号变化：{alert['message']}")
        if notify_url:
            ok, message = lab.send_webhook(notify_url, "\n".join(alert["message"] for alert in alerts))
            print(f"Webhook 提醒：{'成功' if ok else '失败'}：{message}")
    lab.save_signal_snapshot(close, fast, slow)

    print("报告已写入 reports/。")


if __name__ == "__main__":
    try:
        main()
    except (lab.DataValidationError, lab.DataUnavailableError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
