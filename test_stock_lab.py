import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import stock_lab as lab


class StockLabTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_paths = (
            lab.DATA_DIR,
            lab.NAME_CACHE,
            lab.WATCHLIST_FILE,
            lab.HOLDINGS_FILE,
            lab.TRADES_FILE,
            lab.UPDATE_LOG_FILE,
            lab.SIGNAL_SNAPSHOT_FILE,
            lab.SIGNAL_ALERTS_FILE,
        )
        lab.DATA_DIR = Path(self.temp_dir.name)
        lab.NAME_CACHE = lab.DATA_DIR / "symbol_names.json"
        lab.WATCHLIST_FILE = lab.DATA_DIR / "watchlist.json"
        lab.HOLDINGS_FILE = lab.DATA_DIR / "holdings.json"
        lab.TRADES_FILE = lab.DATA_DIR / "trades.json"
        lab.UPDATE_LOG_FILE = lab.DATA_DIR / "update_log.json"
        lab.SIGNAL_SNAPSHOT_FILE = lab.DATA_DIR / "signal_snapshot.json"
        lab.SIGNAL_ALERTS_FILE = lab.DATA_DIR / "signal_alerts.json"

    def tearDown(self):
        (
            lab.DATA_DIR,
            lab.NAME_CACHE,
            lab.WATCHLIST_FILE,
            lab.HOLDINGS_FILE,
            lab.TRADES_FILE,
            lab.UPDATE_LOG_FILE,
            lab.SIGNAL_SNAPSHOT_FILE,
            lab.SIGNAL_ALERTS_FILE,
        ) = self.original_paths
        self.temp_dir.cleanup()

    def test_asset_detection_and_schema_normalisation(self):
        stock = pd.DataFrame(
            {
                "date": ["2026-01-02"],
                "open": [10],
                "high": [11],
                "low": [9],
                "close": [10.5],
            }
        )
        etf = pd.DataFrame(
            {
                "日期": ["2026-01-02"],
                "开盘": [1.0],
                "最高": [1.1],
                "最低": [0.9],
                "收盘": [1.05],
            }
        )
        self.assertEqual(list(lab._normalise_ohlc(stock).columns), ["日期", "开盘", "最高", "最低", "收盘"])
        self.assertEqual(list(lab._normalise_ohlc(etf).columns), ["日期", "开盘", "最高", "最低", "收盘"])
        self.assertEqual(lab.asset_type("002463"), "A股")
        self.assertEqual(lab.asset_type("588000"), "场内ETF")

    def test_execution_signal_is_one_trading_day_after_cross(self):
        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        series = pd.Series([3, 1, 1, 3, 3], index=dates)
        raw_entries, _, executed_entries, _ = lab.ma_signals(series, 2, 3)
        self.assertEqual(raw_entries[raw_entries].index[0], dates[3])
        self.assertEqual(executed_entries[executed_entries].index[0], dates[4])

    def test_a_share_and_etf_fee_rates_are_different(self):
        dates = pd.date_range("2026-01-01", periods=3, freq="D")
        close = pd.DataFrame({"002463": [10, 10, 10], "588000": [1, 1, 1]}, index=dates)
        exits = pd.DataFrame({"002463": [False, True, False], "588000": [False, True, False]}, index=dates)
        fees = lab.fee_matrix(close, exits)
        self.assertAlmostEqual(fees.loc[dates[0], "002463"], 0.0003)
        self.assertAlmostEqual(fees.loc[dates[1], "002463"], 0.0008)
        self.assertAlmostEqual(fees.loc[dates[1], "588000"], 0.0003)

    def test_watchlist_and_holdings_persist(self):
        lab.save_watchlist([{"code": "002463", "name": "沪电股份"}, {"code": "588000", "name": "科创50ETF"}])
        lab.save_holdings(
            [{"code": "588000", "name": "科创50ETF", "quantity": 100, "avg_cost": 1.8, "note": "测试"}]
        )
        self.assertEqual([item["code"] for item in lab.load_watchlist()], ["002463", "588000"])
        self.assertEqual(lab.load_holdings()[0]["quantity"], 100)

    def test_parameter_summary_uses_cross_asset_median(self):
        scan = pd.DataFrame(
            {
                "fast": [20, 20, 50, 50],
                "slow": [60, 60, 120, 120],
                "symbol": ["A", "B", "A", "B"],
                "train_return": [0.1, 0.1, 0.9, -0.9],
                "validation_return": [0.1, 0.1, 0.9, -0.9],
                "full_return": [0.1, 0.1, 0.9, -0.9],
            }
        ).set_index(["fast", "slow", "symbol"])
        summary = lab.aggregate_scan(scan)
        self.assertEqual((summary.loc[0, "fast"], summary.loc[0, "slow"]), (20, 60))
        self.assertEqual(summary.loc[0, "覆盖标的数"], 2)

    def test_fee_rates_reflect_asset_type(self):
        rates = lab.fee_rates({"002463": "A股", "588000": "场内ETF"})
        self.assertAlmostEqual(rates["002463"][0], 0.0003)
        self.assertAlmostEqual(rates["002463"][1], 0.0008)
        self.assertAlmostEqual(rates["588000"][0], 0.0003)
        self.assertAlmostEqual(rates["588000"][1], 0.0003)

    def test_holdings_valuation_applies_exit_fee(self):
        dates = pd.date_range("2026-01-01", periods=100, freq="D")
        price = pd.DataFrame({"日期": dates, "开盘": 10.0, "最高": 10.5, "最低": 9.5, "收盘": 10.0})
        valuation = lab.holdings_valuation(
            [{"code": "002463", "name": "沪电股份", "quantity": 100, "avg_cost": 9.0, "note": ""}],
            {"002463": price},
            5,
            20,
            {"002463": "沪电股份"},
        )
        self.assertAlmostEqual(valuation.loc[0, "市值"], 1000.0)
        self.assertAlmostEqual(valuation.loc[0, "浮动盈亏"], 100.0)
        self.assertAlmostEqual(valuation.loc[0, "卖出费用"], 1000.0 * 0.0008)
        self.assertAlmostEqual(valuation.loc[0, "税后盈亏"], 100.0 - 1000.0 * 0.0008)

    def test_current_ma_state_has_signal_dates(self):
        dates = pd.date_range("2026-01-01", periods=80, freq="D")
        prices = [1.0 if i < 40 else (2.0 if i % 2 else 1.0) for i in range(80)]
        close = pd.DataFrame({"002463": prices}, index=dates)
        state = lab.current_ma_state(close, 5, 20)
        self.assertIn("最近金叉日", state.columns)
        self.assertIn("信号执行日", state.columns)
        self.assertIn("执行状态", state.columns)
        self.assertTrue(state.loc["002463", "MA状态"].startswith(("持有", "空仓等待")))

    def test_portfolio_metrics_returns_metrics_and_correlation(self):
        dates = pd.date_range("2026-01-01", periods=60, freq="D")
        close = pd.DataFrame(
            {
                "002463": np.linspace(10, 12, 60),
                "600519": np.linspace(100, 102, 60),
            },
            index=dates,
        )
        metrics, correlation = lab.portfolio_metrics(close)
        self.assertIn("组合累计收益", metrics.index)
        self.assertIn("组合最大回撤", metrics.index)
        self.assertEqual(correlation.shape, (2, 2))

    def test_holdings_from_trades_weighted_average_cost(self):
        trades = [
            {"date": "2026-01-02", "code": "002463", "name": "沪电股份", "side": "买", "price": 10, "quantity": 100, "fee": 1.0, "note": ""},
            {"date": "2026-01-03", "code": "002463", "name": "沪电股份", "side": "买", "price": 20, "quantity": 100, "fee": 1.0, "note": ""},
            {"date": "2026-01-05", "code": "002463", "name": "沪电股份", "side": "卖", "price": 15, "quantity": 50, "fee": 0.0, "note": ""},
        ]
        holdings = lab.holdings_from_trades(trades)
        self.assertEqual(len(holdings), 1)
        self.assertEqual(holdings[0]["quantity"], 150)
        # 平均成本 = (1000+1 + 2000+1) / 200 = 15.01
        self.assertAlmostEqual(holdings[0]["avg_cost"], 3002 / 200)

    def test_trades_journal_persist(self):
        lab.save_trades([{"date": "2026-01-02", "code": "588000", "name": "科创50ETF", "side": "买", "price": 1.5, "quantity": 200, "fee": 0.3, "note": ""}])
        self.assertEqual(len(lab.load_trades()), 1)

    def test_cache_meta_records_updated_at(self):
        lab._record_cache_meta("002463")
        updated = lab.cache_updated_at("002463")
        self.assertRegex(updated, r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

    def test_load_config_falls_back_to_defaults(self):
        config = lab.load_config(Path(self.temp_dir.name) / "nonexistent.toml")
        self.assertEqual(config["fast"], 20)
        self.assertEqual(config["symbols"], lab.DEFAULT_SYMBOLS)

    def test_load_watchlist_fills_name_from_cache(self):
        lab.NAME_CACHE.write_text(
            json.dumps({"000725": "京东方Ａ"}, ensure_ascii=False),
            encoding="utf-8",
        )
        lab.WATCHLIST_FILE.write_text(
            json.dumps([{"code": "000725", "name": "000725"}], ensure_ascii=False),
            encoding="utf-8",
        )
        loaded = lab.load_watchlist()
        self.assertEqual(loaded[0]["code"], "000725")
        self.assertEqual(loaded[0]["name"], "京东方Ａ")

    def test_loaders_skip_non_object_records(self):
        lab.WATCHLIST_FILE.write_text(
            json.dumps(["坏记录", {"code": "002463", "name": "沪电股份"}], ensure_ascii=False),
            encoding="utf-8",
        )
        self.assertEqual([item["code"] for item in lab.load_watchlist()], ["002463"])
        lab.TRADES_FILE.write_text(
            json.dumps(
                [
                    "坏记录",
                    {"date": "2026-01-02", "code": "002463", "side": "买", "price": 10, "quantity": 100},
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.assertEqual(len(lab.load_trades()), 1)

    def test_fee_matrix_fills_missing_asset_types(self):
        dates = pd.date_range("2026-01-01", periods=3, freq="D")
        close = pd.DataFrame({"002463": [10, 10, 10], "588000": [1, 1, 1]}, index=dates)
        exits = pd.DataFrame({"002463": [False, True, False], "588000": [False, True, False]}, index=dates)
        fees = lab.fee_matrix(close, exits, {"002463": "A股"})
        self.assertAlmostEqual(fees.loc[dates[1], "588000"], 0.0003)

    def test_summarize_applies_fees_to_buy_and_hold(self):
        dates = pd.date_range("2026-01-01", periods=80, freq="B")
        close = pd.DataFrame({"002463": np.full(80, 10.0)}, index=dates)
        portfolio = lab.backtest(close, 5, 20)
        summary = lab.summarize(portfolio, close, {"002463": "A股"})
        expected = (1 - 0.0003) * (1 - 0.0008) - 1
        self.assertAlmostEqual(summary.loc["002463", "buy_hold_return"], expected, places=10)

    def test_scan_params_train_validation_split(self):
        dates = pd.date_range("2026-01-01", periods=100, freq="B")
        close = pd.DataFrame(
            {
                "002463": np.linspace(10, 15, 100) + np.sin(np.arange(100)) * 0.1,
                "600519": np.linspace(100, 110, 100) + np.cos(np.arange(100)) * 0.1,
            },
            index=dates,
        )
        scan = lab.scan_params(close, [10], [30])
        self.assertIn("train_return", scan.columns)
        self.assertIn("validation_return", scan.columns)
        self.assertIn("full_return", scan.columns)

    def test_fee_rates_accepts_none_asset_types(self):
        self.assertEqual(lab.fee_rates(None), {})

    def test_market_prefix_and_fund_detection(self):
        self.assertEqual(lab._market_prefix("510300"), "sh")
        self.assertEqual(lab._market_prefix("588000"), "sh")
        self.assertEqual(lab._market_prefix("159659"), "sz")
        self.assertEqual(lab._market_prefix("430047"), "bj")
        self.assertEqual(lab._market_prefix("830799"), "bj")
        self.assertEqual(lab.asset_type("501300"), "场内ETF")
        self.assertEqual(lab.asset_type("159659"), "场内ETF")

    def test_load_watchlist_skips_invalid_records(self):
        lab.WATCHLIST_FILE.write_text(
            json.dumps(
                [
                    {"code": "bad!!", "name": "坏记录"},
                    {"code": "002463", "name": "沪电股份"},
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.assertEqual([item["code"] for item in lab.load_watchlist()], ["002463"])

    def test_save_holdings_allows_multiple_lots(self):
        saved = lab.save_holdings(
            [
                {"code": "002463", "quantity": 100, "avg_cost": 10},
                {"code": "002463", "quantity": 200, "avg_cost": 11},
            ]
        )
        self.assertEqual(len(saved), 2)
        self.assertEqual([item["code"] for item in saved], ["002463", "002463"])

    def test_clean_trade_rejects_invalid_date(self):
        with self.assertRaises(lab.DataValidationError):
            lab._clean_trade(
                {"date": "2026-13-99", "code": "002463", "side": "买", "price": 10, "quantity": 100}
            )

    def test_holdings_from_trades_sorts_by_date(self):
        trades = [
            {"date": "2026-01-05", "code": "002463", "name": "沪电股份", "side": "卖", "price": 15, "quantity": 50, "fee": 0.0, "note": ""},
            {"date": "2026-01-02", "code": "002463", "name": "沪电股份", "side": "买", "price": 10, "quantity": 100, "fee": 1.0, "note": ""},
            {"date": "2026-01-03", "code": "002463", "name": "沪电股份", "side": "买", "price": 20, "quantity": 100, "fee": 1.0, "note": ""},
        ]
        holdings = lab.holdings_from_trades(trades)
        self.assertEqual(len(holdings), 1)
        self.assertEqual(holdings[0]["quantity"], 150)
        self.assertAlmostEqual(holdings[0]["avg_cost"], 15.01)

    def test_holdings_from_trades_rejects_oversell(self):
        trades = [
            {"date": "2026-01-02", "code": "002463", "name": "沪电股份", "side": "买", "price": 10, "quantity": 100, "fee": 0.0, "note": ""},
            {"date": "2026-01-03", "code": "002463", "name": "沪电股份", "side": "卖", "price": 10, "quantity": 150, "fee": 0.0, "note": ""},
        ]
        with self.assertRaises(lab.DataValidationError):
            lab.holdings_from_trades(trades)

    def test_validate_strategy_inputs_rejects_bad_windows(self):
        close = pd.DataFrame({"002463": np.linspace(10, 12, 80)})
        with self.assertRaises(lab.DataValidationError):
            lab.validate_strategy_inputs(close, 0, 20)
        with self.assertRaises(lab.DataValidationError):
            lab.validate_strategy_inputs(close, 20, 20)
        self.assertEqual(lab.validate_strategy_inputs(close, 5.0, 20.0), (5, 20))

    def test_backtest_honours_init_cash(self):
        dates = pd.date_range("2026-01-01", periods=80, freq="B")
        close = pd.DataFrame({"002463": np.linspace(10, 12, 80)}, index=dates)
        portfolio = lab.backtest(close, 5, 20, init_cash=50_000)
        self.assertAlmostEqual(portfolio.value().iloc[0, 0], 50_000)

    def test_fetch_price_filters_cached_data_by_start_date(self):
        dates = pd.date_range("2024-01-02", periods=4, freq="B")
        frame = pd.DataFrame({"日期": dates, "开盘": 10.0, "最高": 11.0, "最低": 9.0, "收盘": 10.0})
        frame.to_csv(lab.DATA_DIR / "002463_daily_qfq.csv", index=False, encoding="utf-8-sig")
        with mock.patch.object(lab, "_fetch_stock_raw", side_effect=AssertionError("不应访问网络")):
            out = lab.fetch_price("002463", start_date="2024-01-04")
        self.assertEqual(len(out), 2)
        self.assertEqual(out["日期"].iloc[0], pd.Timestamp("2024-01-04"))
        self.assertTrue((out["日期"] >= pd.Timestamp("2024-01-04")).all())

    def test_fetch_price_refetches_when_cache_does_not_cover_start(self):
        dates = pd.date_range("2024-01-02", periods=4, freq="B")
        frame = pd.DataFrame({"日期": dates, "开盘": 10.0, "最高": 11.0, "最低": 9.0, "收盘": 10.0})
        frame.to_csv(lab.DATA_DIR / "002463_daily_qfq.csv", index=False, encoding="utf-8-sig")
        full = frame.copy()
        full["日期"] = pd.date_range("2023-12-01", periods=4, freq="B")
        with mock.patch.object(lab, "_fetch_stock_raw", return_value=full) as fetcher:
            out = lab.fetch_price("002463", start_date="2023-12-01")
        self.assertEqual(fetcher.call_count, 1)
        self.assertEqual(len(out), 4)

    def test_load_config_sanitises_bad_types(self):
        target = Path(self.temp_dir.name) / "config.toml"
        target.write_text(
            'fast = "20"\ninit_cash = -1\nscan_fasts = [5, "10", 30]\nstock_commission = -0.1\n',
            encoding="utf-8",
        )
        config = lab.load_config(target)
        self.assertEqual(config["fast"], 20)
        self.assertEqual(config["init_cash"], lab.DEFAULT_INIT_CASH)
        self.assertEqual(config["scan_fasts"], [5, 10, 30])
        self.assertEqual(config["stock_commission"], lab.DEFAULT_STOCK_COMMISSION)

    def test_portfolio_size_table_splits_same_day_signals(self):
        dates = pd.date_range("2026-01-01", periods=20, freq="B")
        close = pd.DataFrame({"A": [10.0] * 20, "B": [20.0] * 20}, index=dates)
        entries = pd.DataFrame(False, index=dates, columns=["A", "B"])
        entries.loc[dates[5]] = True
        size = lab._portfolio_size_table(close, entries, None)
        self.assertAlmostEqual(size.loc[dates[5], "A"], 0.5)
        self.assertAlmostEqual(size.loc[dates[5], "B"], 1.0)

    def test_backtest_portfolio_returns_grouped_value(self):
        dates = pd.date_range("2026-01-01", periods=120, freq="B")
        close = pd.DataFrame(
            {
                "002463": np.linspace(10, 12, 120) + np.sin(np.arange(120)) * 0.3,
                "600519": np.linspace(20, 22, 120) + np.cos(np.arange(120)) * 0.3,
            },
            index=dates,
        )
        portfolio = lab.backtest_portfolio(close, 5, 20, init_cash=100_000)
        self.assertIsInstance(portfolio.value(), pd.Series)
        self.assertEqual(len(portfolio.value()), len(close))
        self.assertIn("组合累计收益", lab.portfolio_summary(portfolio).index)

    def test_stop_loss_exits_next_day_after_breach(self):
        dates = pd.date_range("2026-01-01", periods=10, freq="B")
        close = pd.DataFrame({"A": [10.0, 10.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]}, index=dates)
        entries = pd.DataFrame(False, index=dates, columns=["A"])
        entries.loc[dates[1]] = True
        exits = pd.DataFrame(False, index=dates, columns=["A"])
        merged = lab.apply_stop_loss_exits(close, entries, exits, 0.05)
        self.assertFalse(merged.loc[dates[2], "A"])
        self.assertTrue(merged.loc[dates[3], "A"])

    def test_trade_summary_realized_pnl(self):
        trades = [
            {"date": "2026-01-02", "code": "002463", "name": "沪电股份", "side": "买", "price": 10, "quantity": 100, "fee": 1.0, "note": ""},
            {"date": "2026-01-03", "code": "002463", "name": "沪电股份", "side": "卖", "price": 15, "quantity": 50, "fee": 0.5, "note": ""},
        ]
        summary = lab.trade_summary(trades)
        self.assertAlmostEqual(summary.loc[0, "已实现盈亏"], 15 * 50 - 0.5 - (10 * 100 + 1) / 100 * 50)

    def test_holdings_valuation_supports_multiple_lots(self):
        dates = pd.date_range("2026-01-01", periods=100, freq="D")
        price = pd.DataFrame({"日期": dates, "开盘": 10.0, "最高": 10.5, "最低": 9.5, "收盘": 10.0})
        valuation = lab.holdings_valuation(
            [
                {"code": "002463", "name": "沪电股份", "quantity": 100, "avg_cost": 9.0, "note": "第一批"},
                {"code": "002463", "name": "沪电股份", "quantity": 200, "avg_cost": 10.0, "note": "第二批"},
            ],
            {"002463": price},
            5,
            20,
        )
        self.assertEqual(len(valuation), 2)
        self.assertEqual(list(valuation["批次"]), [1, 2])

    def test_annual_returns_by_calendar_year(self):
        dates = pd.date_range("2024-06-01", periods=200, freq="B")
        values = pd.Series(np.linspace(100, 200, len(dates)), index=dates)
        annual = lab.annual_returns(values)
        self.assertIn(2024, annual.index)
        self.assertIn(2025, annual.index)
        self.assertEqual(list(annual.columns), ["组合"])

    def test_signal_snapshot_detect_changes(self):
        dates = pd.date_range("2026-01-01", periods=80, freq="D")
        prices = [1.0 if i < 40 else (2.0 if i % 2 else 1.0) for i in range(80)]
        close = pd.DataFrame({"002463": prices}, index=dates)
        lab.save_signal_snapshot(close, 5, 20)
        self.assertEqual(lab.detect_signal_changes(close, 5, 20), [])
        alerts = lab.record_signal_alerts([{"code": "002463", "direction": "金叉", "date": "2026-03-01", "fast": 5, "slow": 20}])
        self.assertEqual(len(lab.load_signal_alerts()), 1)
        self.assertIn("金叉", alerts[0]["message"])

    def test_update_log_roundtrip(self):
        lab._append_update_log("002463", "增量更新成功", "新增 1 条日线。")
        entries = lab.load_update_log()
        self.assertEqual(entries[0]["code"], "002463")
        self.assertEqual(entries[0]["status"], "增量更新成功")

    def test_fetch_price_incremental_merges_new_rows(self):
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        frame = pd.DataFrame({"日期": dates, "开盘": 10.0, "最高": 11.0, "最低": 9.0, "收盘": 10.0})
        frame.to_csv(lab.DATA_DIR / "002463_daily_qfq.csv", index=False, encoding="utf-8-sig")
        new_dates = pd.date_range(dates[-1], periods=2, freq="B")
        fresh = pd.DataFrame({"日期": new_dates, "开盘": 10.0, "最高": 11.0, "最低": 9.0, "收盘": 10.0})
        with mock.patch.object(lab, "_fetch_stock_raw", return_value=fresh) as fetcher:
            out = lab.fetch_price("002463", refresh=True, start_date="2024-01-01", retries=1)
        self.assertEqual(len(out), 5)
        call_args = fetcher.call_args.args
        self.assertEqual(call_args[0], "002463")
        self.assertLess(call_args[1], "20240105")

    def test_benchmark_stats_aligns_to_close_index(self):
        dates = pd.date_range("2026-01-01", periods=60, freq="B")
        close = pd.DataFrame({"A": np.linspace(10, 12, 60)}, index=dates)
        benchmark = pd.DataFrame({"日期": dates, "收盘": np.linspace(100, 110, 60)})
        stats = lab.benchmark_stats(close, benchmark, "沪深300")
        self.assertIn("沪深300累计收益", stats.index)
        self.assertAlmostEqual(stats["沪深300累计收益"], 110 / 100 - 1, places=10)

    def test_walk_forward_scan_produces_out_of_sample_windows(self):
        dates = pd.date_range("2026-01-01", periods=160, freq="B")
        close = pd.DataFrame(
            {
                "002463": np.linspace(10, 12, 160) + np.sin(np.arange(160)) * 0.1,
                "600519": np.linspace(20, 24, 160) + np.cos(np.arange(160)) * 0.1,
            },
            index=dates,
        )
        wf = lab.walk_forward_scan(close, [5], [20], train_days=80, step_days=40)
        self.assertEqual(len(lab.walk_forward_summary(wf)), 2)
        self.assertIn("选中次数", lab.walk_forward_stability(wf).columns)

    def test_send_webhook_without_url_is_noop(self):
        ok, message = lab.send_webhook("", "test")
        self.assertFalse(ok)
        self.assertIn("未配置", message)


if __name__ == "__main__":
    unittest.main()
