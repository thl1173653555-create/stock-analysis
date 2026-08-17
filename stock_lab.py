"""A 股与场内 ETF 的日线研究、均线回测和本地持仓工具。"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    tomllib = None  # type: ignore[assignment]

import akshare as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"
NAME_CACHE = DATA_DIR / "symbol_names.json"
WATCHLIST_FILE = DATA_DIR / "watchlist.json"
HOLDINGS_FILE = DATA_DIR / "holdings.json"
TRADES_FILE = DATA_DIR / "trades.json"
CONFIG_FILE = BASE_DIR / "config.toml"
UPDATE_LOG_FILE = DATA_DIR / "update_log.json"
SIGNAL_SNAPSHOT_FILE = DATA_DIR / "signal_snapshot.json"
SIGNAL_ALERTS_FILE = DATA_DIR / "signal_alerts.json"

_META_LOCK = threading.Lock()
_UPDATE_LOG_LOCK = threading.Lock()

DEFAULT_SYMBOLS = ["002463", "600519", "000001", "300750", "600036"]
START_DATE = "20210101"
TRADING_DAYS_PER_YEAR = 252
DEFAULT_INIT_CASH = 100_000
DEFAULT_UPDATE_WORKERS = 4
DEFAULT_UPDATE_RETRIES = 2
# 缓存覆盖判定容忍的非交易日间隔（春节/国庆长假）。
CACHE_COVERAGE_GRACE_DAYS = 15
# 增量更新时向前多取的自然日数，覆盖停牌与长假造成的断层。
INCREMENTAL_OVERLAP_DAYS = 14

DEFAULT_STOCK_COMMISSION = 0.0003
DEFAULT_STOCK_STAMP_DUTY = 0.0005
DEFAULT_ETF_COMMISSION = 0.0003
# 沪市场内基金（50/51/52/53/56/58 开头）与深市场内基金（15/16/18 开头）。
ETF_PREFIXES = ("15", "16", "18", "50", "51", "52", "53", "56", "58")

# 配置默认值：config.toml 存在时覆盖；缺失字段回落到这里。
DEFAULT_CONFIG: dict[str, object] = {
    "symbols": DEFAULT_SYMBOLS,
    "start_date": START_DATE,
    "stock_commission": DEFAULT_STOCK_COMMISSION,
    "stock_stamp_duty": DEFAULT_STOCK_STAMP_DUTY,
    "etf_commission": DEFAULT_ETF_COMMISSION,
    "fast": 20,
    "slow": 60,
    "scan_fasts": [5, 10, 15, 20, 30, 50],
    "scan_slows": [60, 90, 120, 150, 200],
    "init_cash": DEFAULT_INIT_CASH,
    "benchmark": "000300",
    "update_workers": DEFAULT_UPDATE_WORKERS,
    "update_retries": DEFAULT_UPDATE_RETRIES,
    "notify_url": "",
    "stop_loss": 0.0,
}


def load_config(path: Path | None = None) -> dict[str, object]:
    """读取 config.toml，合并默认值并做类型清洗；无法解析时回落默认配置。"""
    config: dict[str, object] = dict(DEFAULT_CONFIG)
    target = Path(path) if path is not None else CONFIG_FILE
    if not target.exists():
        return config
    if tomllib is None:
        return config
    try:
        data = tomllib.loads(target.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError):
        return config
    if not isinstance(data, dict):
        return config
    int_keys = ("fast", "slow", "init_cash", "update_workers", "update_retries")
    float_keys = ("stock_commission", "stock_stamp_duty", "etf_commission", "stop_loss")
    list_keys = ("scan_fasts", "scan_slows")
    for key, value in data.items():
        try:
            if key in int_keys:
                value = int(value)
                if key in {"fast", "slow", "update_workers", "update_retries"} and value < 1:
                    continue
                if key == "init_cash" and value <= 0:
                    continue
            elif key in float_keys:
                value = float(value)
                if value < 0 or (key == "stop_loss" and value >= 1):
                    continue
            elif key in list_keys:
                value = [int(item) for item in value]
                if not value or any(item < 1 for item in value):
                    continue
            elif key == "symbols":
                if isinstance(value, str):
                    value = value.split(",")
                value = [str(item).strip() for item in value if str(item).strip()]
                if not value:
                    continue
            elif key in {"start_date", "benchmark", "notify_url"}:
                value = str(value).strip()
                if not value and key != "notify_url":
                    continue
            else:
                continue
        except (TypeError, ValueError):
            continue
        config[key] = value
    return config

_BUILTIN_NAMES = {
    "002463": "沪电股份",
    "600519": "贵州茅台",
    "000001": "平安银行",
    "300750": "宁德时代",
    "600036": "招商银行",
    "159659": "纳斯达克100ETF",
    "588000": "科创50ETF",
}

_COLUMN_LABELS = {
    "strategy_return": "策略收益",
    "buy_hold_return": "买入持有",
    "strategy_minus_hold": "超额收益",
    "annualized_return": "年化收益",
    "max_drawdown": "最大回撤",
    "max_drawdown_duration": "回撤持续",
    "downside_volatility": "下行波动",
    "calmar_ratio": "卡玛",
    "sharpe": "夏普",
    "win_rate": "胜率",
    "trades": "交易次数",
    "avg_holding_days": "平均持仓",
    "profit_loss_ratio": "盈亏比",
    "max_loss": "最大单笔亏损",
    "max_consecutive_losses": "最大连亏",
}


class DataValidationError(ValueError):
    """用户输入或历史数据不满足研究条件。"""


class DataUnavailableError(RuntimeError):
    """数据源和本地缓存都无法提供所需数据。"""


def normalize_symbol(value: object) -> str:
    """将 sh600036、600036 等输入统一为 6 位证券代码。"""
    code = str(value).strip().lower()
    if code.endswith(".0") and code[:-2].isdigit():
        code = code[:-2]
    if code[:2] in {"sh", "sz", "bj"}:
        code = code[2:]
    if not (len(code) == 6 and code.isdigit()):
        raise DataValidationError("证券代码必须是 6 位数字，例如 002463 或 588000。")
    return code


def asset_type(symbol: str) -> str:
    """按常见交易代码判断 A 股或场内 ETF。"""
    return "场内ETF" if normalize_symbol(symbol).startswith(ETF_PREFIXES) else "A股"


def _market_prefix(symbol: str) -> str:
    """按代码段映射数据源前缀：沪市（含沪市场内基金）、深市、北交所。"""
    if symbol.startswith(("6", "9", "5")):
        return "sh"
    if symbol.startswith(("4", "8")):
        return "bj"
    return "sz"


def label(symbol: str, names: dict[str, str] | None = None) -> str:
    code = normalize_symbol(symbol)
    name = (names or {}).get(code, _BUILTIN_NAMES.get(code, ""))
    return f"{name}({code})" if name and name != code else code


def load_names(symbols: Iterable[str]) -> dict[str, str]:
    """只使用本地名称缓存，避免打开页面时拉取全市场快照。"""
    cache: dict[str, str] = {}
    if NAME_CACHE.exists():
        try:
            cache = json.loads(NAME_CACHE.read_text(encoding="utf-8"))
            if not isinstance(cache, dict):
                cache = {}
        except (json.JSONDecodeError, OSError):
            cache = {}
    return {
        normalize_symbol(symbol): cache.get(
            normalize_symbol(symbol), _BUILTIN_NAMES.get(normalize_symbol(symbol), normalize_symbol(symbol))
        )
        for symbol in symbols
    }


def _write_json(path: Path, content: object) -> None:
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path, fallback: object) -> object:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return fallback


def default_watchlist() -> list[dict[str, str]]:
    names = load_names(DEFAULT_SYMBOLS)
    return [{"code": code, "name": names[code]} for code in DEFAULT_SYMBOLS]


def _clean_watchlist(records: Iterable[dict[str, object]], *, strict: bool = True) -> list[dict[str, str]]:
    """清洗自选记录。

    strict=True（保存时）遇到非法代码直接报错；strict=False（加载时）跳过坏记录，
    避免手工改坏 JSON 后整个看板无法启动。
    """
    seen: set[str] = set()
    cleaned: list[dict[str, str]] = []
    for record in records:
        if not isinstance(record, dict):
            if strict:
                raise DataValidationError("自选记录必须是包含 code 的对象。")
            continue
        raw_code = str(record.get("code", "")).strip()
        if not raw_code:
            continue
        try:
            code = normalize_symbol(raw_code)
        except DataValidationError:
            if strict:
                raise
            continue
        if code in seen:
            continue
        seen.add(code)
        cleaned.append(
            {"code": code, "name": str(record.get("name", "")).strip()}
        )
    names = load_names(item["code"] for item in cleaned)
    for item in cleaned:
        # 记录里没有名称或名称只是代码本身时，从本地名称缓存补齐。
        if not item["name"] or item["name"] == item["code"]:
            item["name"] = names.get(item["code"], _BUILTIN_NAMES.get(item["code"], item["code"]))
    return cleaned


def load_watchlist() -> list[dict[str, str]]:
    data = _read_json(WATCHLIST_FILE, default_watchlist())
    return _clean_watchlist(data if isinstance(data, list) else default_watchlist(), strict=False)


def save_watchlist(records: Iterable[dict[str, object]]) -> list[dict[str, str]]:
    cleaned = _clean_watchlist(records)
    _write_json(WATCHLIST_FILE, cleaned)
    return cleaned


def _clean_holdings(records: Iterable[dict[str, object]], *, strict: bool = False) -> list[dict[str, object]]:
    """清洗持仓记录。

    同一代码允许出现多条（代表不同买入批次）；严格模式拒绝非法记录，
    宽松模式跳过坏记录。
    """
    cleaned: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, dict):
            if strict:
                raise DataValidationError("持仓记录必须是包含 code 的对象。")
            continue
        raw_code = str(record.get("code", "")).strip()
        if not raw_code:
            continue
        try:
            code = normalize_symbol(raw_code)
        except DataValidationError:
            if strict:
                raise
            continue
        try:
            quantity = float(record.get("quantity", 0))
            avg_cost = float(record.get("avg_cost", 0))
        except (TypeError, ValueError) as exc:
            if strict:
                raise DataValidationError(f"{code} 的数量和成本价必须是数字。") from exc
            continue
        if quantity < 0 or avg_cost < 0:
            if strict:
                raise DataValidationError(f"{code} 的数量和成本价不能为负数。")
            continue
        cleaned.append(
            {
                "code": code,
                "name": str(record.get("name", "")).strip(),
                "quantity": quantity,
                "avg_cost": avg_cost,
                "note": str(record.get("note", "")).strip(),
            }
        )
    names = load_names(str(record["code"]) for record in cleaned)
    for record in cleaned:
        code = str(record["code"])
        if not record["name"] or record["name"] == code:
            record["name"] = names.get(code, _BUILTIN_NAMES.get(code, code))
    return cleaned


def load_holdings() -> list[dict[str, object]]:
    data = _read_json(HOLDINGS_FILE, [])
    return _clean_holdings(data if isinstance(data, list) else [], strict=False)


def save_holdings(records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    cleaned = _clean_holdings(records, strict=True)
    _write_json(HOLDINGS_FILE, cleaned)
    return cleaned


def _clean_trade(record: dict[str, object]) -> dict[str, object]:
    if not isinstance(record, dict):
        raise DataValidationError("成交流水记录必须是包含 code 的对象。")
    code = normalize_symbol(record.get("code", ""))
    raw_date = str(record.get("date", "")).strip()
    if not raw_date:
        raise DataValidationError(f"{code} 缺少成交日期，请填写 YYYY-MM-DD。")
    try:
        trade_date = date.fromisoformat(raw_date)
    except ValueError as exc:
        raise DataValidationError(f"{code} 的成交日期“{raw_date}”无效，请使用 YYYY-MM-DD。") from exc
    side = str(record.get("side", "")).strip()
    if side not in {"买", "卖"}:
        raise DataValidationError(f"{code} 的成交方向必须是“买”或“卖”。")
    try:
        price = float(record.get("price", 0))
        quantity = float(record.get("quantity", 0))
        fee = float(record.get("fee", 0.0) or 0.0)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{code} 的成交价格、数量和费用必须是数字。") from exc
    if price <= 0 or quantity <= 0:
        raise DataValidationError(f"{code} 的成交价格和数量必须大于 0。")
    if fee < 0:
        raise DataValidationError(f"{code} 的成交费用不能为负数。")
    return {
        "date": str(trade_date),
        "code": code,
        "name": str(record.get("name", "")).strip(),
        "side": side,
        "price": price,
        "quantity": quantity,
        "fee": fee,
        "note": str(record.get("note", "")).strip(),
    }


def load_trades() -> list[dict[str, object]]:
    data = _read_json(TRADES_FILE, [])
    records = data if isinstance(data, list) else []
    cleaned: list[dict[str, object]] = []
    for record in records:
        try:
            cleaned.append(_clean_trade(record))
        except DataValidationError:
            continue
    return cleaned


def save_trades(records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    cleaned = [_clean_trade(record) for record in records]
    _write_json(TRADES_FILE, cleaned)
    return cleaned


def append_trade(record: dict[str, object]) -> list[dict[str, object]]:
    trades = load_trades()
    trades.append(_clean_trade(record))
    _write_json(TRADES_FILE, trades)
    return trades


def holdings_from_trades(trades: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """由成交流水按日期顺序和加权平均成本汇总持仓。

    卖出按当前持仓平均成本结转成本，买入费用计入成本；卖出费用不影响剩余成本。
    卖出数量超过持仓或未持仓时抛 DataValidationError，而不是静默丢弃。
    """
    cleaned_trades = sorted((_clean_trade(trade) for trade in trades), key=lambda trade: trade["date"])
    positions: dict[str, dict[str, object]] = {}
    for trade in cleaned_trades:
        code = normalize_symbol(trade["code"])
        side = trade["side"]
        price = float(trade["price"])
        quantity = float(trade["quantity"])
        fee = float(trade.get("fee", 0.0))
        if side == "买":
            position = positions.setdefault(
                code, {"code": code, "name": trade.get("name", ""), "quantity": 0.0, "cost": 0.0}
            )
            notional = price * quantity + fee
            position["quantity"] += quantity
            position["cost"] += notional
        else:
            position = positions.get(code)
            if position is None:
                raise DataValidationError(f"{code} 在 {trade['date']} 卖出时没有持仓，无法汇总。")
            held_quantity = float(position["quantity"])
            if quantity > held_quantity + 1e-9:
                raise DataValidationError(
                    f"{code} 在 {trade['date']} 卖出 {quantity:g} 超过当前持仓 {held_quantity:g}，无法汇总。"
                )
            avg_cost = float(position["cost"]) / held_quantity if held_quantity else 0.0
            position["quantity"] -= quantity
            position["cost"] -= avg_cost * quantity
            if position["quantity"] <= 1e-9:
                positions.pop(code)
    result: list[dict[str, object]] = []
    for position in positions.values():
        quantity = float(position["quantity"])
        if quantity <= 1e-9:
            continue
        result.append(
            {
                "code": position["code"],
                "name": position["name"],
                "quantity": quantity,
                "avg_cost": float(position["cost"]) / quantity,
                "note": "由成交流水汇总",
            }
        )
    return result


def trade_summary(trades: Iterable[dict[str, object]]) -> pd.DataFrame:
    """按代码汇总成交流水：买卖金额、费用、已实现盈亏与笔数。

    已实现盈亏 = 卖出金额 - 卖出费用 - 卖出数量对应的加权平均成本，
    与 holdings_from_trades 的成本结转口径一致；卖出超出持仓时抛错。
    """
    cleaned_trades = sorted((_clean_trade(trade) for trade in trades), key=lambda trade: trade["date"])
    positions: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for trade in cleaned_trades:
        code = normalize_symbol(trade["code"])
        side = trade["side"]
        price = float(trade["price"])
        quantity = float(trade["quantity"])
        fee = float(trade.get("fee", 0.0))
        row = positions.setdefault(
            code,
            {
                "code": code,
                "name": trade.get("name", ""),
                "buy_count": 0,
                "sell_count": 0,
                "buy_amount": 0.0,
                "sell_amount": 0.0,
                "fees": 0.0,
                "realized_pnl": 0.0,
                "quantity": 0.0,
                "cost": 0.0,
            },
        )
        if side == "买":
            row["buy_count"] += 1
            row["buy_amount"] += price * quantity
            row["fees"] += fee
            row["quantity"] += quantity
            row["cost"] += price * quantity + fee
        else:
            held_quantity = float(row["quantity"])
            if held_quantity <= 0 or quantity > held_quantity + 1e-9:
                raise DataValidationError(f"{code} 在 {trade['date']} 卖出时持仓不足，无法统计已实现盈亏。")
            avg_cost = float(row["cost"]) / held_quantity
            row["sell_count"] += 1
            row["sell_amount"] += price * quantity
            row["fees"] += fee
            row["realized_pnl"] += price * quantity - fee - avg_cost * quantity
            row["quantity"] -= quantity
            row["cost"] -= avg_cost * quantity
    if not positions:
        return pd.DataFrame(
            columns=["代码", "名称", "买入笔数", "卖出笔数", "买入金额", "卖出金额", "费用", "已实现盈亏"]
        )
    return pd.DataFrame(
        [
            {
                "代码": row["code"],
                "名称": row["name"] or row["code"],
                "买入笔数": row["buy_count"],
                "卖出笔数": row["sell_count"],
                "买入金额": row["buy_amount"],
                "卖出金额": row["sell_amount"],
                "费用": row["fees"],
                "已实现盈亏": row["realized_pnl"],
            }
            for row in positions.values()
        ]
    )


def _normalise_ohlc(raw: pd.DataFrame) -> pd.DataFrame:
    """统一 AKShare 股票和 ETF 日线的中英文列名。"""
    aliases = {
        "日期": ("日期", "date"),
        "开盘": ("开盘", "open"),
        "最高": ("最高", "high"),
        "最低": ("最低", "low"),
        "收盘": ("收盘", "close"),
    }
    resolved: dict[str, str] = {}
    for target, candidates in aliases.items():
        source = next((column for column in candidates if column in raw.columns), None)
        if source is None:
            raise DataUnavailableError(f"日线数据缺少“{target}”字段，无法绘制 K 线或回测。")
        resolved[target] = source
    df = raw[[resolved[column] for column in aliases]].rename(
        columns={source: target for target, source in resolved.items()}
    )
    # 可选成交量字段（缺失时不影响回测与 K 线）。
    volume_source = next((column for column in ("volume", "成交量") if column in raw.columns), None)
    if volume_source is not None:
        df["成交量"] = pd.to_numeric(raw[volume_source], errors="coerce")
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    for column in ("开盘", "最高", "最低", "收盘"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["日期", "收盘"]).sort_values("日期").drop_duplicates("日期")
    if df.empty:
        raise DataUnavailableError("日线数据为空。")
    return df.reset_index(drop=True)


def _read_cache(cache: Path) -> pd.DataFrame:
    return _normalise_ohlc(pd.read_csv(cache, encoding="utf-8-sig"))


def _call_with_retries(fetcher, retries: int):
    """调用数据源，失败时按次退避重试；重试次数耗尽后抛出最后一次异常。"""
    attempts = max(1, int(retries))
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fetcher()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < attempts:
                time.sleep(min(1.0, 0.25 * attempt))
    assert last_exc is not None
    raise last_exc


def _fetch_stock_raw(
    code: str,
    start_date: str = "19900101",
    end_date: str = "20500101",
    retries: int = DEFAULT_UPDATE_RETRIES,
) -> pd.DataFrame:
    """按优先顺序尝试多个 A 股数据源，返回首个可用结果。

    缓存保存全历史；增量更新时传入更晚的 start_date 以缩小下载范围。
    """
    errors: list[str] = []
    prefix = _market_prefix(code)
    sina_fetcher = lambda: ak.stock_zh_a_daily(
        symbol=f"{prefix}{code}", start_date=start_date, end_date=end_date, adjust="qfq"
    )
    em_fetcher = lambda: ak.stock_zh_a_hist(
        symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq"
    )
    # 沪深优先新浪源（历史输出即新浪日线），失败再试东方财富；北交所新浪源通常无数据，直接东财优先。
    fetchers = (em_fetcher, sina_fetcher) if prefix == "bj" else (sina_fetcher, em_fetcher)
    for attempt, fetcher in enumerate(fetchers):
        try:
            return _call_with_retries(fetcher, retries)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"源{attempt + 1}: {exc}")
    raise DataUnavailableError(f"{code} 所有 A 股数据源均失败：{' | '.join(errors)}")


def _fetch_etf_raw(
    code: str,
    start_date: str = "19900101",
    end_date: str = "20500101",
    retries: int = DEFAULT_UPDATE_RETRIES,
) -> pd.DataFrame:
    """按优先顺序尝试多个场内 ETF 数据源，返回首个可用结果。"""
    errors: list[str] = []
    for attempt, fetcher in enumerate(
        (
            lambda: ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq"),
            lambda: ak.fund_etf_hist_sina(symbol=f"{_market_prefix(code)}{code}"),
        )
    ):
        try:
            return _call_with_retries(fetcher, retries)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"源{attempt + 1}: {exc}")
    raise DataUnavailableError(f"{code} 所有 ETF 数据源均失败：{' | '.join(errors)}")


_cache_meta_file = lambda: DATA_DIR / "cache_meta.json"  # noqa: E731


def _read_cache_meta() -> dict[str, str]:
    data = _read_json(_cache_meta_file(), {})
    return data if isinstance(data, dict) else {}


def _record_cache_meta(code: str) -> None:
    with _META_LOCK:
        meta = _read_cache_meta()
        meta[code] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _write_json(_cache_meta_file(), meta)


def cache_updated_at(code: str) -> str:
    """返回指定标的缓存最后成功更新时间，无记录时返回空字符串。"""
    return _read_cache_meta().get(normalize_symbol(code), "")


def _append_update_log(code: str, status: str, message: str) -> None:
    """记录一次数据更新/失败事件（保留最近 200 条）。"""
    with _UPDATE_LOG_LOCK:
        data = _read_json(UPDATE_LOG_FILE, [])
        records = data if isinstance(data, list) else []
        records.append(
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "code": code,
                "status": status,
                "message": message[:500],
            }
        )
        _write_json(UPDATE_LOG_FILE, records[-200:])


def load_update_log() -> list[dict[str, object]]:
    """返回最近的更新日志，新事件在前。"""
    data = _read_json(UPDATE_LOG_FILE, [])
    return list(reversed(data if isinstance(data, list) else []))


def _merge_with_cache(cached: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([cached, fresh], ignore_index=True)
    return (
        merged.sort_values("日期")
        .drop_duplicates("日期", keep="last")
        .reset_index(drop=True)
    )


def _coerce_start_date(start_date: object) -> pd.Timestamp:
    if start_date is None:
        start_date = START_DATE
    if isinstance(start_date, (int, float, np.integer, np.floating)) and not isinstance(start_date, bool):
        raise DataValidationError("起始日期请使用字符串，例如 20210101 或 2021-01-01。")
    try:
        timestamp = pd.Timestamp(start_date)
    except Exception as exc:
        raise DataValidationError(f"起始日期“{start_date}”无效，请使用 YYYYMMDD 或 YYYY-MM-DD。") from exc
    if pd.isna(timestamp):
        raise DataValidationError(f"起始日期“{start_date}”无效，请使用 YYYYMMDD 或 YYYY-MM-DD。")
    return timestamp


def _trim_to_start(frame: pd.DataFrame, code: str, start: pd.Timestamp) -> pd.DataFrame:
    trimmed = frame[frame["日期"] >= start].reset_index(drop=True)
    if trimmed.empty:
        raise DataUnavailableError(f"{code} 没有 {start:%Y-%m-%d} 以来的可用日线数据。")
    return trimmed


def fetch_price(
    symbol: str,
    refresh: bool = False,
    start_date: object = None,
    retries: int = DEFAULT_UPDATE_RETRIES,
) -> pd.DataFrame:
    """读取本地日线并截取 start_date 之后的数据。

    refresh=False 时优先本地缓存；缓存缺失或覆盖不到 start_date 时自动尝试补
    数据源（多源回退）。refresh=True 时先尝试增量更新（只拉最近一段再合并），
    增量失败则全量下载；最终失败回落到本地缓存。
    """
    code = normalize_symbol(symbol)
    start = _coerce_start_date(start_date)
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / f"{code}_daily_qfq.csv"

    if cache.exists() and not refresh:
        try:
            cached = _read_cache(cache)
            first, last = cached["日期"].iloc[0], cached["日期"].iloc[-1]
            # 容忍 start_date 落在缓存首日之前的非交易日（长假等）。
            if first <= start + pd.Timedelta(days=CACHE_COVERAGE_GRACE_DAYS) and last >= start:
                return _trim_to_start(cached, code, start)
        except Exception:  # noqa: BLE001 - 缓存损坏按“无缓存”处理，转入数据源更新。
            pass

    cached_before_update: pd.DataFrame | None = None
    increment_failed = False
    if refresh and cache.exists():
        try:
            cached_before_update = _read_cache(cache)
        except Exception:  # noqa: BLE001
            cached_before_update = None

    def fetch_raw(start: str, end: str) -> pd.DataFrame:
        if asset_type(code) == "场内ETF":
            return _fetch_etf_raw(code, start, end, retries)
        return _fetch_stock_raw(code, start, end, retries)

    try:
        if cached_before_update is not None and not cached_before_update.empty:
            # 增量更新：从缓存末日向前多取一小段，合并去重后写回。
            try:
                since = cached_before_update["日期"].iloc[-1] - pd.Timedelta(days=INCREMENTAL_OVERLAP_DAYS)
                fresh = _normalise_ohlc(fetch_raw(since.strftime("%Y%m%d"), "20500101"))
                if not fresh.empty:
                    merged = _merge_with_cache(cached_before_update, fresh)
                    merged.to_csv(cache, index=False, encoding="utf-8-sig")
                    _record_cache_meta(code)
                    _append_update_log(
                        code, "增量更新成功", f"新增 {max(0, len(merged) - len(cached_before_update))} 条日线。"
                    )
                    return _trim_to_start(merged, code, start)
            except Exception as exc:  # noqa: BLE001 - 增量失败自动转全量。
                increment_failed = True
                _append_update_log(code, "增量失败转全量", str(exc))

        full = _normalise_ohlc(fetch_raw("19900101", "20500101"))
        full.to_csv(cache, index=False, encoding="utf-8-sig")
        _record_cache_meta(code)
        if cached_before_update is not None:
            status = "全量更新" if increment_failed else "刷新更新"
        else:
            status = "自动补拉更新"
        _append_update_log(code, status, f"缓存共 {len(full)} 条日线。")
        return _trim_to_start(full, code, start)
    except Exception as exc:
        if cache.exists():
            try:
                cached = _read_cache(cache)
                trimmed = cached[cached["日期"] >= start].reset_index(drop=True)
                if not trimmed.empty:
                    trimmed.attrs["warning"] = f"{code} 更新失败，正在使用本地缓存：{exc}"
                    _append_update_log(code, "失败(使用缓存)", str(exc))
                    return trimmed
            except Exception:  # noqa: BLE001 - 缓存损坏时沿用原始数据源错误。
                pass
        _append_update_log(code, "失败", str(exc))
        if isinstance(exc, DataUnavailableError):
            raise
        raise DataUnavailableError(f"{code} 数据获取失败，请检查代码或网络后重试。") from exc


def load_prices(
    symbols: Iterable[str],
    refresh: bool = False,
    start_date: object = None,
    retries: int = DEFAULT_UPDATE_RETRIES,
    workers: int = DEFAULT_UPDATE_WORKERS,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """并发加载每个标的；非法代码或失败项不阻断其它标的研究。"""
    prices: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    codes: list[str] = []
    for item in symbols:
        try:
            code = normalize_symbol(item)
        except DataValidationError as exc:
            errors[str(item)] = str(exc)
            continue
        if code in codes:
            continue
        codes.append(code)

    if not codes:
        raise DataUnavailableError("没有可用行情数据，请先检查自选代码或点击更新日线。")

    max_workers = max(1, min(int(workers), len(codes)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_price, code, refresh, start_date, retries): code for code in codes
        }
        for future in as_completed(futures):
            code = futures[future]
            try:
                prices[code] = future.result()
            except (DataValidationError, DataUnavailableError) as exc:
                errors[code] = str(exc)

    ordered_prices = {code: prices[code] for code in codes if code in prices}
    if not ordered_prices:
        raise DataUnavailableError("没有可用行情数据，请先检查自选代码或点击更新日线。")
    return ordered_prices, errors


def close_from_prices(prices: dict[str, pd.DataFrame], symbols: Iterable[str] | None = None) -> pd.DataFrame:
    codes: list[str] = []
    for symbol in symbols if symbols is not None else prices.keys():
        try:
            code = normalize_symbol(symbol)
        except DataValidationError:
            continue
        if code in prices and code not in codes:
            codes.append(code)
    if not codes:
        raise DataUnavailableError("没有可用于回测的标的。")
    close = pd.concat(
        {code: prices[code].set_index("日期")["收盘"].astype(float) for code in codes}, axis=1
    ).sort_index()
    return close.dropna(how="all").ffill()


def _fetch_index_raw(code: str, retries: int = DEFAULT_UPDATE_RETRIES) -> pd.DataFrame:
    errors: list[str] = []
    for attempt, fetcher in enumerate(
        (
            lambda: ak.index_zh_a_hist(symbol=code, period="daily", start_date="19900101", end_date="20500101"),
            lambda: ak.stock_zh_index_daily(symbol=f"sh{code}"),
        )
    ):
        try:
            return _call_with_retries(fetcher, retries)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"源{attempt + 1}: {exc}")
    raise DataUnavailableError(f"指数 {code} 所有数据源均失败：{' | '.join(errors)}")


def load_benchmark(
    code: str = "000300",
    start_date: object = None,
    refresh: bool = False,
    retries: int = DEFAULT_UPDATE_RETRIES,
) -> pd.DataFrame:
    """加载指数日线（默认沪深300），缓存于 data/{code}_index_daily.csv。"""
    benchmark_code = normalize_symbol(code)
    start = _coerce_start_date(start_date)
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / f"{benchmark_code}_index_daily.csv"

    if cache.exists() and not refresh:
        try:
            cached = _normalise_ohlc(pd.read_csv(cache, encoding="utf-8-sig"))
            first, last = cached["日期"].iloc[0], cached["日期"].iloc[-1]
            if first <= start + pd.Timedelta(days=CACHE_COVERAGE_GRACE_DAYS) and last >= start:
                return _trim_to_start(cached, benchmark_code, start)
        except Exception:  # noqa: BLE001
            pass

    try:
        full = _normalise_ohlc(_fetch_index_raw(benchmark_code, retries))
        full.to_csv(cache, index=False, encoding="utf-8-sig")
        _append_update_log(benchmark_code, "指数更新成功", f"缓存共 {len(full)} 条日线。")
        return _trim_to_start(full, benchmark_code, start)
    except Exception as exc:
        if cache.exists():
            try:
                cached = _normalise_ohlc(pd.read_csv(cache, encoding="utf-8-sig"))
                trimmed = cached[cached["日期"] >= start].reset_index(drop=True)
                if not trimmed.empty:
                    _append_update_log(benchmark_code, "指数失败(使用缓存)", str(exc))
                    return trimmed
            except Exception:  # noqa: BLE001
                pass
        _append_update_log(benchmark_code, "指数失败", str(exc))
        if isinstance(exc, DataUnavailableError):
            raise
        raise DataUnavailableError(f"指数 {benchmark_code} 数据获取失败。") from exc


def benchmark_stats(
    close: pd.DataFrame,
    benchmark: pd.DataFrame,
    index_name: str = "基准指数",
) -> pd.Series:
    """把指数收盘对齐到研究区间，返回累计/年化收益、最大回撤与夏普。"""
    aligned = (
        benchmark.set_index("日期")["收盘"].astype(float)
        .reindex(close.index)
        .ffill()
        .dropna()
    )
    if len(aligned) < 2:
        raise DataValidationError("基准指数与研究区间没有足够重叠数据。")
    returns = aligned.pct_change().dropna()
    total = aligned.iloc[-1] / aligned.iloc[0] - 1
    years = len(aligned) / TRADING_DAYS_PER_YEAR
    annualized = (1 + total) ** (1 / years) - 1 if total > -1 else np.nan
    drawdown = aligned / aligned.cummax() - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if returns.std() > 0 else np.nan
    return pd.Series(
        {
            f"{index_name}累计收益": total,
            f"{index_name}年化收益": annualized,
            f"{index_name}最大回撤": drawdown.min(),
            f"{index_name}夏普": sharpe,
        }
    )


def signal_snapshot(close: pd.DataFrame, fast: int, slow: int) -> dict[str, object]:
    """当前每个标的最近一次原始交叉信号（金叉/死叉）。"""
    snapshot: dict[str, object] = {}
    for code in close.columns:
        raw_entries, raw_exits, _, _ = ma_signals(close[code], fast, slow)
        entry_dates = raw_entries.index[raw_entries]
        exit_dates = raw_exits.index[raw_exits]
        if len(entry_dates) and (not len(exit_dates) or entry_dates[-1] > exit_dates[-1]):
            direction, signal_date = "金叉", entry_dates[-1]
        elif len(exit_dates):
            direction, signal_date = "死叉", exit_dates[-1]
        else:
            direction, signal_date = "无", None
        snapshot[code] = {
            "direction": direction,
            "date": signal_date.date().isoformat() if signal_date is not None else None,
            "fast": int(fast),
            "slow": int(slow),
        }
    return snapshot


def detect_signal_changes(close: pd.DataFrame, fast: int, slow: int) -> list[dict[str, object]]:
    """与上次快照比较，返回新增的 MA 状态变化事件。"""
    current = signal_snapshot(close, fast, slow)
    previous = _read_json(SIGNAL_SNAPSHOT_FILE, {})
    previous = previous if isinstance(previous, dict) else {}
    changes: list[dict[str, object]] = []
    for code, info in current.items():
        if not isinstance(info, dict):
            continue
        prev = previous.get(code)
        if (
            isinstance(prev, dict)
            and prev.get("direction") == info.get("direction")
            and prev.get("date") == info.get("date")
        ):
            continue
        if info.get("direction") in {"金叉", "死叉"}:
            changes.append({"code": code, **info})
    return changes


def save_signal_snapshot(close: pd.DataFrame, fast: int, slow: int) -> None:
    _write_json(SIGNAL_SNAPSHOT_FILE, signal_snapshot(close, fast, slow))


def record_signal_alerts(changes: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """把信号变化写入提醒记录，返回包含时间戳的全部记录。"""
    stamped = [
        {
            **change,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": f"{change.get('code')} 出现 {change.get('direction')}（MA{change.get('fast')}/{change.get('slow')}，信号日 {change.get('date')}）",
        }
        for change in changes
    ]
    if not stamped:
        return []
    data = _read_json(SIGNAL_ALERTS_FILE, [])
    records = data if isinstance(data, list) else []
    records.extend(stamped)
    _write_json(SIGNAL_ALERTS_FILE, records[-200:])
    return stamped


def load_signal_alerts() -> list[dict[str, object]]:
    data = _read_json(SIGNAL_ALERTS_FILE, [])
    return list(reversed(data if isinstance(data, list) else []))


def send_webhook(url: str, text: str) -> tuple[bool, str]:
    """向通用 webhook（Server酱/钉钉/自定义机器人等）发送 JSON 文本提醒。"""
    if not url:
        return False, "未配置 notify_url，跳过发送。"
    payload = json.dumps({"title": "MA 信号提醒", "text": text}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8", errors="replace")
            return True, f"HTTP {response.status}：{body[:120]}"
    except Exception as exc:  # noqa: BLE001 - 提醒失败不应中断报告。
        return False, f"发送失败：{exc}"


def _coerce_ma_window(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise DataValidationError(f"MA {name} 必须是正整数。")
    if isinstance(value, (int, np.integer)):
        window = int(value)
    elif isinstance(value, (float, np.floating)) and value.is_integer():
        window = int(value)
    else:
        raise DataValidationError(f"MA {name} 必须是正整数。")
    if window < 1:
        raise DataValidationError(f"MA {name} 必须是正整数。")
    return window


def validate_strategy_inputs(close: pd.DataFrame, fast: int, slow: int) -> tuple[int, int]:
    """校验并归一化 MA 参数，返回 (fast, slow) 整数窗口。"""
    fast = _coerce_ma_window(fast, "快线")
    slow = _coerce_ma_window(slow, "慢线")
    if fast >= slow:
        raise DataValidationError("MA 快线必须小于慢线。")
    if close.empty:
        raise DataValidationError("没有可用收盘价数据。")
    if len(close) <= slow:
        raise DataValidationError(f"历史数据不足：MA{slow} 至少需要 {slow + 1} 个交易日。")
    return fast, slow


def ma_signals(series: pd.Series, fast: int, slow: int) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """返回原始交叉与下一交易日执行的信号。"""
    fast, slow = validate_strategy_inputs(series.to_frame(), fast, slow)
    ma_fast = series.rolling(fast).mean()
    ma_slow = series.rolling(slow).mean()
    raw_entries = (ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))
    raw_exits = (ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))
    return raw_entries, raw_exits, raw_entries.shift(1, fill_value=False), raw_exits.shift(1, fill_value=False)


def _ma_signal_tables(close: pd.DataFrame, fast: int, slow: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """多标的 MA 交叉信号，返回下一交易日收盘执行的 entries/exits。"""
    fast, slow = validate_strategy_inputs(close, fast, slow)
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()
    raw_entries = (ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))
    raw_exits = (ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))
    return raw_entries.shift(1, fill_value=False), raw_exits.shift(1, fill_value=False)


def apply_stop_loss_exits(
    close: pd.DataFrame,
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    stop_loss: float | None,
) -> pd.DataFrame:
    """按“跌破买入价 (1 - stop_loss) 后下一交易日收盘卖出”合并止损出场信号。

    MA 出场与止损出场在同一天时只记一次；止损日与 MA 买回日冲突时交由 vectorbt
    的冲突处理逻辑解决。
    """
    if stop_loss is None or float(stop_loss) <= 0:
        return exits.copy()
    stop_loss = float(stop_loss)
    if not np.isfinite(stop_loss) or stop_loss >= 1:
        raise DataValidationError("止损比例必须在 0 到 1 之间。")
    stop_exits = pd.DataFrame(False, index=close.index, columns=close.columns)
    for code in close.columns:
        holding = False
        entry_price = np.nan
        n = len(close.index)
        for position in range(n):
            if exits[code].iloc[position] and holding:
                holding = False
                entry_price = np.nan
            if entries[code].iloc[position] and not holding:
                holding = True
                entry_price = float(close[code].iloc[position])
            if holding and float(close[code].iloc[position]) <= entry_price * (1 - stop_loss):
                holding = False
                if position + 1 < n:
                    stop_exits.iloc[position + 1, stop_exits.columns.get_loc(code)] = True
    return exits | stop_exits


def fee_matrix(
    close: pd.DataFrame,
    exits: pd.DataFrame,
    asset_types: dict[str, str] | None = None,
    stock_commission: float = DEFAULT_STOCK_COMMISSION,
    stock_stamp_duty: float = DEFAULT_STOCK_STAMP_DUTY,
    etf_commission: float = DEFAULT_ETF_COMMISSION,
) -> pd.DataFrame:
    """按成交日生成费用率：A 股卖出额外收印花税，ETF 不收。"""
    resolved_types = {code: (asset_types or {}).get(code, asset_type(code)) for code in close.columns}
    rates = fee_rates(resolved_types, stock_commission, stock_stamp_duty, etf_commission)
    fees = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for code in close.columns:
        entry_fee, exit_fee = rates[code]
        fees[code] = np.where(exits[code], exit_fee, entry_fee)
    return fees


def fee_rates(
    asset_types: dict[str, str] | None = None,
    stock_commission: float = DEFAULT_STOCK_COMMISSION,
    stock_stamp_duty: float = DEFAULT_STOCK_STAMP_DUTY,
    etf_commission: float = DEFAULT_ETF_COMMISSION,
) -> dict[str, tuple[float, float]]:
    """返回每个标的的 (买入费率, 卖出费率)。ETF 双边佣金，A 股卖出额外收印花税。"""
    asset_types = asset_types or {}
    return {
        code: (
            (etf_commission, etf_commission)
            if asset_types.get(code) == "场内ETF"
            else (stock_commission, stock_commission + stock_stamp_duty)
        )
        for code in asset_types
    }


def backtest(
    close: pd.DataFrame,
    fast: int = 20,
    slow: int = 60,
    asset_types: dict[str, str] | None = None,
    stock_commission: float = DEFAULT_STOCK_COMMISSION,
    stock_stamp_duty: float = DEFAULT_STOCK_STAMP_DUTY,
    etf_commission: float = DEFAULT_ETF_COMMISSION,
    init_cash: float = DEFAULT_INIT_CASH,
    stop_loss: float = 0.0,
) -> vbt.Portfolio:
    """MA 交叉策略，信号于下一交易日收盘执行；可选跌破买入价比例止损。"""
    fast, slow = validate_strategy_inputs(close, fast, slow)
    try:
        init_cash = float(init_cash)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("初始资金必须是数字。") from exc
    if not np.isfinite(init_cash) or init_cash <= 0:
        raise DataValidationError("初始资金必须大于 0。")
    entries, ma_exits = _ma_signal_tables(close, fast, slow)
    exits = apply_stop_loss_exits(close, entries, ma_exits, stop_loss)
    fees = fee_matrix(
        close, exits, asset_types, stock_commission, stock_stamp_duty, etf_commission
    )
    return vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        init_cash=init_cash,
        fees=fees,
        freq="1D",
    )


def _portfolio_size_table(close: pd.DataFrame, entries: pd.DataFrame, weights: dict[str, float] | None) -> pd.DataFrame:
    """为共享资金组合生成 size_type='percent' 的资金分配表。

    同一天出现多个买入信号时，按权重把“当时可用现金”顺序切分：
    第 j 个信号使用剩余现金的 w_j / 剩余权重之和，等价于同日信号按权重等比例分配。
    """
    columns = list(close.columns)
    if weights is None:
        weights = {code: 1.0 for code in columns}
    resolved = {code: float(weights.get(code, 0.0)) for code in columns}
    if any(value < 0 for value in resolved.values()):
        raise DataValidationError("组合权重不能为负数。")
    size = pd.DataFrame(np.nan, index=close.index, columns=columns, dtype=float)
    for position in range(len(close.index)):
        today = [code for code in columns if entries[code].iloc[position]]
        if not today:
            continue
        remaining = sum(resolved[code] for code in today)
        if remaining <= 0:
            raise DataValidationError("当天出现买入信号的所有标的权重之和必须大于 0。")
        for code in today:
            weight = resolved[code]
            if weight <= 0:
                continue
            size.iloc[position, columns.index(code)] = weight / remaining
            remaining -= weight
    return size


def backtest_portfolio(
    close: pd.DataFrame,
    fast: int = 20,
    slow: int = 60,
    asset_types: dict[str, str] | None = None,
    stock_commission: float = DEFAULT_STOCK_COMMISSION,
    stock_stamp_duty: float = DEFAULT_STOCK_STAMP_DUTY,
    etf_commission: float = DEFAULT_ETF_COMMISSION,
    init_cash: float = DEFAULT_INIT_CASH,
    weights: dict[str, float] | None = None,
    stop_loss: float = 0.0,
) -> vbt.Portfolio:
    """单账户共享资金的多标的 MA 组合回测。

    同一账户只有一份初始资金，各标的信号共用现金池；同日多个买入信号按
    weights 切分当时可用现金（缺省等权）。返回组合级 Portfolio（value() 为单列）。
    """
    fast, slow = validate_strategy_inputs(close, fast, slow)
    try:
        init_cash = float(init_cash)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("初始资金必须是数字。") from exc
    if not np.isfinite(init_cash) or init_cash <= 0:
        raise DataValidationError("初始资金必须大于 0。")
    entries, ma_exits = _ma_signal_tables(close, fast, slow)
    exits = apply_stop_loss_exits(close, entries, ma_exits, stop_loss)
    fees = fee_matrix(
        close, exits, asset_types, stock_commission, stock_stamp_duty, etf_commission
    )
    size = _portfolio_size_table(close, entries, weights)
    return vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        init_cash=init_cash,
        fees=fees,
        size=size,
        size_type="percent",
        cash_sharing=True,
        call_seq=list(close.columns),
        freq="1D",
    )


def _extra_trade_stats(pf: vbt.Portfolio, columns: list[str]) -> pd.DataFrame:
    """从成交记录计算平均持仓天数、盈亏比、最大单笔亏损与最大连续亏损次数。"""
    empty = pd.DataFrame(
        {
            "avg_holding_days": np.nan,
            "profit_loss_ratio": np.nan,
            "max_loss": np.nan,
            "max_consecutive_losses": 0,
        },
        index=columns,
    )
    records = pf.trades.records_readable
    if records is None or records.empty:
        return empty
    rows: dict[str, object] = {}
    for column in columns:
        sub = records[records["Column"] == column].sort_values("Exit Timestamp")
        if sub.empty:
            rows[column] = {
                "avg_holding_days": np.nan,
                "profit_loss_ratio": np.nan,
                "max_loss": np.nan,
                "max_consecutive_losses": 0,
            }
            continue
        durations = (sub["Exit Timestamp"] - sub["Entry Timestamp"]).dt.days
        returns = sub["Return"].astype(float)
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        ratio = wins.mean() / abs(losses.mean()) if len(wins) and len(losses) else np.nan
        max_consecutive = 0
        run = 0
        for pnl in sub["PnL"].astype(float):
            if pnl < 0:
                run += 1
                max_consecutive = max(max_consecutive, run)
            else:
                run = 0
        rows[column] = {
            "avg_holding_days": float(durations.mean()) if len(durations) else np.nan,
            "profit_loss_ratio": float(ratio),
            "max_loss": float(returns.min()) if len(returns) else np.nan,
            "max_consecutive_losses": max_consecutive,
        }
    return pd.DataFrame(rows).T


def summarize(
    pf: vbt.Portfolio,
    close: pd.DataFrame,
    asset_types: dict[str, str] | None = None,
    stock_commission: float = DEFAULT_STOCK_COMMISSION,
    stock_stamp_duty: float = DEFAULT_STOCK_STAMP_DUTY,
    etf_commission: float = DEFAULT_ETF_COMMISSION,
) -> pd.DataFrame:
    total_return = pf.total_return()
    years = len(close) / TRADING_DAYS_PER_YEAR
    first_valid = close.apply(
        lambda series: series.loc[series.first_valid_index()] if series.first_valid_index() is not None else np.nan
    )
    gross_hold = close.iloc[-1] / first_valid - 1
    # 买入持有也按同口径计算双边费用，保证与策略收益可比。
    rates = fee_rates(
        {code: (asset_types or {}).get(code, asset_type(code)) for code in close.columns},
        stock_commission,
        stock_stamp_duty,
        etf_commission,
    )
    buy_hold = gross_hold.copy()
    for code in close.columns:
        entry_fee, exit_fee = rates[code]
        buy_hold[code] = (1 + gross_hold[code]) * (1 - entry_fee) * (1 - exit_fee) - 1
    values = pf.value()
    returns = values.pct_change()
    drawdown = values / values.cummax() - 1

    def drawdown_duration(dd: pd.Series) -> int:
        """最大回撤区间长度（连续水下交易日数）。"""
        if dd.empty:
            return 0
        underwater = dd < 0
        best = 0
        run = 0
        for flag in underwater:
            if flag:
                run += 1
                best = max(best, run)
            else:
                run = 0
        return best

    downside = returns.where(returns < 0).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return_values = total_return.to_numpy(dtype=float)
    annualized_values = np.where(
        return_values > -1,
        (1 + return_values) ** (1 / years) - 1,
        np.nan,
    )
    annualized = pd.Series(annualized_values, index=total_return.index)
    calmar = annualized / drawdown.min().abs().replace(0, np.nan)
    extra = _extra_trade_stats(pf, list(close.columns))

    return pd.DataFrame(
        {
            "strategy_return": total_return,
            "buy_hold_return": buy_hold,
            "strategy_minus_hold": total_return - buy_hold,
            "annualized_return": annualized,
            "max_drawdown": pf.max_drawdown(),
            "max_drawdown_duration": [drawdown_duration(drawdown[col]) for col in drawdown.columns],
            "downside_volatility": downside,
            "calmar_ratio": calmar,
            "sharpe": pf.sharpe_ratio(),
            "win_rate": pf.trades.win_rate(),
            "trades": pf.trades.count(),
            "avg_holding_days": extra["avg_holding_days"],
            "profit_loss_ratio": extra["profit_loss_ratio"],
            "max_loss": extra["max_loss"],
            "max_consecutive_losses": extra["max_consecutive_losses"].astype(int),
        }
    ).rename_axis("symbol")


def portfolio_summary(pf: vbt.Portfolio, init_cash: float = DEFAULT_INIT_CASH) -> pd.Series:
    """共享资金组合的绩效汇总（单账户口径）。"""
    values = pf.value()
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    returns = values.pct_change().dropna()
    total_return = float(values.iloc[-1] / values.iloc[0] - 1) if len(values) > 1 else np.nan
    years = len(values) / TRADING_DAYS_PER_YEAR
    annualized = (1 + total_return) ** (1 / years) - 1 if total_return > -1 and years > 0 else np.nan
    drawdown = values / values.cummax() - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if returns.std() > 0 else np.nan
    records = pf.trades.records_readable
    win_rate = float((records["PnL"] > 0).mean()) if records is not None and len(records) else np.nan
    durations = (
        (records["Exit Timestamp"] - records["Entry Timestamp"]).dt.days
        if records is not None and len(records)
        else pd.Series(dtype=float)
    )
    return pd.Series(
        {
            "组合累计收益": total_return,
            "组合年化收益": annualized,
            "组合最大回撤": float(drawdown.min()) if len(drawdown) else np.nan,
            "组合夏普": float(sharpe),
            "组合胜率": win_rate,
            "组合交易次数": int(pf.trades.count().sum()) if hasattr(pf.trades.count(), "sum") else int(pf.trades.count()),
            "组合平均持仓天数": float(durations.mean()) if len(durations) else np.nan,
            "组合初始资金": float(init_cash),
        }
    )


def annual_returns(values: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """按自然年汇总净值/收盘序列的年度收益率。"""
    frame = values if isinstance(values, pd.DataFrame) else values.to_frame("组合")
    years = pd.Series(frame.index.year, index=frame.index, name="年度")
    result = frame.groupby(years).apply(lambda group: group.iloc[-1] / group.iloc[0] - 1)
    result.index.name = "年度"
    return result


def fmt_perf(summary: pd.DataFrame, names: dict[str, str] | None = None) -> pd.DataFrame:
    out = summary.copy()
    for column in (
        "strategy_return",
        "buy_hold_return",
        "strategy_minus_hold",
        "annualized_return",
        "max_drawdown",
        "downside_volatility",
        "win_rate",
        "max_loss",
    ):
        out[column] = out[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "-")
    for column in ("sharpe", "calmar_ratio", "profit_loss_ratio"):
        out[column] = out[column].map(lambda value: f"{value:.2f}" if pd.notna(value) else "-")
    if "max_drawdown_duration" in out.columns:
        out["max_drawdown_duration"] = out["max_drawdown_duration"].map(
            lambda value: f"{int(value)} 天" if pd.notna(value) else "-"
        )
    if "avg_holding_days" in out.columns:
        out["avg_holding_days"] = out["avg_holding_days"].map(
            lambda value: f"{int(value)} 天" if pd.notna(value) else "-"
        )
    out["trades"] = out["trades"].astype(int)
    if "max_consecutive_losses" in out.columns:
        out["max_consecutive_losses"] = out["max_consecutive_losses"].astype(int)
    out = out.rename(columns=_COLUMN_LABELS)
    if names:
        out.index = [label(symbol, names) for symbol in summary.index]
    return out


def trades_table(
    pf: vbt.Portfolio, recent: int | None = None, names: dict[str, str] | None = None
) -> pd.DataFrame:
    columns = ["标的", "买入日期", "买入价", "卖出日期", "卖出价", "收益率", "盈亏"]
    trades = pf.trades.records_readable
    if trades is None or trades.empty:
        return pd.DataFrame(columns=columns)
    trades = trades.rename(
        columns={
            "Column": "标的",
            "Entry Timestamp": "买入日期",
            "Avg Entry Price": "买入价",
            "Exit Timestamp": "卖出日期",
            "Avg Exit Price": "卖出价",
            "Return": "收益率",
            "PnL": "盈亏",
        }
    )[columns]
    if names:
        trades["标的"] = trades["标的"].map(lambda symbol: label(symbol, names))
    if recent is not None:
        trades = trades.groupby("标的", group_keys=False).tail(recent)
    return trades.sort_values(["标的", "买入日期"]).reset_index(drop=True)


def current_ma_state(
    close: pd.DataFrame,
    fast: int,
    slow: int,
    names: dict[str, str] | None = None,
    asset_types: dict[str, str] | None = None,
) -> pd.DataFrame:
    fast, slow = validate_strategy_inputs(close, fast, slow)
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()
    latest = (ma_fast > ma_slow).iloc[-1]
    asset_types = asset_types or {code: asset_type(code) for code in close.columns}
    last_golden: list[object] = []
    last_death: list[object] = []
    execute_dates: list[object] = []
    execute_status: list[str] = []
    latest_date = close.index[-1].date()
    for code in close.columns:
        raw_entries, raw_exits, _, _ = ma_signals(close[code], fast, slow)
        entry_dates = raw_entries.index[raw_entries]
        exit_dates = raw_exits.index[raw_exits]
        last_golden.append(entry_dates[-1].date() if len(entry_dates) else None)
        last_death.append(exit_dates[-1].date() if len(exit_dates) else None)
        # 最近一次交叉的信号于下一交易日收盘执行。
        if len(entry_dates) and (not len(exit_dates) or entry_dates[-1] > exit_dates[-1]):
            last_cross = entry_dates[-1]
        elif len(exit_dates):
            last_cross = exit_dates[-1]
        else:
            last_cross = None
        candidate: object = None
        status = "无信号"
        if last_cross is not None:
            position = close.index.get_loc(last_cross)
            if position + 1 < len(close.index):
                candidate = close.index[position + 1].date()
                status = "已执行" if candidate <= latest_date else "待执行"
            else:
                status = "待下一交易日执行"
        execute_dates.append(candidate)
        execute_status.append(status)
    ma_statuses: list[str] = []
    for code in close.columns:
        has_data = pd.notna(ma_fast[code].iloc[-1]) and pd.notna(ma_slow[code].iloc[-1])
        if not has_data:
            ma_statuses.append(f"数据不足（需 MA{slow}）")
        elif bool(latest[code]):
            ma_statuses.append(f"持有（MA{fast} 在 MA{slow} 上方）")
        else:
            ma_statuses.append(f"空仓等待（MA{fast} 在 MA{slow} 下方）")
    state = pd.DataFrame(
        {
            "类型": [asset_types.get(code, asset_type(code)) for code in close.columns],
            "最新收盘": close.iloc[-1].round(3),
            f"MA{fast}": ma_fast.iloc[-1].round(3),
            f"MA{slow}": ma_slow.iloc[-1].round(3),
            "MA状态": ma_statuses,
            "最近金叉日": last_golden,
            "最近死叉日": last_death,
            "信号执行日": execute_dates,
            "执行状态": execute_status,
        },
        index=close.columns,
    )
    state.index.name = "标的"
    if names:
        state.index = [label(code, names) for code in close.columns]
    return state


def holdings_valuation(
    holdings: Iterable[dict[str, object]],
    prices: dict[str, pd.DataFrame],
    fast: int,
    slow: int,
    names: dict[str, str] | None = None,
    stock_commission: float = DEFAULT_STOCK_COMMISSION,
    stock_stamp_duty: float = DEFAULT_STOCK_STAMP_DUTY,
    etf_commission: float = DEFAULT_ETF_COMMISSION,
) -> pd.DataFrame:
    """持仓估值：每个代码允许出现多个批次，逐批计算市值、盈亏与仓位占比。"""
    rows: list[dict[str, object]] = []
    records: list[tuple[str, dict[str, object]]] = []
    for holding in holdings:
        if not isinstance(holding, dict):
            continue
        raw_code = str(holding.get("code", "")).strip()
        if not raw_code:
            continue
        records.append((normalize_symbol(raw_code), holding))
    asset_types = {code: asset_type(code) for code, _ in records}
    rates = fee_rates(asset_types, stock_commission, stock_stamp_duty, etf_commission)
    lot_counters: dict[str, int] = {}
    for code, holding in records:
        lot_counters[code] = lot_counters.get(code, 0) + 1
        lot_no = lot_counters[code]
        quantity = float(holding["quantity"])
        avg_cost = float(holding["avg_cost"])
        price = prices.get(code)
        latest = np.nan
        end_date = ""
        signal = "无日线数据"
        if price is not None and len(price):
            latest = float(price["收盘"].iloc[-1])
            end_date = str(price["日期"].iloc[-1].date())
            if len(price) > slow:
                ma_fast = price["收盘"].rolling(fast).mean().iloc[-1]
                ma_slow = price["收盘"].rolling(slow).mean().iloc[-1]
                signal = "持有" if ma_fast > ma_slow else "空仓等待"
            else:
                signal = f"数据不足（需 MA{slow}）"
        cost = quantity * avg_cost
        market_value = quantity * latest if pd.notna(latest) else np.nan
        pnl = market_value - cost if pd.notna(market_value) else np.nan
        _, exit_rate = rates[code]
        exit_fee = market_value * exit_rate if pd.notna(market_value) else np.nan
        tax_adjusted_pnl = pnl - exit_fee if pd.notna(pnl) else np.nan
        rows.append(
            {
                "代码": code,
                "批次": lot_no,
                "名称": str(holding.get("name") or (names or {}).get(code, code)),
                "类型": asset_type(code),
                "数量": quantity,
                "成本价": avg_cost,
                "最新收盘": latest,
                "市值": market_value,
                "浮动盈亏": pnl,
                "卖出费用": exit_fee,
                "税后盈亏": tax_adjusted_pnl,
                "收益率": pnl / cost if cost else np.nan,
                "税后收益率": tax_adjusted_pnl / cost if cost else np.nan,
                "组合占比": np.nan,
                "MA状态": signal,
                "数据截至": end_date,
                "备注": str(holding.get("note", "")),
            }
        )
    columns = [
        "代码",
        "批次",
        "名称",
        "类型",
        "数量",
        "成本价",
        "最新收盘",
        "市值",
        "浮动盈亏",
        "卖出费用",
        "税后盈亏",
        "收益率",
        "税后收益率",
        "组合占比",
        "MA状态",
        "数据截至",
        "备注",
    ]
    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return pd.DataFrame(columns=columns)
    total_value = result["市值"].sum(min_count=1)
    if pd.notna(total_value) and total_value:
        result["组合占比"] = result["市值"] / total_value
    return result


def portfolio_metrics(
    close: pd.DataFrame,
    weights: dict[str, float] | None = None,
    risk_free: float = 0.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """组合层指标与持仓日收益相关性矩阵。

    权重缺省时按等权；返回 (组合指标 Series, 相关性矩阵 DataFrame)。
    """
    if close.empty or not len(close.columns):
        raise DataValidationError("组合概览至少需要一个标的。")
    returns = close.pct_change().dropna(how="all")
    codes = list(close.columns)
    if weights is None:
        weights = {code: 1.0 / len(codes) for code in codes}
    weight_vector = pd.Series({code: weights.get(code, 0.0) for code in codes})
    total_weight = weight_vector.sum()
    if total_weight:
        weight_vector = weight_vector / total_weight
    portfolio_return = returns.mul(weight_vector, axis=1).sum(axis=1, min_count=1)
    level = (1 + portfolio_return.fillna(0)).cumprod()
    level_return = level.iloc[-1] - 1 if len(level) else np.nan
    years = len(level) / TRADING_DAYS_PER_YEAR
    annualized = (1 + level_return) ** (1 / years) - 1 if years > 0 and level_return > -1 else np.nan
    drawdown = level / level.cummax() - 1
    max_dd = drawdown.min() if len(drawdown) else np.nan
    excess = portfolio_return - risk_free / TRADING_DAYS_PER_YEAR
    sharpe = (
        excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        if excess.std() > 0
        else np.nan
    )
    metrics = pd.Series(
        {
            "组合累计收益": level_return,
            "组合年化收益": annualized,
            "组合最大回撤": max_dd,
            "组合夏普": sharpe,
        }
    )
    correlation = returns.corr()
    return metrics, correlation


def scan_params(
    close: pd.DataFrame,
    fasts: Iterable[int],
    slows: Iterable[int],
    asset_types: dict[str, str] | None = None,
    stock_commission: float = DEFAULT_STOCK_COMMISSION,
    stock_stamp_duty: float = DEFAULT_STOCK_STAMP_DUTY,
    etf_commission: float = DEFAULT_ETF_COMMISSION,
    init_cash: float = DEFAULT_INIT_CASH,
    stop_loss: float = 0.0,
) -> pd.DataFrame:
    """参数扫描：前 70% 训练、后 30% 验证，保留每个标的的结果。"""
    rows: list[pd.DataFrame] = []
    split = int(len(close) * 0.7)
    for fast in fasts:
        for slow in slows:
            if fast >= slow or split <= slow:
                continue
            portfolio = backtest(
                close,
                fast,
                slow,
                asset_types,
                stock_commission,
                stock_stamp_duty,
                etf_commission,
                init_cash,
                stop_loss,
            )
            values = portfolio.value()
            train_return = values.iloc[split - 1] / values.iloc[0] - 1
            validation_return = values.iloc[-1] / values.iloc[split - 1] - 1
            full_return = values.iloc[-1] / values.iloc[0] - 1
            rows.append(
                pd.DataFrame(
                    {
                        "fast": fast,
                        "slow": slow,
                        "symbol": close.columns,
                        "train_return": train_return.to_numpy(),
                        "validation_return": validation_return.to_numpy(),
                        "full_return": full_return.to_numpy(),
                    }
                )
            )
    if not rows:
        raise DataValidationError("没有满足快线小于慢线且数据充足的参数组合。")
    return pd.concat(rows, ignore_index=True).set_index(["fast", "slow", "symbol"])


def aggregate_scan(scan: pd.DataFrame) -> pd.DataFrame:
    """按多标的样本外收益中位数排序，避免单一标的主导结论。"""
    result = (
        scan.reset_index()
        .groupby(["fast", "slow"], as_index=False)
        .agg(
            训练期中位收益=("train_return", "median"),
            验证期中位收益=("validation_return", "median"),
            验证期平均收益=("validation_return", "mean"),
            全样本中位收益=("full_return", "median"),
            覆盖标的数=("symbol", "nunique"),
        )
        .sort_values(["验证期中位收益", "验证期平均收益"], ascending=False)
        .reset_index(drop=True)
    )
    return result


def walk_forward_scan(
    close: pd.DataFrame,
    fasts: Iterable[int],
    slows: Iterable[int],
    asset_types: dict[str, str] | None = None,
    stock_commission: float = DEFAULT_STOCK_COMMISSION,
    stock_stamp_duty: float = DEFAULT_STOCK_STAMP_DUTY,
    etf_commission: float = DEFAULT_ETF_COMMISSION,
    init_cash: float = DEFAULT_INIT_CASH,
    train_days: int = 504,
    step_days: int = 126,
    stop_loss: float = 0.0,
) -> pd.DataFrame:
    """滚动窗口 walk-forward 参数扫描。

    每个窗口用前 train_days 个交易日训练（按多标的收益中位数选参数），
    随后 step_days 个交易日样本外验证；窗口按 step_days 平移，验证段不参与选参。
    """
    if close.empty:
        raise DataValidationError("没有可用于滚动扫描的数据。")
    train_days = int(train_days)
    step_days = int(step_days)
    if train_days < 20 or step_days < 10:
        raise DataValidationError("滚动窗口训练期至少 20 个交易日，验证期至少 10 个交易日。")
    combos = [(int(fast), int(slow)) for fast in fasts for slow in slows if int(fast) < int(slow)]
    if not combos:
        raise DataValidationError("没有满足快线小于慢线的参数组合。")
    max_slow = max(slow for _, slow in combos)
    if train_days <= max_slow or step_days <= max_slow:
        raise DataValidationError(f"滚动窗口长度不足以支持 MA{max_slow}。")

    rows: list[dict[str, object]] = []
    start = 0
    while start + train_days + step_days <= len(close):
        train = close.iloc[start : start + train_days]
        validation = close.iloc[start + train_days : start + train_days + step_days]
        best: tuple[int, int] | None = None
        best_score = -np.inf
        for fast, slow in combos:
            portfolio = backtest(
                train,
                fast,
                slow,
                asset_types,
                stock_commission,
                stock_stamp_duty,
                etf_commission,
                init_cash,
                stop_loss,
            )
            returns = portfolio.value().iloc[-1] / portfolio.value().iloc[0] - 1
            score = float(returns.median())
            if score > best_score:
                best_score = score
                best = (fast, slow)
        assert best is not None
        fast, slow = best
        portfolio = backtest(
            validation,
            fast,
            slow,
            asset_types,
            stock_commission,
            stock_stamp_duty,
            etf_commission,
            init_cash,
            stop_loss,
        )
        values = portfolio.value()
        validation_returns = values.iloc[-1] / values.iloc[0] - 1
        for code in close.columns:
            rows.append(
                {
                    "window_start": close.index[start].date(),
                    "train_end": close.index[start + train_days - 1].date(),
                    "window_end": close.index[start + train_days + step_days - 1].date(),
                    "fast": fast,
                    "slow": slow,
                    "train_median_return": best_score,
                    "symbol": code,
                    "validation_return": float(validation_returns[code]),
                }
            )
        start += step_days
    if not rows:
        raise DataValidationError("数据长度不足以生成至少一个滚动验证窗口。")
    return pd.DataFrame(rows)


def walk_forward_summary(wf: pd.DataFrame) -> pd.DataFrame:
    """把 walk-forward 明细按窗口汇总为样本外表现。"""
    return (
        wf.groupby(["window_start", "train_end", "window_end", "fast", "slow"], as_index=False)
        .agg(
            训练期中位收益=("train_median_return", "first"),
            验证期中位收益=("validation_return", "median"),
            验证期平均收益=("validation_return", "mean"),
            覆盖标的数=("symbol", "nunique"),
        )
        .sort_values("window_start")
        .reset_index(drop=True)
    )


def walk_forward_stability(wf: pd.DataFrame) -> pd.DataFrame:
    """统计各参数组合被滚动窗口选中为最优的次数（稳健性参考）。"""
    return (
        wf.groupby(["fast", "slow"], as_index=False)
        .agg(选中次数=("window_start", "nunique"))
        .sort_values(["选中次数", "slow", "fast"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def dataframe_to_csv(dataframe: pd.DataFrame, index: bool = False) -> bytes:
    return dataframe.to_csv(index=index, encoding="utf-8-sig").encode("utf-8-sig")


def plot_equity(
    close: pd.DataFrame,
    portfolio: vbt.Portfolio,
    out_path: Path,
    init_cash: float = DEFAULT_INIT_CASH,
    benchmark: pd.DataFrame | None = None,
    benchmark_label: str = "沪深300",
) -> None:
    values = portfolio.value() / float(init_cash)
    first_valid = close.apply(
        lambda series: series.loc[series.first_valid_index()] if series.first_valid_index() is not None else np.nan
    )
    buy_hold = close / first_valid
    benchmark_norm: pd.Series | None = None
    if benchmark is not None:
        aligned = benchmark.set_index("日期")["收盘"].astype(float).reindex(values.index).ffill().dropna()
        if len(aligned):
            benchmark_norm = aligned / aligned.iloc[0]
    count = len(close.columns)
    cols = 3
    rows = int(np.ceil(count / cols))
    figure, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
    for axis, symbol in zip(axes.ravel(), close.columns):
        axis.plot(values.index, values[symbol], label="MA 策略", lw=1.6)
        axis.plot(buy_hold.index, buy_hold[symbol], label="买入持有", lw=1.2, ls="--", alpha=0.7)
        if benchmark_norm is not None:
            axis.plot(benchmark_norm.index, benchmark_norm, label=benchmark_label, lw=1.0, alpha=0.7, color="gray")
        axis.set_title(symbol)
        axis.legend(loc="upper left", fontsize=8)
        axis.grid(alpha=0.3)
    for axis in axes.ravel()[count:]:
        axis.axis("off")
    figure.suptitle("策略净值 vs 买入持有（起点 = 1.0）", fontsize=13)
    figure.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def export_report(
    summary: pd.DataFrame,
    scan: pd.DataFrame | None,
    close: pd.DataFrame,
    portfolio: vbt.Portfolio,
    fast: int,
    slow: int,
    names: dict[str, str] | None = None,
    init_cash: float = DEFAULT_INIT_CASH,
    portfolio_summary: pd.Series | None = None,
) -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    summary.to_csv(REPORT_DIR / "performance_summary.csv", encoding="utf-8-sig")
    portfolio.value().to_csv(REPORT_DIR / "equity_curve.csv", encoding="utf-8-sig")
    annual_returns(portfolio.value()).to_csv(REPORT_DIR / "annual_returns.csv", encoding="utf-8-sig")
    trades_table(portfolio, names=names).to_csv(REPORT_DIR / "trades.csv", index=False, encoding="utf-8-sig")
    if portfolio_summary is not None:
        portfolio_summary.to_frame("组合绩效").to_csv(REPORT_DIR / "portfolio_summary.csv", encoding="utf-8-sig")
    if scan is not None:
        scan.to_csv(REPORT_DIR / "param_scan.csv", encoding="utf-8-sig")
        aggregate_scan(scan).to_csv(REPORT_DIR / "param_scan_grid.csv", index=False, encoding="utf-8-sig")
    report = {
        "数据范围": f"{close.index[0].date()} 至 {close.index[-1].date()}",
        "交易日数": int(len(close)),
        "策略参数": {"fast": fast, "slow": slow, "init_cash": init_cash},
        "说明": "参数扫描按多标的样本外收益中位数排序，不代表未来收益。",
    }
    _write_json(REPORT_DIR / "report.json", report)


def parse_args() -> argparse.Namespace:
    config = load_config()
    parser = argparse.ArgumentParser(description="A 股与场内 ETF MA 研究工具")
    parser.add_argument("--symbols", default=",".join(config["symbols"]), help="6 位代码，逗号分隔")
    parser.add_argument("--fast", type=int, default=config["fast"], help="MA 快线窗口")
    parser.add_argument("--slow", type=int, default=config["slow"], help="MA 慢线窗口")
    parser.add_argument("--start-date", default=str(config["start_date"]), help="研究起始日期 YYYYMMDD/YYYY-MM-DD")
    parser.add_argument("--init-cash", type=float, default=float(config["init_cash"]), help="回测初始资金")
    parser.add_argument("--stop-loss", type=float, default=float(config["stop_loss"]), help="跌破买入价的比例止损，0 表示关闭")
    parser.add_argument("--benchmark", default=str(config["benchmark"]), help="基准指数代码，空字符串表示关闭")
    parser.add_argument("--scan", action="store_true", help="执行参数扫描")
    parser.add_argument("--walk-forward", action="store_true", help="执行滚动窗口 walk-forward 扫描（较慢）")
    parser.add_argument("--refresh", action="store_true", help="更新日线缓存")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
    stock_commission = float(config["stock_commission"])
    stock_stamp_duty = float(config["stock_stamp_duty"])
    etf_commission = float(config["etf_commission"])
    try:
        prices, errors = load_prices(
            symbols,
            refresh=args.refresh,
            start_date=args.start_date,
            retries=int(config["update_retries"]),
            workers=int(config["update_workers"]),
        )
        for error in errors.values():
            print(f"警告：{error}")
        close = close_from_prices(prices, symbols)
        validate_strategy_inputs(close, args.fast, args.slow)
        names = load_names(close.columns)
        types = {code: asset_type(code) for code in close.columns}
        portfolio = backtest(
            close,
            args.fast,
            args.slow,
            types,
            stock_commission,
            stock_stamp_duty,
            etf_commission,
            args.init_cash,
            args.stop_loss,
        )
    except (DataValidationError, DataUnavailableError) as exc:
        raise SystemExit(f"错误：{exc}") from exc

    summary = summarize(
        portfolio,
        close,
        types,
        stock_commission,
        stock_stamp_duty,
        etf_commission,
    )
    print(f"数据范围：{close.index[0].date()} 至 {close.index[-1].date()}")
    print(f"\n===== 回测绩效（MA{args.fast}/{args.slow}）=====")
    print(fmt_perf(summary, names).to_string())
    print(f"\n===== 当前 MA 信号（MA{args.fast}/{args.slow}）=====")
    print(current_ma_state(close, args.fast, args.slow, names, types).to_string())

    benchmark_frame = None
    if args.benchmark:
        try:
            benchmark_frame = load_benchmark(args.benchmark, start_date=args.start_date)
            print(f"\n===== 基准对比（{args.benchmark}）=====")
            print(benchmark_stats(close, benchmark_frame, f"基准{args.benchmark}").to_string())
        except (DataValidationError, DataUnavailableError) as exc:
            print(f"警告：基准加载失败：{exc}")

    combo_portfolio = backtest_portfolio(
        close,
        args.fast,
        args.slow,
        types,
        stock_commission,
        stock_stamp_duty,
        etf_commission,
        args.init_cash,
        stop_loss=args.stop_loss,
    )
    combo_summary = portfolio_summary(combo_portfolio, args.init_cash)
    print(f"\n===== 共享资金组合回测（MA{args.fast}/{args.slow}）=====")
    print(combo_summary.to_string())
    annual = annual_returns(combo_portfolio.value())
    print(f"\n===== 组合分年度收益 =====")
    print(annual.to_string(float_format=lambda value: f"{value:.2%}"))

    scan = None
    if args.scan:
        scan = scan_params(
            close,
            config["scan_fasts"],
            config["scan_slows"],
            types,
            stock_commission,
            stock_stamp_duty,
            etf_commission,
            args.init_cash,
        )
        print("\n===== 参数扫描：按样本外中位收益排序 =====")
        print(aggregate_scan(scan).head(10).to_string(index=False))

    if args.walk_forward:
        wf = walk_forward_scan(
            close,
            config["scan_fasts"],
            config["scan_slows"],
            types,
            stock_commission,
            stock_stamp_duty,
            etf_commission,
            args.init_cash,
            stop_loss=args.stop_loss,
        )
        print("\n===== 滚动窗口 walk-forward：每个窗口的最优参数与样本外表现 =====")
        print(walk_forward_summary(wf).to_string(index=False))
        print("\n===== 参数被选中的次数 =====")
        print(walk_forward_stability(wf).to_string(index=False))

    if not args.no_plot:
        plot_equity(close, portfolio, REPORT_DIR / "equity_curve.png", args.init_cash, benchmark_frame, f"基准{args.benchmark}")
    if not args.no_export:
        export_report(
            summary,
            scan,
            close,
            portfolio,
            args.fast,
            args.slow,
            names,
            args.init_cash,
            portfolio_summary=combo_summary,
        )


if __name__ == "__main__":
    main()
