# 股票研究与持仓看板

本地运行的 Streamlit 看板，用于研究 A 股与场内 ETF 的日线、MA 均线策略和本地持仓记录。仅供学习研究，不构成投资建议。

## 安装

Python 3.11+，在项目目录执行：

~~~powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
~~~

## 启动

双击 run_dashboard.bat，或在本目录执行：

~~~powershell
.venv\Scripts\python.exe -m streamlit run dashboard.py
~~~

浏览器打开 http://localhost:8501 后即可使用。

## 功能

- 自选：新增、删除、调整顺序；支持 A 股与场内 ETF 六位代码（坏记录在加载时自动跳过，不会让看板崩溃）。
- 数据：默认读取本地日线缓存；点击“更新日线数据”才会强制刷新。刷新优先增量更新（只拉最近一段并合并），失败自动转全量；多标的并发下载、数据源自动重试，更新成功/失败记录写入看板的“数据更新记录”。缓存缺失或覆盖不到 config.toml 的 start_date 时自动补拉（A 股新浪源优先、东财兜底；ETF 东财优先、新浪兜底）。
- 研究：MA 信号（含最近金叉/死叉日、信号执行日与执行状态）、K线（含成交量副图与成交标记）、策略与买入持有对比、风险指标（回撤持续、下行波动、卡玛）、交易统计（平均持仓、盈亏比、最大单笔亏损、最大连亏）与 CSV 导出。
- 组合回测：单账户共享资金的多标的组合回测，展示组合净值/年化/回撤/夏普与分年度收益；同日多个买入信号按等权切分当时可用现金。
- 组合概览：多标的等权静态组合与日收益相关性矩阵。
- 基准：沪深300 等指数日线对比（累计/年化/回撤/夏普与净值曲线）。
- 扫描：固定前 70% 训练、后 30% 验证，按多标的验证期中位收益排序，支持参数下钻；另有滚动窗口 walk-forward 稳健性扫描（每个窗口用训练期选参数，验证期不参与选参）。
- 持仓：支持同一代码多批次录入，逐批显示市值、税后浮亏/浮盈与仓位占比；可用成交流水按日期顺序自动汇总生成持仓，并统计已实现盈亏。
- 提醒：`daily_report.py` 收盘更新时对比 MA 信号快照，新金叉/死叉写入提醒记录，并可推送到 config.toml 配置的 webhook。

## 本地文件

- config.toml：默认标的、起始日期、费率、MA 与扫描参数、初始资金、止损、基准指数、更新并发/重试、提醒 webhook。
- data/watchlist.json：自选列表。
- data/holdings.json：手动维护（或由流水生成）的持仓，允许同一代码多批次。
- data/trades.json：成交流水（日期/代码/方向/价格/数量/费用/备注）。
- data/cache_meta.json：各标的日线缓存最后成功更新时间。
- data/update_log.json：最近 200 条数据更新/失败记录。
- data/signal_snapshot.json 与 signal_alerts.json：MA 信号快照与提醒记录。
- data/*_daily_qfq.csv：日线缓存（保存全历史，按需截取 start_date 之后的数据）。
- data/*_index_daily.csv：基准指数缓存。
- reports/：命令行回测导出的报告（绩效、净值、分年度收益、交易、组合绩效、扫描等）。

## 回测口径

- MA 交叉当天产生信号，于下一交易日收盘执行。
- A 股默认买入佣金 0.03%，卖出佣金 0.03% + 印花税 0.05%；ETF 默认双边佣金 0.03%，不收印花税。
- 买入持有基准同样按标的类型扣除双边费用，与策略收益口径一致。
- 止损（可选）：持仓期跌破买入价 × (1 - stop_loss) 后，于下一交易日收盘卖出。
- 组合回测为单账户共享资金：同日多个买入信号按等权切分当时可用现金，无现金时放弃买入。
- 初始资金默认 100,000，可在 config.toml 或看板侧栏调整。
- 未模拟最低佣金、涨跌停、停牌和 T+1 成交限制；费率可在侧栏或 config.toml 调整。

## 每日自动报告

收盘后可用 `daily_report.py` 刷新日线并生成报告（支持 `--scan` 与 `--walk-forward`），可配合系统计划任务运行：

~~~powershell
.venv\Scripts\python.exe daily_report.py --scan
~~~

首次运行只初始化信号快照；之后每次运行发现新金叉/死叉会记录提醒，并在配置了 notify_url 时推送。

## 命令行研究

~~~powershell
.venv\Scripts\python.exe stock_lab.py --symbols 002463,600519 --stop-loss 0.08 --scan
~~~

常用参数：`--start-date`、`--init-cash`、`--stop-loss`、`--benchmark`（留空关闭）、`--refresh`、`--scan`、`--walk-forward`。

## 检查

~~~powershell
.venv\Scripts\python.exe -m unittest -v test_stock_lab.py
~~~
