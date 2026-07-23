# AIShareTxt

**English** | [中文](README.md)

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MulanPSL2-blue.svg)](LICENSE)

**A Chinese stock technical indicator text generator** — turns market data into structured text context for AI agents.

- [**AIShareTxt on PyPI**](https://pypi.org/project/aishare-txt/) provides accurate and comprehensive technical indicator context for AI agents working on Chinese-stock analysis tasks.

## ✨ Features

- 📊 **Stock data fetching** — akshare (default) and Goldminer (gm) SDK as priority source when configured, with automatic fallback
- 📈 **Technical indicators** — 50+ indicators computed via TA-Lib (MA/MACD/RSI/KDJ/ADX/OBV/Bollinger/…)
- 🌐 **Market environment analysis** — independent breadth / index / sector report (`aishare-market`), decoupled from single-stock analysis
- 🤖 **AI processing advice** — DeepSeek and ZhipuAI integration
- 📋 **Report generation** — structured text reports ready to feed into LLMs
- 🔧 **Modular design** — clean module boundaries, easy to extend

## 🚀 Installation

### Requirements

- Python 3.10+
- Windows / Linux / macOS

### Install

```bash
# From source (for development)
git clone https://github.com/chaofanat/AIShareTxt
cd AIShareTxt
pip install -e .

# Or install from PyPI (recommended)
pip install aishare-txt
```

### Dependencies

Core dependencies (installed automatically):
- `akshare>=1.9.0` — stock data (default source)
- `gm>=3.0.0` — Goldminer SDK (optional, requires Goldminer Terminal; `pip install aishare-txt[gm]`)
- `TA-Lib>=0.4.26` — technical indicator computation (**requires system-level TA-Lib binary**, see note below)
- `pandas>=1.5.0`, `numpy>=1.21.0`
- `pandas_market_calendars>=1.1.0` — SSE trading calendar
- `scipy>=1.9.0` — OBV extremum detection
- `requests>=2.28.0`
- `openai>=1.0.0`, `zhipuai>=2.0.0` — optional AI analysis

> ⚠️ **TA-Lib system install**
> Linux/macOS:
> ```bash
> sudo apt-get install -y build-essential python3-dev ta-lib   # Debian/Ubuntu
> brew install ta-lib                                           # macOS
> ```
> Windows: download the prebuilt wheel matching your Python version from [cgohlke/talib-build](https://github.com/cgohlke/talib-build) and install it manually. Without the underlying C library, `pip install TA-Lib` will fail.

## 📖 Quick Start

### Stock analysis

```python
from AIShareTxt import StockDataProcessor

processor = StockDataProcessor()
report = processor.generate_stock_report("000001")  # Ping An Bank
print(report)
```

Or the convenience function:

```python
from AIShareTxt import analyze_stock
print(analyze_stock("000001"))
```

### Command line

```bash
aishare 000001        # analyze Ping An Bank
aishare 600036        # analyze China Merchants Bank
aishare               # interactive mode (no args)
```

### Market environment analysis

`aishare-market` produces a market-level report (indexes + breadth + phase classification), independent from individual stocks. Pass a stock code to also include that stock's industry/concept sectors:

```bash
aishare-market             # market-only report
aishare-market 000001      # market report + sectors for 000001
```

Via API:

```python
from AIShareTxt import analyze_market

print(analyze_market())           # market only
print(analyze_market('000001'))   # market + sectors
```

> When `GM_TOKEN` is set, the market module fetches index/snapshot/turnover data via the Goldminer SDK first and falls back to akshare if gm fails or is not installed. `DATA_SOURCE` is not required for this path.

### AI-powered processing advice

```python
from AIShareTxt.ai.client import AIClient

ai = AIClient(api_key="...", provider="deepseek")
if ai.is_available():
    advice = ai.generate_data_processing_recommendation(
        technical_report="<report text>",
        stock_code="000001",
    )
    print(ai.get_recommendation_text(advice))
```

## 📁 Project Structure

```
AIShareTxt/
├── core/                       # Coordination layer
│   ├── data_processor.py      # StockDataProcessor (main entry coordinator)
│   └── config.py              # IndicatorConfig
├── indicators/                 # Stock technical indicators
│   ├── data_fetcher.py        # Stock data fetcher (akshare/gm dispatch)
│   ├── data_sources/          # Data source implementations
│   ├── technical_indicators.py
│   └── report_generator.py
├── market/                     # Market environment analysis (independent)
│   ├── market_fetcher.py      # Market-level data + TTL cache
│   ├── market_analyzer.py     # 3-layer phase classification (trend × sentiment)
│   ├── market_report_generator.py
│   ├── sector_resolver.py     # stock → industry/concepts
│   ├── processor.py           # MarketEnvironmentProcessor orchestrator
│   └── cli.py                 # aishare-market entry point
├── ai/                         # AI processing-advice module
│   └── client.py
├── utils/                      # Utilities
│   ├── utils.py
│   ├── stock_list.py
│   ├── cache.py               # TTLCache (intraday/post-close differentiated)
│   └── trading_calendar.py    # SSE calendar + market-open detection
└── examples/
```

## 📊 Supported Indicators

- **Trend**: MA5/10/20/60, EMA, WMA, Bollinger Bands
- **Momentum**: MACD, RSI, KDJ, Williams %R, CCI
- **Volume**: OBV, VWMA, volume ratio
- **Volatility**: ATR, historical volatility
- **Money flow**: main-force net inflow, 5-day flow trend, DMI (+DI, -DI, ADX)
- **Market-level** (via `aishare-market`): advance/decline counts, limit-up/down, seal ratio, median pct change, total turnover MA20 (5-day series)

## ⚙️ Configuration

### Data source

akshare is the default and needs no configuration. To enable the gm (Goldminer) source:

```powershell
# Windows
$env:DATA_SOURCE="gm"
$env:GM_TOKEN="your_gm_token"
```

```bash
# Linux/macOS
export DATA_SOURCE="gm"
export GM_TOKEN="your_gm_token"
```

> Get your token from Goldminer Terminal → System Settings → Key Management.
> The gm source requires the Goldminer Terminal to be online; failures fall back to akshare automatically.
> `DATA_SOURCE=gm` toggles the **stock** data source. The **market** module uses gm whenever `GM_TOKEN` is detected — no need to set `DATA_SOURCE`.

### AI providers

Set one of the following environment variables:

| Provider | Env var | Get key |
|----------|---------|---------|
| DeepSeek (default) | `DEEPSEEK_API_KEY` | https://platform.deepseek.com/ |
| ZhipuAI | `ZHIPUAI_API_KEY` | https://open.bigmodel.cn/ |

### Tunable parameters

All thresholds and periods live on `IndicatorConfig`:

```python
from AIShareTxt.core.config import IndicatorConfig

config = IndicatorConfig()
config.MA_PERIODS = {'short': [5, 10, 20], 'medium': [60], 'long': [120, 250]}
config.MACD_CONFIG = {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9}
```

Market-environment thresholds (ADX trending/ranging, breadth panic/hot, limit-up/down rules) are in `IndicatorConfig.MARKET_ENVIRONMENT_CONFIG`.

## 🔧 Development

### Regression test

Before submitting a PR:

```bash
python scripts/regression_test.py
```

This verifies the two stock-analysis entry points (`analyze_stock("000001")` and `aishare 000001`) still run end-to-end. See [`docs/regression-test-guide.md`](docs/regression-test-guide.md) for details.

### Running tests

```bash
python -m pytest AIShareTxt/tests/
```

The `tests/market/` suite covers the market module (fetcher, analyzer, report generator, CLI) with all akshare/gm calls monkeypatched — no live network needed.

## 📄 License

Mulan Permissive Software License v2 — see [LICENSE](LICENSE).

## ⚠️ Disclaimer

All outputs are objective computation results for reference only and do not constitute investment advice.

## 📞 Contact

- Project: https://github.com/chaofanat/AIShareTxt
- Issues: https://github.com/chaofanat/AIShareTxt/issues
- Email: chaofanat@gmail.com

## 🙏 Acknowledgements

- [akshare](https://github.com/akfamily/akshare) — financial data interface
- [Goldminer](https://www.myquant.cn/) — quantitative data service
- [TA-Lib](https://mrjbq7.github.io/ta-lib/) — technical analysis library
- [pandas](https://pandas.pydata.org/) · [numpy](http://www.numpy.org/)
