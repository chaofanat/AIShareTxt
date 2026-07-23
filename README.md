# AIShareTxt

[English](README.en.md) | **中文**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MulanPSL2-blue.svg)](LICENSE)

**中国股票技术指标文本生成工具包**
- [**AIShareTxt**](https://pypi.org/project/aishare-txt/) 用于为金融分析相关领域的AI智能体提供准确且全面的中国股票技术指标上下文服务。

<a href="https://www.star-history.com/?repos=chaofanat%2FAIShareTxt&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=chaofanat/AIShareTxt&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=chaofanat/AIShareTxt&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=chaofanat/AIShareTxt&type=date&legend=top-left" />
 </picture>
</a>



## ✨ 主要功能

- 📊 **股票数据获取** - 支持akshare（默认）和掘金量化(gm)双数据源，自动降级
- 📈 **技术指标计算** - 基于TA-Lib，支持50+种技术指标计算
- 🌐 **市场环境分析** - 独立的大盘/市场宽度/板块报告（`aishare-market`），与个股分析解耦
- 🤖 **AI数据处理建议** - 集成DeepSeek和智谱AI，提供数据质量处理建议
- 📋 **详细报告生成** - 自动生成专业的股票数据报告
- 🔧 **模块化设计** - 清晰的模块结构，易于扩展和定制
- ⚡ **高性能计算** - 优化的算法，支持批量处理

## 🚀 快速安装

### 环境要求

- Python 3.10+
- Windows/Linux/macOS

### 安装方法

```bash
# 从源码安装
git clone https://gitee.com/chaofanat/aishare-txt
cd aishare-txt
pip install -e .

# 或者直接安装使用项目（推荐）
pip install aishare-txt
```


### 依赖说明

项目会自动安装以下核心依赖：
- `akshare>=1.9.0` - 股票数据获取（默认数据源）
- `gm>=3.0.0` - 掘金量化SDK（可选数据源，需掘金终端，`pip install aishare-txt[gm]`）
- `TA-Lib>=0.4.26` - 技术指标计算（**需先系统级安装 TA-Lib 二进制库，详见下方提示**）
- `pandas>=1.5.0` - 数据处理
- `numpy>=1.21.0` - 数值计算
- `pandas_market_calendars>=1.1.0` - 交易日历
- `scipy>=1.9.0` - 科学计算，用于OBV能量潮指标的极值检测
- `requests>=2.28.0` - HTTP请求
- `openai>=1.0.0` - AI分析（可选）
- `zhipuai>=2.0.0` - AI分析（可选）

> ⚠️ **(https://ta-lib.org/install/)[TA-Lib] 系统级安装提示**  
> 在 Linux/macOS 上请先执行：  
> ```bash
> # Ubuntu/Debian
> sudo apt-get install -y build-essential python3-dev ta-lib
> # macOS (Homebrew)
> brew install ta-lib
> ```  
> Windows 用户请下载对应 Python 版本的 [TA-Lib 预编译 whl](https://github.com/cgohlke/talib-build) 后手动安装。  
> 否则 `pip install TA-Lib` 会因缺少底层 C 库而失败。

## 📖 快速开始

### 基本使用

```python
from AIShareTxt import StockDataProcessor

# 创建数据处理器实例
processor = StockDataProcessor()

# 生成股票数据报告（直接返回报告文本）
report = processor.generate_stock_report("000001")  # 平安银行
print(report)
```

### 使用便捷函数

```python
from AIShareTxt import analyze_stock

# 简单分析（直接返回报告）
report = analyze_stock("000001")
print(report)
```

### 命令行使用

安装完成后，可以在终端中使用 `aishare` 命令：

```bash
# 分析指定股票
aishare 000001        # 分析平安银行
aishare 600036        # 分析招商银行
aishare 603259        # 分析药明康德

# 显示帮助信息
aishare --help
aishare -h

# 交互模式（不带参数运行）
aishare
```

**命令行参数说明：**
- `[股票代码]` - 要分析的6位股票代码
- `-h, --help` - 显示帮助信息
- 不带参数运行将进入交互模式，可以输入股票代码进行分析

**交互模式示例：**
```bash
$ aishare
股票数据报告生成器
============================================================
提示：可以在命令行直接运行 aishare 000001 来快速测试

验证处理环境...
环境验证通过

============================================================
请输入股票代码（如：000001，输入 'quit' 退出）：000001

# [显示完整的股票数据报告]
```

### 市场环境分析

`aishare-market` 提供独立于个股的大盘环境报告（指数行情 + 市场宽度 + 阶段判定），传入股票代码时附加该股的板块涨跌：

```bash
# 仅市场环境
aishare-market

# 市场环境 + 指定股票的板块信息
aishare-market 000001
```

也可通过 API 调用：

```python
from AIShareTxt import analyze_market

# 仅市场环境
print(analyze_market())

# 含个股板块
print(analyze_market('000001'))
```

> 配置 `GM_TOKEN` 后，市场数据（指数/快照/成交额中枢）优先走掘金 SDK，未配置或失败时自动回退 akshare。

### AI智能分析

```python
from AIShareTxt.ai.client import AIClient

# 创建AI客户端（需要配置API密钥）
ai_client = AIClient(api_key="your_api_key", provider="deepseek")

# 进行AI数据处理建议
if ai_client.is_available():
    advice = ai_client.generate_data_processing_recommendation(
        technical_report="技术数据报告内容",
        stock_code="000001"
    )
    print(f"AI数据处理建议: {ai_client.get_recommendation_text(advice)}")
else:
    print("AI功能不可用，请检查API配置")
```

### 技术指标计算

```python
from AIShareTxt.indicators.technical_indicators import TechnicalIndicators
import pandas as pd
import numpy as np

# 创建技术指标数据处理器
ti = TechnicalIndicators()

# 准备股票数据（OHLCV格式）
data = pd.DataFrame({
    'open': [100, 102, 101, 103, 104],
    'high': [105, 106, 104, 107, 108],
    'low': [99, 101, 100, 102, 103],
    'close': [102, 101, 103, 104, 105],
    'volume': [1000, 1200, 800, 1500, 900]
})

# 处理所有技术指标
indicators = ti.process_all_indicators(data)

# 计算单个指标
bias = ti.calculate_bias(data['close'], timeperiod=20)
ma_patterns = ti.analyze_ma_patterns(data['close'])
```

### 获取股票列表

```python
from AIShareTxt.utils.stock_list import get_stock_list

# 获取沪深300主板成分股
stocks = get_stock_list()
print(f"获取到 {len(stocks)} 只股票")
print(stocks.head())
```

## 📁 项目结构

```
AIShareTxt/
├── core/                      # 核心协调层
│   ├── data_processor.py     # 股票数据处理器（主要入口协调器）
│   └── config.py             # 配置管理
├── indicators/                # 个股技术指标层
│   ├── data_fetcher.py       # 个股数据获取调度器（akshare/gm）
│   ├── data_sources/         # 数据源模块
│   │   ├── base.py           # 数据源抽象基类
│   │   ├── akshare_source.py # akshare数据源（东方财富/新浪/腾讯）
│   │   └── gm_source.py      # 掘金量化(gm)数据源
│   ├── technical_indicators.py # 技术指标计算
│   └── report_generator.py   # 技术指标报告生成
├── market/                    # 市场环境分析层（独立于个股）
│   ├── market_fetcher.py     # 市场级数据获取（指数/快照/板块）+ TTL 缓存
│   ├── market_analyzer.py    # 三层阶段判定（趋势 × 情绪 → 合成）
│   ├── market_report_generator.py # 市场环境报告生成
│   ├── sector_resolver.py    # 个股 → 行业/概念板块映射
│   ├── processor.py          # MarketEnvironmentProcessor 编排类
│   └── cli.py                # aishare-market 命令入口
├── ai/                        # AI数据处理建议模块
│   └── client.py             # AI客户端
├── utils/                     # 工具模块
│   ├── utils.py              # 通用工具类
│   ├── stock_list.py         # 股票列表工具
│   ├── cache.py              # TTLCache（市场数据按更新频率分级缓存）
│   └── trading_calendar.py   # 交易日历（SSE）+ 盘中/盘后判定
└── examples/                  # 示例代码
    └── legacy_api.py         # 传统API示例
```

## 📊 支持的技术指标

### 趋势指标
- 移动平均线（MA5, MA10, MA20, MA60）
- 指数移动平均线（EMA5, EMA10, EMA12, EMA20, EMA26）
- 加权移动平均线（WMA10, WMA20）
- 布林带（BOLL）

### 动量指标
- MACD指标
- RSI相对强弱指标（RSI9, RSI14）
- KDJ随机指标
- 威廉指标（Williams %R）
- CCI商品通道指标

### 成交量指标
- OBV能量潮指标
- VWMA成交量加权移动平均
- 量比指标

### 波动率指标
- ATR平均真实波幅
- 历史波动率

### 资金流向指标
- 主力资金净流入
- 5日资金流向趋势
- DMI动向指标（+DI, -DI, ADX）

## ⚙️ 配置说明

### AI配置

在使用AI功能前，需要配置API密钥。项目支持DeepSeek和智谱AI两种AI服务：

1. **DeepSeek配置**
   - 需要设置环境变量 `DEEPSEEK_API_KEY`
   - 获取API密钥：访问 [DeepSeek官网](https://platform.deepseek.com/) 注册并获取API密钥
   
   **Windows系统设置方法：**
   ```powershell
   # 临时设置（当前会话有效）
   $env:DEEPSEEK_API_KEY="your_deepseek_api_key"
   
   # 永久设置（需要管理员权限）
   setx DEEPSEEK_API_KEY "your_deepseek_api_key"
   ```
   
   **Linux/macOS系统设置方法：**
   ```bash
   # 临时设置（当前会话有效）
   export DEEPSEEK_API_KEY="your_deepseek_api_key"
   
   # 永久设置（添加到~/.bashrc或~/.zshrc）
   echo 'export DEEPSEEK_API_KEY="your_deepseek_api_key"' >> ~/.bashrc
   source ~/.bashrc
   ```

2. **智谱AI配置**
   - 需要设置环境变量 `ZHIPUAI_API_KEY`
   - 获取API密钥：访问 [智谱AI官网](https://open.bigmodel.cn/) 注册并获取API密钥
   
   **Windows系统设置方法：**
   ```powershell
   # 临时设置（当前会话有效）
   $env:ZHIPUAI_API_KEY="your_zhipuai_api_key"
   
   # 永久设置（需要管理员权限）
   setx ZHIPUAI_API_KEY "your_zhipuai_api_key"
   ```
   
   **Linux/macOS系统设置方法：**
   ```bash
   # 临时设置（当前会话有效）
   export ZHIPUAI_API_KEY="your_zhipuai_api_key"
   
   # 永久设置（添加到~/.bashrc或~/.zshrc）
   echo 'export ZHIPUAI_API_KEY="your_zhipuai_api_key"' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **验证环境变量设置**
   ```python
   import os
   
   # 检查环境变量是否设置成功
   deepseek_key = os.environ.get('DEEPSEEK_API_KEY')
   zhipuai_key = os.environ.get('ZHIPUAI_API_KEY')
   
   print(f"DeepSeek API Key: {'已设置' if deepseek_key else '未设置'}")
   print(f"智谱AI API Key: {'已设置' if zhipuai_key else '未设置'}")
   ```

4. **在代码中使用**
   ```python
   from AIShareTxt.ai.client import AIClient
   
   # 使用DeepSeek（默认）
   ai_client = AIClient(provider="deepseek")
   
   # 使用智谱AI
   ai_client = AIClient(provider="zhipuai")
   
   # 进行AI数据处理建议
   if ai_client.is_available():
       advice = ai_client.generate_data_processing_recommendation(
           technical_report="技术数据报告内容",
           stock_code="000001"
       )
       print(f"AI数据处理建议: {ai_client.get_recommendation_text(advice)}")
   else:
       print("AI功能不可用，请检查API配置")
   ```

### 数据源配置

默认使用 akshare 数据源（无需额外配置）。如需使用掘金量化(gm)数据源，需安装掘金终端并配置密钥：

**Windows系统：**
```powershell
# 设置环境变量
$env:DATA_SOURCE="gm"
$env:GM_TOKEN="your_gm_token"
```

**Linux/macOS系统：**
```bash
export DATA_SOURCE="gm"
export GM_TOKEN="your_gm_token"
```

> 掘金密钥获取方式：掘金终端 → 系统设置 → 密钥管理  
> gm 数据源需要掘金量化终端在线，连接失败时自动降级到 akshare  
> `DATA_SOURCE=gm` 控制个股数据源切换；市场环境模块（`aishare-market`）只要检测到 `GM_TOKEN` 就优先走 gm，无需设置 `DATA_SOURCE`

### 分析配置

可以通过配置文件调整分析参数：

```python
from AIShareTxt.core.config import IndicatorConfig

config = IndicatorConfig()

# 调整均线周期
config.MA_PERIODS = {
    'short': [5, 10, 20],    # 短期均线
    'medium': [60],           # 中期均线
    'long': [120, 250]        # 长期均线
}

# 调整MACD参数
config.MACD_CONFIG = {
    'fastperiod': 12,
    'slowperiod': 26,
    'signalperiod': 9
}
```

## 🔧 开发指南

### 贡献代码

欢迎贡献代码！请遵循以下步骤：

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改前运行回归测试：`python scripts/regression_test.py`
4. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
5. 推送到分支 (`git push origin feature/AmazingFeature`)
6. 创建 Pull Request

### 回归测试

在提交代码前，请运行回归测试确保核心功能正常：

```bash
# 运行回归测试
python scripts/regression_test.py
```

回归测试会验证以下核心入口：
- `analyze_stock("000001")` - API 方式
- `aishare 000001` - 命令行方式

> 💡 **注意**：只有修改 `AIShareTxt/` 目录下的代码时才需要运行回归测试。文档、配置文件等修改不会触发测试。

详细说明请查看：[回归测试使用指南](docs/regression-test-guide.md)

## 📈 使用示例

### 完整分析流程

```python
from AIShareTxt import StockAnalyzer

def analyze_example():
    """完整的股票数据处理示例"""

    # 1. 创建数据处理器
    processor = StockDataProcessor()

    # 2. 处理指定股票数据
    stock_code = "000001"  # 平安银行
    report = processor.generate_stock_report(stock_code)

    # 3. 输出数据报告
    print(f"股票 {stock_code} 数据报告：")
    print("=" * 60)
    print(report)

# 运行示例
if __name__ == "__main__":
    analyze_example()
```

### 批量分析

```python
from AIShareTxt.utils.stock_list import get_stock_list
from AIShareTxt import StockDataProcessor

def batch_analysis():
    """批量数据处理示例"""

    # 获取股票列表
    stocks = get_stock_list()
    if stocks is None:
        print("无法获取股票列表")
        return

    # 处理前5只股票
    processor = StockDataProcessor()

    for idx, stock in stocks.head(5).iterrows():
        stock_code = stock['代码']
        stock_name = stock['名称']

        print(f"\n处理 {stock_name} ({stock_code}) 数据...")
        print("=" * 50)

        try:
            report = processor.generate_stock_report(stock_code)
            print(report)

        except Exception as e:
            print(f"  数据处理失败: {e}")

batch_analysis()
```

## 📄 许可证

本项目采用木兰宽松许可证 第2版 - 查看 [LICENSE](LICENSE) 文件了解详情。

## ⚠️ 免责声明

本工具提供的所有信息均为客观的数据处理结果，仅供参考，不构成投资建议。投资有风险，入市需谨慎。

## 📞 联系方式

- 项目主页: https://github.com/chaofanat/AIShareTxt
- 问题反馈: https://github.com/chaofanat/AIShareTxt/issues
- 邮箱: chaofanat@gmail.com

## 🙏 致谢

感谢以下开源项目的支持：
- [akshare](https://github.com/akfamily/akshare) - 金融数据接口
- [掘金量化](https://www.myquant.cn/) - 量化交易数据服务
- [TA-Lib](https://mrjbq7.github.io/ta-lib/) - 技术分析库
- [pandas](https://pandas.pydata.org/) - 数据分析库
- [numpy](http://www.numpy.org/) - 科学计算库
