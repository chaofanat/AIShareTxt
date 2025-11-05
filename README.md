# AIShareTxt

股票技术指标分析工具包

## 简介

AIShareTxt是一个专业的股票技术指标分析工具包，提供全面的股票数据获取、技术指标计算和AI分析功能。

## 主要功能

- 股票数据获取（基于akshare）
- 多种技术指标计算（基于talib）
- AI驱动的股票分析建议
- 详细的股票分析报告生成
- 支持多种数据源和分析模式

## 安装

### 从源码安装

```bash
git clone https://github.com/example/aishare-txt.git
cd aishare-txt
pip install -e .
```

### 使用pip安装

```bash
pip install aishare-txt
```

## 快速开始

### 基本使用

```python
from AIShareTxt import analyze_stock

# 分析单只股票
result = analyze_stock("000001")
print(result)
```

### 高级使用

```python
from AIShareTxt.core.analyzer import StockAnalyzer

# 创建分析器实例
analyzer = StockAnalyzer()

# 分析股票
result = analyzer.analyze_stock("000001")

# 生成报告
report = analyzer.generate_report("000001", result["indicators"])
print(report)
```

### AI分析

```python
from AIShareTxt.ai.client import AIStockAnalyzer

# 创建AI分析器
ai_analyzer = AIStockAnalyzer(api_key="your_api_key", provider="deepseek")

# 获取AI分析建议
advice = ai_analyzer.analyze_stock("000001")
print(advice)
```

## 项目结构

```
AIShareTxt/
├── core/              # 核心模块
│   ├── analyzer.py    # 股票分析器
│   ├── data_fetcher.py # 数据获取器
│   ├── report_generator.py # 报告生成器
│   └── config.py      # 配置模块
├── ai/                # AI分析模块
│   ├── client.py      # AI客户端
│   └── providers/     # AI提供商
├── indicators/        # 技术指标模块
│   └── technical_indicators.py # 技术指标计算
├── utils/             # 工具模块
│   ├── utils.py       # 通用工具
│   └── stock_list.py  # 股票列表工具
├── examples/          # 示例代码
├── tests/             # 测试代码
└── docs/              # 文档
```

## 支持的技术指标

- 移动平均线（MA、EMA）
- MACD指标
- 布林带（Bollinger Bands）
- RSI相对强弱指标
- KDJ随机指标
- 威廉指标（Williams %R）
- CCI商品通道指标
- ATR平均真实波幅
- OBV能量潮指标
- 更多指标...

## 开发

### 安装开发依赖

```bash
pip install -e ".[dev]"
```

### 运行测试

```bash
pytest
```

### 代码格式化

```bash
black AIShareTxt/
flake8 AIShareTxt/
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

- 邮箱：aishare@example.com
- 项目主页：https://github.com/example/aishare-txt
