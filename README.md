# AIShareTxt

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MulanPSL2-blue.svg)](LICENSE)

**专业的股票技术指标分析工具包**

AIShareTxt是一个功能强大的Python股票技术指标分析工具包，提供全面的股票数据获取、技术指标计算、AI智能分析和详细报告生成功能。

## ✨ 主要功能

- 📊 **股票数据获取** - 基于akshare，支持多数据源获取实时和历史数据
- 📈 **技术指标计算** - 基于TA-Lib，支持50+种技术指标计算
- 🤖 **AI智能分析** - 集成DeepSeek和智谱AI，提供智能投资建议
- 📋 **详细报告生成** - 自动生成专业的股票分析报告
- 🔧 **模块化设计** - 清晰的模块结构，易于扩展和定制
- ⚡ **高性能计算** - 优化的算法，支持批量处理

## 🚀 快速安装

### 环境要求

- Python 3.8+
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
- `akshare>=1.9.0` - 股票数据获取
- `TA-Lib>=0.4.26` - 技术指标计算（**需先系统级安装 TA-Lib 二进制库，详见下方提示**）
- `pandas>=1.5.0` - 数据处理
- `numpy>=1.21.0` - 数值计算
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
from AIShareTxt import StockAnalyzer

# 创建分析器实例
analyzer = StockAnalyzer()

# 分析股票（直接返回报告文本）
report = analyzer.analyze_stock("000001")  # 平安银行
print(report)
```

### 使用便捷函数

```python
from AIShareTxt import analyze_stock

# 简单分析（直接返回报告）
report = analyze_stock("000001")
print(report)
```

### AI智能分析

```python
from AIShareTxt.ai.client import AIClient

# 创建AI客户端（需要配置API密钥）
ai_client = AIClient(api_key="your_api_key", provider="deepseek")

# 进行AI分析
if ai_client.is_available():
    advice = ai_client.analyze_investment_recommendation(
        technical_report="技术分析报告内容",
        stock_code="000001"
    )
    print(f"AI投资建议: {ai_client.get_recommendation_text(advice)}")
else:
    print("AI功能不可用，请检查API配置")
```

### 技术指标计算

```python
from AIShareTxt.indicators.technical_indicators import TechnicalIndicators
import pandas as pd
import numpy as np

# 创建技术指标计算器
ti = TechnicalIndicators()

# 准备股票数据（OHLCV格式）
data = pd.DataFrame({
    'open': [100, 102, 101, 103, 104],
    'high': [105, 106, 104, 107, 108],
    'low': [99, 101, 100, 102, 103],
    'close': [102, 101, 103, 104, 105],
    'volume': [1000, 1200, 800, 1500, 900]
})

# 计算所有技术指标
indicators = ti.calculate_all_indicators(data)

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
├── core/                      # 核心功能模块
│   ├── analyzer.py           # 股票分析器（主要入口）
│   ├── data_fetcher.py       # 数据获取器
│   ├── report_generator.py   # 报告生成器
│   └── config.py             # 配置管理
├── ai/                        # AI分析模块
│   └── client.py             # AI客户端
├── indicators/                # 技术指标模块
│   └── technical_indicators.py # 技术指标计算
├── utils/                     # 工具模块
│   ├── utils.py              # 通用工具类
│   └── stock_list.py         # 股票列表工具
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
- VWAP成交量加权平均价
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
   
   # 进行AI分析
   if ai_client.is_available():
       advice = ai_client.analyze_investment_recommendation(
           technical_report="技术分析报告内容",
           stock_code="000001"
       )
       print(f"AI投资建议: {ai_client.get_recommendation_text(advice)}")
   else:
       print("AI功能不可用，请检查API配置")
   ```

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

### 安装开发依赖

```bash
pip install -e ".[dev]"
```

### 运行示例

```bash
cd examples
python legacy_api.py
```

### 代码风格

项目使用以下工具确保代码质量：

```bash
# 代码格式化
black AIShareTxt/

# 代码检查
flake8 AIShareTxt/

# 运行测试
pytest
```

## 📈 使用示例

### 完整分析流程

```python
from AIShareTxt import StockAnalyzer

def analyze_example():
    """完整的股票分析示例"""

    # 1. 创建分析器
    analyzer = StockAnalyzer()

    # 2. 分析指定股票
    stock_code = "000001"  # 平安银行
    report = analyzer.analyze_stock(stock_code)

    # 3. 输出分析报告
    print(f"股票 {stock_code} 分析报告：")
    print("=" * 60)
    print(report)

# 运行示例
if __name__ == "__main__":
    analyze_example()
```

### 批量分析

```python
from AIShareTxt.utils.stock_list import get_stock_list
from AIShareTxt import StockAnalyzer

def batch_analysis():
    """批量分析示例"""

    # 获取股票列表
    stocks = get_stock_list()
    if stocks is None:
        print("无法获取股票列表")
        return

    # 分析前5只股票
    analyzer = StockAnalyzer()

    for idx, stock in stocks.head(5).iterrows():
        stock_code = stock['代码']
        stock_name = stock['名称']

        print(f"\n分析 {stock_name} ({stock_code})...")
        print("=" * 50)

        try:
            report = analyzer.analyze_stock(stock_code)
            print(report)

        except Exception as e:
            print(f"  分析失败: {e}")

batch_analysis()
```

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用木兰宽松许可证 第2版 - 查看 [LICENSE](LICENSE) 文件了解详情。

## ⚠️ 免责声明

本工具提供的所有信息仅供参考，不构成投资建议。投资有风险，入市需谨慎。

## 📞 联系方式

- 项目主页: https://github.com/your-repo/aishare-txt
- 问题反馈: https://github.com/your-repo/aishare-txt/issues
- 邮箱: aishare@example.com

## 🙏 致谢

感谢以下开源项目的支持：
- [akshare](https://github.com/akfamily/akshare) - 金融数据接口
- [TA-Lib](https://mrjbq7.github.io/ta-lib/) - 技术分析库
- [pandas](https://pandas.pydata.org/) - 数据分析库
- [numpy](http://www.numpy.org/) - 科学计算库