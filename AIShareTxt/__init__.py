"""
AIShareTxt - 股票技术指标分析工具包

一个功能强大的股票技术指标分析工具，支持多种技术指标计算、
AI集成分析和报告生成功能。
"""

__version__ = "2026.04.27.78"
__author__ = "AIShareTxt Team"

# 导入核心类
from .core.data_processor import StockDataProcessor
from .core.config import IndicatorConfig
# 向后兼容
StockAnalyzer = StockDataProcessor

# 从indicators模块导入
from .indicators.data_fetcher import StockDataFetcher
from .indicators.report_generator import ReportGenerator

# 导入技术指标
from .indicators.technical_indicators import TechnicalIndicators

# 导入AI客户端
from .ai.client import AIClient

# 导入市场环境模块
from .market import (
    MarketEnvironmentProcessor,
    MarketDataFetcher,
    MarketEnvironmentAnalyzer,
    MarketReportGenerator,
    SectorResolver,
    analyze_market,
)

# 导入工具
from .utils.utils import Logger
from .utils.stock_list import get_stock_list

# 定义公共API
__all__ = [
    # 核心类
    "StockDataProcessor",
    "StockAnalyzer",  # 向后兼容
    "StockDataFetcher",
    "ReportGenerator",
    "IndicatorConfig",

    # 技术指标
    "TechnicalIndicators",

    # AI客户端
    "AIClient",

    # 市场环境
    "MarketEnvironmentProcessor",
    "MarketDataFetcher",
    "MarketEnvironmentAnalyzer",
    "MarketReportGenerator",
    "SectorResolver",
    "analyze_market",

    # 工具
    "Logger",
    "get_stock_list",
]

# 便捷函数
def analyze_stock(symbol):
    """
    便捷函数：分析股票技术指标

    参数:
        symbol: 股票代码

    返回:
        分析结果字典
    """
    analyzer = StockAnalyzer()
    return analyzer.generate_stock_report(symbol)


