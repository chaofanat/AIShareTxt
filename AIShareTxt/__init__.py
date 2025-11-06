"""
AIShareTxt - 股票技术指标分析工具包

一个功能强大的股票技术指标分析工具，支持多种技术指标计算、
AI集成分析和报告生成功能。
"""

__version__ = "1.0.4"
__author__ = "AIShareTxt Team"

# 导入核心类
from .core.analyzer import StockAnalyzer
from .core.data_fetcher import StockDataFetcher
from .core.report_generator import ReportGenerator
from .core.config import IndicatorConfig

# 导入技术指标
from .indicators.technical_indicators import TechnicalIndicators

# 导入AI客户端
from .ai.client import AIClient

# 导入工具
from .utils.utils import Logger
from .utils.stock_list import get_stock_list

# 定义公共API
__all__ = [
    # 核心类
    "StockAnalyzer",
    "StockDataFetcher",
    "ReportGenerator",
    "IndicatorConfig",
    
    # 技术指标
    "TechnicalIndicators",
    
    # AI客户端
    "AIClient",
    
    # 工具
    "Logger",
    "get_stock_list",
]

# 便捷函数
def analyze_stock(symbol, period="1y", use_ai=False, ai_provider="deepseek"):
    """
    便捷函数：分析股票技术指标

    参数:
        symbol: 股票代码
        period: 数据周期，默认1年
        use_ai: 是否使用AI分析
        ai_provider: AI提供商，支持"deepseek"或"zhipu"

    返回:
        分析结果字典
    """
    analyzer = StockAnalyzer()
    return analyzer.analyze_stock(symbol)


