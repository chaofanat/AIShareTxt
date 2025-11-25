"""
AIShareTxt核心模块

包含股票分析器、数据获取器、报告生成器和配置管理等核心功能。
"""

from .data_processor import StockDataProcessor
from .config import IndicatorConfig
# 向后兼容
StockAnalyzer = StockDataProcessor

__all__ = [
    "StockDataProcessor",
    "StockAnalyzer",  # 向后兼容
    "IndicatorConfig",
]