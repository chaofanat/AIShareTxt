"""
AIShareTxt market 子包

提供市场环境维度数据的获取、分析、报告生成。
与个股技术指标（indicators）独立分离，互不依赖。

公共 API:
    MarketDataFetcher          - 数据获取层
    MarketEnvironmentAnalyzer  - 分析与阶段判定
    MarketReportGenerator      - 文本报告生成
    MarketEnvironmentProcessor - 编排器
    SectorResolver             - 个股 → 板块映射
    analyze_market             - 便捷函数
"""

from .market_fetcher import MarketDataFetcher
from .market_analyzer import MarketEnvironmentAnalyzer
from .market_report_generator import MarketReportGenerator
from .sector_resolver import SectorResolver
from .processor import MarketEnvironmentProcessor


__all__ = [
    "MarketDataFetcher",
    "MarketEnvironmentAnalyzer",
    "MarketReportGenerator",
    "SectorResolver",
    "MarketEnvironmentProcessor",
    "analyze_market",
]


def analyze_market(stock_code=None):
    """
    便捷函数：生成市场环境报告。

    Args:
        stock_code: 可选，6 位股票代码。传入则附加个股板块信息。

    Returns:
        str: 完整的市场环境报告文本
    """
    processor = MarketEnvironmentProcessor()
    return processor.generate_market_report(stock_code)
