#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场环境处理器（编排器）

类比 StockDataProcessor：串联 fetcher → analyzer → report_generator。

对外入口：
    processor = MarketEnvironmentProcessor()
    report = processor.generate_market_report()              # 仅市场环境
    report = processor.generate_market_report('000001')      # 含个股板块
"""

from typing import Optional

from ..core.config import IndicatorConfig as Config
from ..utils.utils import LoggerManager, ErrorHandler
from .market_fetcher import MarketDataFetcher
from .market_analyzer import MarketEnvironmentAnalyzer
from .market_report_generator import MarketReportGenerator


class MarketEnvironmentProcessor:
    """市场环境报告编排器"""

    def __init__(self):
        self.config = Config()
        self.logger = LoggerManager.get_logger('market.processor')
        self.fetcher = MarketDataFetcher()
        self.analyzer = MarketEnvironmentAnalyzer(self.fetcher)
        self.report_generator = MarketReportGenerator()

    def generate_market_report(self, stock_code: Optional[str] = None) -> str:
        """
        生成市场环境报告。

        Args:
            stock_code: 可选，6 位股票代码。传入则附加板块信息。

        Returns:
            str: 完整报告文本。出错时返回用户可读的错误信息。
        """
        try:
            self.logger.info(f"开始生成市场环境报告（stock_code={stock_code}）")

            market_data = self.analyzer.analyze(stock_code)

            # 三大块全失败时整体降级
            has_index = bool(market_data.get('indexes'))
            has_breadth = market_data.get('breadth') is not None
            if not has_index and not has_breadth:
                self.logger.error("市场环境所有数据源均失败")
                return (
                    "无法生成市场环境报告：指数和市场宽度数据均获取失败。\n"
                    "可能原因：网络异常、akshare 接口暂时不可用、非交易日且无缓存。"
                )

            report = self.report_generator.generate_report(market_data, stock_code)
            self.logger.info("市场环境报告生成完成")
            return report

        except Exception as e:
            ErrorHandler.handle_api_error(e, "市场环境报告生成")
            return f"市场环境报告生成失败：{str(e)}"
