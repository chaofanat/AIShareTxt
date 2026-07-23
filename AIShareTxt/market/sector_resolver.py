#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
个股 → 板块映射解析器

输入 stock_code，返回其主行业 + 关联概念板块（含涨跌信息）。

P0 限制：
- 主行业通过 akshare stock_individual_info_em 获取，industry_boards 查当日涨跌
- 概念板块暂不实现（需要遍历 stock_board_concept_cons_em 反查，性能差），留 P2
"""

from typing import Dict, Any, Optional
import pandas as pd

from ..core.config import IndicatorConfig as Config
from ..utils.utils import LoggerManager
from .market_fetcher import MarketDataFetcher


class SectorResolver:
    """解析个股所属板块及当日涨跌"""

    def __init__(self, fetcher: Optional[MarketDataFetcher] = None):
        self.fetcher = fetcher or MarketDataFetcher()
        self.config = Config()
        self.logger = LoggerManager.get_logger('market.sector_resolver')

    def resolve(self, stock_code: str) -> Dict[str, Any]:
        """
        Args:
            stock_code: 6 位股票代码

        Returns:
            {
                'industry': Optional[str],              # 主行业名（如"银行"），查不到为 None
                'industry_detail': Optional[dict],      # 主行业当日涨跌详情
                'concepts': List[dict],                 # 关联概念列表（P0 为空）
            }
        """
        industry = self.fetcher.get_stock_industry(stock_code)
        if not industry:
            self.logger.info(f"个股 {stock_code} 未查到行业信息")
            return {'industry': None, 'industry_detail': None, 'concepts': []}

        industry_detail = self._lookup_board_detail(industry, self.fetcher.get_industry_boards())
        concepts = self._resolve_concepts(stock_code)  # P0 返回空

        return {
            'industry': industry,
            'industry_detail': industry_detail,
            'concepts': concepts,
        }

    def _lookup_board_detail(self, board_name: str, boards_df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """在板块列表 DataFrame 中查指定板块的当日数据。"""
        if boards_df is None or boards_df.empty or 'name' not in boards_df.columns:
            return None

        row = boards_df[boards_df['name'] == board_name]
        if row.empty:
            self.logger.debug(f"板块 '{board_name}' 在当日列表中未找到")
            return None

        r = row.iloc[0]
        detail = {'name': board_name}

        if 'pct_chg' in r.index and pd.notna(r['pct_chg']):
            detail['pct_chg'] = float(r['pct_chg'])
        if 'advance_count' in r.index and pd.notna(r.get('advance_count')):
            detail['advance_count'] = int(r['advance_count'])
        if 'decline_count' in r.index and pd.notna(r.get('decline_count')):
            detail['decline_count'] = int(r['decline_count'])

        return detail

    def _resolve_concepts(self, stock_code: str):  # pragma: no cover - P0 留空
        """P0 未实现：反查个股所属概念需要遍历所有概念板块成分股，性能差。

        P2 计划方案：
        1. 一次性构建 stock_code → concepts 反向映射（每日缓存）
        2. 或使用东方财富直接 API（push2.eastmoney.com）
        """
        return []
