#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场环境分析器

组合 MarketDataFetcher 拉取的数据，输出结构化分析结果：
- 各主指数的 OHLCV + 技术指标（复用 TechnicalIndicators）+ 近 N 日涨跌
- 市场宽度数据
- 个股板块（如传入 stock_code）
- 三层阶段判定（趋势 + 情绪 + 合成）
"""

from typing import Dict, Any, Optional, List, Tuple

import pandas as pd

from ..core.config import IndicatorConfig as Config
from ..utils.utils import LoggerManager
from ..indicators.technical_indicators import TechnicalIndicators
from .market_fetcher import MarketDataFetcher
from .sector_resolver import SectorResolver


class MarketEnvironmentAnalyzer:
    """市场环境分析器"""

    def __init__(self, fetcher: Optional[MarketDataFetcher] = None):
        self.fetcher = fetcher or MarketDataFetcher()
        self.config = Config()
        self.logger = LoggerManager.get_logger('market.analyzer')
        self._market_config = self.config.MARKET_ENVIRONMENT_CONFIG
        self._indicators_calc = TechnicalIndicators()

    # ==================== 主入口 ====================

    def analyze(self, stock_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Args:
            stock_code: 6 位股票代码。传入则附加个股板块信息。

        Returns:
            {
                'indexes': Dict[code, dict],
                'breadth': Optional[dict],
                'sector': Optional[dict],
                'phase': dict,   # {phase, trend, sentiment, confidence, reason_tags}
            }
        """
        result: Dict[str, Any] = {
            'indexes': {},
            'breadth': None,
            'sector': None,
            'data_date': None,
            'total_amount_ma20_series': None,
            'phase': {
                'phase': '未知', 'trend': 'unknown',
                'sentiment': 'unknown', 'confidence': 0.0, 'reason_tags': [],
            },
        }

        # 1. 主要指数
        for code in self._market_config['primary_indexes']:
            idx = self._analyze_index(code)
            if idx is not None:
                result['indexes'][code] = idx

        # 数据日期：取主指数最新交易日（参考个股报告 indicators['date'] 口径）
        for idx in result['indexes'].values():
            d = (idx.get('indicators') or {}).get('date')
            if d:
                result['data_date'] = d
                break

        # 全市场成交额中枢 MA20（近 5 日）
        try:
            result['total_amount_ma20_series'] = self.fetcher.get_total_amount_ma20_series(5)
        except Exception as e:
            self.logger.warning(f"获取全市场成交额 MA20 序列失败：{e}")
            result['total_amount_ma20_series'] = None

        # 2. 市场宽度
        result['breadth'] = self.fetcher.get_market_snapshot()

        # 3. 个股板块
        if stock_code:
            try:
                resolver = SectorResolver(self.fetcher)
                result['sector'] = resolver.resolve(stock_code)
            except Exception as e:
                self.logger.warning(f"解析个股 {stock_code} 板块失败：{e}")
                result['sector'] = None

        # 4. 阶段判定（以上证指数为基准）
        primary = result['indexes'].get('sh000001')
        result['phase'] = self._classify_market_phase(primary, result['breadth'])

        return result

    # ==================== 指数分析 ====================

    def _analyze_index(self, code: str) -> Optional[Dict[str, Any]]:
        """拉取单个指数日线 + 委托 TechnicalIndicators 计算指标 + 近 N 日涨跌幅。"""
        df = self.fetcher.get_index_data(code)
        if df is None or len(df) < 30:
            self.logger.warning(f"指数 {code} 数据不足 30 行，跳过")
            return None

        # 委托复用现有指标计算器
        try:
            indicators = self._indicators_calc.process_all_indicators(df)
        except Exception as e:
            self.logger.warning(f"指数 {code} 指标计算失败：{e}")
            return None

        if not indicators:
            return None

        # 近 N 日累计涨跌幅
        recent_pct = self._compute_recent_pct(df, [1, 3, 5, 10, 20])

        return {
            'code': code,
            'name': self._market_config['index_codes'].get(code, code),
            'indicators': indicators,
            'recent_pct': recent_pct,
        }

    @staticmethod
    def _compute_recent_pct(df: pd.DataFrame, periods: List[int]) -> Dict[str, Optional[float]]:
        """累计涨跌幅（最近 N 个交易日）。"""
        result: Dict[str, Optional[float]] = {}
        if 'close' not in df.columns or len(df) < 2:
            return {f'{n}d': None for n in periods}

        closes = df['close'].values
        for n in periods:
            if len(closes) > n:
                start = closes[-n - 1]
                if start > 0:
                    result[f'{n}d'] = round(float((closes[-1] - start) / start * 100), 2)
                else:
                    result[f'{n}d'] = None
            else:
                result[f'{n}d'] = None
        return result

    # ==================== 阶段判定（三层） ====================

    def _classify_market_phase(
        self, primary_index: Optional[Dict[str, Any]], breadth: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """三层阶段判定：趋势维度 + 情绪维度 + 合成。"""
        if not primary_index:
            return {
                'phase': '未知', 'trend': 'unknown', 'sentiment': 'unknown',
                'confidence': 0.0, 'reason_tags': ['主指数数据缺失'],
            }

        indicators = primary_index.get('indicators', {})

        trend, trend_tags = self._classify_trend(indicators)
        sentiment, sentiment_tags = self._classify_sentiment(breadth)
        phase = self._compose_phase(trend, sentiment)

        # 置信度：趋势 + 情绪都偏离"中性"时提高
        confidence = 0.5
        if trend != 'ranging' and sentiment not in ('normal', 'unknown'):
            confidence = 0.8
        elif trend != 'ranging' or sentiment not in ('normal', 'unknown'):
            confidence = 0.65

        return {
            'phase': phase,
            'trend': trend,
            'sentiment': sentiment,
            'confidence': confidence,
            'reason_tags': trend_tags + sentiment_tags,
        }

    def _classify_trend(self, indicators: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Layer 1: 基于 MA 排列 + ADX 判定趋势方向。"""
        adx = float(indicators.get('ADX', 0) or 0)
        trend_pattern = str(indicators.get('trend_pattern', ''))
        arrangement = str(indicators.get('arrangement_pattern', ''))
        thresholds = self._market_config['phase_adx_thresholds']

        tags = [f"ADX={adx:.1f}", f"MA: {trend_pattern or arrangement or '未知'}"]

        if adx >= thresholds['trending']:
            if '多头' in trend_pattern or '>' in arrangement and '<' not in arrangement:
                return 'trending_up', tags
            if '空头' in trend_pattern or '<' in arrangement:
                return 'trending_down', tags

        if adx < thresholds['ranging']:
            return 'ranging', tags

        # ADX 处于 20-25 中间态，看 MA 排列兜底
        if '多头' in trend_pattern:
            return 'trending_up', tags
        if '空头' in trend_pattern:
            return 'trending_down', tags
        return 'ranging', tags

    def _classify_sentiment(self, breadth: Optional[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """Layer 2: 基于市场宽度判定情绪。"""
        if not breadth:
            return 'unknown', ['宽度数据缺失']

        cfg = self._market_config['breadth_thresholds']
        limit_down = int(breadth.get('limit_down_count', 0) or 0)
        limit_up = int(breadth.get('limit_up_count', 0) or 0)
        median_pct = float(breadth.get('median_pct', 0) or 0)
        seal_ratio = float(breadth.get('seal_ratio', 0) or 0)

        tags = [f"涨停{limit_up}/跌停{limit_down}", f"中位数{median_pct:.2f}%"]

        # 恐慌：跌停多 + 中位数跌幅大
        if limit_down >= cfg['panic_limit_down_count'] and median_pct <= cfg['panic_median_pct']:
            tags.append(f"触发恐慌阈值（跌停≥{cfg['panic_limit_down_count']}且中位数≤{cfg['panic_median_pct']}%）")
            return 'panic', tags

        # 过热：涨停多 + 封板率高
        if limit_up >= cfg['hot_limit_up_count'] and seal_ratio >= cfg['hot_seal_ratio']:
            tags.append(f"触发过热阈值（涨停≥{cfg['hot_limit_up_count']}且封板率≥{cfg['hot_seal_ratio']*100:.0f}%）")
            return 'hot', tags

        return 'normal', tags

    @staticmethod
    def _compose_phase(trend: str, sentiment: str) -> str:
        """Layer 3: 趋势 × 情绪 合成最终阶段标签。"""
        table = {
            ('trending_up', 'hot'): '过热上涨',
            ('trending_up', 'normal'): '趋势上涨',
            ('trending_up', 'panic'): '背离上涨（指数涨但个股恐慌）',
            ('trending_up', 'unknown'): '趋势上涨',
            ('trending_down', 'panic'): '恐慌下跌',
            ('trending_down', 'normal'): '趋势下跌',
            ('trending_down', 'hot'): '背离下跌（指数跌但个股过热）',
            ('trending_down', 'unknown'): '趋势下跌',
            ('ranging', 'normal'): '震荡',
            ('ranging', 'hot'): '偏强震荡',
            ('ranging', 'panic'): '偏弱震荡',
            ('ranging', 'unknown'): '震荡',
        }
        return table.get((trend, sentiment), '未知')
