#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场环境文本报告生成器

模仿 indicators/report_generator.py 的风格：
- report.extend(section) + "\n".join 模式
- 中文数字标题 + section_separator
- 失败降级：数据缺失时显式输出"暂无 xxx 数据"，不静默跳过
"""

from datetime import datetime
from typing import Dict, Any, Optional, List

from ..core.config import IndicatorConfig as Config
from ..utils.utils import LoggerManager


class MarketReportGenerator:
    """市场环境文本报告生成器"""

    def __init__(self):
        self.config = Config()
        self.logger = LoggerManager.get_logger('market.report_generator')

    # ==================== 主入口 ====================

    def generate_report(self, market_data: Dict[str, Any], stock_code: Optional[str] = None) -> str:
        """
        Args:
            market_data: MarketEnvironmentAnalyzer.analyze() 返回的字典
            stock_code: 可选，传入时生成板块 section

        Returns:
            str: 完整的市场环境报告文本
        """
        if not market_data:
            return "无法生成市场环境报告：分析结果为空"

        report: List[str] = []
        report.extend(self._generate_header(market_data))
        report.extend(self._generate_index_section(market_data))
        report.extend(self._generate_breadth_section(market_data))
        report.extend(self._generate_phase_section(market_data))
        if stock_code:
            report.extend(self._generate_sector_section(market_data, stock_code))
        report.extend(self._generate_footer())
        return "\n".join(report)

    # ==================== 各 section ====================

    def _generate_header(self, market_data: Dict[str, Any]) -> List[str]:
        data_date = market_data.get('data_date')
        if not data_date:
            breadth = market_data.get('breadth') or {}
            as_of = breadth.get('as_of')
            data_date = as_of.strftime('%Y-%m-%d') if as_of else 'N/A'
        return [
            self.config.REPORT_CONFIG['title_separator'],
            "市场环境报告",
            f"数据日期: {data_date}",
            "数据来源: 掘金量化(gm) / akshare",
            self.config.REPORT_CONFIG['title_separator'],
        ]

    def _generate_index_section(self, market_data: Dict[str, Any]) -> List[str]:
        section: List[str] = ["\n【一、大盘指数】", self.config.REPORT_CONFIG['section_separator']]

        indexes = market_data.get('indexes') or {}
        if not indexes:
            section.append("暂无大盘指数数据")
            return section

        for code, data in indexes.items():
            section.extend(self._render_single_index(code, data))

        return section

    def _render_single_index(self, code: str, data: Dict[str, Any]) -> List[str]:
        name = data.get('name', code)
        indicators = data.get('indicators', {}) or {}
        recent_pct = data.get('recent_pct', {}) or {}

        price = indicators.get('current_price', 0) or 0
        pct_1d = recent_pct.get('1d')
        trend_pattern = indicators.get('trend_pattern', '')
        arrangement = indicators.get('arrangement_pattern', '')
        adx = indicators.get('ADX', 0) or 0
        rsi = indicators.get('RSI_14', 0) or 0

        # 涨跌幅展示（近 1 日累计）
        pct_str = f"{pct_1d:+.2f}%" if pct_1d is not None else "N/A"

        lines = [
            f"{name} {price:.2f} {pct_str}",
            f"  MA: {trend_pattern or arrangement or '未知'}  ADX: {adx:.1f}  RSI: {rsi:.1f}",
            self._format_recent_pct_line(recent_pct),
        ]
        return lines

    def _format_recent_pct_line(self, recent_pct: Dict[str, Any]) -> str:
        parts = []
        for n in [1, 3, 5, 10, 20]:
            v = recent_pct.get(f'{n}d')
            if v is not None:
                parts.append(f"{n}日 {v:+.2f}%")
        return f"  近: {' / '.join(parts)}" if parts else "  近: N/A"

    def _generate_breadth_section(self, market_data: Dict[str, Any]) -> List[str]:
        section: List[str] = ["\n【二、市场宽度（情绪指标）】", self.config.REPORT_CONFIG['section_separator']]

        breadth = market_data.get('breadth')
        if not breadth:
            section.append("暂无市场宽度数据")
            return section

        advance = breadth.get('advance_count', 0)
        decline = breadth.get('decline_count', 0)
        unchanged = breadth.get('unchanged_count', 0)
        limit_up = breadth.get('limit_up_count', 0)
        limit_down = breadth.get('limit_down_count', 0)
        seal_ratio = breadth.get('seal_ratio', 0) or 0
        median_pct = breadth.get('median_pct', 0) or 0
        amount_yi = breadth.get('total_amount_yi', 0) or 0

        section.extend([
            f"涨跌家数: 上涨 {advance} / 下跌 {decline} / 平盘 {unchanged}",
            f"涨跌停: 涨停 {limit_up} 家 / 跌停 {limit_down} 家（封板率 {seal_ratio*100:.0f}%）",
            f"涨跌幅中位数: {median_pct:+.2f}%",
            f"全市场成交额: {amount_yi:.0f} 亿元",
        ])

        # 成交额中枢 MA20 近5日（独立于 breadth，由 analyzer 顶层传入）
        ma20_line = self._format_total_amount_ma20_line(market_data)
        if ma20_line:
            section.append(ma20_line)

        # 简易情绪标注
        if limit_down >= 30 and median_pct <= -2:
            section.append("情绪判定: 偏恐慌（跌停增多 + 中位数为负）")
        elif limit_up >= 80 and seal_ratio >= 0.6:
            section.append("情绪判定: 偏过热（涨停密集 + 封板率高）")
        else:
            section.append("情绪判定: 正常")

        return section

    @staticmethod
    def _format_total_amount_ma20_line(market_data: Dict[str, Any]) -> Optional[str]:
        """格式化成交额中枢 MA20 近5日展示。

        格式：成交额中枢 MA20 近5日 (MM-DD→MM-DD): v1 / v2 / ... / v5 亿（升序：旧 → 新）[（上证代理）]
        日期范围让顺序明确，避免歧义。
        """
        series = market_data.get('total_amount_ma20_series')
        if not series:
            return None
        ma20_values = [int(s.get('ma20_yi') or 0) for s in series]
        if not ma20_values:
            return None
        # 日期范围（取首末），格式 MM-DD
        first_date = (series[0].get('date') or '')[5:]  # 截掉年份
        last_date = (series[-1].get('date') or '')[5:]
        date_range = f" ({first_date}→{last_date})" if first_date and last_date else ""
        source_tag = ''
        src = (series[-1] or {}).get('source', '')
        if src == 'akshare_index_proxy':
            source_tag = '（上证代理）'
        return (
            f"成交额中枢 MA20 近{len(ma20_values)}日{date_range}: "
            f"{' / '.join(str(v) for v in ma20_values)} 亿{source_tag}"
        )

    def _generate_phase_section(self, market_data: Dict[str, Any]) -> List[str]:
        section: List[str] = ["\n【三、阶段判定】", self.config.REPORT_CONFIG['section_separator']]

        phase = market_data.get('phase') or {}
        phase_label = phase.get('phase', '未知')
        confidence = phase.get('confidence', 0) or 0
        tags = phase.get('reason_tags', []) or []

        section.append(f"当前阶段: {phase_label}")
        section.append(f"置信度: {confidence*100:.0f}%")
        if tags:
            section.append(f"判定理由: {'; '.join(tags)}")

        return section

    def _generate_sector_section(self, market_data: Dict[str, Any], stock_code: str) -> List[str]:
        section: List[str] = [
            f"\n【四、个股板块（{stock_code}）】",
            self.config.REPORT_CONFIG['section_separator'],
        ]

        sector = market_data.get('sector')
        if not sector:
            section.append(f"暂无 {stock_code} 的板块数据")
            return section

        industry = sector.get('industry')
        industry_detail = sector.get('industry_detail') or {}
        concepts = sector.get('concepts') or []

        if not industry:
            section.append("所属行业: 未查询到")
        else:
            pct = industry_detail.get('pct_chg')
            pct_str = f" {pct:+.2f}%" if pct is not None else ""
            advance = industry_detail.get('advance_count')
            decline = industry_detail.get('decline_count')
            inner = ""
            if advance is not None and decline is not None:
                inner = f"（板块内上涨 {advance} / 下跌 {decline}）"
            section.append(f"所属行业: {industry}{pct_str}{inner}")

        if concepts:
            concept_lines = []
            for c in concepts[:3]:  # P0 不会进入此分支
                name = c.get('name', '')
                pct = c.get('pct_chg')
                if pct is not None:
                    concept_lines.append(f"{name} {pct:+.2f}%")
                else:
                    concept_lines.append(name)
            section.append(f"关联概念: {' / '.join(concept_lines)}")
        else:
            section.append("关联概念: 暂未提供（计划在下一版本支持）")

        return section

    def _generate_footer(self) -> List[str]:
        return [
            "",
            self.config.REPORT_CONFIG['title_separator'],
            f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            self.config.REPORT_CONFIG['title_separator'],
        ]
