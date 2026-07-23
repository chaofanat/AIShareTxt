"""MarketReportGenerator 单测。

测试 focus：
- 各 section 的渲染逻辑（含缺失数据降级）
- 整体报告结构（header + 4 个 section + footer）
- 带或不带 stock_code 的输出差异
"""

from datetime import datetime
import re

import pandas as pd
import pytest

from AIShareTxt.market.market_report_generator import MarketReportGenerator


def _make_market_data(include_index=True, include_breadth=True, sector=None):
    """构造测试用 market_data（结构同 MarketEnvironmentAnalyzer.analyze()）。"""
    indexes = {}
    if include_index:
        indexes = {
            'sh000001': {
                'code': 'sh000001',
                'name': '上证指数',
                'indicators': {
                    'current_price': 3210.45,
                    'ADX': 28.5,
                    'RSI_14': 55.2,
                    'trend_pattern': '多头排列',
                    'arrangement_pattern': 'MA5>MA10>MA20',
                    'MACD_HIST': 5.2,
                    'date': '2026-07-22',
                },
                'recent_pct': {'1d': 0.85, '3d': 1.5, '5d': -0.3, '10d': 1.2, '20d': 3.2},
            },
            'sz399001': {
                'code': 'sz399001',
                'name': '深证成指',
                'indicators': {
                    'current_price': 10520.3,
                    'ADX': 25.0,
                    'RSI_14': 50.0,
                    'trend_pattern': '多头排列',
                    'arrangement_pattern': 'MA5>MA10>MA20',
                    'date': '2026-07-22',
                },
                'recent_pct': {'1d': 1.2, '3d': 2.1, '5d': -0.5, '10d': None, '20d': 4.0},
            },
            'sz399006': {
                'code': 'sz399006',
                'name': '创业板指',
                'indicators': {
                    'current_price': 2150.6,
                    'ADX': 22.0,
                    'RSI_14': 48.5,
                    'trend_pattern': '均线缠绕',
                    'arrangement_pattern': '均线无规律排列',
                    'date': '2026-07-22',
                },
                'recent_pct': {'1d': -0.3, '3d': 0.5, '5d': 0.0, '10d': 1.0, '20d': 2.5},
            },
        }

    breadth = None
    if include_breadth:
        breadth = {
            'as_of': datetime(2026, 7, 22, 14, 30, 0),
            'advance_count': 2356,
            'decline_count': 2340,
            'unchanged_count': 89,
            'limit_up_count': 38,
            'limit_down_count': 12,
            'touched_limit_up': 50,
            'touched_limit_down': 15,
            'seal_ratio': 0.76,
            'median_pct': 0.12,
            'total_amount_yi': 8234.0,
            'spot_df': pd.DataFrame(),
        }

    return {
        'indexes': indexes,
        'breadth': breadth,
        'sector': sector,
        'data_date': '2026-07-22',
        'phase': {
            'phase': '趋势上涨',
            'trend': 'trending_up',
            'sentiment': 'normal',
            'confidence': 0.65,
            'reason_tags': ['ADX=28.5', 'MA: 多头排列', '涨停38/跌停12', '中位数+0.12%'],
        },
    }


# ==================== 基础渲染 ====================

def test_full_report_has_all_sections():
    gen = MarketReportGenerator()
    report = gen.generate_report(_make_market_data())

    assert "市场环境报告" in report
    assert "【一、大盘指数】" in report
    assert "【二、市场宽度（情绪指标）】" in report
    assert "【三、阶段判定】" in report
    # 没传 stock_code，不应有第四 section
    assert "【四、个股板块" not in report


def test_report_with_stock_code_has_sector_section():
    gen = MarketReportGenerator()
    data = _make_market_data(sector={
        'industry': '银行',
        'industry_detail': {'name': '银行', 'pct_chg': 1.25, 'advance_count': 30, 'decline_count': 10},
        'concepts': [],
    })
    report = gen.generate_report(data, stock_code='000001')
    assert "【四、个股板块（000001）】" in report
    assert "所属行业: 银行 +1.25%" in report
    assert "板块内上涨 30 / 下跌 10" in report


def test_index_section_renders_all_three_indexes():
    gen = MarketReportGenerator()
    report = gen.generate_report(_make_market_data())
    assert "上证指数" in report
    assert "深证成指" in report
    assert "创业板指" in report


def test_breadth_section_format():
    gen = MarketReportGenerator()
    report = gen.generate_report(_make_market_data())
    assert "上涨 2356 / 下跌 2340 / 平盘 89" in report
    assert "涨停 38 家 / 跌停 12 家" in report
    assert "封板率 76%" in report
    assert "中位数: +0.12%" in report
    assert "成交额: 8234 亿元" in report


def test_phase_section_rendered():
    gen = MarketReportGenerator()
    report = gen.generate_report(_make_market_data())
    assert "当前阶段: 趋势上涨" in report
    assert "置信度: 65%" in report
    assert "ADX=28.5" in report


# ==================== 缺失数据降级 ====================

def test_missing_indexes_shows_fallback():
    gen = MarketReportGenerator()
    data = _make_market_data(include_index=False)
    report = gen.generate_report(data)
    assert "暂无大盘指数数据" in report


def test_missing_breadth_shows_fallback():
    gen = MarketReportGenerator()
    data = _make_market_data(include_breadth=False)
    report = gen.generate_report(data)
    assert "暂无市场宽度数据" in report


def test_missing_sector_shows_fallback():
    gen = MarketReportGenerator()
    data = _make_market_data(sector=None)
    report = gen.generate_report(data, stock_code='000001')
    assert "暂无 000001 的板块数据" in report


def test_missing_industry_in_sector():
    gen = MarketReportGenerator()
    data = _make_market_data(sector={
        'industry': None,
        'industry_detail': None,
        'concepts': [],
    })
    report = gen.generate_report(data, stock_code='000001')
    assert "所属行业: 未查询到" in report


def test_empty_market_data_returns_error():
    gen = MarketReportGenerator()
    report = gen.generate_report({})
    assert "无法生成市场环境报告" in report


# ==================== 情绪标签 ====================

def test_panic_emotion_label():
    gen = MarketReportGenerator()
    data = _make_market_data()
    data['breadth']['limit_down_count'] = 50
    data['breadth']['median_pct'] = -3.0
    report = gen.generate_report(data)
    assert "情绪判定: 偏恐慌" in report


def test_hot_emotion_label():
    gen = MarketReportGenerator()
    data = _make_market_data()
    data['breadth']['limit_up_count'] = 100
    data['breadth']['seal_ratio'] = 0.75
    report = gen.generate_report(data)
    assert "情绪判定: 偏过热" in report


def test_normal_emotion_label():
    gen = MarketReportGenerator()
    report = gen.generate_report(_make_market_data())
    assert "情绪判定: 正常" in report


# ==================== 数据日期（参考个股报告口径） ====================

def test_header_uses_data_date_when_present():
    """market_data 顶层提供 data_date 时，header 应直接使用（不取 breadth.as_of）。"""
    gen = MarketReportGenerator()
    data = _make_market_data()
    data['data_date'] = '2026-07-22'
    report = gen.generate_report(data)
    assert "数据日期: 2026-07-22" in report
    # 不应再出现带时间的旧格式
    assert "数据时间:" not in report


def test_header_falls_back_to_breadth_as_of_date():
    """data_date 缺失时，从 breadth.as_of 提取日期部分。"""
    gen = MarketReportGenerator()
    data = _make_market_data()
    data.pop('data_date')
    report = gen.generate_report(data)
    assert "数据日期: 2026-07-22" in report


def test_header_shows_na_when_no_data_source():
    """data_date 与 breadth 都缺失时，header 显示 N/A。"""
    gen = MarketReportGenerator()
    data = _make_market_data(include_breadth=False)
    data.pop('data_date', None)
    report = gen.generate_report(data)
    assert "数据日期: N/A" in report


# ==================== 成交额中枢 MA20 ====================

def _make_ma20_series(values, source='gm'):
    """构造 MA20 序列 fixture。values: [(date_str, total, ma20), ...]"""
    return [
        {'date': d, 'total_amount_yi': t, 'ma20_yi': m, 'source': source}
        for d, t, m in values
    ]


def test_breadth_renders_ma20_series_gm():
    gen = MarketReportGenerator()
    data = _make_market_data()
    data['total_amount_ma20_series'] = _make_ma20_series([
        ('2026-07-18', 7400, 7500),
        ('2026-07-21', 7600, 7550),
        ('2026-07-22', 7700, 7580),
        ('2026-07-23', 7800, 7620),
        ('2026-07-24', 7900, 7650),
    ])
    report = gen.generate_report(data)
    assert "成交额中枢 MA20 近5日 (07-18→07-24)" in report
    # 升序展示，最新值在末尾
    assert "7500 / 7550 / 7580 / 7620 / 7650 亿" in report
    # gm 源不加标注
    assert "上证代理" not in report


def test_breadth_renders_ma20_series_akshare_proxy_tag():
    """akshare 备源时附加（上证代理）标注。"""
    gen = MarketReportGenerator()
    data = _make_market_data()
    data['total_amount_ma20_series'] = _make_ma20_series([
        ('2026-07-20', 5000, 4900),
        ('2026-07-21', 5100, 4950),
        ('2026-07-22', 5200, 5000),
        ('2026-07-23', 5300, 5050),
        ('2026-07-24', 5400, 5100),
    ], source='akshare_index_proxy')
    report = gen.generate_report(data)
    assert "成交额中枢 MA20 近5日 (07-20→07-24)" in report
    assert "（上证代理）" in report


def test_breadth_omits_ma20_line_when_missing():
    """total_amount_ma20_series 缺失或空时，不输出 MA20 行（不影响其他 breadth 展示）。"""
    gen = MarketReportGenerator()
    data = _make_market_data()
    data['total_amount_ma20_series'] = None
    report = gen.generate_report(data)
    assert "成交额中枢 MA20" not in report
    # 其他 breadth 字段仍存在
    assert "全市场成交额: 8234 亿元" in report


def test_breadth_omits_ma20_line_when_empty_list():
    gen = MarketReportGenerator()
    data = _make_market_data()
    data['total_amount_ma20_series'] = []
    report = gen.generate_report(data)
    assert "成交额中枢 MA20" not in report
