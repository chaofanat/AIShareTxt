"""MarketEnvironmentAnalyzer 单元测试

重点覆盖三层阶段判定逻辑（趋势 × 情绪 → 合成）。
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from AIShareTxt.market.market_analyzer import MarketEnvironmentAnalyzer
from AIShareTxt.utils.cache import get_global_cache


# ==================== fixtures ====================

@pytest.fixture(autouse=True)
def clear_cache():
    cache = get_global_cache()
    cache.clear()
    yield
    cache.clear()


def _make_index_df(n: int, trend: str = 'up') -> pd.DataFrame:
    """构造指数 OHLCV。

    trend:
        'up'   - 持续上涨（MA 多头排列）
        'down' - 持续下跌（MA 空头排列）
        'flat' - 震荡（围绕均值波动）
    """
    dates = pd.date_range(end=datetime.now().date(), periods=n, freq='B')
    base = 3000.0
    if trend == 'up':
        # 单边上涨：每天涨 0.5%
        closes = [base * (1.005 ** i) for i in range(n)]
    elif trend == 'down':
        closes = [base * (0.995 ** i) for i in range(n)]
    else:
        # 震荡：sin 波动 ±1%
        closes = [base * (1 + 0.01 * np.sin(i / 3)) for i in range(n)]

    closes = np.array(closes)
    return pd.DataFrame({
        'date': dates,
        'open': closes * 0.999,
        'close': closes,
        'high': closes * 1.003,
        'low': closes * 0.997,
        'volume': 1e8 + np.arange(n) * 1e6,
    })


def _make_breadth(advance=2000, decline=2000, limit_up=30, limit_down=10,
                  median_pct=0.1, seal_ratio=0.5):
    return {
        'as_of': datetime.now(),
        'advance_count': advance,
        'decline_count': decline,
        'unchanged_count': 100,
        'limit_up_count': limit_up,
        'limit_down_count': limit_down,
        'touched_limit_up': limit_up,
        'touched_limit_down': limit_down,
        'seal_ratio': seal_ratio,
        'median_pct': median_pct,
        'total_amount_yi': 8000.0,
        'spot_df': pd.DataFrame(),
    }


def _patch_fetcher(monkeypatch, analyzer, index_trend='up', breadth=None,
                   index_codes=None, ma20_series=None):
    """统一 monkeypatch fetcher 的网络方法。"""
    if index_codes is None:
        index_codes = ['sh000001', 'sz399001', 'sz399006']

    def fake_get_index_data(code, days=None):
        if code in index_codes:
            return _make_index_df(60, trend=index_trend)
        return None

    def fake_get_market_snapshot():
        return breadth

    def fake_get_total_amount_ma20_series(display_days=5):
        return ma20_series

    analyzer.fetcher.get_index_data = fake_get_index_data
    analyzer.fetcher.get_market_snapshot = fake_get_market_snapshot
    analyzer.fetcher.get_total_amount_ma20_series = fake_get_total_amount_ma20_series


# ==================== _classify_trend ====================

def test_classify_trend_up(monkeypatch):
    a = MarketEnvironmentAnalyzer()
    _patch_fetcher(monkeypatch, a, index_trend='up', breadth=_make_breadth())
    result = a.analyze()
    # 单边上涨序列 ADX 通常 > 25，trend_pattern 含"多头"
    assert result['phase']['trend'] == 'trending_up'
    assert result['phase']['sentiment'] == 'normal'


def test_classify_trend_down_with_panic(monkeypatch):
    a = MarketEnvironmentAnalyzer()
    _patch_fetcher(
        monkeypatch, a,
        index_trend='down',
        breadth=_make_breadth(limit_down=80, median_pct=-3.5),
    )
    result = a.analyze()
    assert result['phase']['trend'] == 'trending_down'
    assert result['phase']['sentiment'] == 'panic'
    assert result['phase']['phase'] == '恐慌下跌'
    assert result['phase']['confidence'] >= 0.8


def test_classify_flat_market_normal(monkeypatch):
    a = MarketEnvironmentAnalyzer()
    _patch_fetcher(
        monkeypatch, a,
        index_trend='flat',
        breadth=_make_breadth(limit_up=20, limit_down=15, median_pct=0.05),
    )
    result = a.analyze()
    # 震荡 + normal
    assert result['phase']['trend'] == 'ranging'
    assert result['phase']['sentiment'] == 'normal'
    assert '震荡' in result['phase']['phase']


def test_hot_market_phase(monkeypatch):
    a = MarketEnvironmentAnalyzer()
    _patch_fetcher(
        monkeypatch, a,
        index_trend='up',
        breadth=_make_breadth(limit_up=120, seal_ratio=0.75, median_pct=2.5),
    )
    result = a.analyze()
    assert result['phase']['sentiment'] == 'hot'
    assert result['phase']['phase'] == '过热上涨'


# ==================== 缺失数据降级 ====================

def test_missing_primary_index_returns_unknown(monkeypatch):
    """主指数 sh000001 缺失时阶段判定为"未知"。"""
    a = MarketEnvironmentAnalyzer()
    _patch_fetcher(monkeypatch, a, index_codes=[])  # 不返回任何指数
    result = a.analyze()
    assert result['phase']['phase'] == '未知'
    assert result['phase']['confidence'] == 0
    assert '主指数数据缺失' in result['phase']['reason_tags']


def test_missing_breadth_keeps_trend_only(monkeypatch):
    """宽度缺失时，sentiment 为 unknown，但 trend 仍能判定。"""
    a = MarketEnvironmentAnalyzer()
    _patch_fetcher(monkeypatch, a, index_trend='up', breadth=None)
    result = a.analyze()
    assert result['phase']['trend'] == 'trending_up'
    assert result['phase']['sentiment'] == 'unknown'
    # trending_up + unknown → 趋势上涨
    assert result['phase']['phase'] == '趋势上涨'


# ==================== 完整 analyze() 流程 ====================

def test_analyze_returns_full_structure(monkeypatch):
    a = MarketEnvironmentAnalyzer()
    _patch_fetcher(monkeypatch, a, index_trend='up', breadth=_make_breadth())
    result = a.analyze()

    assert 'indexes' in result
    assert 'breadth' in result
    assert 'sector' in result
    assert 'phase' in result

    # 三个主指数都被填充
    assert set(result['indexes'].keys()) == {'sh000001', 'sz399001', 'sz399006'}

    # 每个指数含必要字段
    sh = result['indexes']['sh000001']
    assert sh['name'] == '上证指数'
    assert 'indicators' in sh
    assert 'recent_pct' in sh
    assert 'current_price' in sh['indicators']
    assert '1d' in sh['recent_pct']

    assert result['breadth'] is not None
    assert result['sector'] is None  # 没传 stock_code


def test_analyze_populates_data_date_from_index(monkeypatch):
    """analyze() 应从主指数 indicators['date'] 提取 data_date（参考个股报告口径）。"""
    a = MarketEnvironmentAnalyzer()
    _patch_fetcher(monkeypatch, a, index_trend='up', breadth=_make_breadth())
    result = a.analyze()
    assert result['data_date'] is not None
    # 格式应为 YYYY-MM-DD
    assert len(result['data_date']) == 10
    # 与 sh000001 指数的最新日期一致
    expected = result['indexes']['sh000001']['indicators']['date']
    assert result['data_date'] == expected


def test_analyze_data_date_none_when_no_index(monkeypatch):
    """主指数全部缺失时，data_date 为 None。"""
    a = MarketEnvironmentAnalyzer()
    _patch_fetcher(monkeypatch, a, index_codes=[], breadth=_make_breadth())
    result = a.analyze()
    assert result['data_date'] is None


# ==================== 全市场成交额 MA20 透传 ====================

def test_analyze_passthrough_total_amount_ma20_series(monkeypatch):
    """analyze() 应将 fetcher.get_total_amount_ma20_series 结果透传到顶层。"""
    a = MarketEnvironmentAnalyzer()
    fake_series = [
        {'date': '2026-07-18', 'total_amount_yi': 8000, 'ma20_yi': 7500, 'source': 'gm'},
        {'date': '2026-07-21', 'total_amount_yi': 8200, 'ma20_yi': 7550, 'source': 'gm'},
        {'date': '2026-07-22', 'total_amount_yi': 8100, 'ma20_yi': 7580, 'source': 'gm'},
        {'date': '2026-07-23', 'total_amount_yi': 8300, 'ma20_yi': 7620, 'source': 'gm'},
        {'date': '2026-07-24', 'total_amount_yi': 8400, 'ma20_yi': 7650, 'source': 'gm'},
    ]
    _patch_fetcher(monkeypatch, a, index_trend='up', breadth=_make_breadth(),
                   ma20_series=fake_series)
    result = a.analyze()
    assert result['total_amount_ma20_series'] is not None
    assert len(result['total_amount_ma20_series']) == 5
    assert result['total_amount_ma20_series'][-1]['ma20_yi'] == 7650


def test_analyze_ma20_series_none_when_fetch_fails(monkeypatch):
    """fetcher 返回 None 时，result['total_amount_ma20_series'] 为 None，不阻塞其他字段。"""
    a = MarketEnvironmentAnalyzer()
    _patch_fetcher(monkeypatch, a, index_trend='up', breadth=_make_breadth(),
                   ma20_series=None)
    result = a.analyze()
    assert result['total_amount_ma20_series'] is None
    # 其他字段仍正常
    assert result['phase']['trend'] == 'trending_up'
    assert result['breadth'] is not None


def test_analyze_with_stock_code_invokes_sector_resolver(monkeypatch):
    """传入 stock_code 时应触发 sector resolver。"""
    a = MarketEnvironmentAnalyzer()
    _patch_fetcher(monkeypatch, a, index_trend='up', breadth=_make_breadth())

    # mock sector resolver
    called = {'n': 0}

    def fake_resolve(code):
        called['n'] += 1
        return {'industry': '银行', 'industry_detail': {'name': '银行', 'pct_chg': 1.2}, 'concepts': []}

    import AIShareTxt.market.market_analyzer as mod
    original = mod.SectorResolver
    try:
        class FakeSR:
            def __init__(self, fetcher): pass
            def resolve(self, code): return fake_resolve(code)
        mod.SectorResolver = FakeSR

        result = a.analyze('000001')
        assert called['n'] == 1
        assert result['sector'] is not None
        assert result['sector']['industry'] == '银行'
    finally:
        mod.SectorResolver = original


# ==================== recent_pct 计算 ====================

def test_recent_pct_calculation(monkeypatch):
    """近 N 日涨跌幅使用 close[-N-1] 作为起点（N 日累计涨跌）。"""
    a = MarketEnvironmentAnalyzer()
    _patch_fetcher(monkeypatch, a, index_trend='up', breadth=_make_breadth())
    result = a.analyze()
    sh = result['indexes']['sh000001']

    # 单边每日涨 0.5%，5 日累计应该约 +2.5%
    assert sh['recent_pct']['1d'] is not None
    assert sh['recent_pct']['5d'] > 0
    # 涨幅应该随周期增长（牛市中长周期涨幅 > 短周期）
    assert sh['recent_pct']['5d'] > sh['recent_pct']['1d']
    assert sh['recent_pct']['10d'] > sh['recent_pct']['5d']
