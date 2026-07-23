"""MarketDataFetcher 单元测试。

所有 akshare 调用均 monkeypatch，不触发真实网络请求。
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from AIShareTxt.market import market_fetcher as mf_module
from AIShareTxt.market.market_fetcher import MarketDataFetcher
from AIShareTxt.utils.cache import get_global_cache


# ==================== fixtures ====================

@pytest.fixture(autouse=True)
def clear_cache(monkeypatch):
    """每个测试前后清空全局缓存，避免相互影响。

    默认禁用 gm 优先逻辑，让现有 akshare mock 测试保持原意。
    gm 专用测试在自己的用例中显式启用。
    """
    cache = get_global_cache()
    cache.clear()
    # 重置东财熔断状态，避免上一个测试触发的熔断影响后续
    mf_module._eastmoney_blackout_until = datetime.min
    # 禁用 gm：让 _init_gm 成为 noop，gm 相关属性置默认
    def _stub_init_gm(self):
        self._gm_token = ''
        self._gm_available = False
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.MarketDataFetcher._init_gm',
        _stub_init_gm,
    )
    yield
    cache.clear()
    mf_module._eastmoney_blackout_until = datetime.min


def _make_index_df(n: int = 30) -> pd.DataFrame:
    """构造 n 行的指数日线 DataFrame（中文列名，模拟 akshare 返回）。"""
    dates = pd.date_range(end=datetime.now().date(), periods=n, freq='B')
    return pd.DataFrame({
        '日期': dates,
        '开盘': 3000.0 + pd.Series(range(n)) * 5,
        '收盘': 3010.0 + pd.Series(range(n)) * 5,
        '最高': 3020.0 + pd.Series(range(n)) * 5,
        '最低': 2990.0 + pd.Series(range(n)) * 5,
        '成交量': 1000000 + pd.Series(range(n)) * 1000,
    })


def _make_index_df_en(n: int = 30) -> pd.DataFrame:
    """新浪 stock_zh_index_daily 返回英文列名。"""
    df = _make_index_df(n)
    return df.rename(columns={
        '日期': 'date', '开盘': 'open', '收盘': 'close',
        '最高': 'high', '最低': 'low', '成交量': 'volume',
    })


def _make_spot_df() -> pd.DataFrame:
    """构造全市场快照 DataFrame，含主板/创业板/科创板/ST 各类样本。"""
    rows = [
        # 主板涨停 + 曾封板
        {'代码': '000001', '名称': '平安银行', '最新价': 13.2, '涨跌幅': 10.0, '最高': 13.2, '最低': 12.0, '昨收': 12.0, '成交额': 1e9},
        # 主板上涨未涨停
        {'代码': '000002', '名称': '万科A', '最新价': 9.5, '涨跌幅': 3.2, '最高': 9.6, '最低': 9.2, '昨收': 9.2, '成交额': 5e8},
        # 主板下跌
        {'代码': '600000', '名称': '浦发银行', '最新价': 7.8, '涨跌幅': -1.5, '最高': 8.0, '最低': 7.7, '昨收': 7.92, '成交额': 3e8},
        # 主板跌停
        {'代码': '600001', '名称': '某股', '最新价': 5.4, '涨跌幅': -10.0, '最高': 6.0, '最低': 5.4, '昨收': 6.0, '成交额': 2e8},
        # 创业板（20% 涨跌停）
        {'代码': '300001', '名称': '特锐德', '最新价': 15.0, '涨跌幅': 19.5, '最高': 15.0, '最低': 12.5, '昨收': 12.55, '成交额': 4e8},
        # 科创板（20% 涨跌停）
        {'代码': '688001', '名称': '华兴源创', '最新价': 50.0, '涨跌幅': -19.0, '最高': 62.0, '最低': 50.0, '昨收': 61.73, '成交额': 6e8},
        # ST 股（5% 涨跌停）
        {'代码': '000002', '名称': 'ST 某股', '最新价': 2.1, '涨跌幅': -4.95, '最高': 2.2, '最低': 2.1, '昨收': 2.21, '成交额': 1e7},
        # 平盘
        {'代码': '000003', '名称': '平盘股', '最新价': 10.0, '涨跌幅': 0.0, '最高': 10.0, '最低': 10.0, '昨收': 10.0, '成交额': 5e7},
    ]
    return pd.DataFrame(rows)


def _make_industry_df() -> pd.DataFrame:
    return pd.DataFrame([
        {'板块名称': '银行', '涨跌幅': 1.5, '上涨家数': 30, '下跌家数': 10, '成交额': 5e9},
        {'板块名称': '半导体', '涨跌幅': -2.3, '上涨家数': 20, '下跌家数': 80, '成交额': 8e9},
    ])


def _make_industry_df_sina() -> pd.DataFrame:
    """新浪 stock_sector_spot 返回列名。"""
    return pd.DataFrame([
        {'板块': '银行', '公司家数': 40, '涨跌幅': 1.5, '总成交额': 5e9},
        {'板块': '半导体', '公司家数': 100, '涨跌幅': -2.3, '总成交额': 8e9},
    ])


def _disable_sina(monkeypatch):
    """让新浪备源调用也失败，用于测试"双源全挂"场景。"""
    def boom(*args, **kwargs):
        raise RuntimeError("sina disabled")
    monkeypatch.setattr('AIShareTxt.market.market_fetcher.ak.stock_zh_index_daily', boom)
    monkeypatch.setattr('AIShareTxt.market.market_fetcher.ak.stock_zh_a_spot', boom)
    monkeypatch.setattr('AIShareTxt.market.market_fetcher.ak.stock_sector_spot', boom)


# ==================== get_index_data ====================

def test_get_index_data_normal(monkeypatch):
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_zh_index_daily_em',
        lambda symbol: _make_index_df(50)
    )
    _disable_sina(monkeypatch)
    f = MarketDataFetcher()
    df = f.get_index_data('sh000001', days=30)
    assert df is not None
    assert len(df) == 30
    assert list(df.columns) == ['date', 'open', 'high', 'low', 'close', 'volume']
    # 升序
    assert df['date'].is_monotonic_increasing


def test_get_index_data_caches_within_ttl(monkeypatch):
    call_count = {'n': 0}

    def fake(symbol):
        call_count['n'] += 1
        return _make_index_df(50)

    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_zh_index_daily_em', fake
    )
    _disable_sina(monkeypatch)
    f = MarketDataFetcher()
    f.get_index_data('sh000001')
    f.get_index_data('sh000001')
    assert call_count['n'] == 1  # 第二次命中缓存


def test_get_index_data_failure_returns_none(monkeypatch):
    def boom(symbol):
        raise RuntimeError("network error")
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_zh_index_daily_em', boom
    )
    _disable_sina(monkeypatch)
    f = MarketDataFetcher()
    assert f.get_index_data('sh000001') is None


def test_get_index_data_missing_columns(monkeypatch):
    """如果返回的 DataFrame 缺少必需列，应返回 None 而非抛异常。"""
    bad = pd.DataFrame({'date': [datetime.now()], 'open': [1.0]})
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_zh_index_daily_em',
        lambda symbol: bad
    )
    _disable_sina(monkeypatch)
    f = MarketDataFetcher()
    assert f.get_index_data('sh000001') is None


def test_get_index_data_fallback_to_sina(monkeypatch):
    """东财失败时走新浪备源。"""
    def boom(symbol):
        raise RuntimeError("eastmoney down")
    monkeypatch.setattr('AIShareTxt.market.market_fetcher.ak.stock_zh_index_daily_em', boom)
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_zh_index_daily',
        lambda symbol: _make_index_df_en(50)
    )
    f = MarketDataFetcher()
    df = f.get_index_data('sh000001', days=20)
    assert df is not None
    assert len(df) == 20
    assert list(df.columns) == ['date', 'open', 'high', 'low', 'close', 'volume']
    # 东财失败应触发熔断
    assert mf_module._eastmoney_blackout_until > datetime.now()


def test_get_index_data_skips_eastmoney_during_blackout(monkeypatch):
    """熔断期间直接跳过东财，走新浪。"""
    em_calls = {'n': 0}

    def em(symbol):
        em_calls['n'] += 1
        raise RuntimeError("should not be called")

    monkeypatch.setattr('AIShareTxt.market.market_fetcher.ak.stock_zh_index_daily_em', em)
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_zh_index_daily',
        lambda symbol: _make_index_df_en(50)
    )
    # 提前触发熔断
    mf_module._trigger_blackout("test")
    f = MarketDataFetcher()
    df = f.get_index_data('sh000001', days=10)
    assert df is not None
    assert em_calls['n'] == 0  # 东财没被调用


# ==================== get_market_snapshot ====================

def test_get_market_snapshot_counts(monkeypatch):
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_zh_a_spot_em',
        lambda: _make_spot_df()
    )
    _disable_sina(monkeypatch)
    f = MarketDataFetcher()
    snap = f.get_market_snapshot()

    assert snap is not None
    assert snap['advance_count'] == 3
    assert snap['decline_count'] == 4
    assert snap['unchanged_count'] == 1
    assert snap['limit_up_count'] == 1
    assert snap['limit_down_count'] == 2  # 600001 + ST


def test_get_market_snapshot_seal_ratio(monkeypatch):
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_zh_a_spot_em',
        lambda: _make_spot_df()
    )
    _disable_sina(monkeypatch)
    f = MarketDataFetcher()
    snap = f.get_market_snapshot()
    assert snap['touched_limit_up'] == 1
    assert snap['limit_up_count'] == 1
    assert snap['seal_ratio'] == 1.0


def test_get_market_snapshot_median_pct(monkeypatch):
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_zh_a_spot_em',
        lambda: _make_spot_df()
    )
    _disable_sina(monkeypatch)
    f = MarketDataFetcher()
    snap = f.get_market_snapshot()
    pct_values = [10.0, 3.2, -1.5, -10.0, 19.5, -19.0, -4.95, 0.0]
    expected = pd.Series(pct_values).median()
    assert snap['median_pct'] == expected


def test_get_market_snapshot_caches(monkeypatch):
    call_count = {'n': 0}

    def fake():
        call_count['n'] += 1
        return _make_spot_df()

    monkeypatch.setattr('AIShareTxt.market.market_fetcher.ak.stock_zh_a_spot_em', fake)
    _disable_sina(monkeypatch)
    f = MarketDataFetcher()
    f.get_market_snapshot()
    f.get_market_snapshot()
    assert call_count['n'] == 1


def test_get_market_snapshot_failure(monkeypatch):
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_zh_a_spot_em',
        lambda: (_ for _ in ()).throw(RuntimeError("oops"))
    )
    _disable_sina(monkeypatch)
    f = MarketDataFetcher()
    assert f.get_market_snapshot() is None


def test_get_market_snapshot_fallback_to_sina(monkeypatch):
    """东财快照失败时走新浪 stock_zh_a_spot。"""
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_zh_a_spot_em',
        lambda: (_ for _ in ()).throw(RuntimeError("eastmoney down"))
    )
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_zh_a_spot',
        lambda: _make_spot_df()
    )
    f = MarketDataFetcher()
    snap = f.get_market_snapshot()
    assert snap is not None
    assert snap['advance_count'] == 3


# ==================== get_industry_boards ====================

def test_get_industry_boards(monkeypatch):
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_board_industry_name_em',
        lambda: _make_industry_df()
    )
    _disable_sina(monkeypatch)
    f = MarketDataFetcher()
    df = f.get_industry_boards()
    assert df is not None
    assert len(df) == 2
    assert 'name' in df.columns
    assert 'pct_chg' in df.columns


def test_get_industry_boards_failure(monkeypatch):
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_board_industry_name_em',
        lambda: (_ for _ in ()).throw(RuntimeError("oops"))
    )
    _disable_sina(monkeypatch)
    f = MarketDataFetcher()
    assert f.get_industry_boards() is None


def test_get_industry_boards_fallback_to_sina(monkeypatch):
    """东财行业板块失败时走新浪 stock_sector_spot。"""
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_board_industry_name_em',
        lambda: (_ for _ in ()).throw(RuntimeError("eastmoney down"))
    )
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_sector_spot',
        lambda indicator: _make_industry_df_sina()
    )
    f = MarketDataFetcher()
    df = f.get_industry_boards()
    assert df is not None
    assert len(df) == 2
    assert '银行' in df['name'].values
    # 新浪源不含 advance_count/decline_count，列应不存在
    assert 'advance_count' not in df.columns


# ==================== get_stock_industry ====================

def test_get_stock_industry_dataframe_format(monkeypatch):
    """akshare 较新版本返回 item/value 两列的 DataFrame。"""
    info = pd.DataFrame({
        'item': ['股票代码', '股票简称', '行业', '上市时间'],
        'value': ['000001', '平安银行', '银行', '1991-04-03'],
    })
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_individual_info_em',
        lambda symbol: info
    )
    f = MarketDataFetcher()
    assert f.get_stock_industry('000001') == '银行'


def test_get_stock_industry_no_industry_field(monkeypatch):
    info = pd.DataFrame({
        'item': ['股票代码', '股票简称'],
        'value': ['000001', '某股'],
    })
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_individual_info_em',
        lambda symbol: info
    )
    f = MarketDataFetcher()
    assert f.get_stock_industry('000001') is None


def test_get_stock_industry_cached_as_none_sentinel(monkeypatch):
    """查不到行业时缓存 '__NONE__' 标识，避免重复请求。"""
    call_count = {'n': 0}

    def fake(symbol):
        call_count['n'] += 1
        return pd.DataFrame({'item': ['code'], 'value': ['000001']})

    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_individual_info_em', fake
    )
    f = MarketDataFetcher()
    assert f.get_stock_industry('000001') is None
    assert f.get_stock_industry('000001') is None
    assert call_count['n'] == 1


def test_get_stock_industry_failure(monkeypatch):
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_individual_info_em',
        lambda symbol: (_ for _ in ()).throw(RuntimeError("oops"))
    )
    f = MarketDataFetcher()
    assert f.get_stock_industry('000001') is None


def test_get_stock_industry_skipped_during_blackout(monkeypatch):
    """东财熔断期间直接返回 None，不调用 akshare。"""
    em_calls = {'n': 0}

    def em(symbol):
        em_calls['n'] += 1
        return pd.DataFrame()

    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_individual_info_em', em
    )
    mf_module._trigger_blackout("test")
    f = MarketDataFetcher()
    assert f.get_stock_industry('000001') is None
    assert em_calls['n'] == 0


# ==================== gm 优先逻辑 ====================

def _enable_gm(monkeypatch):
    """覆盖 autouse 的禁用 fixture，让 fetcher 启用 gm。"""
    def _stub(self):
        self._gm_token = 'test-token'
        self._gm_available = True
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.MarketDataFetcher._init_gm',
        _stub,
    )


def _make_gm_index_df(n: int = 50) -> pd.DataFrame:
    """模拟 gm history 返回（eob + OHLCV）。"""
    dates = pd.date_range(end=datetime.now().date(), periods=n, freq='B')
    seq = pd.Series(range(n))
    return pd.DataFrame({
        'eob': dates,
        'open': 3000.0 + seq * 5,
        'high': 3020.0 + seq * 5,
        'low': 2990.0 + seq * 5,
        'close': 3010.0 + seq * 5,
        'volume': 1000000 + seq * 1000,
    })


def test_get_index_data_gm_priority(monkeypatch):
    """gm 可用且成功时走 gm，akshare 不被调用。"""
    _enable_gm(monkeypatch)

    gm_calls = {'n': 0}

    def fake_history(**kwargs):
        gm_calls['n'] += 1
        return _make_gm_index_df(50)

    monkeypatch.setattr('gm.api.history', fake_history)

    def boom_em(*a, **kw):
        raise AssertionError("东财不应被调用")
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_zh_index_daily_em', boom_em
    )
    _disable_sina(monkeypatch)

    f = MarketDataFetcher()
    df = f.get_index_data('sh000001', days=30)
    assert df is not None
    assert len(df) == 30
    assert list(df.columns) == ['date', 'open', 'high', 'low', 'close', 'volume']
    assert df['date'].is_monotonic_increasing
    assert gm_calls['n'] == 1


def test_get_index_data_gm_failure_falls_back(monkeypatch):
    """gm 失败时回落到东财。"""
    _enable_gm(monkeypatch)

    monkeypatch.setattr(
        'gm.api.history',
        lambda **kw: (_ for _ in ()).throw(RuntimeError("gm network error"))
    )
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_zh_index_daily_em',
        lambda symbol: _make_index_df(50)
    )
    _disable_sina(monkeypatch)

    f = MarketDataFetcher()
    df = f.get_index_data('sh000001', days=30)
    assert df is not None
    assert len(df) == 30


def test_get_stock_industry_gm_priority(monkeypatch):
    """个股行业 gm 可用时优先走 gm，akshare 不被调用。"""
    _enable_gm(monkeypatch)

    gm_calls = {'n': 0}

    def fake(*a, **kw):
        gm_calls['n'] += 1
        return pd.DataFrame([{
            'symbol': 'SZSE.000001', 'sec_name': '平安银行',
            'industry_code': '480000', 'industry_name': '银行',
        }])

    monkeypatch.setattr('gm.api.stk_get_symbol_industry', fake)
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_individual_info_em',
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("东财不应被调用"))
    )

    f = MarketDataFetcher()
    assert f.get_stock_industry('000001') == '银行'
    assert gm_calls['n'] == 1


def test_get_stock_industry_gm_failure_falls_back(monkeypatch):
    """gm 行业查询失败时回落到东财。"""
    _enable_gm(monkeypatch)

    monkeypatch.setattr(
        'gm.api.stk_get_symbol_industry',
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("gm down"))
    )
    info = pd.DataFrame({
        'item': ['股票代码', '行业'],
        'value': ['000001', '银行'],
    })
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_individual_info_em',
        lambda symbol: info
    )

    f = MarketDataFetcher()
    assert f.get_stock_industry('000001') == '银行'


def test_get_market_snapshot_gm_post_close(monkeypatch):
    """盘后 + gm 可用 → 走 gm，akshare 不被调用。"""
    _enable_gm(monkeypatch)
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.is_market_open', lambda: False
    )

    monkeypatch.setattr('gm.api.get_symbols', lambda **kw: pd.DataFrame([
        {'symbol': 'SHSE.600000', 'sec_name': '浦发银行'},
        {'symbol': 'SZSE.000001', 'sec_name': '平安银行'},
    ]))

    def fake_history(**kwargs):
        now = datetime.now()
        return pd.DataFrame([
            {'symbol': 'SHSE.600000', 'eob': now, 'close': 10.0,
             'high': 10.5, 'low': 9.5, 'pre_close': 9.8, 'amount': 1e9},
            {'symbol': 'SZSE.000001', 'eob': now, 'close': 13.0,
             'high': 13.2, 'low': 12.5, 'pre_close': 12.0, 'amount': 2e9},
        ])
    monkeypatch.setattr('gm.api.history', fake_history)

    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_zh_a_spot_em',
        lambda: (_ for _ in ()).throw(AssertionError("东财不应被调用"))
    )
    _disable_sina(monkeypatch)

    f = MarketDataFetcher()
    snap = f.get_market_snapshot()
    assert snap is not None
    # 两只都涨（600000: +2.04%, 000001: +8.33%）
    assert snap['advance_count'] == 2
    assert snap['decline_count'] == 0
    assert snap['median_pct'] > 0


def test_get_market_snapshot_gm_skipped_intraday(monkeypatch):
    """盘中 gm 跳过（history 拿不到当日数据），走 akshare。"""
    _enable_gm(monkeypatch)
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.is_market_open', lambda: True
    )

    gm_calls = {'n': 0}

    def boom_history(**kw):
        gm_calls['n'] += 1
        raise AssertionError("gm 盘中不应被调用")
    monkeypatch.setattr('gm.api.history', boom_history)
    monkeypatch.setattr('gm.api.get_symbols', lambda **kw: pd.DataFrame())

    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_zh_a_spot_em',
        lambda: _make_spot_df()
    )
    _disable_sina(monkeypatch)

    f = MarketDataFetcher()
    snap = f.get_market_snapshot()
    assert snap is not None
    assert gm_calls['n'] == 0


# ==================== 北交所过滤 ====================

def test_normalize_spot_filters_bse_eastmoney(monkeypatch):
    """东财格式：6 位裸码，过滤 43/83/87/92 开头的北交所标的。"""
    raw = pd.DataFrame([
        {'代码': '000001', '名称': '平安银行', '涨跌幅': 1.0, '昨收': 10.0},
        {'代码': '600000', '名称': '浦发银行', '涨跌幅': -0.5, '昨收': 10.0},
        {'代码': '920000', '名称': '北交所1', '涨跌幅': 9.0, '昨收': 10.0},  # 北交所
        {'代码': '830000', '名称': '北交所2', '涨跌幅': 12.0, '昨收': 10.0},  # 北交所
        {'代码': '430001', '名称': '北交所3', '涨跌幅': 5.0, '昨收': 10.0},  # 北交所
    ])
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_zh_a_spot_em',
        lambda: raw
    )
    _disable_sina(monkeypatch)

    f = MarketDataFetcher()
    snap = f.get_market_snapshot()
    assert snap is not None
    # 只剩 2 只沪深 A 股
    assert snap['advance_count'] + snap['decline_count'] + snap['unchanged_count'] == 2


def test_normalize_spot_filters_bse_sina(monkeypatch):
    """新浪格式：bj 前缀，过滤掉。"""
    raw = pd.DataFrame([
        {'代码': 'sh600000', '名称': '浦发银行', '涨跌幅': 1.0, '昨收': 10.0},
        {'代码': 'sz000001', '名称': '平安银行', '涨跌幅': -1.0, '昨收': 10.0},
        {'代码': 'bj920000', '名称': '北交所1', '涨跌幅': 9.0, '昨收': 10.0},
        {'代码': 'bj830001', '名称': '北交所2', '涨跌幅': 12.0, '昨收': 10.0},
    ])
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_zh_a_spot_em',
        lambda: (_ for _ in ()).throw(RuntimeError("eastmoney down"))
    )
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_zh_a_spot',
        lambda: raw
    )
    f = MarketDataFetcher()
    snap = f.get_market_snapshot()
    assert snap is not None
    assert snap['advance_count'] == 1
    assert snap['decline_count'] == 1


# ==================== gm 数据时效性检查 ====================

def test_gm_index_data_stale_falls_back(monkeypatch):
    """_is_gm_data_fresh 返回 False 时，gm 拿到数据也会回退 akshare。"""
    _enable_gm(monkeypatch)
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.MarketDataFetcher._is_gm_data_fresh',
        lambda self, df: False,
    )

    gm_calls = {'n': 0}
    def fake_history(**kwargs):
        gm_calls['n'] += 1
        return _make_gm_index_df(50)
    monkeypatch.setattr('gm.api.history', fake_history)

    ak_calls = {'n': 0}
    def ak_em(symbol):
        ak_calls['n'] += 1
        return _make_index_df(50)
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_zh_index_daily_em', ak_em
    )
    _disable_sina(monkeypatch)

    f = MarketDataFetcher()
    df = f.get_index_data('sh000001', days=30)
    assert df is not None
    assert gm_calls['n'] == 1  # gm 被调用过
    assert ak_calls['n'] == 1  # 但回退到 akshare


def test_gm_index_data_fresh_used(monkeypatch):
    """_is_gm_data_fresh 返回 True 时，gm 数据正常使用，akshare 不被调用。"""
    _enable_gm(monkeypatch)
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.MarketDataFetcher._is_gm_data_fresh',
        lambda self, df: True,
    )

    monkeypatch.setattr('gm.api.history', lambda **kw: _make_gm_index_df(50))
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.ak.stock_zh_index_daily_em',
        lambda symbol: (_ for _ in ()).throw(AssertionError("akshare 不应被调用"))
    )
    _disable_sina(monkeypatch)

    f = MarketDataFetcher()
    df = f.get_index_data('sh000001', days=30)
    assert df is not None
    assert len(df) == 30


def test_is_gm_data_fresh_post_close_trading_day(monkeypatch):
    """盘后交易日：gm 数据含今日 → True；含 T-1 → False。"""
    monkeypatch.setattr(
        'AIShareTxt.utils.trading_calendar.is_trading_day', lambda d: True
    )

    f = MarketDataFetcher()
    post_close = datetime(2026, 7, 23, 16, 0, 0)
    today = post_close.date()
    yesterday = today - timedelta(days=1)

    fresh_df = pd.DataFrame({'date': [pd.Timestamp(today)]})
    stale_df = pd.DataFrame({'date': [pd.Timestamp(yesterday)]})

    assert f._is_gm_data_fresh(fresh_df, now=post_close) is True
    assert f._is_gm_data_fresh(stale_df, now=post_close) is False


def test_is_gm_data_fresh_non_trading_day_no_check(monkeypatch):
    """非交易日：gm 数据即使 T-1 也视为可接受。"""
    monkeypatch.setattr(
        'AIShareTxt.utils.trading_calendar.is_trading_day', lambda d: False
    )

    f = MarketDataFetcher()
    saturday = datetime(2026, 7, 25, 16, 0, 0)
    friday = saturday.date() - timedelta(days=1)
    df = pd.DataFrame({'date': [pd.Timestamp(friday)]})
    assert f._is_gm_data_fresh(df, now=saturday) is True


def test_is_gm_data_fresh_intraday_no_check():
    """盘中（< 15:30）：不检查，直接返回 True。"""
    f = MarketDataFetcher()
    intraday = datetime(2026, 7, 23, 10, 30, 0)
    # 即使是 1 年前的数据，盘中也接受（gm 本就拿不到今日）
    df = pd.DataFrame({'date': [pd.Timestamp('2020-01-01')]})
    assert f._is_gm_data_fresh(df, now=intraday) is True


# ==================== 全市场成交额中枢 MA20 ====================

def test_normalize_index_df_keeps_amount_when_present():
    """_normalize_index_df 应保留 amount 列并转 numeric。"""
    raw = pd.DataFrame({
        '日期': pd.date_range(end=datetime.now().date(), periods=25, freq='B'),
        '开盘': 3000.0, '收盘': 3010.0, '最高': 3020.0, '最低': 2990.0,
        '成交量': 1000000,
        '成交额': [1e10 + i * 1e8 for i in range(25)],
    })
    from AIShareTxt.market.market_fetcher import _normalize_index_df, _INDEX_FIELD_ALIASES_EASTMONEY
    df = _normalize_index_df(raw, _INDEX_FIELD_ALIASES_EASTMONEY, 20)
    assert df is not None
    assert 'amount' in df.columns
    assert df['amount'].dtype.kind in 'iuf'  # numeric
    assert len(df) == 20


def test_normalize_index_df_works_without_amount():
    """无 amount 列时仍正常返回（仅 OHLCV）。"""
    raw = pd.DataFrame({
        '日期': pd.date_range(end=datetime.now().date(), periods=25, freq='B'),
        '开盘': 3000.0, '收盘': 3010.0, '最高': 3020.0, '最低': 2990.0,
        '成交量': 1000000,
    })
    from AIShareTxt.market.market_fetcher import _normalize_index_df, _INDEX_FIELD_ALIASES_EASTMONEY
    df = _normalize_index_df(raw, _INDEX_FIELD_ALIASES_EASTMONEY, 20)
    assert df is not None
    assert 'amount' not in df.columns


def test_get_total_amount_ma20_via_index_proxy(monkeypatch):
    """无 gm 时，用 sh000001 日线 amount 作为代理算 MA20。"""
    f = MarketDataFetcher()

    # 构造 30 日 sh000001 日线（含 amount）
    n = 30
    dates = pd.date_range(end=datetime.now().date(), periods=n, freq='B')
    fake_index = pd.DataFrame({
        'date': dates,
        'open': 3000.0, 'high': 3020.0, 'low': 2990.0, 'close': 3010.0,
        'volume': 1e8,
        'amount': [1e10 + i * 5e8 for i in range(n)],  # 100亿 → 240亿 线性递增
    })
    monkeypatch.setattr(f, 'get_index_data', lambda code, days=None: fake_index.copy())

    result = f.get_total_amount_ma20_series(display_days=5)
    assert result is not None
    assert len(result) == 5
    # 每个元素结构
    first = result[0]
    assert set(first.keys()) >= {'date', 'total_amount_yi', 'ma20_yi', 'source'}
    assert first['source'] == 'akshare_index_proxy'
    # MA20 应在合理范围内（amount 单位/1e8 = 亿元，应在 100-250 亿之间）
    assert 100 <= first['ma20_yi'] <= 300
    # 序列按日期升序，MA20 应递增（因 amount 线性递增）
    assert result[-1]['ma20_yi'] > result[0]['ma20_yi']


def test_get_total_amount_ma20_via_index_insufficient_data(monkeypatch):
    """sh000001 数据不足 20 日时返回 None。"""
    f = MarketDataFetcher()
    short_df = pd.DataFrame({
        'date': pd.date_range(end=datetime.now().date(), periods=15, freq='B'),
        'open': 3000.0, 'high': 3020.0, 'low': 2990.0, 'close': 3010.0,
        'volume': 1e8, 'amount': 1e10,
    })
    monkeypatch.setattr(f, 'get_index_data', lambda code, days=None: short_df.copy())
    assert f.get_total_amount_ma20_series(display_days=5) is None


def test_get_total_amount_ma20_via_index_no_amount_column(monkeypatch):
    """sh000001 没有 amount 列时返回 None（兼容旧数据源）。"""
    f = MarketDataFetcher()
    no_amount_df = pd.DataFrame({
        'date': pd.date_range(end=datetime.now().date(), periods=30, freq='B'),
        'open': 3000.0, 'high': 3020.0, 'low': 2990.0, 'close': 3010.0,
        'volume': 1e8,
    })
    monkeypatch.setattr(f, 'get_index_data', lambda code, days=None: no_amount_df.copy())
    assert f.get_total_amount_ma20_series(display_days=5) is None


def test_get_total_amount_ma20_gm_path_priority(monkeypatch):
    """gm 可用时优先使用，且不调用 akshare 备源。"""
    _enable_gm(monkeypatch)
    f = MarketDataFetcher()

    gm_called = {'n': 0}
    akshare_called = {'n': 0}

    def fake_gm(self_inner, display_days):
        gm_called['n'] += 1
        return [{'date': '2026-07-24', 'total_amount_yi': 8000, 'ma20_yi': 7500, 'source': 'gm'}]

    def fake_via_index(self_inner, display_days):
        akshare_called['n'] += 1
        return None

    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.MarketDataFetcher._get_total_amount_ma20_gm',
        fake_gm,
    )
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.MarketDataFetcher._get_total_amount_ma20_via_index',
        fake_via_index,
    )

    result = f.get_total_amount_ma20_series(display_days=1)
    assert result is not None
    assert result[0]['source'] == 'gm'
    assert gm_called['n'] == 1
    assert akshare_called['n'] == 0


def test_get_total_amount_ma20_gm_failure_falls_back(monkeypatch):
    """gm 返回 None 时，回退 akshare 备源。"""
    _enable_gm(monkeypatch)
    f = MarketDataFetcher()

    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.MarketDataFetcher._get_total_amount_ma20_gm',
        lambda self_inner, d: None,
    )
    monkeypatch.setattr(
        'AIShareTxt.market.market_fetcher.MarketDataFetcher._get_total_amount_ma20_via_index',
        lambda self_inner, d: [{'date': '2026-07-24', 'total_amount_yi': 5000,
                                'ma20_yi': 4800, 'source': 'akshare_index_proxy'}],
    )

    result = f.get_total_amount_ma20_series(display_days=1)
    assert result is not None
    assert result[0]['source'] == 'akshare_index_proxy'


def test_get_total_amount_ma20_gm_batch_computation(monkeypatch):
    """gm 路径：批量 history 返回 → 按日汇总 → MA20。验证计算逻辑。"""
    _enable_gm(monkeypatch)
    f = MarketDataFetcher()

    # 构造 25 个交易日 × 3 只股票的 history 返回
    n_days = 25
    dates = pd.date_range(end=datetime.now().date(), periods=n_days, freq='B')
    rows = []
    for d in dates:
        for sym in ['SHSE.600000', 'SHSE.600001', 'SZSE.000001']:
            rows.append({
                'symbol': sym,
                'eob': d,
                'amount': 1e9,  # 每只 10 亿，3 只共 30 亿/日
            })
    fake_raw = pd.DataFrame(rows)

    fake_stocks = pd.DataFrame({
        'symbol': ['SHSE.600000', 'SHSE.600001', 'SZSE.000001'],
        'sec_name': ['A', 'B', 'C'],
    })

    monkeypatch.setattr('gm.api.get_symbols', lambda **kw: fake_stocks)
    monkeypatch.setattr('gm.api.history', lambda **kw: fake_raw.copy())

    result = f.get_total_amount_ma20_series(display_days=5)
    assert result is not None
    assert len(result) == 5
    assert result[0]['source'] == 'gm'
    # 3 只股票 × 10 亿 = 30 亿/日，MA20 = 30
    assert result[-1]['ma20_yi'] == 30.0
    assert result[-1]['total_amount_yi'] == 30
