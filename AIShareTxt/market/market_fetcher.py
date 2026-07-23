#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场环境数据获取层

封装 akshare 的市场级 API 调用，提供：
- 大盘指数日线
- 全市场宽度（涨跌家数、涨跌停、中位数、封板率）—— 单次 spot 调用派生
- 行业 / 概念板块涨跌
- 个股 → 行业映射

数据源策略（主源失败自动 fallback）：
- 指数日线：东财 stock_zh_index_daily_em → 新浪 stock_zh_index_daily
- 全市场快照：东财 stock_zh_a_spot_em → 新浪 stock_zh_a_spot（慢但通）
- 行业板块：东财 stock_board_industry_name_em → 新浪 stock_sector_spot
- 个股行业：东财 stock_individual_info_em → 无可靠备源（雪球失败率高）

熔断：东财连续失败触发 5 分钟熔断，期间直接走备源，避免每次都等 timeout。
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import akshare as ak
import pandas as pd
import numpy as np

from ..core.config import IndicatorConfig as Config
from ..utils.utils import LoggerManager
from ..utils.cache import get_global_cache
from ..utils.trading_calendar import is_market_open


# ==================== 字段名兼容映射 ====================
# akshare 不同版本、不同数据源字段名差异

_SPOT_FIELD_ALIASES_EASTMONEY = {
    '代码': 'code', '名称': 'name', '最新价': 'last',
    '涨跌幅': 'pct_chg', '最高': 'high', '最低': 'low',
    '今开': 'open', '昨收': 'pre_close',
    '成交额': 'amount',
}
_SPOT_FIELD_ALIASES_SINA = {
    '代码': 'code', '名称': 'name', '最新价': 'last',
    '涨跌幅': 'pct_chg', '最高': 'high', '最低': 'low',
    '今开': 'open', '昨收': 'pre_close',
    '成交额': 'amount',
}

_INDEX_FIELD_ALIASES_EASTMONEY = {
    'date': 'date', '日期': 'date',
    'open': 'open', '开盘': 'open',
    'close': 'close', '收盘': 'close',
    'high': 'high', '最高': 'high',
    'low': 'low', '最低': 'low',
    'volume': 'volume', '成交量': 'volume',
    'amount': 'amount', '成交额': 'amount',
}
_INDEX_FIELD_ALIASES_SINA = _INDEX_FIELD_ALIASES_EASTMONEY  # 新浪 stock_zh_index_daily 列名已与东财标准化后一致

_BOARD_FIELD_ALIASES_EASTMONEY = {
    '板块名称': 'name', '涨跌幅': 'pct_chg',
    '上涨家数': 'advance_count', '下跌家数': 'decline_count',
    '成交额': 'amount',
}
_BOARD_FIELD_ALIASES_SINA = {
    '板块': 'name', '涨跌幅': 'pct_chg',
    '公司家数': 'company_count',
    '总成交额': 'amount',
}


# ==================== 东财熔断 ====================
# 模块级共享：任一东财接口失败触发熔断，期间所有东财调用直接跳过

_BLACKOUT_MINUTES = 5
_eastmoney_blackout_until: datetime = datetime.min


def _eastmoney_available() -> bool:
    return datetime.now() >= _eastmoney_blackout_until


def _trigger_blackout(reason: str) -> None:
    global _eastmoney_blackout_until
    _eastmoney_blackout_until = datetime.now() + timedelta(minutes=_BLACKOUT_MINUTES)
    LoggerManager.get_logger('market.fetcher').warning(
        f"东财接口失败（{reason}），触发 {_BLACKOUT_MINUTES} 分钟熔断，期间走备源"
    )


def _map_columns(df: pd.DataFrame, aliases: Dict[str, str]) -> pd.DataFrame:
    """按 alias 表把 df 的中文列名映射到英文，未命中的列保留原名。"""
    if df is None or df.empty:
        return df
    rename = {col: aliases[col] for col in df.columns if col in aliases}
    return df.rename(columns=rename)


def _normalize_index_df(raw: pd.DataFrame, aliases: Dict[str, str], days: int) -> Optional[pd.DataFrame]:
    """统一清洗指数日线：列名映射 → 数值化 → 按日期升序 → 截取最近 N 天。

    amount 列为可选（部分数据源/指数不返回），存在时保留并转 numeric。
    """
    df = _map_columns(raw.copy(), aliases)
    required = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None

    cols = required + (['amount'] if 'amount' in df.columns else [])
    df = df[cols].copy()
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'amount' in df.columns:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.sort_values('date').reset_index(drop=True)
    return df.tail(days).reset_index(drop=True)


class MarketDataFetcher:
    """市场环境数据获取器"""

    def __init__(self):
        self.config = Config()
        self.logger = LoggerManager.get_logger('market.fetcher')
        self._market_config = self.config.MARKET_ENVIRONMENT_CONFIG
        self._cache = get_global_cache()
        self._init_gm()

    def _init_gm(self) -> None:
        """检测 GM_TOKEN 并初始化。有 token 时设为优先数据源。"""
        self._gm_token = self.config.DATA_SOURCE_CONFIG.get('gm', {}).get('token', '')
        self._gm_available = False
        if not self._gm_token:
            return
        try:
            from gm.api import set_token
            set_token(self._gm_token)
            self._gm_available = True
            self.logger.info("检测到 GM_TOKEN，市场数据优先使用掘金 SDK")
        except ImportError:
            self.logger.info("gm SDK 未安装，市场数据走 akshare")
        except Exception as e:
            self.logger.warning(f"gm 初始化失败，回退 akshare：{e}")

    # ==================== TTL 工具 ====================

    def _ttl(self, intraday_key: str, post_close_key: str) -> timedelta:
        """根据是否盘中返回对应 TTL。"""
        ttl_cfg = self._market_config['cache_ttl']
        key = intraday_key if is_market_open() else post_close_key
        return timedelta(seconds=ttl_cfg[key])

    def _cache_get(self, key: str, ttl: timedelta):
        return self._cache.get(key, ttl)

    def _cache_set(self, key: str, value) -> None:
        self._cache.set(key, value)

    # ==================== 大盘指数日线 ====================

    def get_index_data(self, code: str, days: Optional[int] = None) -> Optional[pd.DataFrame]:
        """获取指数日线 OHLCV。优先级：gm → 东财 → 新浪。

        Args:
            code: akshare 格式指数代码，如 'sh000001'
            days: 回溯天数，None 取 config 默认

        Returns:
            DataFrame[date, open, high, low, close, volume] 按日期升序；失败返回 None
        """
        if days is None:
            days = self._market_config['index_history_days']

        cache_key = f'index_daily:{code}'
        ttl = self._ttl('index_daily_intraday', 'index_daily_post_close')
        cached = self._cache_get(cache_key, ttl)
        if cached is not None:
            self.logger.debug(f"指数 {code} 命中缓存")
            return cached

        # 优先源：gm
        if self._gm_available:
            df = self._get_index_data_gm(code, days)
            if df is not None and not df.empty:
                self._cache_set(cache_key, df)
                self.logger.info(f"指数 {code} 获取成功（gm），共 {len(df)} 条")
                return df

        # 兜底 1：东财
        if _eastmoney_available():
            try:
                self.logger.info(f"获取指数日线（东财）: {code}")
                raw = ak.stock_zh_index_daily_em(symbol=code)
                df = _normalize_index_df(raw, _INDEX_FIELD_ALIASES_EASTMONEY, days)
                if df is not None and not df.empty:
                    self._cache_set(cache_key, df)
                    self.logger.info(f"指数 {code} 获取成功（东财），共 {len(df)} 条")
                    return df
                self.logger.warning(f"指数 {code} 东财返回空数据或列不匹配")
            except Exception as e:
                self.logger.warning(f"指数 {code} 东财获取失败：{e}")
                _trigger_blackout(f"index_daily {code}: {e}")

        # 备源：新浪
        try:
            self.logger.info(f"获取指数日线（新浪）: {code}")
            raw = ak.stock_zh_index_daily(symbol=code)
            df = _normalize_index_df(raw, _INDEX_FIELD_ALIASES_SINA, days)
            if df is not None and not df.empty:
                self._cache_set(cache_key, df)
                self.logger.info(f"指数 {code} 获取成功（新浪），共 {len(df)} 条")
                return df
            self.logger.warning(f"指数 {code} 新浪返回空数据或列不匹配")
        except Exception as e:
            self.logger.warning(f"指数 {code} 新浪获取失败：{e}")

        return None

    def _get_index_data_gm(self, code: str, days: int) -> Optional[pd.DataFrame]:
        """gm 实现：history() 直接拉指数日线。"""
        try:
            from gm.api import history
        except ImportError:
            return None

        gm_symbol = self._to_gm_index_symbol(code)
        if gm_symbol is None:
            self.logger.warning(f"指数 {code} 无法转换为 gm symbol")
            return None

        try:
            end = datetime.now()
            # 多拉 buffer：节假日 + 周末，实际交易日约为一半
            start = end - timedelta(days=days * 2 + 10)
            self.logger.info(f"获取指数日线（gm）: {gm_symbol}")
            raw = history(
                symbol=gm_symbol,
                start_time=f'{start.strftime("%Y-%m-%d")} 09:30:00',
                end_time=f'{end.strftime("%Y-%m-%d")} 23:59:59',
                frequency='1d',
                fields='eob,open,high,low,close,volume,amount',
                df=True,
            )
        except Exception as e:
            self.logger.warning(f"指数 {code} gm 获取失败：{e}")
            return None

        if raw is None or raw.empty:
            return None

        df = raw.rename(columns={'eob': 'date'})
        for c in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        df = df.sort_values('date').reset_index(drop=True)
        df = df.tail(days).reset_index(drop=True)

        # 盘后交易日场景下，gm 数据应包含当日；若仍是 T-1（清算未完成），回退 akshare
        if not self._is_gm_data_fresh(df):
            self.logger.info(
                f"gm 指数 {code} 最新数据早于今日（清算未完成），回退 akshare"
            )
            return None
        return df

    def _is_gm_data_fresh(self, df: Optional[pd.DataFrame], now: Optional[datetime] = None) -> bool:
        """15:30 后的交易日场景下检查 gm 数据是否包含当日。

        gm history 盘后清算完成后才更新当日数据（实测约 16:00-17:00），
        在此之前 gm 拿到 T-1 数据，应让上层 fallback 到 akshare。

        - 盘前/盘中（< 15:30）：不检查（gm 本就拿不到当日，akshare 也未必有）
        - 非交易日：不检查
        - 盘后交易日（>= 15:30）：期望 gm 含当日数据

        Args:
            now: 测试注入用，默认 datetime.now()
        """
        from datetime import time as dt_time
        now = now or datetime.now()
        if now.time() < dt_time(15, 30):
            return True
        from ..utils.trading_calendar import is_trading_day
        today = now.date()
        if not is_trading_day(today):
            return True
        if df is None or df.empty or 'date' not in df.columns:
            return False
        latest = pd.to_datetime(df['date']).max().date()
        return latest >= today

    @staticmethod
    def _to_gm_index_symbol(code: str) -> Optional[str]:
        """'sh000001' → 'SHSE.000001', 'sz399001' → 'SZSE.399001'"""
        if not code or len(code) < 3:
            return None
        prefix = code[:2].upper()
        num = code[2:]
        if prefix == 'SH':
            return f'SHSE.{num}'
        if prefix == 'SZ':
            return f'SZSE.{num}'
        return None

    # ==================== 全市场快照（派生市场宽度） ====================

    def get_market_snapshot(self) -> Optional[Dict[str, Any]]:
        """全市场快照。优先级：gm（仅盘后） → 东财 → 新浪。

        Returns:
            {
                'as_of': datetime,
                'advance_count': int,         # 上涨家数
                'decline_count': int,         # 下跌家数
                'unchanged_count': int,       # 平盘家数
                'limit_up_count': int,        # 涨停家数（非 ST，封板中）
                'limit_down_count': int,      # 跌停家数（非 ST）
                'touched_limit_up': int,      # 曾触及涨停（含已开板）
                'touched_limit_down': int,    # 曾触及跌停
                'seal_ratio': float,          # 封板率 = limit_up / touched_limit_up
                'median_pct': float,          # 涨跌幅中位数（%）
                'total_amount_yi': float,     # 全市场成交额（亿元）
                'spot_df': pd.DataFrame,      # 原始快照（供下游板块归属分析复用）
            }
            失败返回 None
        """
        cache_key = 'market_snapshot'
        ttl = self._ttl('market_snapshot_intraday', 'market_snapshot_post_close')
        cached = self._cache_get(cache_key, ttl)
        if cached is not None:
            self.logger.debug("市场快照命中缓存")
            return cached

        # 优先源：gm（仅盘后，盘中数据未清算 history 拿不到当日）
        if self._gm_available:
            snapshot = self._get_market_snapshot_gm()
            if snapshot is not None:
                self._cache_set(cache_key, snapshot)
                self.logger.info(
                    f"市场快照构建成功（gm）: 上涨{snapshot['advance_count']}/"
                    f"下跌{snapshot['decline_count']}/"
                    f"涨停{snapshot['limit_up_count']}/"
                    f"跌停{snapshot['limit_down_count']}/"
                    f"中位数{snapshot['median_pct']:.2f}%"
                )
                return snapshot

        # 兜底：东财
        spot = None
        if _eastmoney_available():
            try:
                self.logger.info("拉取全市场快照 stock_zh_a_spot_em ...")
                raw = ak.stock_zh_a_spot_em()
                spot = self._normalize_spot(raw, _SPOT_FIELD_ALIASES_EASTMONEY)
            except Exception as e:
                self.logger.warning(f"东财市场快照失败：{e}")
                _trigger_blackout(f"spot_em: {e}")

        # 备源：新浪
        if spot is None:
            try:
                self.logger.info("拉取全市场快照 stock_zh_a_spot（新浪）...")
                raw = ak.stock_zh_a_spot()
                spot = self._normalize_spot(raw, _SPOT_FIELD_ALIASES_SINA)
            except Exception as e:
                self.logger.warning(f"新浪市场快照失败：{e}")

        if spot is None:
            return None

        snapshot = self._derive_breadth(spot)
        if snapshot is None:
            return None

        self._cache_set(cache_key, snapshot)
        self.logger.info(
            f"市场快照构建成功: 上涨{snapshot['advance_count']}/下跌{snapshot['decline_count']}/"
            f"涨停{snapshot['limit_up_count']}/跌停{snapshot['limit_down_count']}/"
            f"中位数{snapshot['median_pct']:.2f}%"
        )
        return snapshot

    def _get_market_snapshot_gm(self) -> Optional[Dict[str, Any]]:
        """gm 实现：盘后用 history 批量拉所有 A 股 OHLC + pre_close，派生宽度。

        盘中（is_market_open 为 True）返回 None，让上层走 akshare。
        gm history 盘中拿不到当日数据（清算未完成），实测在 15:30 后才稳定。
        """
        if is_market_open():
            return None

        try:
            from gm.api import get_symbols, history
        except ImportError:
            return None

        # 1. 全市场 A 股列表 + 名称映射
        today_str = datetime.now().strftime('%Y-%m-%d')
        try:
            self.logger.info("拉取 A 股列表（gm get_symbols）...")
            stocks = get_symbols(
                trade_date=today_str, sec_type1=1010, sec_type2=101001, df=True
            )
        except Exception as e:
            self.logger.warning(f"gm get_symbols 失败：{e}")
            return None

        if stocks is None or len(stocks) == 0:
            return None

        symbol_list = stocks['symbol'].tolist()
        name_map = dict(zip(stocks['symbol'], stocks.get('sec_name', pd.Series(dtype=str))))

        # 2. history 批量拉最近 1 个交易日的 OHLC + pre_close + amount
        try:
            end = datetime.now()
            start = end - timedelta(days=10)
            self.logger.info(
                f"拉取全市场快照（gm history，{len(symbol_list)} 只）..."
            )
            raw = history(
                symbol=','.join(symbol_list),
                start_time=f'{start.strftime("%Y-%m-%d")} 09:30:00',
                end_time=f'{end.strftime("%Y-%m-%d")} 23:59:59',
                frequency='1d',
                fields='symbol,eob,close,high,low,pre_close,amount',
                df=True,
            )
        except Exception as e:
            self.logger.warning(f"gm history 批量失败：{e}")
            return None

        if raw is None or raw.empty:
            return None

        # 3. 每个 symbol 取最新一行
        raw['eob'] = pd.to_datetime(raw['eob'])
        latest = (
            raw.sort_values('eob')
            .groupby('symbol', as_index=False)
            .tail(1)
            .copy()
        )

        # 盘后交易日场景：检查 gm 数据是否含当日，否则回退 akshare
        freshness_df = latest[['eob']].rename(columns={'eob': 'date'})
        if not self._is_gm_data_fresh(freshness_df):
            self.logger.info("gm 快照最新数据早于今日（清算未完成），回退 akshare")
            return None

        # 4. 派生 pct_chg（gm history 不返回涨跌幅字段，自己算）
        latest = latest.dropna(subset=['close', 'pre_close'])
        latest = latest[latest['pre_close'] > 0]
        if latest.empty:
            return None
        latest['pct_chg'] = (
            (latest['close'] - latest['pre_close']) / latest['pre_close'] * 100
        )

        # 5. 列名标准化，对齐 _derive_breadth 期望（code 为 6 位裸码，便于板块判定）
        latest = latest.rename(columns={'symbol': '_gm_symbol'})
        latest['code'] = latest['_gm_symbol'].str.split('.').str[-1]
        latest['name'] = latest['_gm_symbol'].map(name_map).fillna('')
        latest = latest.drop(columns=['_gm_symbol'])

        # 6. 派生宽度（复用 _derive_breadth）
        return self._derive_breadth(latest)

    def _normalize_spot(self, raw: pd.DataFrame, aliases: Dict[str, str]) -> Optional[pd.DataFrame]:
        """把 akshare spot 返回的 DataFrame 列名标准化、数值列转 numeric、过滤北交所。"""
        if raw is None or len(raw) == 0:
            return None
        spot = _map_columns(raw.copy(), aliases)
        for c in ['pct_chg', 'high', 'low', 'pre_close']:
            if c in spot.columns:
                spot[c] = pd.to_numeric(spot[c], errors='coerce')
        spot = self._filter_bse(spot)
        valid = spot.dropna(subset=['pct_chg'])
        if len(valid) == 0:
            self.logger.warning("市场快照所有 pct_chg 为空")
            return None
        return spot

    @staticmethod
    def _filter_bse(spot: pd.DataFrame) -> pd.DataFrame:
        """过滤北交所标的（gm SDK 不支持北交所，akshare 需对齐口径）。

        北交所代码段：
        - 新浪格式：'bj' 前缀（如 'bj920000'）
        - 东财格式：6 位裸码，以 43/83/87/92 开头（如 '920000'、'830000'）
        """
        if spot is None or spot.empty or 'code' not in spot.columns:
            return spot
        code = spot['code'].astype(str).str.strip()
        naked = code.str.replace(r'^(bj|sh|sz|BJ|SH|SZ)', '', regex=True)
        bse_mask = naked.str.match(r'^(43|83|87|92)')
        return spot[~bse_mask].reset_index(drop=True)

    def _derive_breadth(self, spot: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """从标准化后的全市场 spot DataFrame 派生宽度指标。"""
        valid = spot.dropna(subset=['pct_chg']).copy()
        if len(valid) == 0:
            return None

        advance = int((valid['pct_chg'] > 0).sum())
        decline = int((valid['pct_chg'] < 0).sum())
        unchanged = int((valid['pct_chg'] == 0).sum())

        thresholds = self._market_config['limit_thresholds']
        limit_up_count, limit_down_count = 0, 0
        touched_up, touched_down = 0, 0

        for _, row in valid.iterrows():
            code = str(row.get('code', ''))
            name = str(row.get('name', ''))
            pct = float(row.get('pct_chg', 0))
            high = row.get('high')
            low = row.get('low')
            pre_close = row.get('pre_close')

            # 涨跌停阈值：创业板/科创板 20%，ST 5%，主板 10%
            if 'ST' in name or '*ST' in name:
                up_th, down_th = thresholds['st_up'], thresholds['st_down']
            elif code.startswith(('300', '688')):
                up_th, down_th = thresholds['star_gem_up'], thresholds['star_gem_down']
            else:
                up_th, down_th = thresholds['main_board_up'], thresholds['main_board_down']

            if pct >= up_th:
                limit_up_count += 1
            if pct <= down_th:
                limit_down_count += 1

            touched_up_pct = self._high_pct(high, pre_close)
            touched_down_pct = self._low_pct(low, pre_close)
            if touched_up_pct is not None and touched_up_pct >= up_th:
                touched_up += 1
            if touched_down_pct is not None and touched_down_pct <= down_th:
                touched_down += 1

        seal_ratio = (limit_up_count / touched_up) if touched_up > 0 else 0.0
        median_pct = float(valid['pct_chg'].median())

        # 全市场成交额（akshare 返回成交额单位：元）；'成交额' 列在 alias 映射后已改名 amount
        total_amount_yi = 0.0
        amt_col = 'amount' if 'amount' in spot.columns else '成交额'
        if amt_col in spot.columns:
            amt = pd.to_numeric(spot[amt_col], errors='coerce').sum()
            total_amount_yi = float(amt / 1e8)

        return {
            'as_of': datetime.now(),
            'advance_count': advance,
            'decline_count': decline,
            'unchanged_count': unchanged,
            'limit_up_count': limit_up_count,
            'limit_down_count': limit_down_count,
            'touched_limit_up': touched_up,
            'touched_limit_down': touched_down,
            'seal_ratio': seal_ratio,
            'median_pct': median_pct,
            'total_amount_yi': total_amount_yi,
            'spot_df': spot,
        }

    @staticmethod
    def _high_pct(high, pre_close) -> Optional[float]:
        """从最高价和昨收推算当日最高涨幅（%）。"""
        try:
            if pd.isna(high) or pd.isna(pre_close) or pre_close <= 0:
                return None
            return float((high - pre_close) / pre_close * 100)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _low_pct(low, pre_close) -> Optional[float]:
        try:
            if pd.isna(low) or pd.isna(pre_close) or pre_close <= 0:
                return None
            return float((low - pre_close) / pre_close * 100)
        except (TypeError, ValueError):
            return None

    # ==================== 全市场成交额中枢 MA20 ====================

    def get_total_amount_ma20_series(self, display_days: int = 5) -> Optional[List[Dict[str, Any]]]:
        """A 股全市场成交额中枢（MA20）近 N 日序列。

        - gm 路径：批量拉所有 A 股近 30 日 amount，按交易日汇总，算 MA20（口径与
          breadth.total_amount_yi 一致，覆盖 SHSE/SZSE 全部 A 股，不含北交所）
        - akshare 备源：用 sh000001 日线 amount 作为代理（仅覆盖上交所，标注 source）

        Args:
            display_days: 返回最近几日的 MA20 值，默认 5

        Returns:
            [{'date': 'YYYY-MM-DD', 'total_amount_yi': float, 'ma20_yi': float,
              'source': 'gm'|'akshare_index_proxy'}, ...]；按日期升序；失败返回 None
        """
        cache_key = f'total_amount_ma20:{display_days}'
        ttl = self._ttl('market_snapshot_intraday', 'market_snapshot_post_close')
        cached = self._cache_get(cache_key, ttl)
        if cached is not None:
            self.logger.debug("全市场成交额 MA20 序列命中缓存")
            return cached

        result: Optional[List[Dict[str, Any]]] = None
        if self._gm_available:
            result = self._get_total_amount_ma20_gm(display_days)

        if result is None:
            result = self._get_total_amount_ma20_via_index(display_days)

        if result is not None:
            self._cache_set(cache_key, result)
        return result

    def _get_total_amount_ma20_gm(self, display_days: int) -> Optional[List[Dict[str, Any]]]:
        """gm 实现：批量 history 拉所有 A 股 amount，按日汇总算 MA20。"""
        try:
            from gm.api import get_symbols, history
        except ImportError:
            return None

        today_str = datetime.now().strftime('%Y-%m-%d')
        try:
            self.logger.info("拉取 A 股列表（gm get_symbols）...")
            stocks = get_symbols(
                trade_date=today_str, sec_type1=1010, sec_type2=101001, df=True
            )
        except Exception as e:
            self.logger.warning(f"gm get_symbols 失败：{e}")
            return None

        if stocks is None or len(stocks) == 0:
            return None

        symbol_list = stocks['symbol'].tolist()

        end = datetime.now()
        # 45 自然日 ≈ 30 交易日，足够 MA20 + display_days
        start = end - timedelta(days=45)
        try:
            self.logger.info(
                f"拉取全市场 amount 序列（gm history，{len(symbol_list)} 只 ×30 日）..."
            )
            raw = history(
                symbol=','.join(symbol_list),
                start_time=f'{start.strftime("%Y-%m-%d")} 09:30:00',
                end_time=f'{end.strftime("%Y-%m-%d")} 23:59:59',
                frequency='1d',
                fields='symbol,eob,amount',
                df=True,
            )
        except Exception as e:
            self.logger.warning(f"gm history amount 批量失败：{e}")
            return None

        if raw is None or raw.empty:
            return None

        raw['amount'] = pd.to_numeric(raw['amount'], errors='coerce')
        raw = raw.dropna(subset=['amount'])
        raw = raw[raw['amount'] > 0]
        if raw.empty:
            return None

        raw['date'] = pd.to_datetime(raw['eob']).dt.normalize()
        daily = raw.groupby('date', as_index=False)['amount'].sum()
        daily['amount_yi'] = daily['amount'] / 1e8
        daily = daily.sort_values('date').reset_index(drop=True)

        if len(daily) < 20:
            self.logger.warning(f"gm amount 序列不足 20 日（{len(daily)} 日），无法算 MA20")
            return None

        daily['ma20_yi'] = daily['amount_yi'].rolling(window=20).mean()
        daily = daily.dropna(subset=['ma20_yi'])
        if daily.empty:
            return None

        recent = daily.tail(display_days)
        self.logger.info(
            f"全市场成交额 MA20 序列构建成功（gm），最新 {recent.iloc[-1]['date'].date()} "
            f"MA20={recent.iloc[-1]['ma20_yi']:.0f} 亿"
        )
        return [
            {
                'date': row['date'].strftime('%Y-%m-%d'),
                'total_amount_yi': round(float(row['amount_yi']), 0),
                'ma20_yi': round(float(row['ma20_yi']), 0),
                'source': 'gm',
            }
            for _, row in recent.iterrows()
        ]

    def _get_total_amount_ma20_via_index(self, display_days: int) -> Optional[List[Dict[str, Any]]]:
        """akshare 备源：用 sh000001 日线 amount 作为全市场代理。

        上证综指 amount = 上交所全部 A 股成交额，约占全市场 60-70%，
        作为趋势/中枢参考可接受，但绝对值偏低，source 字段标注来源。
        """
        days_need = max(30, 20 + display_days + 5)
        df = self.get_index_data('sh000001', days=days_need)
        if df is None or len(df) < 20 or 'amount' not in df.columns:
            self.logger.warning("sh000001 amount 序列获取失败，无法算 MA20 代理")
            return None

        df = df.copy()
        df['amount_yi'] = pd.to_numeric(df['amount'], errors='coerce') / 1e8
        df = df.dropna(subset=['amount_yi'])
        if len(df) < 20:
            return None

        df['ma20_yi'] = df['amount_yi'].rolling(window=20).mean()
        df = df.dropna(subset=['ma20_yi'])
        if df.empty:
            return None

        recent = df.tail(display_days)
        self.logger.info(
            f"全市场成交额 MA20 序列构建成功（akshare 上证代理），"
            f"最新 {recent.iloc[-1]['date'].date()} MA20={recent.iloc[-1]['ma20_yi']:.0f} 亿"
        )
        return [
            {
                'date': row['date'].strftime('%Y-%m-%d'),
                'total_amount_yi': round(float(row['amount_yi']), 0),
                'ma20_yi': round(float(row['ma20_yi']), 0),
                'source': 'akshare_index_proxy',
            }
            for _, row in recent.iterrows()
        ]

    # ==================== 行业 / 概念板块 ====================

    def get_industry_boards(self) -> Optional[pd.DataFrame]:
        """行业板块列表 + 涨跌幅。主源东财失败走新浪。

        Returns DataFrame[name, pct_chg, advance_count, decline_count, amount] or None.
        新浪源不含 advance/decline 计数，对应列为空。
        """
        cache_key = 'industry_boards'
        ttl = self._ttl('industry_boards_intraday', 'industry_boards_post_close')
        cached = self._cache_get(cache_key, ttl)
        if cached is not None:
            return cached

        # 主源：东财
        if _eastmoney_available():
            try:
                self.logger.info("拉取行业板块列表 stock_board_industry_name_em ...")
                raw = ak.stock_board_industry_name_em()
                df = self._normalize_board_df(raw, _BOARD_FIELD_ALIASES_EASTMONEY)
                if df is not None and not df.empty:
                    self._cache_set(cache_key, df)
                    self.logger.info(f"行业板块获取成功（东财），共 {len(df)} 个板块")
                    return df
            except Exception as e:
                self.logger.warning(f"东财行业板块失败：{e}")
                _trigger_blackout(f"industry_board: {e}")

        # 备源：新浪
        try:
            self.logger.info("拉取行业板块列表 stock_sector_spot（新浪）...")
            raw = ak.stock_sector_spot(indicator='新浪行业')
            df = self._normalize_board_df(raw, _BOARD_FIELD_ALIASES_SINA)
            if df is not None and not df.empty:
                self._cache_set(cache_key, df)
                self.logger.info(f"行业板块获取成功（新浪），共 {len(df)} 个板块")
                return df
        except Exception as e:
            self.logger.warning(f"新浪行业板块失败：{e}")

        return None

    @staticmethod
    def _normalize_board_df(raw: pd.DataFrame, aliases: Dict[str, str]) -> Optional[pd.DataFrame]:
        """统一板块 DataFrame：列名映射 + 保留可用列 + pct_chg 转数值。"""
        if raw is None or len(raw) == 0:
            return None
        df = _map_columns(raw.copy(), aliases)
        keep = [c for c in ['name', 'pct_chg', 'advance_count', 'decline_count', 'amount'] if c in df.columns]
        df = df[keep].copy()
        if 'pct_chg' in df.columns:
            df['pct_chg'] = pd.to_numeric(df['pct_chg'], errors='coerce')
        return df

    def get_concept_boards(self) -> Optional[pd.DataFrame]:
        """概念板块列表 + 涨跌幅。Columns 同 industry。无可靠备源，东财失败/熔断期间直接返回 None。"""
        cache_key = 'concept_boards'
        ttl = self._ttl('concept_boards_intraday', 'concept_boards_post_close')
        cached = self._cache_get(cache_key, ttl)
        if cached is not None:
            return cached

        if not _eastmoney_available():
            self.logger.info("东财熔断中，跳过概念板块拉取")
            return None

        try:
            self.logger.info("拉取概念板块列表 stock_board_concept_name_em ...")
            raw = ak.stock_board_concept_name_em()
            df = self._normalize_board_df(raw, _BOARD_FIELD_ALIASES_EASTMONEY)
            if df is None or df.empty:
                self.logger.warning("概念板块返回空")
                return None

            self._cache_set(cache_key, df)
            self.logger.info(f"概念板块获取成功，共 {len(df)} 个板块")
            return df
        except Exception as e:
            self.logger.warning(f"获取概念板块失败：{e}")
            _trigger_blackout(f"concept_board: {e}")
            return None

    # ==================== 个股 → 行业映射 ====================

    def get_stock_industry(self, stock_code: str) -> Optional[str]:
        """查个股所属行业。优先级：gm → 东财（无可靠新浪备源）。"""
        cache_key = f'stock_industry:{stock_code}'
        ttl = timedelta(seconds=self._market_config['cache_ttl']['stock_industry_mapping'])
        cached = self._cache_get(cache_key, ttl)
        if cached is not None:
            return cached if cached != '__NONE__' else None

        # 优先源：gm
        if self._gm_available:
            industry = self._get_stock_industry_gm(stock_code)
            if industry:
                self._cache_set(cache_key, industry)
                return industry
            # gm 没拿到不直接 return，继续试东财

        if not _eastmoney_available():
            self.logger.info(f"东财熔断中，跳过个股 {stock_code} 行业查询")
            return None

        try:
            self.logger.info(f"查询个股 {stock_code} 所属行业（东财）")
            info = ak.stock_individual_info_em(symbol=stock_code)

            industry = self._extract_industry_from_info(info)
            # 缓存时区分 "未获取到" 和 "未缓存"
            self._cache_set(cache_key, industry if industry else '__NONE__')
            return industry
        except Exception as e:
            self.logger.warning(f"获取个股 {stock_code} 行业失败：{e}")
            _trigger_blackout(f"stock_industry {stock_code}: {e}")
            return None

    def _get_stock_industry_gm(self, stock_code: str) -> Optional[str]:
        """gm 实现：stk_get_symbol_industry 拿申万一级行业名。"""
        try:
            from gm.api import stk_get_symbol_industry
        except ImportError:
            return None

        gm_symbol = self._to_gm_stock_symbol(stock_code)
        if gm_symbol is None:
            return None

        try:
            self.logger.info(f"查询个股 {stock_code} 所属行业（gm）")
            df = stk_get_symbol_industry(
                symbols=gm_symbol, source='sw2021', level=1
            )
        except Exception as e:
            self.logger.warning(f"个股 {stock_code} gm 行业查询失败：{e}")
            return None

        if df is None or len(df) == 0:
            return None
        row = df.iloc[0]
        name = row.get('industry_name')
        return str(name) if pd.notna(name) and str(name).strip() else None

    @staticmethod
    def _to_gm_stock_symbol(stock_code: str) -> Optional[str]:
        """'000001' → 'SZSE.000001', '600000' → 'SHSE.600000'"""
        if not stock_code:
            return None
        code = str(stock_code).strip().zfill(6)
        if len(code) != 6 or not code.isdigit():
            return None
        if code[0] in ('6', '9'):
            return f'SHSE.{code}'
        return f'SZSE.{code}'

    @staticmethod
    def _extract_industry_from_info(info) -> Optional[str]:
        """stock_individual_info_em 在不同 akshare 版本可能返回 DataFrame 或 dict。"""
        if info is None:
            return None
        # DataFrame 形式：两列 item / value
        if isinstance(info, pd.DataFrame):
            if 'item' in info.columns and 'value' in info.columns:
                row = info[info['item'] == '行业']
                if not row.empty:
                    return str(row['value'].iloc[0])
            # 中文列名兼容
            if '行业' in info.columns and len(info) > 0:
                return str(info['行业'].iloc[0])
        # dict 形式
        if isinstance(info, dict):
            return str(info.get('行业')) if info.get('行业') else None
        return None
