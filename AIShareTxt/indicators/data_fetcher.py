#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票数据获取调度模块
根据配置选择数据源，统一数据清洗流程
"""

import pandas as pd
import pandas_market_calendars as mcal
from typing import Optional, cast
from datetime import datetime, timedelta, time, date
from ..core.config import IndicatorConfig as Config
from ..utils.utils import LoggerManager
from .data_sources import AkshareDataSource, create_gm_source
import warnings
warnings.filterwarnings('ignore')


class StockDataFetcher:
    """股票数据获取器（调度器）"""

    def __init__(self):
        self.config = Config()
        self.logger = LoggerManager.get_logger('data_fetcher')

        # 初始化多市场日历
        self.calendars = {
            'SSE': mcal.get_calendar('SSE'),
            'HKEX': mcal.get_calendar('HKEX'),
        }
        self.sse_calendar = self.calendars['SSE']

        # 初始化数据源
        self._akshare = AkshareDataSource()
        self._primary = self._akshare
        self._used_source = 'akshare'

        source_name = self.config.DATA_SOURCE_CONFIG.get('default_source', 'akshare')
        if source_name == 'gm':
            gm_token = self.config.DATA_SOURCE_CONFIG.get('gm', {}).get('token', '')
            if gm_token:
                try:
                    self._primary = create_gm_source(self.config.DATA_SOURCE_CONFIG)
                    self.logger.info("数据源已切换为 gm（掘金量化）")
                except Exception as e:
                    self.logger.warning(f"gm 数据源初始化失败，回退到 akshare：{str(e)}")
                    self._primary = self._akshare
            else:
                self.logger.warning("gm 数据源未配置 GM_TOKEN，使用 akshare")

    # ==================== 公开接口 ====================

    def fetch_stock_data(self, stock_code: str, period: Optional[str] = None,
                         adjust: Optional[str] = None, start_date: Optional[str] = None,
                         market: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        获取股票历史价格数据（支持多市场）

        根据配置选择主数据源，失败后 fallback 到 akshare
        """
        if market is None:
            market = self.config.identify_market(stock_code)

        if period is None:
            period = self.config.DATA_CONFIG['default_period']
        if adjust is None:
            adjust = self.config.DATA_CONFIG['default_adjust']
        if start_date is None:
            months_back = self.config.DATA_CONFIG.get('default_months_back', 4)
            default_start = datetime.now() - timedelta(days=months_back * 30)
            start_date = default_start.strftime('%Y%m%d')

        if market == 'HK':
            raw_data = self._akshare.fetch_hk_stock_data(stock_code, start_date, cast(str, adjust))
        elif market == 'CN':
            raw_data = self._primary.fetch_stock_data(stock_code, period, start_date, adjust)
            self._used_source = self._primary.name
            if (raw_data is None or (isinstance(raw_data, pd.DataFrame) and len(raw_data) == 0)):
                if self._primary is not self._akshare:
                    self.logger.info("主数据源失败，fallback 到 akshare")
                    raw_data = self._akshare.fetch_stock_data(stock_code, period, start_date, adjust)
                    self._used_source = 'akshare'
        else:
            self.logger.error(f"不支持的市场类型: {market}")
            return None

        if raw_data is None or (isinstance(raw_data, pd.DataFrame) and len(raw_data) == 0):
            self.logger.error(f"所有数据源均获取失败：{stock_code}")
            return None

        return self._process_stock_data(raw_data, stock_code)

    def get_fund_flow_data(self, stock_code, target_date=None, market: Optional[str] = None):
        """获取主力资金流数据，含日期一致性校验"""
        if market is None:
            market = self.config.identify_market(stock_code)
        if market == 'HK':
            self.logger.info("港股不支持主力资金流数据")
            return {}

        data = self._primary.get_fund_flow_data(stock_code, target_date)

        # 校验返回数据的日期是否与请求日期一致
        if data and target_date:
            data_date = data.get('日期', '')
            expected_date = target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date)
            if data_date != expected_date:
                self.logger.warning(
                    f"资金流数据日期不一致：请求 {expected_date}，返回 {data_date}，降级使用 akshare"
                )
                data = None

        if not data and self._primary is not self._akshare:
            self.logger.info("主数据源资金流失败，fallback 到 akshare")
            data = self._akshare.get_fund_flow_data(stock_code, target_date)

        return data

    def get_stock_basic_info(self, stock_code, market: Optional[str] = None):
        """获取股票基本信息"""
        if market is None:
            market = self.config.identify_market(stock_code)
        if market == 'HK':
            return self._akshare.get_hk_stock_basic_info(stock_code)

        info = self._primary.get_stock_basic_info(stock_code)

        if (not info or info.get('股票简称') == '未知') and self._primary is not self._akshare:
            self.logger.info("主数据源基本信息失败，fallback 到 akshare")
            info = self._akshare.get_stock_basic_info(stock_code)

        return info

    # ==================== 共享数据清洗 ====================

    def _process_stock_data(self, data: pd.DataFrame, stock_code: str) -> Optional[pd.DataFrame]:
        """处理和清洗股票数据（所有数据源共享）"""
        try:
            if data is None or len(data) == 0:
                self.logger.error(f"{self.config.ERROR_MESSAGES['no_data']}: {stock_code}")
                return None

            self.logger.debug(f"获取到数据，形状: {data.shape}")

            # 标准化列名（如果数据源未做映射）
            data = data.rename(columns=self.config.COLUMN_MAPPING)

            # 检查必要的列是否存在
            required_columns = self.config.DATA_CONFIG['required_columns']
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                self.logger.error(f"数据缺少必要列 {missing_columns}")
                return None

            # 确保数据类型正确
            for col in ['open', 'close', 'high', 'low', 'volume']:
                data[col] = pd.to_numeric(data[col], errors='coerce')

            # 删除包含NaN的行
            data = data.dropna(subset=['open', 'close', 'high', 'low', 'volume'])

            # 数据清洗：处理异常价格数据
            original_len = len(data)

            price_cols = ['open', 'close', 'high', 'low']
            for col in price_cols:
                mask = data[col] >= 0
                data = cast(pd.DataFrame, data[mask].copy())

            volume_mask = data['volume'] >= 0
            data = cast(pd.DataFrame, data[volume_mask].copy())

            high_low_mask = data['high'] >= data['low']
            data = cast(pd.DataFrame, data[high_low_mask].copy())

            close_range_mask = (data['close'] >= data['low']) & (data['close'] <= data['high'])
            data = cast(pd.DataFrame, data[close_range_mask].copy())

            open_range_mask = (data['open'] >= data['low']) & (data['open'] <= data['high'])
            data = cast(pd.DataFrame, data[open_range_mask].copy())

            cleaned_count = original_len - len(data)
            if cleaned_count > 0:
                self.logger.info(f"数据清洗：移除了 {cleaned_count} 条异常数据")

            if len(data) == 0:
                self.logger.error("数据清洗后为空")
                return None

            # 按日期排序
            if not isinstance(data, pd.DataFrame):
                self.logger.error("数据类型错误：期望DataFrame类型")
                return None
            data = data.sort_values('date').reset_index(drop=True)

            # 检查并处理未收盘的不完整数据
            original_length = len(data)
            data = self._remove_incomplete_trading_data(data)

            if len(data) == 0:
                self.logger.error("处理后数据为空")
                return None

            self.logger.info(f"成功处理股票 {stock_code} 的数据，共 {len(data)} 条记录")
            if original_length != len(data):
                self.logger.info(f"已移除 {original_length - len(data)} 条不完整交易数据")

            self.logger.debug(f"数据日期范围: {data['date'].iloc[0]} 到 {data['date'].iloc[-1]}")
            self.logger.info(f"最新收盘价: {data['close'].iloc[-1]:.{self.config.DISPLAY_PRECISION['price']}f}")

            return data

        except Exception as e:
            self.logger.error(f"处理股票数据时出错：{str(e)}")
            return None

    def _remove_incomplete_trading_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """如果今天是交易日且未收盘，移除最后一行当天数据"""
        if self._is_trading_day_and_not_closed():
            if len(data) > 0:
                last_date = data['date'].iloc[-1]
                if hasattr(last_date, 'date'):
                    last_date = last_date.date()

                today = datetime.now().date()

                if last_date == today:
                    self.logger.info("检测到今天的不完整数据，移除")
                    data = data.iloc[:-1].copy()
                    self.logger.info(f"已移除今天的不完整数据，剩余 {len(data)} 条记录")
                else:
                    self.logger.info(f"最新数据是 {last_date}（不是今天），不移除")
            else:
                self.logger.info("数据为空，无法移除")
        else:
            self.logger.debug("当前为非交易日或已收盘，保留所有数据")

        return data

    # ==================== 交易日判断 ====================

    def _is_trading_day_and_not_closed(self) -> bool:
        """判断今天是否是交易日且未收盘"""
        try:
            now = datetime.now()
            today = now.date()
            current_time = now.time()

            is_trading_day = self._is_trading_day(today)
            if not is_trading_day:
                self.logger.debug(f"今天 {today} 不是交易日")
                return False

            market_open = time(9, 25)
            is_market_opened = current_time >= market_open
            if not is_market_opened:
                self.logger.debug(f"今天 {today} 还未开盘")
                return False

            market_close = time(15, 0)
            is_market_closed = current_time >= market_close

            self.logger.debug(f"当前时间: {now}, 交易日: {is_trading_day}, 已开盘: {is_market_opened}, 已收盘: {is_market_closed}")
            return not is_market_closed

        except Exception as e:
            self.logger.warning(f"判断交易时间时出错：{str(e)}")
            return False

    def _get_nearest_trading_date(self, input_date: date) -> Optional[date]:
        """获取指定日期最近的交易日（向前查找）"""
        try:
            start_date = input_date - timedelta(days=30)
            end_date = input_date + timedelta(days=7)

            schedule = self.sse_calendar.schedule(start_date=start_date, end_date=end_date)

            if schedule.empty:
                self.logger.debug(f"无法获取 {start_date} 到 {end_date} 的交易日历")
                return self._fallback_nearest_trading_date(input_date)

            trading_days = sorted(schedule.index.date)

            for trading_day in reversed(trading_days):
                if trading_day <= input_date:
                    self.logger.debug(f"日期 {input_date} 的最近交易日是 {trading_day}")
                    return trading_day

            if trading_days:
                last_trading_day = trading_days[-1]
                self.logger.debug(f"日期 {input_date} 超出范围，返回最后一个交易日 {last_trading_day}")
                return last_trading_day

            return None

        except Exception as e:
            self.logger.warning(f"使用pandas_market_calendars获取最近交易日失败：{str(e)}")
            return self._fallback_nearest_trading_date(input_date)

    def _fallback_nearest_trading_date(self, input_date: date) -> Optional[date]:
        """备用的最近交易日获取方法"""
        try:
            current_date = input_date
            max_lookback = 7

            for _ in range(max_lookback):
                weekday = current_date.weekday()
                if weekday < 5:
                    self.logger.debug(f"使用备用方法：日期 {input_date} 的最近交易日是 {current_date}")
                    return current_date
                current_date = current_date - timedelta(days=1)

            self.logger.debug(f"备用方法未找到交易日，返回输入日期 {input_date}")
            return input_date

        except Exception as e:
            self.logger.warning(f"备用方法获取最近交易日失败：{str(e)}")
            return None

    def _is_trading_day(self, date_to_check) -> bool:
        """判断指定日期是否为交易日"""
        try:
            nearest_trading_date = self._get_nearest_trading_date(date_to_check)
            if nearest_trading_date is None:
                self.logger.debug(f"无法获取 {date_to_check} 的最近交易日")
                return False
            is_trading = nearest_trading_date == date_to_check
            self.logger.debug(f"使用日历检查 {date_to_check}: {'是交易日' if is_trading else '非交易日'}")
            return is_trading
        except Exception as e:
            self.logger.warning(f"判断交易日失败：{str(e)}")
            weekday = date_to_check.weekday()
            is_trading = weekday < 5
            self.logger.debug(f"使用备用方法检查 {date_to_check}: {'是交易日' if is_trading else '非交易日'}")
            return is_trading
