#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票数据获取模块
负责从各种数据源获取股票相关数据
"""

import akshare as ak
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from typing import Optional, Union, cast
from datetime import datetime, timedelta, time
from ..core.config import IndicatorConfig as Config
from ..utils.utils import LoggerManager
import warnings
warnings.filterwarnings('ignore')


class StockDataFetcher:
    """股票数据获取器"""
    
    def __init__(self):
        self.config = Config()
        self.logger = LoggerManager.get_logger('data_fetcher')
        # 获取上交所（SSE）日历
        self.sse_calendar = mcal.get_calendar('SSE')
    
    def _is_trading_day_and_not_closed(self) -> bool:
        """
        判断今天是否是交易日且未收盘

        Returns:
            bool: True表示今天是交易日且未收盘，False表示非交易日或已收盘
        """
        try:
            now = datetime.now()
            today = now.date()
            current_time = now.time()

            # 首先使用 pandas_market_calendars 判断是否为交易日
            is_trading_day = self._is_trading_day(today)

            if not is_trading_day:
                self.logger.debug(f"今天 {today} 不是交易日")
                return False

            # 收盘时间：15:00
            market_close = time(15, 0)

            # 开盘时间：9:30
            market_open = time(9, 25)
            # 判断今天是否已开盘
            is_market_opened = current_time >= market_open
            # 如果今天未开盘，返回False
            if not is_market_opened:
                self.logger.debug(f"今天 {today} 还未开盘")
                return False

            # 判断今天是否已收盘
            is_market_closed = current_time >= market_close 

            self.logger.debug(f"当前时间: {now}")
            self.logger.debug(f"是否为交易日: {is_trading_day}")
            self.logger.debug(f"市场是否已开盘: {is_market_opened}")
            self.logger.debug(f"市场是否已收盘: {is_market_closed}")

            # 如果是交易日且已开盘但是未收盘，返回True
            return not is_market_closed

        except Exception as e:
            self.logger.warning(f"判断交易时间时出错：{str(e)}")
            # 如果无法判断，返回False（保守处理）
            return False

    def _is_trading_day(self, date_to_check) -> bool:
        """
        判断指定日期是否为交易日

        Args:
            date_to_check: 要检查的日期

        Returns:
            bool: True表示是交易日，False表示非交易日
        """
        try:
            # 获取日期前后的交易日历（扩大范围以确保能获取到数据）
            start_date = date_to_check - timedelta(days=7)  # 往前7天
            end_date = date_to_check + timedelta(days=7)    # 往后7天

            schedule = self.sse_calendar.schedule(start_date=start_date, end_date=end_date)

            if schedule.empty:
                self.logger.debug(f"无法获取 {start_date} 到 {end_date} 的交易日历")
                return self._fallback_trading_day_check(date_to_check)

            # 检查指定日期是否在交易日历中
            trading_days = schedule.index.date
            is_trading = date_to_check in trading_days

            self.logger.debug(f"使用日历检查 {date_to_check}: {'是交易日' if is_trading else '非交易日'}")
            return is_trading

        except Exception as e:
            self.logger.warning(f"使用pandas_market_calendars判断交易日失败：{str(e)}")
            # 如果 pandas_market_calendars 失败，回退到简单的周一到周五判断
            return self._fallback_trading_day_check(date_to_check)

    def _fallback_trading_day_check(self, date_to_check) -> bool:
        """
        备用的交易日判断方法（简单的周一到周五判断）

        Args:
            date_to_check: 要检查的日期

        Returns:
            bool: True表示是交易日，False表示非交易日
        """
        weekday = date_to_check.weekday()
        is_trading = weekday < 5  # 0-4 是周一到周五
        self.logger.debug(f"使用备用方法检查 {date_to_check}: {'是交易日' if is_trading else '非交易日'}")
        return is_trading

    def _remove_incomplete_trading_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        如果今天是交易日且已经开盘但是未收盘，移除最后一个不完整的交易日数据

        Args:
            data: 股票数据DataFrame

        Returns:
            处理后的DataFrame
        """
        if self._is_trading_day_and_not_closed():
            self.logger.info("检测到当前为交易日且未收盘，移除最新不完整数据")
            # 移除最后一行数据
            if len(data) > 0:
                data = data.iloc[:-1].copy()
                self.logger.info(f"已移除最新数据，剩余 {len(data)} 条记录")
        else:
            self.logger.debug("当前为非交易日或已收盘，保留所有数据")

        return data

    def _add_stock_prefix(self, stock_code: str) -> str:
        """
        为股票代码添加市场前缀

        Args:
            stock_code (str): 6位股票代码，如 '000001'

        Returns:
            str: 带前缀的股票代码，如 'sz000001' 或 'sh600000'
        """
        if stock_code.startswith('6'):
            return f'sh{stock_code}'
        else:
            return f'sz{stock_code}'
    
    def _process_stock_data(self, data: pd.DataFrame, stock_code: str) -> Optional[pd.DataFrame]:
        """
        处理和清洗股票数据
        
        Args:
            data: 原始股票数据
            stock_code: 股票代码
            
        Returns:
            处理后的数据或None
        """
        try:
            if data is None or len(data) == 0:
                self.logger.error(f"{self.config.ERROR_MESSAGES['no_data']}: {stock_code}")
                return None
            
            self.logger.debug(f"获取到数据，形状: {data.shape}")
            
            # 标准化列名
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
            
            # 删除价格为负数的行
            price_cols = ['open', 'close', 'high', 'low']
            for col in price_cols:
                mask = data[col] >= 0
                data = cast(pd.DataFrame, data[mask].copy())
            
            # 删除成交量为负数的行
            volume_mask = data['volume'] >= 0
            data = cast(pd.DataFrame, data[volume_mask].copy())
            
            # 删除价格逻辑错误的行（最高价小于最低价等）
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

            self.logger.info(f"✓ 成功处理股票 {stock_code} 的数据，共 {len(data)} 条记录")
            if original_length != len(data):
                self.logger.info(f"已移除 {original_length - len(data)} 条不完整交易数据")

            self.logger.debug(f"数据日期范围: {data['date'].iloc[0]} 到 {data['date'].iloc[-1]}")
            self.logger.info(f"最新收盘价: {data['close'].iloc[-1]:.{self.config.DISPLAY_PRECISION['price']}f}")

            return data
            
        except Exception as e:
            self.logger.error(f"处理股票数据时出错：{str(e)}")
            return None
    
    def _fetch_from_default_api(self, stock_code: str, period: str, start_date: str, adjust: str) -> Optional[pd.DataFrame]:
        """
        从默认API获取数据（东方财富）
        
        Args:
            stock_code: 股票代码
            period: 周期
            start_date: 开始日期
            adjust: 复权类型
            
        Returns:
            原始数据或None
        """
        try:
            self.logger.info(f"[方法1] 尝试从东方财富获取股票 {stock_code} 的数据（从 {start_date} 开始）...")
            raw_data = ak.stock_zh_a_hist(
                symbol=stock_code,
                period=cast(str, period),
                start_date=start_date,
                adjust=cast(str, adjust)
            )

            # 对成交量列需要进行数量转换*100，从手转换为股
            if '成交量' in raw_data.columns:
                raw_data['成交量'] *= 100
            
            return cast(pd.DataFrame, raw_data)
        except Exception as e:
            self.logger.warning(f"[方法1] 东方财富API获取失败：{str(e)}")
            return None
    
    def _fetch_from_sina_api(self, stock_code: str, start_date: str, adjust: str) -> Optional[pd.DataFrame]:
        """
        从新浪API获取数据（备用方案1）
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            adjust: 复权类型
            
        Returns:
            原始数据或None
        """
        try:
            stock_code_with_prefix = self._add_stock_prefix(stock_code)
            end_date = datetime.now().strftime('%Y%m%d')
            
            self.logger.info(f"[方法2] 尝试从新浪获取股票 {stock_code_with_prefix} 的数据...")
            raw_data = ak.stock_zh_a_daily(
                symbol=stock_code_with_prefix,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust
            )

            # amount列更名为turnover
            if 'amount' in raw_data.columns:
                raw_data.rename(columns={'amount': 'turnover'}, inplace=True)
            
            return cast(pd.DataFrame, raw_data)
        except Exception as e:
            self.logger.warning(f"[方法2] 新浪API获取失败：{str(e)}")
            return None
    
    def _fetch_from_tencent_api(self, stock_code: str, start_date: str, adjust: str) -> Optional[pd.DataFrame]:
        """
        从腾讯API获取数据（备用方案2）
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            adjust: 复权类型
            
        Returns:
            原始数据或None
        """
        try:
            stock_code_with_prefix = self._add_stock_prefix(stock_code)
            end_date = datetime.now().strftime('%Y%m%d')
            
            self.logger.info(f"[方法3] 尝试从腾讯获取股票 {stock_code_with_prefix} 的数据...")
            raw_data = ak.stock_zh_a_hist_tx(
                symbol=stock_code_with_prefix,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust
            )
            
            # 对成交量列需要进行数量转换*100，从手转换为股，并且列名从amount改为volume
            if 'amount' in raw_data.columns:
                raw_data['volume'] = raw_data['amount'] * 100
                # 移除原始amount列
                raw_data.drop(columns=['amount'], inplace=True)
            
            return cast(pd.DataFrame, raw_data)
        except Exception as e:
            self.logger.warning(f"[方法3] 腾讯API获取失败：{str(e)}")
            return None
    
    def fetch_stock_data(self, stock_code: str, period: Optional[str] = None, adjust: Optional[str] = None, start_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        获取股票历史价格数据（支持多数据源备用方案）
        
        Args:
            stock_code (str): 股票代码，如 '000001'
            period (str): 周期，默认为 'daily'
            adjust (str): 复权类型，默认为 'qfq'
            start_date (str): 开始日期，如 '20230101',默认为当前时间往前推4个月
        
        Returns:
            pd.DataFrame or None: 股票数据DataFrame,失败返回None
        """
        if period is None:
            period = self.config.DATA_CONFIG['default_period']
        if adjust is None:
            adjust = self.config.DATA_CONFIG['default_adjust']
        
        # 如果没有指定开始日期，默认为当前时间往前推配置的月数
        if start_date is None:
            months_back = self.config.DATA_CONFIG.get('default_months_back', 4)
            default_start = datetime.now() - timedelta(days=months_back * 30)  # 每月按30天计算
            start_date = default_start.strftime('%Y%m%d')
        
        # 尝试多个数据源
        raw_data = None
        
        # 方法1：东方财富（默认）
        raw_data = self._fetch_from_default_api(stock_code, cast(str, period), start_date, cast(str, adjust))
        
        # 方法2：新浪（备用1）
        if raw_data is None or len(raw_data) == 0:
            self.logger.info("尝试使用备用方案1：新浪数据源")
            raw_data = self._fetch_from_sina_api(stock_code, start_date, cast(str, adjust))
        
        # 方法3：腾讯（备用2）
        if raw_data is None or len(raw_data) == 0:
            self.logger.info("尝试使用备用方案2：腾讯数据源")
            raw_data = self._fetch_from_tencent_api(stock_code, start_date, cast(str, adjust))
        
        # 如果所有方法都失败
        if raw_data is None or len(raw_data) == 0:
            self.logger.error(f"所有数据源均获取失败：{stock_code}")
            self.logger.info("可能的原因：")
            self.logger.info("1. 股票代码格式错误（请确保是6位数字，如 000001）")
            self.logger.info("2. 网络连接问题")
            self.logger.info("3. 所有数据源API暂时不可用")
            self.logger.info("4. 股票代码不存在或已退市")
            return None
        

        
        # 处理数据
        return self._process_stock_data(raw_data, stock_code)
    
    def get_fund_flow_data(self, stock_code):
        """
        获取主力资金流数据

        Args:
            stock_code (str): 股票代码

        Returns:
            dict: 资金流数据字典

        Note:
            akshare 的 stock_individual_fund_flow API 已经处理了交易日不完整数据的问题，
            不需要额外移除当日数据
        """
        try:
            self.logger.info("正在获取主力资金流数据...")

            fund_flow_data = {}

            # 获取个股资金流数据
            try:
                # 根据股票代码判断市场
                market = self._determine_market(stock_code)
                fund_df = ak.stock_individual_fund_flow(stock=stock_code, market=market)

                if fund_df is not None and len(fund_df) > 0:
                    # 获取最新一天的数据
                    latest_row = fund_df.iloc[-1]

                    # 解析资金流数据
                    fund_flow_data = self._parse_fund_flow_data(latest_row)

                    # 计算5日累计数据
                    if len(fund_df) >= 5:
                        fund_flow_data.update(self._calculate_5day_fund_flow(fund_df))

                    self.logger.info("✓ 主力资金流数据获取成功")
                    self.logger.debug(f"资金流数据范围: {fund_df.iloc[0].get('日期', 'N/A')} 到 {fund_df.iloc[-1].get('日期', 'N/A')}")

            except Exception as e:
                self.logger.warning(f"获取个股资金流数据失败：{str(e)}")

            # 数据清洗和格式化
            fund_flow_data = self._clean_fund_flow_data(fund_flow_data)

            return fund_flow_data

        except Exception as e:
            self.logger.error(f"获取主力资金流数据失败：{str(e)}")
            return {}
    
    def get_stock_basic_info(self, stock_code):
        """
        获取股票基本信息
        
        Args:
            stock_code (str): 股票代码
            
        Returns:
            dict: 股票基本信息字典
        """
        try:
            self.logger.info("正在获取股票基本信息...")
            
            # 获取个股信息
            df = ak.stock_individual_info_em(symbol=stock_code)
            
            # 创建字段映射
            info_dict = {}
            for i, row in df.iterrows():
                key = str(row.iloc[0]).strip()
                value = str(row.iloc[1]).strip()
                info_dict[key] = value
            
            # 解析基本信息
            stock_info = self._parse_basic_info(info_dict, stock_code)
            
            # 格式化数值
            stock_info = self._format_market_values(stock_info)
            
            self.logger.info("✓ 股票基本信息获取成功")
            return stock_info
            
        except Exception as e:
            self.logger.warning(f"获取股票基本信息失败：{str(e)}")
            # 返回基础信息
            return self._get_default_basic_info(stock_code)
    
    def _determine_market(self, stock_code):
        """根据股票代码判断市场"""
        first_digit = stock_code[0]
        return self.config.MARKET_MAPPING.get(first_digit, 'sz')
    
    def _parse_fund_flow_data(self, latest_row):
        """解析资金流数据"""
        # 处理日期字段，确保为字符串格式
        date_value = latest_row.get('日期', '')
        if hasattr(date_value, 'strftime'):
            date_str = date_value.strftime('%Y-%m-%d')
        else:
            date_str = str(date_value)
            
        return {
            '日期': date_str,
            '主力净流入额': latest_row.get('主力净流入-净额', 0),
            '主力净流入占比': latest_row.get('主力净流入-净占比', 0),
            '超大单净流入额': latest_row.get('超大单净流入-净额', 0),
            '超大单净流入占比': latest_row.get('超大单净流入-净占比', 0),
            '大单净流入额': latest_row.get('大单净流入-净额', 0),
            '大单净流入占比': latest_row.get('大单净流入-净占比', 0),
            '中单净流入额': latest_row.get('中单净流入-净额', 0),
            '中单净流入占比': latest_row.get('中单净流入-净占比', 0),
            '小单净流入额': latest_row.get('小单净流入-净额', 0),
            '小单净流入占比': latest_row.get('小单净流入-净占比', 0),
            '收盘价': latest_row.get('收盘价', 0),
            '涨跌幅': latest_row.get('涨跌幅', 0)
        }
    
    def _calculate_5day_fund_flow(self, fund_df):
        """计算5日累计资金流"""
        recent_5_days = fund_df.tail(5)
        
        result = {
            '5日主力净流入额': recent_5_days['主力净流入-净额'].sum() if '主力净流入-净额' in recent_5_days.columns else 0,
            '5日超大单净流入额': recent_5_days['超大单净流入-净额'].sum() if '超大单净流入-净额' in recent_5_days.columns else 0,
            '5日大单净流入额': recent_5_days['大单净流入-净额'].sum() if '大单净流入-净额' in recent_5_days.columns else 0,
            '5日中单净流入额': recent_5_days['中单净流入-净额'].sum() if '中单净流入-净额' in recent_5_days.columns else 0,
            '5日小单净流入额': recent_5_days['小单净流入-净额'].sum() if '小单净流入-净额' in recent_5_days.columns else 0,
        }
        
        # 计算5日平均占比
        for col_name, avg_key in [
            ('主力净流入-净占比', '5日主力净流入占比'),
            ('超大单净流入-净占比', '5日超大单净流入占比'),
            ('大单净流入-净占比', '5日大单净流入占比'),
            ('中单净流入-净占比', '5日中单净流入占比'),
            ('小单净流入-净占比', '5日小单净流入占比')
        ]:
            if col_name in recent_5_days.columns:
                result[avg_key] = recent_5_days[col_name].mean()
        
        return result
    
    def _clean_fund_flow_data(self, fund_flow_data):
        """清洗和格式化资金流数据"""
        for key, value in fund_flow_data.items():
            if key == '日期':  # 日期字段保持字符串
                continue
            try:
                if isinstance(value, str):
                    # 去除百分号并转换为数值
                    if '%' in str(value):
                        fund_flow_data[key] = float(str(value).replace('%', ''))
                    else:
                        fund_flow_data[key] = float(value) if value != '-' else 0.0
                else:
                    fund_flow_data[key] = float(value) if value is not None else 0.0
            except (ValueError, TypeError):
                fund_flow_data[key] = 0.0
        
        return fund_flow_data
    
    def _parse_basic_info(self, info_dict, stock_code):
        """解析基本信息"""
        return {
            '股票代码': info_dict.get('代码', info_dict.get('股票代码', stock_code)),
            '股票简称': info_dict.get('简称', info_dict.get('股票简称', '未知')),
            '行业': info_dict.get('行业', info_dict.get('所属行业', '未知')),
            '流通股本': self._safe_float_conversion(info_dict.get('流通股本', 0)),
            '流通市值': self._safe_float_conversion(info_dict.get('流通市值', 0)),
            '总股本': self._safe_float_conversion(info_dict.get('总股本', 0)),
            '总市值': self._safe_float_conversion(info_dict.get('总市值', 0)),
            '市盈率': info_dict.get('市盈率-动态', info_dict.get('市盈率', '未知')),
            '市净率': info_dict.get('市净率', '未知')
        }
    
    def _safe_float_conversion(self, value):
        """安全的浮点数转换"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _format_market_values(self, stock_info):
        """格式化市值数据（转换为亿为单位）"""
        # 格式化数值（亿为单位）
        for field, unit_field in [
            ('流通市值', '流通市值_亿'),
            ('总市值', '总市值_亿'),
            ('流通股本', '流通股本_亿股'),
            ('总股本', '总股本_亿股')
        ]:
            value = stock_info.get(field, 0)
            if value > 0:
                stock_info[unit_field] = value / 100000000
            else:
                stock_info[unit_field] = 0
        
        return stock_info
    
    def _get_default_basic_info(self, stock_code):
        """获取默认基本信息"""
        return {
            '股票代码': stock_code,
            '股票简称': "未知",
            '行业': "未知",
            '流通市值_亿': 0,
            '总市值_亿': 0,
            '流通股本_亿股': 0,
            '总股本_亿股': 0,
            '市盈率': "未知",
            '市净率': "未知"
        }
