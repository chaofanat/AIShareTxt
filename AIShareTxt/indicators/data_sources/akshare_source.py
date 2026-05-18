import akshare as ak
import pandas as pd
import numpy as np
import requests
from typing import Optional, cast
from datetime import datetime, timedelta
from .base import BaseDataSource
from ...core.config import IndicatorConfig as Config
from ...utils.utils import LoggerManager

import warnings
warnings.filterwarnings('ignore')


class AkshareDataSource(BaseDataSource):
    """基于 akshare 的数据源（东方财富/新浪/腾讯多级 fallback）"""

    def __init__(self):
        self.config = Config()
        self.logger = LoggerManager.get_logger('data_fetcher')

    @property
    def name(self) -> str:
        return 'akshare'

    # ==================== 日线数据 ====================

    def fetch_stock_data(self, stock_code: str, period: str, start_date: str, adjust: str) -> Optional[pd.DataFrame]:
        """通过东方财富→新浪→腾讯三级 fallback 获取日线数据"""

        # 方法1：东方财富（默认）
        raw_data = self._fetch_from_default_api(stock_code, period, start_date, adjust)

        # 方法2：新浪（备用1）
        if raw_data is None or len(raw_data) == 0:
            self.logger.info("尝试使用备用方案1：新浪数据源")
            raw_data = self._fetch_from_sina_api(stock_code, start_date, adjust)

        # 方法3：腾讯（备用2）
        if raw_data is None or len(raw_data) == 0:
            self.logger.info("尝试使用备用方案2：腾讯数据源")
            raw_data = self._fetch_from_tencent_api(stock_code, start_date, adjust)

        if raw_data is None or len(raw_data) == 0:
            self.logger.error(f"所有数据源均获取失败：{stock_code}")
            return None

        return raw_data

    def _fetch_from_default_api(self, stock_code: str, period: str, start_date: str, adjust: str) -> Optional[pd.DataFrame]:
        """从东方财富获取数据"""
        try:
            self.logger.info(f"[方法1] 尝试从东方财富获取股票 {stock_code} 的数据（从 {start_date} 开始）...")
            raw_data = ak.stock_zh_a_hist(
                symbol=stock_code,
                period=cast(str, period),
                start_date=start_date,
                adjust=cast(str, adjust)
            )
            if '成交量' in raw_data.columns:
                raw_data['成交量'] *= 100
            return cast(pd.DataFrame, raw_data)
        except Exception as e:
            self.logger.warning(f"[方法1] 东方财富API获取失败：{str(e)}")
            return None

    def _fetch_from_sina_api(self, stock_code: str, start_date: str, adjust: str) -> Optional[pd.DataFrame]:
        """从新浪获取数据（备用方案1）"""
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

            if 'turnover' in raw_data.columns:
                raw_data['turnover'] = raw_data['turnover'] * 100
                raw_data.rename(columns={'turnover': 'turnover_rate'}, inplace=True)
            if 'amount' in raw_data.columns:
                raw_data.rename(columns={'amount': 'turnover'}, inplace=True)

            return cast(pd.DataFrame, raw_data)
        except Exception as e:
            self.logger.warning(f"[方法2] 新浪API获取失败：{str(e)}")
            return None

    def _fetch_from_tencent_api(self, stock_code: str, start_date: str, adjust: str) -> Optional[pd.DataFrame]:
        """从腾讯获取数据（备用方案2）"""
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

            if 'amount' in raw_data.columns:
                raw_data['volume'] = raw_data['amount'] * 100
                raw_data.drop(columns=['amount'], inplace=True)

            return cast(pd.DataFrame, raw_data)
        except Exception as e:
            self.logger.warning(f"[方法3] 腾讯API获取失败：{str(e)}")
            return None

    # ==================== 资金流 ====================

    def get_fund_flow_data(self, stock_code: str, target_date=None) -> dict:
        """获取主力资金流数据"""
        try:
            self.logger.info("正在获取主力资金流数据...")
            fund_flow_data = {}

            try:
                market = self._determine_market(stock_code)
                fund_df = ak.stock_individual_fund_flow(stock=stock_code, market=market)

                if fund_df is not None and len(fund_df) > 0:
                    latest_row = self._find_fund_flow_row(fund_df, target_date)

                    if latest_row is not None:
                        fund_flow_data = self._parse_fund_flow_data(latest_row)
                        if len(fund_df) >= 5:
                            fund_flow_data.update(self._calculate_5day_fund_flow(fund_df))
                        self.logger.info("主力资金流数据获取成功")

            except Exception as e:
                self.logger.warning(f"获取个股资金流数据失败：{str(e)}")

            fund_flow_data = self._clean_fund_flow_data(fund_flow_data)
            return fund_flow_data

        except Exception as e:
            self.logger.error(f"获取主力资金流数据失败：{str(e)}")
            return {}

    def _find_fund_flow_row(self, fund_df, target_date):
        """在资金流 DataFrame 中查找目标日期的行"""
        if target_date is not None:
            matched_row = None
            for idx, row in fund_df.iterrows():
                row_date = row.get('日期')
                if hasattr(row_date, 'date'):
                    row_date = row_date.date()
                elif isinstance(row_date, str):
                    try:
                        row_date = datetime.strptime(row_date, '%Y-%m-%d').date()
                    except:
                        continue
                if row_date == target_date:
                    matched_row = row
                    self.logger.info(f"找到目标日期 {target_date} 的资金流数据")
                    break

            if matched_row is None:
                self.logger.warning(f"未找到目标日期 {target_date} 的资金流数据，使用最接近的日期")
                for idx in range(len(fund_df) - 1, -1, -1):
                    row = fund_df.iloc[idx]
                    row_date = row.get('日期')
                    if hasattr(row_date, 'date'):
                        row_date = row_date.date()
                    elif isinstance(row_date, str):
                        try:
                            row_date = datetime.strptime(row_date, '%Y-%m-%d').date()
                        except:
                            continue
                    if row_date <= target_date:
                        matched_row = row
                        self.logger.info(f"使用最接近日期 {row_date} 的资金流数据")
                        break

            if matched_row is not None:
                return matched_row
            else:
                earliest_row = fund_df.iloc[0]
                earliest_date = earliest_row.get('日期')
                if hasattr(earliest_date, 'date'):
                    earliest_date = earliest_date.date()
                elif isinstance(earliest_date, str):
                    try:
                        earliest_date = datetime.strptime(earliest_date, '%Y-%m-%d').date()
                    except:
                        earliest_date = "未知"
                self.logger.warning(f"未找到 <= {target_date} 的资金流数据，使用最早可用日期 {earliest_date}")
                return earliest_row
        else:
            return fund_df.iloc[-1]

    def _parse_fund_flow_data(self, latest_row):
        """解析资金流数据"""
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
            if key == '日期':
                continue
            try:
                if isinstance(value, str):
                    if '%' in str(value):
                        fund_flow_data[key] = float(str(value).replace('%', ''))
                    else:
                        fund_flow_data[key] = float(value) if value != '-' else 0.0
                else:
                    fund_flow_data[key] = float(value) if value is not None else 0.0
            except (ValueError, TypeError):
                fund_flow_data[key] = 0.0
        return fund_flow_data

    # ==================== 基本信息 ====================

    def get_stock_basic_info(self, stock_code: str) -> dict:
        """获取股票基本信息（支持降级机制）"""
        # 方案1: 东方财富 API
        self.logger.info("正在获取股票基本信息...")
        try:
            df = ak.stock_individual_info_em(symbol=stock_code)
            info_dict = {}
            for i, row in df.iterrows():
                key = str(row.iloc[0]).strip()
                value = str(row.iloc[1]).strip()
                info_dict[key] = value
            stock_info = self._parse_basic_info(info_dict, stock_code)
            stock_info = self._format_market_values(stock_info)
            self.logger.info("股票基本信息获取成功（东方财富 API）")
            return stock_info
        except Exception as e:
            self.logger.warning(f"东方财富 API 失败：{str(e)}")

        # 方案2: 雪球 API
        self.logger.info("尝试降级方案：雪球 API 组合...")
        try:
            stock_info = self._get_stock_basic_info_from_xq(stock_code)
            if stock_info and stock_info.get('股票简称') != '未知':
                self.logger.info("股票基本信息获取成功（雪球 API 降级）")
                return stock_info
        except Exception as e:
            self.logger.warning(f"雪球 API 降级失败：{str(e)}")

        # 方案3: 默认值
        self.logger.warning("所有数据源均失败，返回默认值")
        return self._get_default_basic_info(stock_code)

    def _get_stock_basic_info_from_xq(self, stock_code):
        """从雪球 API 获取股票基本信息（降级方案）"""
        if stock_code.startswith('6'):
            xq_code = f'SH{stock_code}'
        else:
            xq_code = f'SZ{stock_code}'

        basic_info = self._get_xq_basic_info(xq_code)
        spot_info = self._get_xq_spot_info(xq_code)

        stock_info = {
            '股票代码': stock_code,
            '股票简称': basic_info.get('股票简称', '未知'),
            '行业': basic_info.get('行业', '未知'),
            '总市值': spot_info.get('总市值', 0),
            '流通市值': spot_info.get('流通市值', 0),
            '总股本': spot_info.get('流通股', 0),
            '流通股本': spot_info.get('流通股', 0),
            '市盈率': spot_info.get('市盈率', '未知'),
            '市净率': spot_info.get('市净率', '未知'),
            '当前价格': spot_info.get('现价', 0),
        }
        stock_info = self._format_market_values(stock_info)
        return stock_info

    def _get_xq_basic_info(self, xq_code):
        """从雪球获取基本信息（股票简称、行业）"""
        import pandas as pd
        import requests
        import os

        retry_exceptions = (
            requests.exceptions.ProxyError,
            requests.exceptions.SSLError,
            KeyError,
            requests.exceptions.ConnectionError,
        )

        for attempt in range(2):
            try:
                if attempt == 1:
                    self.logger.info("检测到代理错误，尝试禁用代理后重试...")
                    import akshare.stock.cons as cons
                    original_session = getattr(cons, '_session', None)
                    session = requests.Session()
                    session.trust_env = False
                    session.proxies = {'http': None, 'https': None}
                    original_get = requests.get
                    requests.get = lambda *args, **kwargs: original_get(
                        *args, proxies={'http': None, 'https': None}, **kwargs
                    )

                try:
                    df = ak.stock_individual_basic_info_xq(symbol=xq_code)
                    name = '未知'
                    name_row = df[df['item'] == 'org_short_name_cn']
                    if not name_row.empty:
                        val = name_row.iloc[0]['value']
                        if pd.notna(val) and val:
                            name = val

                    industry = '未知'
                    industry_row = df[df['item'] == 'affiliate_industry']
                    if not industry_row.empty:
                        val = industry_row.iloc[0]['value']
                        if isinstance(val, dict) and 'ind_name' in val:
                            industry = val['ind_name']

                    return {'股票简称': name, '行业': industry}

                finally:
                    if attempt == 1:
                        requests.get = original_get

            except retry_exceptions as e:
                if attempt == 0:
                    if isinstance(e, KeyError):
                        self.logger.warning(f"雪球API返回格式异常(KeyError): {str(e)}")
                    else:
                        self.logger.warning(f"雪球API请求失败(可能是代理问题): {type(e).__name__}: {str(e)}")
                else:
                    self.logger.error(f"雪球API重试后仍然失败: {type(e).__name__}: {str(e)}")
                    raise

            except Exception as e:
                self.logger.error(f"雪球API获取基本信息时发生未预期错误: {type(e).__name__}: {str(e)}")
                raise

    def _get_xq_spot_info(self, xq_code):
        """从雪球获取实时行情（市值、市盈率等）"""
        import requests
        import pandas as pd

        retry_exceptions = (
            requests.exceptions.ProxyError,
            requests.exceptions.SSLError,
            KeyError,
            requests.exceptions.ConnectionError,
        )

        for attempt in range(2):
            try:
                if attempt == 1:
                    self.logger.info("检测到代理错误，尝试禁用代理后重试...")
                    original_get = requests.get
                    requests.get = lambda *args, **kwargs: original_get(
                        *args, proxies={'http': None, 'https': None}, **kwargs
                    )

                try:
                    df = ak.stock_individual_spot_xq(symbol=xq_code)
                    data_dict = dict(zip(df['item'], df['value']))

                    total_market_cap = self._safe_float_conversion(data_dict.get('资产净值/总市值', 0))
                    float_market_cap = self._safe_float_conversion(data_dict.get('流通值', 0))
                    float_shares = self._safe_float_conversion(data_dict.get('流通股', 0))
                    pe_ratio = data_dict.get('市盈率(TTM)', '未知')
                    pb_ratio = data_dict.get('市净率', '未知')
                    current_price = self._safe_float_conversion(data_dict.get('现价', 0))

                    return {
                        '总市值': total_market_cap,
                        '流通市值': float_market_cap,
                        '流通股': float_shares,
                        '市盈率': pe_ratio,
                        '市净率': pb_ratio,
                        '现价': current_price,
                    }

                finally:
                    if attempt == 1:
                        requests.get = original_get

            except retry_exceptions as e:
                if attempt == 0:
                    if isinstance(e, KeyError):
                        self.logger.warning(f"雪球API返回格式异常(KeyError): {str(e)}")
                    else:
                        self.logger.warning(f"雪球API请求失败(可能是代理问题): {type(e).__name__}: {str(e)}")
                else:
                    self.logger.error(f"雪球API重试后仍然失败: {type(e).__name__}: {str(e)}")
                    raise

            except Exception as e:
                self.logger.error(f"雪球API获取实时行情时发生未预期错误: {type(e).__name__}: {str(e)}")
                raise

    # ==================== 辅助方法 ====================

    def _add_stock_prefix(self, stock_code: str) -> str:
        if stock_code.startswith('6'):
            return f'sh{stock_code}'
        else:
            return f'sz{stock_code}'

    def _determine_market(self, stock_code):
        first_digit = stock_code[0]
        return self.config.MARKET_MAPPING.get(first_digit, 'sz')

    def _parse_basic_info(self, info_dict, stock_code):
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
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _format_market_values(self, stock_info):
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

    # ==================== 港股数据 ====================

    def fetch_hk_stock_data(self, stock_code: str, start_date: str, adjust: str) -> Optional[pd.DataFrame]:
        """获取港股历史数据（东方财富 → stock_hk_daily fallback）"""
        raw_data = self._fetch_from_hk_hist_em(stock_code, start_date, adjust)
        if raw_data is None or len(raw_data) == 0:
            self.logger.error(f"港股数据获取失败：{stock_code}")
            return None
        return raw_data

    def _fetch_from_hk_hist_em(self, stock_code: str, start_date: str, adjust: str) -> Optional[pd.DataFrame]:
        """从东方财富获取港股历史数据"""
        try:
            self.logger.info(f"[港股] 从东方财富获取港股 {stock_code} 的数据（从 {start_date} 开始）...")
            end_date = datetime.now().strftime('%Y%m%d')
            raw_data = ak.stock_hk_hist(
                symbol=stock_code,
                period='daily',
                start_date=start_date,
                end_date=end_date,
                adjust=adjust
            )
            return cast(pd.DataFrame, raw_data)
        except Exception as e:
            self.logger.warning(f"[港股] stock_hk_hist获取失败：{str(e)}，尝试备用API...")

        try:
            self.logger.info(f"[港股] 尝试备用API stock_hk_daily 获取港股 {stock_code} 的数据...")
            raw_data = ak.stock_hk_daily(symbol=stock_code)
            return cast(pd.DataFrame, raw_data)
        except Exception as e:
            self.logger.warning(f"[港股] stock_hk_daily获取失败：{str(e)}")
            return None

    def get_hk_stock_basic_info(self, stock_code: str) -> dict:
        """获取港股基本信息"""
        info = {'股票代码': stock_code}

        try:
            df = ak.stock_hk_company_profile_em(symbol=stock_code)
            if not df.empty:
                info.update({
                    '股票简称': df.iloc[0].get('公司名称', '未知'),
                    '行业': df.iloc[0].get('所属行业', '未知'),
                })
                self.logger.info("✓ 港股公司信息获取成功（东方财富 API）")
        except Exception as e:
            self.logger.warning(f"[港股] 东方财富港股基本信息失败：{str(e)}")

        if info.get('股票简称') == '未知':
            try:
                xq_info = self._get_hk_basic_info_from_xq(stock_code)
                if xq_info.get('股票简称') != '未知':
                    info.update(xq_info)
                    self.logger.info("✓ 港股公司信息获取成功（雪球 API）")
            except Exception as e:
                self.logger.warning(f"[港股] 雪球港股基本信息失败：{str(e)}")

        try:
            df = ak.stock_hk_financial_indicator_em(symbol=stock_code)
            if not df.empty:
                info.update({
                    '总市值': df.iloc[0].get('总市值(港元)', 0),
                    '市盈率': df.iloc[0].get('市盈率', '未知'),
                    '市净率': df.iloc[0].get('市净率', '未知'),
                    '股息率': df.iloc[0].get('股息率TTM(%)', 0),
                    '每股净资产': df.iloc[0].get('每股净资产(元)', 0),
                })
        except Exception as e:
            self.logger.warning(f"[港股] 财务指标获取失败：{str(e)}")

        return self._format_market_values(info)

    def _get_hk_basic_info_from_xq(self, stock_code: str) -> dict:
        """从雪球获取港股基本信息"""
        xq_code = f'HK{stock_code.zfill(5)}'
        try:
            df = ak.stock_individual_basic_info_hk_xq(symbol=xq_code)
            info = {}
            name = '未知'
            name_row = df[df['item'] == 'name']
            if not name_row.empty:
                val = name_row.iloc[0]['value']
                if pd.notna(val) and val:
                    name = val
            info['股票简称'] = name

            industry = '未知'
            industry_row = df[df['item'] == 'industry']
            if not industry_row.empty:
                val = industry_row.iloc[0]['value']
                if isinstance(val, dict) and 'name' in val:
                    industry = val['name']
            info['行业'] = industry
            return info
        except Exception as e:
            self.logger.warning(f"[港股] 雪球API获取失败：{str(e)}")
            return {'股票简称': '未知', '行业': '未知'}
