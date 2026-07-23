#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易日历工具模块

提供交易日判断、最近交易日查询、市场开盘状态等工具函数。
模块级单例 SSE 日历，跨模块共享。
"""

from datetime import datetime, timedelta, time, date
from typing import Optional

import pandas_market_calendars as mcal

from .utils import LoggerManager


_logger = LoggerManager.get_logger('trading_calendar')

# 模块级单例 SSE 日历，惰性初始化
_sse_calendar = None


def _get_sse_calendar():
    """获取（惰性初始化）上交所交易日历单例"""
    global _sse_calendar
    if _sse_calendar is None:
        _sse_calendar = mcal.get_calendar('SSE')
    return _sse_calendar


def is_trading_day(date_to_check) -> bool:
    """判断指定日期是否为交易日"""
    try:
        nearest = get_nearest_trading_date(date_to_check)
        if nearest is None:
            _logger.debug(f"无法获取 {date_to_check} 的最近交易日")
            return False
        is_trading = nearest == date_to_check
        _logger.debug(f"使用日历检查 {date_to_check}: {'是交易日' if is_trading else '非交易日'}")
        return is_trading
    except Exception as e:
        _logger.warning(f"判断交易日失败：{str(e)}")
        weekday = date_to_check.weekday() if hasattr(date_to_check, 'weekday') else 6
        is_trading = weekday < 5
        _logger.debug(f"使用备用方法检查 {date_to_check}: {'是交易日' if is_trading else '非交易日'}")
        return is_trading


def get_nearest_trading_date(input_date: date) -> Optional[date]:
    """获取指定日期最近的交易日（向前查找）"""
    try:
        start_date = input_date - timedelta(days=30)
        end_date = input_date + timedelta(days=7)

        schedule = _get_sse_calendar().schedule(start_date=start_date, end_date=end_date)

        if schedule.empty:
            _logger.debug(f"无法获取 {start_date} 到 {end_date} 的交易日历")
            return _fallback_nearest_trading_date(input_date)

        trading_days = sorted(schedule.index.date)

        for trading_day in reversed(trading_days):
            if trading_day <= input_date:
                _logger.debug(f"日期 {input_date} 的最近交易日是 {trading_day}")
                return trading_day

        if trading_days:
            last_trading_day = trading_days[-1]
            _logger.debug(f"日期 {input_date} 超出范围，返回最后一个交易日 {last_trading_day}")
            return last_trading_day

        return None

    except Exception as e:
        _logger.warning(f"使用pandas_market_calendars获取最近交易日失败：{str(e)}")
        return _fallback_nearest_trading_date(input_date)


def _fallback_nearest_trading_date(input_date: date) -> Optional[date]:
    """备用的最近交易日获取方法（仅按工作日判断，不考虑节假日）"""
    try:
        current_date = input_date
        max_lookback = 7

        for _ in range(max_lookback):
            weekday = current_date.weekday()
            if weekday < 5:
                _logger.debug(f"使用备用方法：日期 {input_date} 的最近交易日是 {current_date}")
                return current_date
            current_date = current_date - timedelta(days=1)

        _logger.debug(f"备用方法未找到交易日，返回输入日期 {input_date}")
        return input_date

    except Exception as e:
        _logger.warning(f"备用方法获取最近交易日失败：{str(e)}")
        return None


def is_trading_day_and_not_closed() -> bool:
    """
    判断当前是否为交易日且市场尚未收盘（开盘后到收盘前）。

    用于决定是否需要移除当日不完整数据。
    """
    try:
        now = datetime.now()
        today = now.date()
        current_time = now.time()

        if not is_trading_day(today):
            _logger.debug(f"今天 {today} 不是交易日")
            return False

        if current_time < time(9, 25):
            _logger.debug(f"今天 {today} 还未开盘")
            return False

        is_closed = current_time >= time(15, 0)
        _logger.debug(
            f"当前时间: {now}, 交易日: True, 已收盘: {is_closed}"
        )
        return not is_closed

    except Exception as e:
        _logger.warning(f"判断交易时间时出错：{str(e)}")
        return False


def is_market_open() -> bool:
    """
    判断当前是否处于盘中时段（9:30-11:30 或 13:00-15:30）。

    15:00-15:30 视为盘中：虽然收盘但清算未完成，数据仍可能变动，
    gm history 接口尚未更新当日数据，TTL 应保持短间隔。
    15:30 后视为盘后：清算完成，可用 gm history 批量拉数据，TTL 可延长。
    """
    try:
        now = datetime.now()
        today = now.date()
        if not is_trading_day(today):
            return False

        t = now.time()
        in_morning = time(9, 30) <= t <= time(11, 30)
        in_afternoon = time(13, 0) <= t <= time(15, 30)
        return in_morning or in_afternoon
    except Exception as e:
        _logger.warning(f"判断盘中时段失败：{str(e)}")
        return False
