#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aishare-market 命令行入口

用法：
    aishare-market               # 仅输出市场环境报告
    aishare-market 000001        # 市场报告 + 该股的板块涨跌
    aishare-market -h            # 查看帮助

设计原则：一次性输出，不做交互式 REPL（与 aishare 命令职责分离）。
"""

import re
import sys
from typing import Optional, List

from .processor import MarketEnvironmentProcessor


USAGE = """aishare-market - 市场环境维度报告生成器

用法:
    aishare-market               输出市场环境报告（大盘指数 + 市场宽度 + 阶段判定）
    aishare-market <股票代码>    额外附加该股的所属板块涨跌
    aishare-market -h            显示本帮助

示例:
    aishare-market
    aishare-market 000001
    aishare-market 600519

说明:
    本命令与 aishare <股票代码>（个股技术指标报告）相互独立，
    可在 AI workflow 中分别调用后自行拼接。
"""

_STOCK_CODE_RE = re.compile(r'^\d{6}$')
_HELP_ARGS = {'-h', '--help', 'help'}


def _parse_args(argv: List[str]) -> tuple[bool, Optional[str]]:
    """解析命令行参数。

    Returns:
        (show_help, stock_code)
    """
    if any(a in _HELP_ARGS for a in argv):
        return True, None

    stock_code = None
    for arg in argv:
        if _STOCK_CODE_RE.match(arg):
            stock_code = arg
            break

    return False, stock_code


def main() -> int:
    """aishare-market 命令入口。"""
    argv = sys.argv[1:]
    show_help, stock_code = _parse_args(argv)

    if show_help:
        print(USAGE)
        return 0

    processor = MarketEnvironmentProcessor()
    report = processor.generate_market_report(stock_code)
    print(report)
    return 0


if __name__ == '__main__':
    sys.exit(main())
