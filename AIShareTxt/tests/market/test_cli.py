"""aishare-market CLI 入口的单元测试。

不触发真实 akshare 请求，全部 mock processor。
"""

import re
import sys
from io import StringIO

import pytest

from AIShareTxt.market import cli


# ==================== _parse_args ====================

def test_parse_args_help_short():
    show_help, code = cli._parse_args(['-h'])
    assert show_help is True
    assert code is None


def test_parse_args_help_long():
    show_help, code = cli._parse_args(['--help'])
    assert show_help is True


def test_parse_args_help_word():
    show_help, code = cli._parse_args(['help'])
    assert show_help is True


def test_parse_args_no_args():
    show_help, code = cli._parse_args([])
    assert show_help is False
    assert code is None


def test_parse_args_stock_code():
    show_help, code = cli._parse_args(['000001'])
    assert show_help is False
    assert code == '000001'


def test_parse_args_picks_first_stock_code():
    show_help, code = cli._parse_args(['300001', '000002'])
    assert code == '300001'


def test_parse_args_ignores_non_stock_strings():
    show_help, code = cli._parse_args(['foo', '600519', 'bar'])
    assert show_help is False
    assert code == '600519'


def test_parse_args_rejects_non_6_digit():
    show_help, code = cli._parse_args(['12345'])  # 5 位不是合法代码
    assert code is None


# ==================== main ====================

def test_main_prints_help_and_returns_zero(capsys):
    rc = cli.main.__wrapped__ if hasattr(cli.main, '__wrapped__') else None
    # 模拟 sys.argv
    sys.argv = ['aishare-market', '-h']
    try:
        rc = cli.main()
        captured = capsys.readouterr()
        assert rc == 0
        assert 'aishare-market' in captured.out
        assert '用法' in captured.out
    finally:
        sys.argv = [sys.argv[0]]


def test_main_invokes_processor_with_stock_code(monkeypatch, capsys):
    """传入 stock_code 时 processor 应收到该参数。"""
    called = {'code': '__NOT_SET__'}

    def fake_generate(self, stock_code=None):
        called['code'] = stock_code
        return "FAKE_REPORT"

    monkeypatch.setattr(
        'AIShareTxt.market.cli.MarketEnvironmentProcessor.generate_market_report',
        fake_generate
    )
    sys.argv = ['aishare-market', '000001']
    try:
        rc = cli.main()
        captured = capsys.readouterr()
        assert rc == 0
        assert called['code'] == '000001'
        assert 'FAKE_REPORT' in captured.out
    finally:
        sys.argv = [sys.argv[0]]


def test_main_invokes_processor_no_code(monkeypatch, capsys):
    called = {'code': '__NOT_SET__'}

    def fake_generate(self, stock_code=None):
        called['code'] = stock_code
        return "MARKET_ONLY"

    monkeypatch.setattr(
        'AIShareTxt.market.cli.MarketEnvironmentProcessor.generate_market_report',
        fake_generate
    )
    sys.argv = ['aishare-market']
    try:
        rc = cli.main()
        captured = capsys.readouterr()
        assert rc == 0
        assert called['code'] is None
        assert 'MARKET_ONLY' in captured.out
    finally:
        sys.argv = [sys.argv[0]]
