#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIShareTxt 快速开始示例

这个示例展示了如何使用AIShareTxt进行基本的股票分析
"""

from AIShareTxt import StockAnalyzer, analyze_stock

def basic_example():
    """基本使用示例"""
    print("=" * 60)
    print("AIShareTxt 基本使用示例")
    print("=" * 60)

    # 方法1：使用StockAnalyzer类
    print("\n1. 使用StockAnalyzer类进行分析")
    analyzer = StockAnalyzer()

    try:
        report = analyzer.analyze_stock("000001")  # 平安银行
        print("   分析成功！报告长度:", len(report), "字符")
        print("   报告预览（前200字符）:")
        print("   " + report[:200] + "...")
    except Exception as e:
        print(f"   分析失败: {e}")

    # 方法2：使用便捷函数
    print("\n2. 使用便捷函数进行分析")
    try:
        report = analyze_stock("000001")
        print("   分析成功！报告长度:", len(report), "字符")
    except Exception as e:
        print(f"   分析失败: {e}")

def technical_indicators_example():
    """技术指标计算示例"""
    print("\n" + "=" * 60)
    print("技术指标计算示例")
    print("=" * 60)

    from AIShareTxt.indicators.technical_indicators import TechnicalIndicators
    import pandas as pd
    import numpy as np

    # 创建技术指标计算器
    ti = TechnicalIndicators()

    # 创建测试数据
    data = pd.DataFrame({
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(200, 300, 100),
        'low': np.random.uniform(50, 100, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    })

    print(f"\n生成测试数据: {len(data)} 条记录")

    # 计算单个指标
    try:
        bias = ti.calculate_bias(data['close'], timeperiod=20)
        print(f"BIAS指标计算成功，最新值: {bias.iloc[-1]:.4f}")
    except Exception as e:
        print(f"BIAS计算失败: {e}")

def stock_list_example():
    """股票列表获取示例"""
    print("\n" + "=" * 60)
    print("股票列表获取示例")
    print("=" * 60)

    from AIShareTxt.utils.stock_list import get_stock_list

    try:
        stocks = get_stock_list()
        if stocks is not None:
            print(f"\n获取股票列表成功！")
            print(f"   总计: {len(stocks)} 只股票")
            print(f"   前5只股票:")
            for idx, stock in stocks.head().iterrows():
                print(f"     {stock['代码']} - {stock['名称']}")
        else:
            print("获取股票列表失败：返回None")
    except Exception as e:
        print(f"获取股票列表失败: {e}")

def ai_analysis_example():
    """AI分析示例"""
    print("\n" + "=" * 60)
    print("AI分析示例")
    print("=" * 60)

    from AIShareTxt.ai.client import AIClient

    # 创建AI客户端（需要配置API密钥）
    try:
        ai_client = AIClient()

        if ai_client.is_available():
            print(f"AI客户端可用，提供商: {ai_client.provider}")
            print("注意：进行实际AI分析需要配置API密钥")
        else:
            print("AI客户端不可用，需要配置API密钥")
    except Exception as e:
        print(f"AI客户端初始化失败: {e}")

def main():
    """主函数"""
    print("AIShareTxt 快速开始示例")
    print("这个示例将展示项目的主要功能")

    # 运行各种示例
    basic_example()
    technical_indicators_example()
    stock_list_example()
    ai_analysis_example()

    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60)
    print("\n更多详细信息请查看 README.md 文档")

if __name__ == "__main__":
    main()