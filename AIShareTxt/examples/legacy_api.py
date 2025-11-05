#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票技术指标分析器 (兼容性包装器)
保持原有接口不变，但内部使用重构后的模块化架构
支持多种技术指标计算，包括均线、MACD、布林带等
使用 akshare 获取数据，talib 计算指标
"""

import warnings
from ..core.analyzer import StockAnalyzer
from ..utils.utils import Utils, LoggerManager

warnings.filterwarnings('ignore')


class StockIndicatorsAnalyzer:
    """股票技术指标分析器 (兼容性包装器)"""
    
    def __init__(self):
        """初始化分析器，使用新的模块化架构"""
        self._analyzer = StockAnalyzer()
        self.logger = LoggerManager.get_logger('compatibility_wrapper')
        
        # 保持向后兼容的属性
        self.data = None
        self.stock_code = None
        self.stock_info = None
        
    def fetch_stock_data(self, stock_code, period="daily", adjust="qfq", start_date=None):
        """
        获取股票数据 (兼容性方法)
        
        Args:
            stock_code (str): 股票代码，如 '000001'
            period (str): 周期，'daily'=日K线
            adjust (str): 复权类型，''=不复权, 'qfq'=前复权, 'hfq'=后复权
            start_date (str): 开始日期，如 '20230101'
        
        Returns:
            bool: 是否成功获取数据
        """
        try:
            self.data = self._analyzer.data_fetcher.fetch_stock_data(
                stock_code, period, adjust, start_date
            )
            
            if self.data is not None:
                self.stock_code = stock_code
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"获取股票数据时出错：{str(e)}")
            return False
    
    def get_fund_flow_data(self, stock_code):
        """
        获取主力资金流数据 (兼容性方法)
        
        Args:
            stock_code (str): 股票代码
            
        Returns:
            dict: 资金流数据字典
        """
        try:
            return self._analyzer.data_fetcher.get_fund_flow_data(stock_code)
        except Exception as e:
            self.logger.error(f"获取主力资金流数据失败：{str(e)}")
            return {}
    
    def get_stock_basic_info(self, stock_code):
        """
        获取股票基本信息 (兼容性方法)
        
        Args:
            stock_code (str): 股票代码
            
        Returns:
            dict: 股票基本信息字典
        """
        try:
            self.stock_info = self._analyzer.data_fetcher.get_stock_basic_info(stock_code)
            return self.stock_info
        except Exception as e:
            self.logger.error(f"获取股票基本信息失败：{str(e)}")
            return {}
    
    def calculate_bias(self, timeperiod=20):
        """
        计算乖离率 (兼容性方法)
        
        Args:
            timeperiod (int): 计算周期
            
        Returns:
            array: 乖离率数组
        """
        if self.data is None:
            return None
            
        try:
            close = self.data['close'].values
            return self._analyzer.indicators_calculator.calculate_bias(close, timeperiod)
        except Exception as e:
            self.logger.error(f"计算乖离率失败：{str(e)}")
            return None
    
    def analyze_ma_patterns(self, close):
        """
        分析均线形态 (兼容性方法)
        
        Args:
            close: 收盘价数组
            
        Returns:
            dict: 均线形态字典
        """
        try:
            return self._analyzer.indicators_calculator.analyze_ma_patterns(close)
        except Exception as e:
            self.logger.error(f"分析均线形态失败：{str(e)}")
            return {}

    def get_latest_indicators(self):
        """
        计算所有技术指标的最新值 (兼容性方法)
        
        Returns:
            dict: 包含所有指标最新值的字典
        """
        if self.data is None:
            return None
            
        try:
            return self._analyzer.indicators_calculator.calculate_all_indicators(self.data)
        except Exception as e:
            self.logger.error(f"计算技术指标失败：{str(e)}")
            return None
    
    def format_indicators_report(self, indicators):
        """
        格式化指标报告 (兼容性方法)
        
        Args:
            indicators (dict): 指标数据字典
            
        Returns:
            str: 格式化的报告文本
        """
        if indicators is None:
            return "无法生成指标报告：数据不足"
        
        try:
            return self._analyzer.report_generator.generate_report(
                self.stock_code, indicators, self.stock_info
            )
        except Exception as e:
            self.logger.error(f"生成报告失败：{str(e)}")
            return f"生成报告时出错：{str(e)}"
    
    def analyze_stock(self, stock_code):
        """
        分析指定股票的技术指标 (兼容性方法)
        
        Args:
            stock_code (str): 股票代码
            
        Returns:
            str: 分析报告
        """
        try:
            # 使用新的分析器
            report = self._analyzer.analyze_stock(stock_code)
            
            # 更新兼容性属性
            self.stock_code = stock_code
            self.data = self._analyzer.get_current_stock_data()
            self.stock_info = self._analyzer.get_current_stock_info()
            
            return report
            
        except Exception as e:
            self.logger.error(f"分析股票失败：{str(e)}")
            return f"分析过程中出错：{str(e)}"


def quick_test(stock_code):
    """快速测试单个股票 (兼容性函数)"""
    analyzer = StockIndicatorsAnalyzer()
    logger = LoggerManager.get_logger('compatibility_wrapper')
    
    print(f"快速测试股票：{stock_code}")
    print("=" * 50)
    logger.info(f"开始兼容性模式快速测试：{stock_code}")
    
    try:
        report = analyzer.analyze_stock(stock_code)
        print("\n" + report)
        logger.info(f"兼容性模式测试完成：{stock_code}")
        return True
    except Exception as e:
        logger.error(f"兼容性模式测试失败：{str(e)}")
        return False


def main():
    """主函数 (兼容性版本)"""
    logger = LoggerManager.get_logger('compatibility_wrapper')
    
    print("股票技术指标分析器 (兼容性版本)")
    print("=" * 50)
    print("注意：此版本使用重构后的模块化架构")
    print("推荐使用新版本：python stock_analyzer.py")
    print("=" * 50)
    logger.info("启动兼容性版本分析器")
    
    analyzer = StockIndicatorsAnalyzer()
    utils = Utils()
    
    # 检查命令行参数
    stock_code = utils.parse_command_line_args()
    if stock_code:
        quick_test(stock_code)
        return
    
    while True:
        try:
            stock_code = utils.get_user_input(
                "\n请输入股票代码（如：000001，输入 'quit' 退出）：",
                ['quit', 'exit', 'q']
            )
            
            if stock_code is None:
                print("谢谢使用！")
                logger.info("用户退出兼容性版本程序")
                break
            
            if not stock_code:
                print("请输入有效的股票代码")
                continue
            
            logger.info(f"兼容性版本分析股票：{stock_code}")
            
            # 分析股票
            report = analyzer.analyze_stock(stock_code)
            print("\n" + report)
            
        except KeyboardInterrupt:
            if utils.handle_keyboard_interrupt():
                break
        except Exception as e:
            utils.log_error(e, "主程序")


if __name__ == "__main__":
    main()
