#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重构后的股票分析器主模块
作为协调器，整合数据获取、指标计算和报告生成功能
"""

import warnings
from .data_fetcher import StockDataFetcher
from ..indicators.technical_indicators import TechnicalIndicators
from .report_generator import ReportGenerator
from ..utils.utils import Logger, LoggerManager, Utils, DataValidator, ErrorHandler
from .config import IndicatorConfig as Config

warnings.filterwarnings('ignore')


class StockAnalyzer:
    """重构后的股票分析器主类"""
    
    def __init__(self):
        """初始化分析器组件"""
        # 初始化日志系统
        LoggerManager.setup_logging()
        self.logger = LoggerManager.get_logger('stock_analyzer')
        
        self.config = Config()
        self.data_fetcher = StockDataFetcher()
        self.indicators_calculator = TechnicalIndicators()
        self.report_generator = ReportGenerator()
        self.utils = Utils()
        
        # 数据存储
        self.stock_data = None
        self.stock_code = None
        self.stock_info = None
        self.fund_flow_data = None
        self.indicators = None
        
    def analyze_stock(self, stock_code, enable_performance_monitor=False):
        """
        分析指定股票的技术指标
        
        Args:
            stock_code (str): 股票代码
            enable_performance_monitor (bool): 是否启用性能监控
            
        Returns:
            str: 分析报告
        """
        # 性能监控
        monitor = None
        if enable_performance_monitor:
            monitor = PerformanceMonitor()
            monitor.start()
        
        try:
            self.logger.info(f"开始分析股票：{stock_code}")
            
            # 验证股票代码
            if not self.utils.validate_stock_code(stock_code):
                error_msg = "错误：股票代码格式不正确，请使用6位数字格式（如：000001）"
                self.logger.error(error_msg)
                return error_msg
            
            self.stock_code = stock_code
            
            # 步骤1：获取股票基本信息
            self.logger.info("步骤1/4：获取股票基本信息...")
            self.stock_info = self.data_fetcher.get_stock_basic_info(stock_code)
            if monitor:
                monitor.checkpoint("获取基本信息")
            
            # 步骤2：获取主力资金流数据
            self.logger.info("步骤2/4：获取主力资金流数据...")
            self.fund_flow_data = self.data_fetcher.get_fund_flow_data(stock_code)
            if monitor:
                monitor.checkpoint("获取资金流数据")
            
            # 步骤3：获取股票价格数据
            self.logger.info("步骤3/4：获取股票价格数据...")
            self.stock_data = self.data_fetcher.fetch_stock_data(stock_code)
            
            if self.stock_data is None:
                return f"{self.config.ERROR_MESSAGES['no_data']}: {stock_code}"
            
            # 验证数据质量
            is_valid, error_msg = DataValidator.validate_price_data(self.stock_data)
            if not is_valid:
                return f"数据质量验证失败: {error_msg}"
            
            # 检查数据长度
            min_length = self.config.DATA_CONFIG.get('min_data_length', 250)
            is_sufficient, length_msg = self.utils.check_data_quality(self.stock_data, min_length)
            if not is_sufficient:
                self.logger.warning(f"警告: {length_msg}")
                
            if monitor:
                monitor.checkpoint("获取价格数据")
            
            # 步骤4：计算技术指标
            self.logger.info("步骤4/4：计算技术指标...")
            self.indicators = self.indicators_calculator.calculate_all_indicators(self.stock_data)
            
            if self.indicators is None:
                return f"{self.config.ERROR_MESSAGES['calculation_failed']}: 可能是数据不足"
            
            # 验证指标数据
            is_valid, error_msg = DataValidator.validate_indicators(self.indicators)
            if not is_valid:
                return f"指标验证失败: {error_msg}"
            
            # 将资金流数据添加到指标中
            if self.fund_flow_data:
                self.indicators.update(self.fund_flow_data)
            
            if monitor:
                monitor.checkpoint("计算技术指标")
            
            # 生成报告
            self.logger.info("生成分析报告...")
            report = self.report_generator.generate_report(
                stock_code, 
                self.indicators, 
                self.stock_info
            )
            
            if monitor:
                monitor.checkpoint("生成报告")
                monitor.finish()
            
            return report
            
        except Exception as e:
            error_context = f"分析股票 {stock_code}"
            self.utils.log_error(e, error_context)
            ErrorHandler.handle_api_error(e, "股票分析")
            return f"分析过程中出错：{str(e)}"
    
    def quick_analyze(self, stock_code):
        """
        快速分析（不包含资金流数据，提高速度）
        
        Args:
            stock_code (str): 股票代码
            
        Returns:
            str: 分析报告
        """
        try:
            self.logger.info(f"快速分析股票：{stock_code}")
            
            # 验证股票代码
            if not self.utils.validate_stock_code(stock_code):
                error_msg = "错误：股票代码格式不正确"
                self.logger.error(error_msg)
                return error_msg
            
            self.stock_code = stock_code
            
            # 获取股票价格数据
            self.stock_data = self.data_fetcher.fetch_stock_data(stock_code)
            
            if self.stock_data is None:
                return f"{self.config.ERROR_MESSAGES['no_data']}: {stock_code}"
            
            # 计算技术指标
            self.indicators = self.indicators_calculator.calculate_all_indicators(self.stock_data)
            
            if self.indicators is None:
                return f"{self.config.ERROR_MESSAGES['calculation_failed']}"
            
            # 生成报告（不包含基本信息和资金流）
            report = self.report_generator.generate_report(
                stock_code, 
                self.indicators, 
                None  # 不包含基本信息
            )
            
            return report
            
        except Exception as e:
            error_context = f"快速分析股票 {stock_code}"
            self.utils.log_error(e, error_context)
            return f"快速分析过程中出错：{str(e)}"
    
    def get_current_indicators(self):
        """
        获取当前计算的指标数据
        
        Returns:
            dict: 指标数据字典
        """
        return self.indicators
    
    def get_current_stock_data(self):
        """
        获取当前的股票数据
        
        Returns:
            pd.DataFrame: 股票数据
        """
        return self.stock_data
    
    def get_current_stock_info(self):
        """
        获取当前的股票基本信息
        
        Returns:
            dict: 股票基本信息
        """
        return self.stock_info
    
    def export_indicators_to_dict(self):
        """
        导出指标数据为字典格式
        
        Returns:
            dict: 包含所有数据的字典
        """
        if not self.indicators:
            return None
        
        # 为了确保JSON序列化安全，对所有数据进行清理
        export_data = {
            'stock_code': self.stock_code,
            'analysis_time': self.utils.get_current_time(),
            'stock_info': self._serialize_data(self.stock_info),
            'indicators': self._serialize_data(self.indicators),
            'fund_flow': self._serialize_data(self.fund_flow_data)
        }
        
        return export_data
    
    def _serialize_data(self, data):
        """
        将数据中的日期对象转换为字符串，确保JSON序列化安全
        
        Args:
            data: 需要序列化的数据
            
        Returns:
            序列化后的数据
        """
        if data is None:
            return None
            
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                result[key] = self._serialize_data(value)
            return result
        elif isinstance(data, (list, tuple)):
            return [self._serialize_data(item) for item in data]
        elif hasattr(data, 'strftime'):  # 日期对象
            return data.strftime('%Y-%m-%d')
        elif hasattr(data, 'isoformat'):  # datetime对象
            return data.isoformat()
        else:
            return data
    
    def validate_analysis_environment(self):
        """
        验证分析环境
        
        Returns:
            tuple: (是否通过验证, 错误消息列表)
        """
        errors = []
        
        try:
            # 检查必要的包
            import akshare as ak
            import talib
            import pandas as pd
            import numpy as np
        except ImportError as e:
            errors.append(f"缺少必要的包: {str(e)}")
        
        # 检查配置
        if not hasattr(self.config, 'MA_PERIODS'):
            errors.append("配置文件不完整")
        
        # 检查组件初始化
        if not self.data_fetcher:
            errors.append("数据获取器初始化失败")
        
        if not self.indicators_calculator:
            errors.append("指标计算器初始化失败")
        
        if not self.report_generator:
            errors.append("报告生成器初始化失败")
        
        return len(errors) == 0, errors
    
    def get_supported_indicators(self):
        """
        获取支持的指标列表
        
        Returns:
            dict: 按类别组织的指标列表
        """
        indicators_info = {
            '移动平均线': {
                'MA': f"周期: {self.config.MA_PERIODS['short'] + self.config.MA_PERIODS['medium'] + self.config.MA_PERIODS['long']}",
                'EMA': f"周期: {self.config.EMA_PERIODS}",
                'WMA': f"周期: {self.config.WMA_PERIODS}"
            },
            '均线衍生指标': {
                'BIAS': f"乖离率，周期: {self.config.BIAS_PERIODS}",
                'MACD': f"参数: {self.config.MACD_CONFIG}",
                'Bollinger Bands': f"参数: {self.config.BOLLINGER_BANDS_CONFIG}"
            },
            '量价指标': {
                'VWMA': f"成交量加权移动平均，周期: {self.config.VWMA_PERIOD}",
                'OBV': "能量潮指标"
            },
            '趋势强度': {
                'ADX/DMI': f"趋势方向指标，周期: {self.config.ADX_PERIOD}"
            },
            '动量振荡': {
                'RSI': f"相对强弱指数，周期: {self.config.RSI_PERIODS}",
                'Stochastic': f"随机振荡器，参数: {self.config.STOCH_CONFIG}"
            },
            '波动率': {
                'ATR': f"平均真实波幅，周期: {self.config.ATR_PERIOD}"
            },
            '成交量': {
                'Volume Ratio': "量比指标",
                'Volume Trend': "成交量趋势"
            },
            '资金流': {
                'Fund Flow': "主力资金流向",
                '5-day Fund Flow': "5日累计资金流"
            }
        }
        
        return indicators_info
    
    def get_analysis_summary(self):
        """
        获取分析摘要
        
        Returns:
            dict: 分析摘要信息
        """
        if not self.indicators:
            return None
        
        summary = {
            'stock_code': self.stock_code,
            'current_price': self.indicators.get('current_price', 0),
            'analysis_date': self.indicators.get('date', ''),
            'data_points': len(self.stock_data) if self.stock_data is not None else 0,
            'indicators_count': len(self.indicators),
            'has_fund_flow': '主力净流入额' in self.indicators,
            'has_basic_info': self.stock_info is not None
        }
        
        return summary


def quick_test(stock_code):
    """快速测试单个股票的便捷函数"""
    analyzer = StockAnalyzer()
    logger = LoggerManager.get_logger('stock_analyzer')
    
    print(f"快速测试股票：{stock_code}")
    print(Utils.create_separator())
    logger.info(f"开始快速测试股票：{stock_code}")
    
    try:
        # 验证分析环境
        is_valid, errors = analyzer.validate_analysis_environment()
        if not is_valid:
            print("环境验证失败:")
            logger.error("环境验证失败")
            for error in errors:
                print(f"  - {error}")
                logger.error(f"环境错误: {error}")
            return False
        
        # 执行分析
        report = analyzer.analyze_stock(stock_code, enable_performance_monitor=True)
        print("\n" + report)
        
        # 显示分析摘要
        summary = analyzer.get_analysis_summary()
        if summary:
            print(f"\n分析摘要:")
            print(f"  数据点数: {summary['data_points']}")
            print(f"  指标数量: {summary['indicators_count']}")
            print(f"  包含资金流: {'是' if summary['has_fund_flow'] else '否'}")
            print(f"  包含基本信息: {'是' if summary['has_basic_info'] else '否'}")
            logger.info(f"快速测试完成 - 数据点数: {summary['data_points']}, 指标数量: {summary['indicators_count']}")
        
        logger.info(f"快速测试股票 {stock_code} 成功完成")
        return True
    except Exception as e:
        Utils.log_error(e, f"测试股票 {stock_code}")
        return False


def main():
    """主函数"""
    analyzer = StockAnalyzer()
    utils = Utils()
    logger = LoggerManager.get_logger('stock_analyzer')
    
    print("股票技术指标分析器 (重构版)")
    print(utils.create_separator())
    print("提示：可以在命令行直接运行 python stock_analyzer.py 000001 来快速测试")
    logger.info("股票技术指标分析器启动")
    
    # 验证分析环境
    print("\n验证分析环境...")
    logger.info("开始验证分析环境")
    is_valid, errors = analyzer.validate_analysis_environment()
    if not is_valid:
        print("环境验证失败:")
        logger.error("环境验证失败")
        for error in errors:
            print(f"  ❌ {error}")
            logger.error(f"环境错误: {error}")
        return
    else:
        print("✅ 环境验证通过")
        logger.info("环境验证通过")
    
    # 检查命令行参数
    stock_code = utils.parse_command_line_args()
    if stock_code:
        logger.info(f"使用命令行参数启动快速测试: {stock_code}")
        quick_test(stock_code)
        return
    
    # 交互模式
    while True:
        try:
            print(f"\n{utils.create_separator('-', 40)}")
            stock_code = utils.get_user_input(
                "请输入股票代码（如：000001，输入 'quit' 退出）：",
                ['quit', 'exit', 'q']
            )
            
            if stock_code is None:
                print("谢谢使用！")
                logger.info("用户退出程序")
                break
            
            if not stock_code:
                print("请输入有效的股票代码")
                continue
            
            logger.info(f"用户输入股票代码: {stock_code}")
            
            # 询问分析模式
            print("\n选择分析模式:")
            print("1. 完整分析（包含基本信息和资金流）")
            print("2. 快速分析（仅技术指标）")
            
            mode_choice = utils.get_user_input("请选择模式 (1/2): ")
            
            if mode_choice == '2':
                print("\n执行快速分析...")
                logger.info(f"开始快速分析股票: {stock_code}")
                report = analyzer.quick_analyze(stock_code)
            else:
                print("\n执行完整分析...")
                logger.info(f"开始完整分析股票: {stock_code}")
                report = analyzer.analyze_stock(stock_code, enable_performance_monitor=True)
            
            print("\n" + report)
            
            # 显示分析摘要
            summary = analyzer.get_analysis_summary()
            if summary:
                print(f"\n{utils.create_separator('-', 30)}")
                print("分析摘要:")
                print(f"  股票代码: {summary['stock_code']}")
                print(f"  当前价格: {summary['current_price']:.2f}")
                print(f"  数据点数: {summary['data_points']}")
                print(f"  指标数量: {summary['indicators_count']}")
                logger.info(f"分析完成 - {summary['stock_code']}: 数据点{summary['data_points']}, 指标{summary['indicators_count']}")
            
        except KeyboardInterrupt:
            if utils.handle_keyboard_interrupt():
                logger.info("程序被用户中断")
                break
        except Exception as e:
            utils.log_error(e, "主程序")


if __name__ == "__main__":
    main()
