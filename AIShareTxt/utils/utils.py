#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具类和辅助函数
提供通用的工具方法和辅助功能
"""

import sys
import os
import logging
import logging.config
import logging.handlers
from datetime import datetime
from ..core.config import IndicatorConfig as Config


class LoggerManager:
    """日志管理器"""
    
    _initialized = False
    
    @classmethod
    def setup_logging(cls, log_level=None, log_to_file=None, log_to_console=None):
        """
        设置日志配置
        
        Args:
            log_level (str): 日志级别
            log_to_file (bool): 是否记录到文件
            log_to_console (bool): 是否显示到控制台
        """
        if cls._initialized:
            return
        
        config = Config()
        
        # 使用参数或默认配置
        if log_level is None:
            log_level = config.DEFAULT_LOG_LEVEL
        if log_to_file is None:
            log_to_file = config.LOG_TO_FILE
        if log_to_console is None:
            log_to_console = config.LOG_TO_CONSOLE
        
        # 创建日志目录
        if log_to_file:
            log_dir = config.LOG_FILE_PATH
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # 更新日志文件路径
            logging_config = config.LOGGING_CONFIG.copy()
            for handler_name, handler_config in logging_config['handlers'].items():
                if 'filename' in handler_config:
                    filename = handler_config['filename']
                    handler_config['filename'] = os.path.join(log_dir, filename)
        else:
            # 如果不记录到文件，移除文件处理器
            logging_config = config.LOGGING_CONFIG.copy()
            for logger_name, logger_config in logging_config['loggers'].items():
                handlers = logger_config.get('handlers', [])
                logger_config['handlers'] = [h for h in handlers if h == 'console']
        
        # 如果不显示到控制台，移除控制台处理器
        if not log_to_console:
            for logger_name, logger_config in logging_config['loggers'].items():
                handlers = logger_config.get('handlers', [])
                logger_config['handlers'] = [h for h in handlers if h != 'console']
        
        # 设置日志级别
        for logger_name, logger_config in logging_config['loggers'].items():
            logger_config['level'] = log_level
        
        # 应用日志配置
        try:
            logging.config.dictConfig(logging_config)
            cls._initialized = True
        except Exception as e:
            # 如果配置失败，使用基本配置
            logging.basicConfig(
                level=getattr(logging, log_level),
                format='[%(levelname)s] %(message)s',
                handlers=[logging.StreamHandler()] if log_to_console else []
            )
            cls._initialized = True
            print(f"警告：日志配置失败，使用基本配置: {e}")
    
    @classmethod
    def get_logger(cls, name=None):
        """
        获取logger实例
        
        Args:
            name (str): logger名称
            
        Returns:
            logging.Logger: logger实例
        """
        if not cls._initialized:
            cls.setup_logging()
        
        if name is None:
            name = 'aishare_txt'
        elif not name.startswith('aishare_txt'):
            name = f'aishare_txt.{name}'
        
        return logging.getLogger(name)
    
    @classmethod
    def set_log_level(cls, level):
        """
        设置日志级别
        
        Args:
            level (str): 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        if not cls._initialized:
            cls.setup_logging()
        
        # 更新所有aishare_txt相关的logger
        for logger_name in ['aishare_txt', 'aishare_txt.data_fetcher',
                           'aishare_txt.technical_indicators', 'aishare_txt.report_generator',
                           'aishare_txt.utils']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(getattr(logging, level.upper()))
    
    @classmethod
    def add_file_handler(cls, filename, level='DEBUG'):
        """
        添加文件处理器
        
        Args:
            filename (str): 文件名
            level (str): 日志级别
        """
        config = Config()
        
        # 创建文件处理器
        handler = logging.handlers.RotatingFileHandler(
            filename=filename,
            maxBytes=config.LOG_MAX_SIZE,
            backupCount=config.LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        handler.setLevel(getattr(logging, level.upper()))
        
        # 添加到所有aishare_txt相关的logger
        for logger_name in ['aishare_txt', 'aishare_txt.data_fetcher',
                           'aishare_txt.technical_indicators', 'aishare_txt.report_generator',
                           'aishare_txt.utils']:
            logger = logging.getLogger(logger_name)
            logger.addHandler(handler)


class Utils:
    """工具类"""
    
    @staticmethod
    def format_number(value, precision=2, default=0):
        """
        安全地格式化数字
        
        Args:
            value: 待格式化的值
            precision (int): 小数位数
            default: 默认值
            
        Returns:
            float: 格式化后的数字
        """
        try:
            if value is None:
                return default
            return round(float(value), precision)
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def safe_divide(numerator, denominator, default=0):
        """
        安全除法，避免除零错误
        
        Args:
            numerator: 分子
            denominator: 分母
            default: 除零时的默认值
            
        Returns:
            float: 除法结果
        """
        try:
            if denominator == 0:
                return default
            return numerator / denominator
        except (TypeError, ZeroDivisionError):
            return default
    
    @staticmethod
    def validate_stock_code(stock_code):
        """
        验证股票代码格式
        
        Args:
            stock_code (str): 股票代码
            
        Returns:
            bool: 是否有效
        """
        if not stock_code or not isinstance(stock_code, str):
            return False
        
        # 去除空格
        stock_code = stock_code.strip()
        
        # 检查长度和数字格式
        if len(stock_code) != 6 or not stock_code.isdigit():
            return False
        
        # 检查首位数字是否合法（0、3、6开头）
        first_digit = stock_code[0]
        if first_digit not in ['0', '3', '6']:
            return False
        
        return True
    
    @staticmethod
    def get_current_time():
        """
        获取当前时间字符串
        
        Returns:
            str: 格式化的当前时间
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def print_progress(message, step=None, total=None):
        """
        打印进度信息
        
        Args:
            message (str): 消息内容
            step (int): 当前步骤
            total (int): 总步骤数
        """
        if step is not None and total is not None:
            progress = f"[{step}/{total}] "
        else:
            progress = ""
        
        print(f"{progress}{message}")
    
    @staticmethod
    def handle_keyboard_interrupt():
        """处理键盘中断"""
        print("\n\n程序被用户中断")
        return True
    
    @staticmethod
    def parse_command_line_args():
        """
        解析命令行参数
        
        Returns:
            str or None: 股票代码，如果没有提供则返回None
        """
        if len(sys.argv) > 1:
            arg = sys.argv[1].strip()
            
            # 处理帮助参数
            if arg in ['-h', '--help', 'help']:
                print("\n股票技术指标分析器 (重构版)")
                print("\n使用方法:")
                print("  aishare [股票代码]     - 分析指定股票")
                print("  aishare -h/--help     - 显示帮助信息")
                print("\n示例:")
                print("  aishare 000001        - 分析平安银行")
                print("  aishare 600036        - 分析招商银行")
                print("\n交互模式:")
                print("  不带参数运行将进入交互模式，可以输入股票代码进行分析")
                print("  输入 'quit' 或 'exit' 退出程序")
                sys.exit(0)
            
            # 处理股票代码
            if Utils.validate_stock_code(arg):
                return arg
            else:
                print(f"错误：股票代码 '{arg}' 格式不正确")
                print("正确格式：6位数字，如 000001")
                print("使用 'aishare --help' 查看更多帮助信息")
                return None
        return None
    
    @staticmethod
    def get_user_input(prompt, quit_words=None):
        """
        获取用户输入
        
        Args:
            prompt (str): 提示信息
            quit_words (list): 退出命令列表
            
        Returns:
            str or None: 用户输入，如果是退出命令则返回None
        """
        if quit_words is None:
            quit_words = ['quit', 'exit', 'q']
        
        user_input = input(prompt).strip()
        
        if user_input.lower() in quit_words:
            return None
        
        return user_input
    
    @staticmethod
    def confirm_action(message):
        """
        确认操作
        
        Args:
            message (str): 确认消息
            
        Returns:
            bool: 是否确认
        """
        response = input(f"{message} (y/N): ").strip().lower()
        return response in ['y', 'yes', '是']
    
    @staticmethod
    def format_large_number(value, unit='万'):
        """
        格式化大数字（转换为万或亿）
        
        Args:
            value (float): 数值
            unit (str): 单位 '万' 或 '亿'
            
        Returns:
            str: 格式化后的字符串
        """
        try:
            if unit == '万':
                return f"{value / 10000:.2f}万"
            elif unit == '亿':
                return f"{value / 100000000:.2f}亿"
            else:
                return f"{value:.2f}"
        except (TypeError, ValueError):
            return "0"
    
    @staticmethod
    def check_data_quality(data, min_length=50):
        """
        检查数据质量
        
        Args:
            data: 数据对象
            min_length (int): 最小数据长度
            
        Returns:
            tuple: (是否通过检查, 错误消息)
        """
        if data is None:
            return False, "数据为空"
        
        if hasattr(data, '__len__'):
            if len(data) < min_length:
                return False, f"数据长度不足，需要至少{min_length}条记录，当前只有{len(data)}条"
        
        return True, "数据质量检查通过"
    
    @staticmethod
    def log_error(error, context=""):
        """
        记录错误信息
        
        Args:
            error: 错误对象或消息
            context (str): 错误上下文
        """
        logger = LoggerManager.get_logger('utils')
        if context:
            logger.error(f"错误在 {context}: {str(error)}")
        else:
            logger.error(f"错误: {str(error)}")
    
    @staticmethod
    def create_separator(char='=', length=60):
        """
        创建分隔符
        
        Args:
            char (str): 分隔符字符
            length (int): 长度
            
        Returns:
            str: 分隔符字符串
        """
        return char * length


class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_price_data(data):
        """
        验证价格数据
        
        Args:
            data: 价格数据
            
        Returns:
            tuple: (是否有效, 错误消息)
        """
        required_columns = ['open', 'close', 'high', 'low', 'volume']
        
        if data is None:
            return False, "数据为空"
        
        # 检查必需列
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"缺少必需列: {missing_columns}"
        
        # 检查数据完整性
        for col in required_columns:
            if data[col].isnull().any():
                return False, f"列 {col} 包含空值"
        
        # 基本数据完整性检查（数据已经在获取阶段被清洗）
        if len(data) == 0:
            return False, "数据为空"
        
        # 检查是否有足够的数据用于技术分析
        if len(data) < 20:
            return False, f"数据量不足，至少需要20条记录进行技术分析，当前只有{len(data)}条"
        
        return True, "价格数据验证通过"
    
    @staticmethod
    def validate_indicators(indicators):
        """
        验证指标数据
        
        Args:
            indicators (dict): 指标字典
            
        Returns:
            tuple: (是否有效, 错误消息)
        """
        if not indicators:
            return False, "指标数据为空"
        
        # 检查关键指标是否存在
        key_indicators = ['current_price', 'date']
        missing_indicators = [key for key in key_indicators if key not in indicators]
        if missing_indicators:
            return False, f"缺少关键指标: {missing_indicators}"
        
        # 检查价格指标的合理性
        current_price = indicators.get('current_price', 0)
        if current_price <= 0:
            return False, "当前价格必须为正数"
        
        return True, "指标数据验证通过"


class ErrorHandler:
    """错误处理器"""
    
    @staticmethod
    def handle_api_error(error, api_name="API"):
        """
        处理API错误
        
        Args:
            error: 错误对象
            api_name (str): API名称
        """
        logger = LoggerManager.get_logger('utils')
        error_msg = str(error)
        logger.error(f"{api_name}调用失败: {error_msg}")
        
        # 提供常见错误的解决建议
        if "network" in error_msg.lower() or "connection" in error_msg.lower():
            logger.info("建议：检查网络连接")
        elif "timeout" in error_msg.lower():
            logger.info("建议：稍后重试，可能是网络超时")
        elif "rate" in error_msg.lower() or "limit" in error_msg.lower():
            logger.info("建议：请求频率过高，稍后重试")
        elif "auth" in error_msg.lower():
            logger.info("建议：检查API认证信息")
        else:
            logger.info("建议：检查输入参数或稍后重试")
    
    @staticmethod
    def handle_calculation_error(error, calculation_name="计算"):
        """
        处理计算错误
        
        Args:
            error: 错误对象
            calculation_name (str): 计算名称
        """
        logger = LoggerManager.get_logger('utils')
        error_msg = str(error)
        logger.error(f"{calculation_name}失败: {error_msg}")
        
        if "insufficient" in error_msg.lower() or "not enough" in error_msg.lower():
            logger.info("建议：数据量不足，需要更多历史数据")
        elif "nan" in error_msg.lower() or "invalid" in error_msg.lower():
            logger.info("建议：数据包含无效值，请检查数据质量")
        else:
            logger.info("建议：检查数据格式和完整性")
    
    @staticmethod
    def handle_file_error(error, operation="文件操作"):
        """
        处理文件错误
        
        Args:
            error: 错误对象
            operation (str): 操作名称
        """
        logger = LoggerManager.get_logger('utils')
        error_msg = str(error)
        logger.error(f"{operation}失败: {error_msg}")
        
        if "permission" in error_msg.lower():
            logger.info("建议：检查文件权限")
        elif "not found" in error_msg.lower():
            logger.info("建议：检查文件路径是否正确")
        elif "disk" in error_msg.lower() or "space" in error_msg.lower():
            logger.info("建议：检查磁盘空间")
        else:
            logger.info("建议：检查文件路径和权限")


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.start_time = None
        self.checkpoints = {}
    
    def start(self):
        """开始计时"""
        logger = LoggerManager.get_logger('utils')
        self.start_time = datetime.now()
        logger.info(f"性能监控开始: {self.start_time.strftime('%H:%M:%S')}")
    
    def checkpoint(self, name):
        """记录检查点"""
        logger = LoggerManager.get_logger('utils')
        if self.start_time is None:
            logger.warning("请先调用start()方法")
            return
        
        current_time = datetime.now()
        elapsed = (current_time - self.start_time).total_seconds()
        self.checkpoints[name] = elapsed
        logger.info(f"检查点 [{name}]: {elapsed:.2f}秒")
    
    def finish(self):
        """结束计时"""
        logger = LoggerManager.get_logger('utils')
        if self.start_time is None:
            logger.warning("请先调用start()方法")
            return
        
        end_time = datetime.now()
        total_elapsed = (end_time - self.start_time).total_seconds()
        logger.info(f"性能监控结束: {end_time.strftime('%H:%M:%S')}")
        logger.info(f"总耗时: {total_elapsed:.2f}秒")
        
        if self.checkpoints:
            logger.info("\n检查点详情:")
            for name, elapsed in self.checkpoints.items():
                percentage = (elapsed / total_elapsed) * 100
                logger.info(f"  {name}: {elapsed:.2f}秒 ({percentage:.1f}%)")

        return total_elapsed


# 向后兼容的别名
Logger = LoggerManager
