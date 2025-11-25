#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票数据处理器配置文件
"""

import os

class IndicatorConfig:
    """股票数据处理配置类，包含所有硬编码的常量和阈值"""
    
    # 均线周期配置
    MA_PERIODS = {
        'short': [5, 10, 20, 30],
        'medium': [60],
        'long': []
    }
    
    # EMA周期配置
    EMA_PERIODS = [5, 10, 12, 20, 26]
    
    # WMA周期配置
    WMA_PERIODS = [10, 20]
    
    # MACD配置
    MACD_CONFIG = {
        'fastperiod': 12,
        'slowperiod': 26,
        'signalperiod': 9
    }
    
    # 布林带配置
    BOLLINGER_BANDS_CONFIG = {
        'timeperiod': 20,
        'nbdevup': 2,
        'nbdevdn': 2,
        'matype': 0
    }
    
    # RSI配置
    RSI_PERIODS = [9, 14]
    
    # KD指标配置
    STOCH_CONFIG = {
        'fastk_period': 9,
        'slowk_period': 3,
        'slowk_matype': 0,
        'slowd_period': 3,
        'slowd_matype': 0
    }
    
    # ADX/DMI配置
    ADX_PERIOD = 14
    
    # ATR配置
    ATR_PERIOD = 14
    
    # VWMA配置
    VWMA_PERIOD = 14
    
    # BIAS乖离率配置
    BIAS_PERIODS = [5, 10, 20]
    
    # OBV分析配置
    OBV_CONFIG = {
        'short_period': 5,
        'medium_period': 20,
        'extrema_order': 3  # 局部极值检测的最小间隔
    }
    
    # 均线形态分析阈值
    MA_THRESHOLDS = {
        'support_resistance': 0.02,  # 2%以内视为支撑/阻力
        'divergence_strong': 0.02,   # 2%以上为强发散
        'divergence_weak': 0.01,     # 1%以下为弱发散
        'adhesion_tight': 0.003,     # 0.3%以下为紧密粘合
        'adhesion_loose': 0.005,     # 0.5%以下为一般粘合
        'adhesion_range': 0.01       # 1%以下为粘合范围
    }
    
    # RSI区间定义
    RSI_ZONES = {
        'overbought': 70,
        'neutral_high': 50,
        'neutral_low': 30,
        'oversold': 0
    }
    
    # ADX强度等级
    ADX_LEVELS = {
        'strong_trend': 40,
        'medium_trend': 25,
        'weak_trend': 0
    }
    
    # 资金流强度等级
    FUND_FLOW_LEVELS = {
        'strong': 5.0,    # 5%以上为强烈
        'active': 2.0,    # 2%以上为活跃
        'moderate': 1.0,  # 1%以上为温和
        'weak': 0         # 1%以下为平淡
    }
    
    # 波动率等级
    VOLATILITY_LEVELS = {
        'high': 5.0,      # 5%以上为高波动
        'medium': 3.0,    # 3-5%为中等波动
        'low': 1.0,       # 1-3%为低波动
        'very_low': 0     # 1%以下为极低波动
    }
    
    # 量比等级
    VOLUME_RATIO_LEVELS = {
        'heavy': 2.0,     # 2倍以上为放量
        'moderate': 1.5,  # 1.5-2倍为温和放量
        'normal': 0.8,    # 0.8-1.5倍为正常
        'light': 0        # 0.8倍以下为缩量
    }
    
    # 市值规模分类（单位：亿元）
    MARKET_CAP_LEVELS = {
        'mega': 1000,     # 超大盘股
        'large': 300,     # 大盘股
        'medium': 100,    # 中盘股
        'small_medium': 50,  # 中小盘股
        'small': 0        # 小盘股
    }
    
    # 显示精度配置
    DISPLAY_PRECISION = {
        'price': 2,       # 价格显示2位小数
        'percentage': 2,  # 百分比显示2位小数
        'ratio': 2,       # 比率显示2位小数
        'volume_wan': 0,  # 成交量（万元）显示整数
        'macd': 4,        # MACD显示4位小数
        'atr': 4          # ATR显示4位小数
    }
    
    # 数据获取配置
    DATA_CONFIG = {
        'default_period': 'daily',
        'default_adjust': 'qfq',    # 前复权
        'required_columns': ['date', 'open', 'close', 'high', 'low', 'volume'],
        'min_data_length': 60,      # 最少需要60个交易日数据
        'default_months_back': 4    # 默认获取往前推4个月的数据
    }
    
    # 列名映射
    COLUMN_MAPPING = {
        '日期': 'date',
        '开盘': 'open', 
        '收盘': 'close',
        '最高': 'high',
        '最低': 'low',
        '成交量': 'volume',
        '成交额': 'turnover',
        '振幅': 'amplitude',
        '涨跌幅': 'pct_chg',
        '涨跌额': 'change',
        '换手率': 'turnover_rate'
    }
    
    # 报告格式配置
    REPORT_CONFIG = {
        'title_separator': '=' * 60,
        'section_separator': '-' * 40,
        'min_fund_display_threshold': 100  # 超过100万才显示的资金流阈值（万元）
    }
    
    # 错误消息
    ERROR_MESSAGES = {
        'no_data': "错误：无法获取股票数据",
        'invalid_code': "错误：股票代码格式错误",
        'network_error': "错误：网络连接问题",
        'api_error': "错误：数据接口暂时不可用",
        'data_insufficient': "错误：数据不足，无法处理指标",
        'calculation_failed': "错误：指标处理失败"
    }
    
    # 市场代码映射
    MARKET_MAPPING = {
        '6': 'sh',  # 上海市场
        '0': 'sz',  # 深圳市场
        '3': 'sz'   # 深圳市场（创业板）
    }
    
    # AI配置
    AI_CONFIG = {
        # 默认使用的AI提供商
        'default_provider': 'deepseek',  # 可选: 'deepseek', 'zhipuai'
        
        # DeepSeek AI配置
        'deepseek': {
            'api_key': os.environ.get('DEEPSEEK_API_KEY'),  # DeepSeek API密钥（从环境变量获取）
            'base_url': 'https://api.deepseek.com',  # DeepSeek API地址
            'model': 'deepseek-chat',  # 使用的模型
            'max_tokens': 100,       # 最大输出token数
            'temperature': 0.6       # 控制随机性，降低以提高一致性
        },
        
        # 智谱AI配置
        'zhipuai': {
            'api_key': os.environ.get('ZHIPUAI_API_KEY'),  # 智谱AI API密钥
            'model': 'glm-4.5-flash',  # 使用的模型
            'max_tokens': 100,       # 最大输出token数
            'temperature': 0.6       # 控制随机性，降低以提高一致性
        },
        
        # 数据处理建议配置
        'recommendation': {
            'prompt_template': """
我打算以最新价格买入，在三个交易日内卖出。基于以下股票技术数据报告，评估数据质量和完整性，返回结果可以是1:数据质量良好可操作、-1:数据质量不佳建议观望、0:数据不充分需要更多观察这三种，请不要返回其他内容,仅返回对应的数字。

股票代码: {stock_code}
技术数据报告:
{technical_report}
""",
            'system_message': '你是一个专业的中国股票数据分析师。',
            # 结果映射（确保符合项目规范）
            'valid_scores': [1, -1, 0],  # 1:数据质量良好, -1:数据质量不佳, 0:数据不充分
            'score_descriptions': {
                1: '数据质量良好',
                -1: '数据质量不佳',
                0: '数据不充分'
            }
        }
    }
    
    # 日志配置
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(asctime)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'console': {
                'format': '[%(levelname)s] %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'console',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': 'aishare_txt.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'encoding': 'utf-8'
            },
            'error_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': 'aishare_txt_errors.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 3,
                'encoding': 'utf-8'
            }
        },
        'loggers': {
            'aishare_txt': {
                'level': 'DEBUG',
                'handlers': ['file', 'error_file'],
                'propagate': False
            },
            'aishare_txt.data_fetcher': {
                'level': 'DEBUG',
                'handlers': ['file'],
                'propagate': False
            },
            'aishare_txt.technical_indicators': {
                'level': 'DEBUG',
                'handlers': ['file'],
                'propagate': False
            },
            'aishare_txt.report_generator': {
                'level': 'DEBUG',
                'handlers': ['file'],
                'propagate': False
            },
            'aishare_txt.utils': {
                'level': 'DEBUG',
                'handlers': ['file'],
                'propagate': False
            }
        },
        'root': {
            'level': 'WARNING',
            'handlers': []
        }
    }
    
    # 日志级别配置
    LOG_LEVELS = {
        'DEBUG': 10,
        'INFO': 20,
        'WARNING': 30,
        'ERROR': 40,
        'CRITICAL': 50
    }
    
    # 默认日志设置
    DEFAULT_LOG_LEVEL = 'INFO'
    LOG_TO_FILE = True
    LOG_TO_CONSOLE = False  # 设置为False禁用控制台输出
    LOG_FILE_PATH = 'logs/'
    LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5


# 向后兼容的别名
Config = IndicatorConfig