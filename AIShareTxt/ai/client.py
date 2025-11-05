#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 客户端封装
支持多种AI提供商（DeepSeek、智谱AI）进行股票投资建议分析
"""

import logging
from typing import Optional, Union
import os

try:
    from openai import OpenAI  # type: ignore[import-untyped]
    HAS_OPENAI = True
except ImportError:
    OpenAI = None  # type: ignore[misc]
    HAS_OPENAI = False

try:
    from zhipuai import ZhipuAI  # type: ignore[import-untyped]
    HAS_ZHIPUAI = True
except ImportError:
    try:
        # 尝试从 zai 包导入
        from zai import ZhipuAiClient as ZhipuAI  # type: ignore[import-untyped]
        HAS_ZHIPUAI = True
    except ImportError:
        ZhipuAI = None  # type: ignore[misc]
        HAS_ZHIPUAI = False

from ..utils.utils import LoggerManager

logger = LoggerManager.get_logger('ai_client')


class AIStockAnalyzer:
    """AI股票分析客户端"""
    
    def __init__(self, api_key: Optional[str] = None, provider: Optional[str] = None):
        """
        初始化AI客户端
        
        Args:
            api_key: AI API密钥（可选，默认从配置文件获取）
            provider: AI提供商（可选，默认从配置文件获取），可选值: 'deepseek', 'zhipuai'
        """
        try:
            from ..core.config import IndicatorConfig as Config
            config = Config()
            
            # 确定使用的提供商
            self.provider = provider or config.AI_CONFIG.get('default_provider', 'deepseek')
            
            # 获取对应提供商的配置
            provider_config = config.AI_CONFIG.get(self.provider, {})
            
            # 优先使用传入的密钥，其次从配置文件获取
            self.api_key = api_key or provider_config.get('api_key', '')
            
            # 保存配置供后续使用
            self.config = config
            
        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            self.provider = provider or 'deepseek'
            self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY', '')
            self.config = None
        
        self.client = None
        
        # 根据提供商初始化对应的客户端
        if self.provider == 'deepseek':
            self._init_deepseek_client()
        elif self.provider == 'zhipuai':
            self._init_zhipuai_client()
        else:
            logger.error(f"不支持的AI提供商: {self.provider}")
    
    def _init_deepseek_client(self):
        """初始化DeepSeek客户端"""
        if not HAS_OPENAI:
            logger.warning("未安装OpenAI客户端库，DeepSeek功能将不可用")
            return
        
        if not self.api_key:
            logger.warning("未提供DeepSeek API密钥，请设置环境变量DEEPSEEK_API_KEY")
            return
        
        try:
            base_url = "https://api.deepseek.com"
            if self.config:
                base_url = self.config.AI_CONFIG.get('deepseek', {}).get('base_url', base_url)
            
            self.client = OpenAI(  # type: ignore[misc]
                api_key=self.api_key,
                base_url=base_url
            )
            logger.info("DeepSeek客户端初始化成功")
        except Exception as e:
            logger.error(f"DeepSeek客户端初始化失败: {e}")
    
    def _init_zhipuai_client(self):
        """初始化智谱AI客户端"""
        if not HAS_ZHIPUAI:
            logger.warning("未安装智谱AI客户端库，智谱AI功能将不可用")
            return
        
        if not self.api_key:
            logger.warning("未提供智谱AI API密钥")
            return
        
        try:
            self.client = ZhipuAI(api_key=self.api_key)  # type: ignore[misc]
            logger.info("智谱AI客户端初始化成功")
        except Exception as e:
            logger.error(f"智谱AI客户端初始化失败: {e}")
    
    def is_available(self) -> bool:
        """检查AI功能是否可用"""
        if self.provider == 'deepseek':
            return self.client is not None and HAS_OPENAI
        elif self.provider == 'zhipuai':
            return self.client is not None and HAS_ZHIPUAI
        return False
    
    def analyze_investment_recommendation(self, technical_report: str, stock_code: str) -> int:
        """
        基于技术分析报告生成投资建议
        
        Args:
            technical_report: 技术分析报告
            stock_code: 股票代码
            
        Returns:
            int: 投资建议 (1: 立即买入, -1: 不适合买入, 0: 观望等待)
        """
        if not self.is_available():
            logger.warning("AI功能不可用，返回默认观望建议")
            return 0
        
        try:
            from .config import Config
            config = Config()
            
            # 根据提供商选择调用不同的方法
            if self.provider == 'deepseek':
                return self._analyze_with_deepseek(technical_report, stock_code, config)
            elif self.provider == 'zhipuai':
                return self._analyze_with_zhipuai(technical_report, stock_code, config)
            else:
                logger.error(f"不支持的AI提供商: {self.provider}")
                return 0
                
        except Exception as e:
            logger.error(f"AI分析失败: {e}")
            return 0  # 出错时默认返回观望
    
    def _analyze_with_deepseek(self, technical_report: str, stock_code: str, config) -> int:
        """使用DeepSeek进行分析"""
        ai_config = config.AI_CONFIG['deepseek']
        
        prompt = config.AI_CONFIG['recommendation']['prompt_template'].format(
            stock_code=stock_code,
            technical_report=technical_report
        )
        
        logger.info(f"发送DeepSeek请求，模型: {ai_config['model']}")
        logger.info(f"系统消息: {config.AI_CONFIG['recommendation']['system_message']}")
        logger.info(f"用户提示: {prompt[:200]}...")  # 只显示前200个字符
        
        response = self.client.chat.completions.create(  # type: ignore[union-attr]
            model=ai_config['model'],
            messages=[
                {"role": "system", "content": config.AI_CONFIG['recommendation']['system_message']},
                {"role": "user", "content": prompt}
            ],
            max_tokens=ai_config['max_tokens'],
            temperature=ai_config['temperature'],
            stream=False
        )
        
        logger.info(f"DeepSeek响应对象: {response}")
        logger.info(f"DeepSeek响应choices: {response.choices}")
        
        result_text = response.choices[0].message.content.strip() if response.choices[0].message.content else ""  # type: ignore[union-attr]
        logger.info(f"DeepSeek分析结果原始输出: '{result_text}'")
        
        # 解析AI返回的结果
        result = self._parse_ai_result(result_text)
        logger.info(f"股票 {stock_code} DeepSeek投资建议: {result}")
        
        return result
    
    def _analyze_with_zhipuai(self, technical_report: str, stock_code: str, config) -> int:
        """使用智谱AI进行分析"""
        ai_config = config.AI_CONFIG['zhipuai']
        
        prompt = config.AI_CONFIG['recommendation']['prompt_template'].format(
            stock_code=stock_code,
            technical_report=technical_report
        )
        
        logger.info(f"发送智谱AI请求，模型: {ai_config['model']}")
        logger.info(f"系统消息: {config.AI_CONFIG['recommendation']['system_message']}")
        logger.info(f"用户提示: {prompt[:200]}...")  # 只显示前200个字符
        
        response = self.client.chat.completions.create(  # type: ignore[union-attr]
            model=ai_config['model'],
            messages=[
                {"role": "system", "content": config.AI_CONFIG['recommendation']['system_message']},
                {"role": "user", "content": prompt}
            ],
            max_tokens=ai_config['max_tokens'],
            temperature=ai_config['temperature'],
            thinking={
                "type": "disabled",    # 禁用深度思考模式
            },
        )
        
        logger.info(f"智谱AI响应对象: {response}")
        logger.info(f"智谱AI响应choices: {response.choices}")
        
        result_text = response.choices[0].message.content.strip() if response.choices[0].message.content else ""  # type: ignore[union-attr]
        logger.info(f"智谱AI分析结果原始输出: '{result_text}'")
        
        # 解析AI返回的结果
        result = self._parse_ai_result(result_text)
        logger.info(f"股票 {stock_code} 智谱AI投资建议: {result}")
        
        return result
    
    def _parse_ai_result(self, ai_response: str) -> int:
        """
        解析AI返回的结果，确保符合规范
        
        Args:
            ai_response: AI的原始响应
            
        Returns:
            int: 标准化的投资建议
        """
        # 清理响应文本
        clean_response = ai_response.strip().replace(" ", "").replace("\n", "")
        
        # 尝试直接提取数字
        if "1" in clean_response and "-1" not in clean_response:
            return 1
        elif "-1" in clean_response:
            return -1
        elif "0" in clean_response:
            return 0
        
        # 如果包含关键词，进行映射
        buy_keywords = ["买入", "立即买入", "建议买入", "可以买入"]
        avoid_keywords = ["不适合", "不建议", "风险", "避免", "卖出"]
        wait_keywords = ["观望", "等待", "持有", "中性"]
        
        for keyword in buy_keywords:
            if keyword in clean_response:
                return 1
        
        for keyword in avoid_keywords:
            if keyword in clean_response:
                return -1
        
        for keyword in wait_keywords:
            if keyword in clean_response:
                return 0
        
        # 默认返回观望
        logger.warning(f"无法解析AI响应: {ai_response}，返回默认观望建议")
        return 0
    
    def get_recommendation_text(self, score: int) -> str:
        """
        获取推荐结果的文本描述
        
        Args:
            score: 推荐分数
            
        Returns:
            str: 文本描述
        """
        score_map = {
            1: "立即买入",
            -1: "不适合买入",
            0: "观望等待"
        }
        return score_map.get(score, "未知")


# 全局AI分析器实例
_ai_analyzer = None

def get_ai_analyzer(api_key: Optional[str] = None, provider: Optional[str] = None) -> AIStockAnalyzer:
    """
    获取全局AI分析器实例
    
    Args:
        api_key: API密钥
        provider: AI提供商（'deepseek' 或 'zhipuai'）
        
    Returns:
        AIStockAnalyzer: AI分析器实例
    """
    global _ai_analyzer
    if _ai_analyzer is None or (api_key and _ai_analyzer.api_key != api_key) or (provider and _ai_analyzer.provider != provider):
        _ai_analyzer = AIStockAnalyzer(api_key, provider)
    return _ai_analyzer


# 向后兼容的别名
AIClient = AIStockAnalyzer