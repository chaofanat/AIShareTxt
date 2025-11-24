#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告生成模块
负责格式化和生成股票分析报告
"""

from .config import IndicatorConfig as Config


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self):
        self.config = Config()
    
    def generate_report(self, stock_code, indicators, stock_info=None, stock_data=None):
        """
        生成完整的股票分析报告

        Args:
            stock_code (str): 股票代码
            indicators (dict): 指标数据字典
            stock_info (dict): 股票基本信息
            stock_data (pd.DataFrame): 股票历史数据

        Returns:
            str: 格式化的报告文本
        """
        if indicators is None:
            return "无法生成指标报告：数据不足"
        
        report = []
        
        # 报告头部
        report.extend(self._generate_header(stock_code, indicators))
        
        # 股票基本信息
        if stock_info:
            report.extend(self._generate_basic_info_section(stock_info))
        
        # 技术指标各个部分
        report.extend(self._generate_ma_section(indicators))
        report.extend(self._generate_ma_derived_section(indicators))
        report.extend(self._generate_volume_price_section(indicators))
        report.extend(self._generate_momentum_section(indicators))
        report.extend(self._generate_volatility_section(indicators))
        report.extend(self._generate_fund_flow_section(indicators))
        report.extend(self._generate_trend_strength_section(indicators))
        report.extend(self._generate_summary_section(indicators))

        # 最近5天OLHCV数据
        if stock_data is not None:
            report.extend(self._generate_recent_ohlcv_section(stock_data))

        # 报告尾部
        report.extend(self._generate_footer())
        
        return "\n".join(report)
    
    def _generate_header(self, stock_code, indicators):
        """生成报告头部"""
        header = []
        header.append(self.config.REPORT_CONFIG['title_separator'])
        header.append(f"股票代码：{stock_code}")
        header.append(f"分析日期：{indicators.get('date', 'N/A')}")
        header.append(f"当前价格：{indicators.get('current_price', 0):.{self.config.DISPLAY_PRECISION['price']}f} 元")
        header.append(self.config.REPORT_CONFIG['title_separator'])
        return header
    
    def _generate_basic_info_section(self, stock_info):
        """生成基本信息部分"""
        section = []
        section.append("\n【股票基本信息】")
        section.append(self.config.REPORT_CONFIG['section_separator'])
        section.append(f"股票简称：{stock_info.get('股票简称', '未知')}")
        section.append(f"所属行业：{stock_info.get('行业', '未知')}")
        
        # 股本信息
        if stock_info.get('总股本_亿股', 0) > 0:
            section.append(f"总股本：  {stock_info.get('总股本_亿股', 0):.{self.config.DISPLAY_PRECISION['ratio']}f} 亿股")
        if stock_info.get('流通股本_亿股', 0) > 0:
            section.append(f"流通股本：{stock_info.get('流通股本_亿股', 0):.{self.config.DISPLAY_PRECISION['ratio']}f} 亿股")
        
        # 市值信息
        if stock_info.get('总市值_亿', 0) > 0:
            section.append(f"总市值：  {stock_info.get('总市值_亿', 0):.{self.config.DISPLAY_PRECISION['ratio']}f} 亿元")
        if stock_info.get('流通市值_亿', 0) > 0:
            section.append(f"流通市值：{stock_info.get('流通市值_亿', 0):.{self.config.DISPLAY_PRECISION['ratio']}f} 亿元")
        
        # 估值信息
        pe_ratio = stock_info.get('市盈率', '未知')
        pb_ratio = stock_info.get('市净率', '未知')
        if pe_ratio != '未知' and pe_ratio != '-':
            section.append(f"市盈率：  {pe_ratio}")
        if pb_ratio != '未知' and pb_ratio != '-':
            section.append(f"市净率：  {pb_ratio}")
        
        # 市值规模判断
        section.append(f"市值规模：{self._get_market_cap_category(stock_info)}")
        
        return section
    
    def _generate_ma_section(self, indicators):
        """生成均线指标部分"""
        section = []
        section.append("\n【二、核心均线指标】")
        section.append(self.config.REPORT_CONFIG['section_separator'])
        
        # 移动平均线
        if 'MA_5' in indicators:
            section.append("移动平均线 (MA) - 标准指标:")
            for period in self.config.MA_PERIODS['short']:
                section.append(f"  MA{period}:      {indicators.get(f'MA_{period}', 0):.{self.config.DISPLAY_PRECISION['price']}f}")
            for period in self.config.MA_PERIODS['medium']:
                section.append(f"  MA{period}:     {indicators.get(f'MA_{period}', 0):.{self.config.DISPLAY_PRECISION['price']}f}")
            for period in self.config.MA_PERIODS['long']:
                section.append(f"  MA{period}:    {indicators.get(f'MA_{period}', 0):.{self.config.DISPLAY_PRECISION['price']}f}")
            
            # 均线形态分析
            section.extend(self._generate_ma_pattern_analysis(indicators))
        
        # EMA
        if 'EMA_5' in indicators:
            section.append("\n指数移动平均线 (EMA):")
            for period in self.config.EMA_PERIODS:
                section.append(f"  EMA{period}:     {indicators.get(f'EMA_{period}', 0):.{self.config.DISPLAY_PRECISION['price']}f}")
            
            # EMA位置关系
            section.append(f"  EMA关系:  {self._get_ema_relationship(indicators)}")
        
        # WMA
        if 'WMA_10' in indicators:
            section.append("\n加权移动平均线 (WMA):")
            for period in self.config.WMA_PERIODS:
                section.append(f"  WMA{period}:    {indicators.get(f'WMA_{period}', 0):.{self.config.DISPLAY_PRECISION['price']}f}")
        
        return section
    
    def _generate_ma_derived_section(self, indicators):
        """生成均线衍生指标部分"""
        section = []
        section.append("\n【三、均线衍生指标】")
        section.append(self.config.REPORT_CONFIG['section_separator'])
        
        # BIAS乖离率
        if 'BIAS_5' in indicators:
            section.append("乖离率 (BIAS):")
            for period in self.config.BIAS_PERIODS:
                section.append(f"  {period}日BIAS:  {indicators.get(f'BIAS_{period}', 0):.{self.config.DISPLAY_PRECISION['percentage']}f}%")
        
        # MACD
        if 'MACD_DIF' in indicators:
            section.append("\nMACD 指标:")
            section.append(f"  MACD DIF: {indicators.get('MACD_DIF', 0):.{self.config.DISPLAY_PRECISION['macd']}f}")
            section.append(f"  MACD DEA: {indicators.get('MACD_DEA', 0):.{self.config.DISPLAY_PRECISION['macd']}f}")
            section.append(f"  MACD柱:   {indicators.get('MACD_HIST', 0):.{self.config.DISPLAY_PRECISION['macd']}f}")
            section.append(f"  MACD状态: {self._get_macd_status(indicators)}")
        
        # 布林带
        if 'BB_UPPER' in indicators:
            section.append("\n布林带 (Bollinger Bands):")
            section.append(f"  上轨:     {indicators.get('BB_UPPER', 0):.{self.config.DISPLAY_PRECISION['price']}f}")
            section.append(f"  中轨:     {indicators.get('BB_MIDDLE', 0):.{self.config.DISPLAY_PRECISION['price']}f}")
            section.append(f"  下轨:     {indicators.get('BB_LOWER', 0):.{self.config.DISPLAY_PRECISION['price']}f}")
            section.append(f"  带宽:     {indicators.get('BB_WIDTH', 0):.{self.config.DISPLAY_PRECISION['percentage']}f}%")
            section.append(f"  价格位置: {self._get_bb_position(indicators)}")
        
        return section
    
    def _generate_volume_price_section(self, indicators):
        """生成量价指标部分"""
        section = []
        section.append("\n【四、量价辅助指标】")
        section.append(self.config.REPORT_CONFIG['section_separator'])
        
        # VWMA
        if 'VWMA_14' in indicators:
            section.append("成交量加权移动平均 (VWMA):")
            section.append(f"  14日VWMA: {indicators.get('VWMA_14', 0):.{self.config.DISPLAY_PRECISION['price']}f}")
            section.append(f"  价格关系: {self._get_vwma_relationship(indicators)}")
        
        # OBV
        if 'OBV_current' in indicators:
            section.extend(self._generate_obv_analysis(indicators))
        
        # 成交量指标
        if 'VOLUME_RATIO' in indicators:
            section.extend(self._generate_volume_analysis(indicators))
        
        return section
    
    def _generate_momentum_section(self, indicators):
        """生成动量振荡指标部分"""
        section = []
        section.append("\n【五、动量振荡指标】")
        section.append(self.config.REPORT_CONFIG['section_separator'])
        
        # RSI
        if 'RSI_14' in indicators:
            section.append("相对强弱指数 (RSI):")
            for period in self.config.RSI_PERIODS:
                section.append(f"  RSI({period}):  {indicators.get(f'RSI_{period}', 0):.{self.config.DISPLAY_PRECISION['ratio']}f}")
            section.append(f"  RSI区间: {self._get_rsi_zone(indicators.get('RSI_14', 0))}")
        
        # KD指标
        if 'STOCH_K' in indicators:
            section.append("\n随机振荡器 (KD指标):")
            section.append(f"  K值:      {indicators.get('STOCH_K', 0):.{self.config.DISPLAY_PRECISION['ratio']}f}")
            section.append(f"  D值:      {indicators.get('STOCH_D', 0):.{self.config.DISPLAY_PRECISION['ratio']}f}")
            section.append(f"  K-D差值:  {indicators.get('STOCH_KD_DIFF', 0):.{self.config.DISPLAY_PRECISION['ratio']}f}")
            section.append(f"  KD关系:   {self._get_kd_relationship(indicators)}")
        
        return section
    
    def _generate_volatility_section(self, indicators):
        """生成波动率指标部分"""
        section = []
        section.append("\n【六、波动率指标】")
        section.append(self.config.REPORT_CONFIG['section_separator'])
        
        if 'ATR_14' in indicators:
            section.append("平均真实波幅 (ATR):")
            section.append(f"  ATR(14):  {indicators.get('ATR_14', 0):.{self.config.DISPLAY_PRECISION['atr']}f}")
            section.append(f"  ATR比率:  {indicators.get('ATR_RATIO', 0):.{self.config.DISPLAY_PRECISION['percentage']}f}%")
            section.append(f"  波动等级: {self._get_volatility_level(indicators.get('ATR_RATIO', 0))}")
        
        return section
    
    def _generate_fund_flow_section(self, indicators):
        """生成主力资金流部分"""
        section = []
        section.append("\n【七、主力资金流指标】")
        section.append(self.config.REPORT_CONFIG['section_separator'])
        
        has_fund_data = '主力净流入额' in indicators
        has_5day_data = '5日主力净流入额' in indicators
        
        if has_fund_data or has_5day_data:
            section.append("主力资金流向:")
            
            # 最新日期的资金流数据
            if has_fund_data:
                section.extend(self._generate_daily_fund_flow(indicators))
            
            # 5日累计数据
            if has_5day_data:
                section.extend(self._generate_5day_fund_flow(indicators))
            
            # 资金流向分析
            section.extend(self._generate_fund_flow_analysis(indicators, has_fund_data, has_5day_data))
        else:
            section.append("暂无主力资金流数据")
        
        return section
    
    def _generate_trend_strength_section(self, indicators):
        """生成趋势强度指标部分"""
        section = []
        section.append("\n【八、趋势强度指标】")
        section.append(self.config.REPORT_CONFIG['section_separator'])
        
        if 'ADX' in indicators:
            section.append("趋势方向指标 (DMI/ADX):")
            section.append(f"  ADX值:    {indicators.get('ADX', 0):.{self.config.DISPLAY_PRECISION['ratio']}f}")
            
            if 'DI_PLUS' in indicators and 'DI_MINUS' in indicators:
                section.append(f"  DI+值:    {indicators.get('DI_PLUS', 0):.{self.config.DISPLAY_PRECISION['ratio']}f}")
                section.append(f"  DI-值:    {indicators.get('DI_MINUS', 0):.{self.config.DISPLAY_PRECISION['ratio']}f}")
                section.append(f"  DI关系:   {self._get_di_relationship(indicators)}")
            
            section.append(f"  ADX范围:  {self._get_adx_range(indicators.get('ADX', 0))}")
            
            if 'ADX_TREND' in indicators:
                section.extend(self._generate_adx_trend_analysis(indicators))
        
        return section
    
    def _generate_summary_section(self, indicators):
        """生成指标状态汇总部分"""
        section = []
        section.append("\n【九、指标状态汇总】")
        section.append(self.config.REPORT_CONFIG['section_separator'])
        
        summary = self._collect_summary_items(indicators)
        
        for item in summary:
            section.append(f"- {item}")
        
        return section
    
    def _generate_footer(self):
        """生成报告尾部"""
        footer = []
        footer.append("\n" + self.config.REPORT_CONFIG['title_separator'])
        return footer
    
    # 辅助方法
    def _get_market_cap_category(self, stock_info):
        """获取市值规模分类"""
        total_market_cap = stock_info.get('总市值_亿', 0)
        levels = self.config.MARKET_CAP_LEVELS
        
        if total_market_cap >= levels['mega']:
            return f"超大盘股（≥{levels['mega']}亿）"
        elif total_market_cap >= levels['large']:
            return f"大盘股（{levels['large']}-{levels['mega']}亿）"
        elif total_market_cap >= levels['medium']:
            return f"中盘股（{levels['medium']}-{levels['large']}亿）"
        elif total_market_cap >= levels['small_medium']:
            return f"中小盘股（{levels['small_medium']}-{levels['medium']}亿）"
        elif total_market_cap > 0:
            return f"小盘股（<{levels['small_medium']}亿）"
        else:
            return "市值未知"
    
    def _generate_ma_pattern_analysis(self, indicators):
        """生成均线形态分析"""
        section = []
        section.append("\n均线形态分析:")
        
        pattern_keys = ['trend_pattern', 'cross_pattern', 'position_pattern', 
                       'arrangement_pattern', 'support_resistance']
        pattern_names = ['趋势形态', '交叉形态', '位置形态', '排列形态', '均线接近']
        
        for key, name in zip(pattern_keys, pattern_names):
            if key in indicators and indicators[key]:
                section.append(f"  {name}: {indicators[key]}")
        
        return section
    
    def _get_ema_relationship(self, indicators):
        """获取EMA关系"""
        ema5 = indicators.get('EMA_5', 0)
        ema10 = indicators.get('EMA_10', 0)
        return "EMA5>EMA10" if ema5 > ema10 else "EMA5<EMA10"
    
    def _get_macd_status(self, indicators):
        """获取MACD状态"""
        macd_hist = indicators.get('MACD_HIST', 0)
        if macd_hist > 0:
            return "HIST>0"
        elif macd_hist < 0:
            return "HIST<0"
        else:
            return "HIST=0"
    
    def _get_bb_position(self, indicators):
        """获取布林带位置"""
        current_price = indicators.get('current_price', 0)
        bb_upper = indicators.get('BB_UPPER', 0)
        bb_lower = indicators.get('BB_LOWER', 0)
        
        if current_price > bb_upper:
            return "价格>上轨"
        elif current_price < bb_lower:
            return "价格<下轨"
        else:
            return "价格在带内"
    
    def _get_vwma_relationship(self, indicators):
        """获取VWMA关系"""
        current_price = indicators.get('current_price', 0)
        vwma = indicators.get('VWMA_14', 0)
        return "价格>VWMA" if current_price > vwma else "价格<VWMA"
    
    def _generate_obv_analysis(self, indicators):
        """生成OBV分析"""
        section = []
        section.append("\n能量潮 (OBV):")
        section.append(f"  当前OBV:     {indicators.get('OBV_current', 0):.{self.config.DISPLAY_PRECISION['volume_wan']}f}")
        section.append(f"  5日变化:     {indicators.get('OBV_5d_change', 0):+.{self.config.DISPLAY_PRECISION['percentage']}f}%")
        section.append(f"  20日变化:    {indicators.get('OBV_20d_change', 0):+.{self.config.DISPLAY_PRECISION['percentage']}f}%")
        
        # OBV变化方向
        obv_5d_change = indicators.get('OBV_5d_change', 0)
        if obv_5d_change > 0:
            obv_signal = "5日OBV上升"
        elif obv_5d_change < 0:
            obv_signal = "5日OBV下降"
        else:
            obv_signal = "5日OBV平稳"
        section.append(f"  OBV趋势:     {obv_signal}")
        
        # OBV斜率
        obv_trend = indicators.get('OBV_trend', 0)
        if obv_trend:
            volume_price_signal = "OBV斜率为正" if obv_trend > 0 else "OBV斜率为负"
            section.append(f"  OBV斜率:     {volume_price_signal}")
        
        # OBV背离分析
        if 'OBV_DIVERGENCE' in indicators:
            section.append(f"  OBV背离:     {indicators['OBV_DIVERGENCE']}")
        
        return section
    
    def _generate_volume_analysis(self, indicators):
        """生成成交量分析"""
        section = []
        section.append("\n量比指标:")
        volume_ratio = indicators.get('VOLUME_RATIO', 0)
        section.append(f"  量比:        {volume_ratio:.{self.config.DISPLAY_PRECISION['ratio']}f}")
        section.append(f"  成交量:      {self._get_volume_level(volume_ratio)}")
        
        if 'VOLUME_TREND' in indicators:
            vol_trend = indicators.get('VOLUME_TREND', 0)
            vol_trend_desc = self._get_volume_trend_description(vol_trend)
            section.append(f"  量能趋势:    {vol_trend_desc}")
        
        return section
    
    def _get_rsi_zone(self, rsi_value):
        """获取RSI区间"""
        zones = self.config.RSI_ZONES
        if rsi_value >= zones['overbought']:
            return f"RSI≥{zones['overbought']}"
        elif rsi_value >= zones['neutral_high']:
            return f"{zones['neutral_high']}≤RSI<{zones['overbought']}"
        elif rsi_value >= zones['neutral_low']:
            return f"{zones['neutral_low']}≤RSI<{zones['neutral_high']}"
        else:
            return f"RSI<{zones['neutral_low']}"
    
    def _get_kd_relationship(self, indicators):
        """获取KD关系"""
        k_val = indicators.get('STOCH_K', 0)
        d_val = indicators.get('STOCH_D', 0)
        if k_val > d_val:
            return "K>D"
        elif k_val < d_val:
            return "K<D"
        else:
            return "K=D"
    
    def _get_volatility_level(self, atr_ratio):
        """获取波动率等级"""
        levels = self.config.VOLATILITY_LEVELS
        if atr_ratio >= levels['high']:
            return f"高波动(≥{levels['high']}%)"
        elif atr_ratio >= levels['medium']:
            return f"中等波动({levels['medium']}-{levels['high']}%)"
        elif atr_ratio >= levels['low']:
            return f"低波动({levels['low']}-{levels['medium']}%)"
        else:
            return f"极低波动(<{levels['low']}%)"
    
    def _get_volume_level(self, volume_ratio):
        """获取成交量等级"""
        levels = self.config.VOLUME_RATIO_LEVELS
        if volume_ratio >= levels['heavy']:
            return f"放量(≥{levels['heavy']}倍)"
        elif volume_ratio >= levels['moderate']:
            return f"温和放量({levels['moderate']}-{levels['heavy']}倍)"
        elif volume_ratio >= levels['normal']:
            return "正常成交量"
        else:
            return f"缩量(<{levels['normal']}倍)"
    
    def _get_volume_trend_description(self, vol_trend):
        """获取成交量趋势描述"""
        if vol_trend > 0:
            return "成交量递增"
        elif vol_trend < 0:
            return "成交量递减"
        else:
            return "成交量平稳"
    
    def _generate_daily_fund_flow(self, indicators):
        """生成当日资金流"""
        section = []
        fund_date = indicators.get('日期', '最新')
        section.append(f"\n{fund_date}资金流:")
        
        fund_items = [
            ('主力净流入额', '主力净流入占比', '主力净流入'),
            ('超大单净流入额', '超大单净流入占比', '超大单'),
            ('大单净流入额', '大单净流入占比', '大单'),
            ('中单净流入额', '中单净流入占比', '中单'),
            ('小单净流入额', '小单净流入占比', '小单')
        ]
        
        for amount_key, ratio_key, label in fund_items:
            amount = indicators.get(amount_key, 0)
            ratio = indicators.get(ratio_key, 0)
            amount_wan = amount / 10000 if amount != 0 else 0
            section.append(f"  {label:<8}: {amount_wan:.{self.config.DISPLAY_PRECISION['volume_wan']}f}万元 ({ratio:+.{self.config.DISPLAY_PRECISION['percentage']}f}%)")
        
        return section
    
    def _generate_5day_fund_flow(self, indicators):
        """生成5日累计资金流"""
        section = []
        section.append("\n5日累计资金流:")
        
        fund_items = [
            ('5日主力净流入额', '5日主力净流入占比', '主力净流入'),
            ('5日超大单净流入额', '5日超大单净流入占比', '超大单'),
            ('5日大单净流入额', '5日大单净流入占比', '大单'),
            ('5日中单净流入额', '5日中单净流入占比', '中单'),
            ('5日小单净流入额', '5日小单净流入占比', '小单')
        ]
        
        for amount_key, ratio_key, label in fund_items:
            amount = indicators.get(amount_key, 0)
            ratio = indicators.get(ratio_key, 0)
            amount_wan = amount / 10000 if amount != 0 else 0
            section.append(f"  {label}: {amount_wan:.{self.config.DISPLAY_PRECISION['volume_wan']}f}万元 (均值:{ratio:+.{self.config.DISPLAY_PRECISION['percentage']}f}%)")
        
        return section
    
    def _generate_fund_flow_analysis(self, indicators, has_fund_data, has_5day_data):
        """生成资金流向分析"""
        section = []
        section.append("\n资金流向分析:")
        
        if has_fund_data:
            section.extend(self._analyze_current_fund_flow(indicators))
        
        if has_5day_data:
            section.extend(self._analyze_5day_fund_flow(indicators))
        
        if has_fund_data and has_5day_data:
            section.extend(self._analyze_fund_flow_trend(indicators))
        
        return section
    
    def _analyze_current_fund_flow(self, indicators):
        """分析当前资金流"""
        section = []
        main_amount = indicators.get('主力净流入额', 0)
        main_ratio = indicators.get('主力净流入占比', 0)
        
        # 资金流向
        if main_amount > 0:
            current_status = "最新日主力资金净流入"
        elif main_amount < 0:
            current_status = "最新日主力资金净流出"
        else:
            current_status = "最新日主力资金平衡"
        
        # 资金强度判断
        current_strength = self._get_fund_flow_strength(abs(main_ratio))
        
        section.append(f"  当前状态: {current_status}")
        section.append(f"  流动强度: {current_strength}")
        
        return section
    
    def _analyze_5day_fund_flow(self, indicators):
        """分析5日资金流"""
        section = []
        day5_main = indicators.get('5日主力净流入额', 0)
        day5_main_ratio = indicators.get('5日主力净流入占比', 0)
        
        if day5_main > 0:
            day5_status = "5日累计主力资金净流入"
        elif day5_main < 0:
            day5_status = "5日累计主力资金净流出"
        else:
            day5_status = "5日累计主力资金平衡"
        
        section.append(f"  5日状态: {day5_status}")
        
        # 5日资金流强度
        day5_strength = self._get_5day_fund_flow_strength(abs(day5_main_ratio))
        section.append(f"  5日强度: {day5_strength}")
        
        return section
    
    def _analyze_fund_flow_trend(self, indicators):
        """分析资金流趋势对比"""
        section = []
        current_main = indicators.get('主力净流入额', 0)
        day5_main = indicators.get('5日主力净流入额', 0)
        
        # 计算5日日均资金流
        day5_avg = day5_main / 5 if day5_main != 0 else 0
        
        trend_analysis = self._determine_trend_analysis(current_main, day5_main, day5_avg)
        section.append(f"  趋势对比: {trend_analysis}")
        
        return section
    
    def _get_fund_flow_strength(self, ratio):
        """获取资金流强度"""
        levels = self.config.FUND_FLOW_LEVELS
        if ratio >= levels['strong']:
            return "资金流动强烈"
        elif ratio >= levels['active']:
            return "资金流动活跃"
        elif ratio >= levels['moderate']:
            return "资金流动温和"
        else:
            return "资金流动平淡"
    
    def _get_5day_fund_flow_strength(self, ratio):
        """获取5日资金流强度"""
        if ratio >= 3:
            return "5日资金流动活跃"
        elif ratio >= 1:
            return "5日资金流动温和"
        else:
            return "5日资金流动平淡"
    
    def _determine_trend_analysis(self, current_main, day5_main, day5_avg):
        """确定趋势分析"""
        if current_main > 0 and day5_main > 0:
            if current_main > abs(day5_avg) * 2:
                return "当日资金流入明显加速"
            elif current_main > abs(day5_avg):
                return "当日资金流入超过近期均值"
            else:
                return "当日资金流入符合近期趋势"
        elif current_main < 0 and day5_main < 0:
            if abs(current_main) > abs(day5_avg) * 2:
                return "当日资金流出明显加速"
            elif abs(current_main) > abs(day5_avg):
                return "当日资金流出超过近期均值"
            else:
                return "当日资金流出符合近期趋势"
        elif current_main > 0 and day5_main < 0:
            return "当日资金流向与近期趋势相反（转为流入）"
        elif current_main < 0 and day5_main > 0:
            return "当日资金流向与近期趋势相反（转为流出）"
        else:
            return "资金流向变化不明显"
    
    def _get_di_relationship(self, indicators):
        """获取DI关系"""
        di_plus = indicators.get('DI_PLUS', 0)
        di_minus = indicators.get('DI_MINUS', 0)
        di_diff = indicators.get('DI_DIFF', 0)
        
        if di_plus > di_minus:
            return f"DI+>DI-（差值：{di_diff:.{self.config.DISPLAY_PRECISION['ratio']}f}）"
        elif di_plus < di_minus:
            return f"DI+<DI-（差值：{di_diff:.{self.config.DISPLAY_PRECISION['ratio']}f}）"
        else:
            return "DI+=DI-"
    
    def _get_adx_range(self, adx_value):
        """获取ADX范围"""
        levels = self.config.ADX_LEVELS
        if adx_value > levels['strong_trend']:
            return f"ADX>{levels['strong_trend']}"
        elif adx_value > levels['medium_trend']:
            return f"{levels['medium_trend']}<ADX≤{levels['strong_trend']}"
        else:
            return f"ADX≤{levels['medium_trend']}"
    
    def _generate_adx_trend_analysis(self, indicators):
        """生成ADX趋势分析"""
        section = []
        adx_trend = indicators.get('ADX_TREND', 0)
        adx_5d_change = indicators.get('ADX_5D_CHANGE', 0)
        
        if adx_trend > 0:
            trend_direction = "ADX上升"
        elif adx_trend < 0:
            trend_direction = "ADX下降"
        else:
            trend_direction = "ADX平稳"
        
        section.append(f"  ADX趋势:  {trend_direction}")
        section.append(f"  5日变化:  {adx_5d_change:+.{self.config.DISPLAY_PRECISION['percentage']}f}%")
        
        return section
    
    def _collect_summary_items(self, indicators):
        """收集汇总项目"""
        summary = []
        
        # 均线状态
        if 'trend_pattern' in indicators and indicators['trend_pattern']:
            summary.append(f"均线形态：{indicators['trend_pattern']}")
        
        if 'cross_pattern' in indicators and indicators['cross_pattern'] != "无交叉信号":
            summary.append(f"均线交叉：{indicators['cross_pattern']}")
        
        if 'arrangement_pattern' in indicators and indicators['arrangement_pattern']:
            summary.append(f"均线排列：{indicators['arrangement_pattern']}")
        
        # MACD状态
        if 'MACD_HIST' in indicators:
            macd_status = "HIST>0" if indicators['MACD_HIST'] > 0 else "HIST<0"
            summary.append(f"MACD：{macd_status}")
        
        # DMI/ADX状态
        if 'ADX' in indicators:
            summary.append(f"ADX：{indicators['ADX']:.1f}")
            
            if 'DI_PLUS' in indicators and 'DI_MINUS' in indicators:
                di_plus = indicators['DI_PLUS']
                di_minus = indicators['DI_MINUS']
                if di_plus > di_minus:
                    summary.append(f"DI+({di_plus:.1f})>DI-({di_minus:.1f})")
                else:
                    summary.append(f"DI+({di_plus:.1f})<DI-({di_minus:.1f})")
        
        # 其他指标状态
        summary.extend(self._collect_other_indicators_summary(indicators))
        
        return summary
    
    def _collect_other_indicators_summary(self, indicators):
        """收集其他指标汇总"""
        summary = []
        
        # OBV状态
        if 'OBV_5d_change' in indicators:
            obv_5d = indicators['OBV_5d_change']
            summary.append(f"OBV 5日变化：{obv_5d:+.{self.config.DISPLAY_PRECISION['percentage']}f}%")
        
        # RSI状态
        if 'RSI_14' in indicators:
            rsi_14 = indicators['RSI_14']
            summary.append(f"RSI(14)：{rsi_14:.1f}")
        
        # KD状态
        if 'STOCH_K' in indicators and 'STOCH_D' in indicators:
            k_val = indicators['STOCH_K']
            d_val = indicators['STOCH_D']
            summary.append(f"KD：K({k_val:.1f}) D({d_val:.1f})")
        
        # ATR状态
        if 'ATR_RATIO' in indicators:
            atr_ratio = indicators['ATR_RATIO']
            summary.append(f"ATR波动率：{atr_ratio:.{self.config.DISPLAY_PRECISION['percentage']}f}%")
        
        # 量比状态
        if 'VOLUME_RATIO' in indicators:
            vol_ratio = indicators['VOLUME_RATIO']
            summary.append(f"量比：{vol_ratio:.{self.config.DISPLAY_PRECISION['ratio']}f}")
        
        # OBV背离状态
        if 'OBV_DIVERGENCE' in indicators and indicators['OBV_DIVERGENCE'] != "无明显背离":
            divergence = indicators['OBV_DIVERGENCE']
            summary.append(f"OBV：{divergence}")
        
        # 资金流状态
        summary.extend(self._collect_fund_flow_summary(indicators))
        
        return summary
    
    def _collect_fund_flow_summary(self, indicators):
        """收集资金流汇总"""
        summary = []
        
        # 最新主力资金流状态
        if '主力净流入额' in indicators:
            main_amount = indicators.get('主力净流入额', 0)
            main_ratio = indicators.get('主力净流入占比', 0)
            main_amount_wan = main_amount / 10000
            
            if main_amount > 0:
                summary.append(f"主力资金：净流入{main_amount_wan:.{self.config.DISPLAY_PRECISION['volume_wan']}f}万元({main_ratio:+.1f}%)")
            elif main_amount < 0:
                summary.append(f"主力资金：净流出{abs(main_amount_wan):.{self.config.DISPLAY_PRECISION['volume_wan']}f}万元({main_ratio:+.1f}%)")
            else:
                summary.append("主力资金：资金平衡")
        
        # 5日主力资金流状态
        if '5日主力净流入额' in indicators:
            day5_main_amount = indicators.get('5日主力净流入额', 0)
            day5_main_ratio = indicators.get('5日主力净流入占比', 0)
            day5_main_wan = day5_main_amount / 10000
            
            if day5_main_amount > 0:
                summary.append(f"5日主力：累计净流入{day5_main_wan:.{self.config.DISPLAY_PRECISION['volume_wan']}f}万元(均值{day5_main_ratio:+.1f}%)")
            elif day5_main_amount < 0:
                summary.append(f"5日主力：累计净流出{abs(day5_main_wan):.{self.config.DISPLAY_PRECISION['volume_wan']}f}万元(均值{day5_main_ratio:+.1f}%)")
            else:
                summary.append("5日主力：资金平衡")
        
        # 超大单资金状态
        if '超大单净流入额' in indicators:
            super_amount = indicators.get('超大单净流入额', 0)
            super_ratio = indicators.get('超大单净流入占比', 0)
            super_amount_wan = super_amount / 10000
            
            threshold = self.config.REPORT_CONFIG['min_fund_display_threshold']
            if abs(super_amount_wan) > threshold:
                if super_amount > 0:
                    summary.append(f"超大单：净流入{super_amount_wan:.{self.config.DISPLAY_PRECISION['volume_wan']}f}万元({super_ratio:+.1f}%)")
                else:
                    summary.append(f"超大单：净流出{abs(super_amount_wan):.{self.config.DISPLAY_PRECISION['volume_wan']}f}万元({super_ratio:+.1f}%)")

        return summary

    def _generate_recent_ohlcv_section(self, stock_data):
        """生成最近5天OLHCV数据部分"""
        section = []
        section.append("\n【十、最近5天交易数据】")
        section.append(self.config.REPORT_CONFIG['section_separator'])

        # 获取最近5天数据
        recent_data = stock_data.tail(5)

        # 表头
        section.append("日期        开盘价    最高价    最低价    收盘价    成交量(万手)  成交额(亿元)")
        section.append("─" * 70)

        for idx, row in recent_data.iterrows():
            # 格式化日期
            date_str = idx.strftime('%m-%d') if hasattr(idx, 'strftime') else str(idx)[:5]

            # 格式化价格数据
            open_price = f"{row['open']:.2f}"
            high_price = f"{row['high']:.2f}"
            low_price = f"{row['low']:.2f}"
            close_price = f"{row['close']:.2f}"

            # 计算涨跌幅
            if idx != recent_data.index[0]:  # 不是第一行
                prev_close = recent_data.loc[recent_data.index[recent_data.index.get_loc(idx) - 1], 'close']
                change_percent = ((row['close'] - prev_close) / prev_close * 100)
                change_sign = "▲" if change_percent > 0 else "▼" if change_percent < 0 else "─"
                close_price = f"{row['close']:.2f}{change_sign}"
            else:
                close_price = f"{row['close']:.2f}"

            # 格式化成交量 (转换为万手)
            volume_wan_hands = row['volume'] / 10000
            volume_str = f"{volume_wan_hands:.1f}"

            # 格式化成交额 (转换为亿元)
            amount = row.get('amount', row['close'] * row['volume'])  # 如果没有amount字段，用估算值
            amount_yi = amount / 100000000
            amount_str = f"{amount_yi:.2f}"

            # 组合一行数据
            line = f"{date_str}  {open_price:>8}  {high_price:>8}  {low_price:>8}  {close_price:>10}  {volume_str:>10}  {amount_str:>12}"
            section.append(line)

        return section
