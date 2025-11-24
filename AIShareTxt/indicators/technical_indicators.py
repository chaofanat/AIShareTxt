#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标计算模块
负责计算各种技术指标
"""

import talib
import pandas as pd
import numpy as np
from ..core.config import IndicatorConfig as Config
from ..utils.utils import LoggerManager
import warnings
warnings.filterwarnings('ignore')


class TechnicalIndicators:
    """技术指标计算器"""
    
    def __init__(self):
        self.config = Config()
        self.logger = LoggerManager.get_logger('technical_indicators')
    
    def calculate_all_indicators(self, data):
        """
        计算所有技术指标
        
        Args:
            data (pd.DataFrame): 股票数据
            
        Returns:
            dict: 包含所有指标的字典
        """
        if data is None or len(data) == 0:
            return None
            
        # 获取价格数据
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values
        open_price = data['open'].values
        
        indicators = {}
        
        # 计算各类指标
        indicators.update(self._calculate_moving_averages(close))
        indicators.update(self._calculate_ma_derived_indicators(close))
        indicators.update(self._calculate_volume_price_indicators(high, low, close, volume))
        indicators.update(self._calculate_trend_strength_indicators(high, low, close))
        indicators.update(self._calculate_momentum_oscillators(high, low, close))
        indicators.update(self._calculate_volatility_indicators(high, low, close))
        indicators.update(self._calculate_volume_indicators(volume))
        
        # 添加基础数据
        indicators['current_price'] = close[-1]
        # 将日期转换为字符串格式以便JSON序列化
        date_value = data['date'].iloc[-1]
        if hasattr(date_value, 'strftime'):
            indicators['date'] = date_value.strftime('%Y-%m-%d')
        else:
            indicators['date'] = str(date_value)
        
        return indicators
    
    def _calculate_moving_averages(self, close):
        """计算移动平均线指标"""
        indicators = {}
        
        try:
            # MA（移动平均线）
            all_periods = (self.config.MA_PERIODS['short'] + 
                          self.config.MA_PERIODS['medium'] + 
                          self.config.MA_PERIODS['long'])
            
            for period in all_periods:
                ma = talib.SMA(close, timeperiod=period)
                indicators[f'MA_{period}'] = ma[-1]
                indicators[f'SMA_{period}'] = ma[-1]  # SMA与MA相同，保留用于兼容
            
            # EMA（指数移动平均线）
            for period in self.config.EMA_PERIODS:
                ema = talib.EMA(close, timeperiod=period)
                indicators[f'EMA_{period}'] = ema[-1]
            
            # WMA（加权移动平均线）
            for period in self.config.WMA_PERIODS:
                wma = talib.WMA(close, timeperiod=period)
                indicators[f'WMA_{period}'] = wma[-1]
            
            # 均线形态分析
            ma_patterns = self.analyze_ma_patterns(close)
            indicators.update(ma_patterns)
            
        except Exception as e:
            self.logger.warning(f"均线指标计算失败：{str(e)}")
        
        return indicators
    
    def _calculate_ma_derived_indicators(self, close):
        """计算均线衍生指标"""
        indicators = {}
        
        try:
            # BIAS 乖离率
            for period in self.config.BIAS_PERIODS:
                bias = self.calculate_bias(close, timeperiod=period)
                if bias is not None:
                    indicators[f'BIAS_{period}'] = bias[-1]
            
            # MACD
            macd_config = self.config.MACD_CONFIG
            macd, macdsignal, macdhist = talib.MACD(
                close, 
                fastperiod=macd_config['fastperiod'],
                slowperiod=macd_config['slowperiod'], 
                signalperiod=macd_config['signalperiod']
            )
            indicators['MACD_DIF'] = macd[-1]
            indicators['MACD_DEA'] = macdsignal[-1]
            indicators['MACD_HIST'] = macdhist[-1]
            
            # 布林带
            bb_config = self.config.BOLLINGER_BANDS_CONFIG
            upperband, middleband, lowerband = talib.BBANDS(
                close, 
                timeperiod=bb_config['timeperiod'],
                nbdevup=bb_config['nbdevup'], 
                nbdevdn=bb_config['nbdevdn'], 
                matype=int(bb_config['matype'])  # type: ignore
            )
            indicators['BB_UPPER'] = upperband[-1]
            indicators['BB_MIDDLE'] = middleband[-1]
            indicators['BB_LOWER'] = lowerband[-1]
            indicators['BB_WIDTH'] = (upperband[-1] - lowerband[-1]) / middleband[-1] * 100
            
        except Exception as e:
            self.logger.warning(f"均线衍生指标计算失败：{str(e)}")
        
        return indicators
    
    def _calculate_volume_price_indicators(self, high, low, close, volume):
        """计算量价指标"""
        indicators = {}
        
        try:
            # VWMA (Volume Weighted Moving Average)
            close_price = close
            volume_weighted_sum = 0
            total_volume = 0
            vwma_period = self.config.VWMA_PERIOD

            if len(close_price) >= vwma_period:
                # 取最近vwma_period周期的数据计算VWMA
                recent_close = close_price[-vwma_period:]
                recent_volume = volume[-vwma_period:]

                # 计算每个周期的成交量权重
                volume_weighted_sum = np.sum(recent_close * recent_volume)
                total_volume = np.sum(recent_volume)

                if total_volume > 0:
                    indicators['VWMA_14'] = volume_weighted_sum / total_volume
            
            # OBV (On Balance Volume)
            close_float = close.astype(np.float64)
            volume_float = volume.astype(np.float64)
            obv = talib.OBV(close_float, volume_float)
            
            indicators.update(self._analyze_obv(obv, close))
            
        except Exception as e:
            self.logger.warning(f"量价指标计算失败：{str(e)}")
        
        return indicators
    
    def _calculate_trend_strength_indicators(self, high, low, close):
        """计算趋势强度指标"""
        indicators = {}
        
        try:
            # ADX (Average Directional Index) 和 DMI 指标
            adx_period = self.config.ADX_PERIOD
            adx = talib.ADX(high, low, close, timeperiod=adx_period)
            plus_di = talib.PLUS_DI(high, low, close, timeperiod=adx_period)
            minus_di = talib.MINUS_DI(high, low, close, timeperiod=adx_period)
            
            indicators['ADX'] = adx[-1]
            indicators['DI_PLUS'] = plus_di[-1]
            indicators['DI_MINUS'] = minus_di[-1]
            indicators['DI_DIFF'] = plus_di[-1] - minus_di[-1]
            
            # ADX趋势强度跟踪
            if len(adx) >= 5:
                adx_recent = adx[-5:]
                adx_slope = np.polyfit(range(len(adx_recent)), adx_recent, 1)[0]
                indicators['ADX_TREND'] = adx_slope
                indicators['ADX_5D_CHANGE'] = ((adx[-1] - adx[-5]) / adx[-5] * 100) if adx[-5] != 0 else 0
            
        except Exception as e:
            self.logger.warning(f"趋势强度指标计算失败：{str(e)}")
        
        return indicators
    
    def _calculate_momentum_oscillators(self, high, low, close):
        """计算动量振荡指标"""
        indicators = {}
        
        try:
            # RSI (Relative Strength Index)
            for period in self.config.RSI_PERIODS:
                rsi = talib.RSI(close, timeperiod=period)
                indicators[f'RSI_{period}'] = rsi[-1]
            
            # Stochastic Oscillator (KD指标)
            stoch_config = self.config.STOCH_CONFIG
            slowk, slowd = talib.STOCH(
                high, low, close,
                fastk_period=stoch_config['fastk_period'],
                slowk_period=stoch_config['slowk_period'],
                slowk_matype=int(stoch_config['slowk_matype']),  # type: ignore
                slowd_period=stoch_config['slowd_period'],
                slowd_matype=int(stoch_config['slowd_matype'])  # type: ignore
            )
            indicators['STOCH_K'] = slowk[-1]
            indicators['STOCH_D'] = slowd[-1]
            indicators['STOCH_KD_DIFF'] = slowk[-1] - slowd[-1]
            
        except Exception as e:
            self.logger.warning(f"动量振荡指标计算失败：{str(e)}")
        
        return indicators
    
    def _calculate_volatility_indicators(self, high, low, close):
        """计算波动率指标"""
        indicators = {}
        
        try:
            # ATR (Average True Range)
            atr_period = self.config.ATR_PERIOD
            atr = talib.ATR(high, low, close, timeperiod=atr_period)
            indicators['ATR_14'] = atr[-1]
            indicators['ATR_RATIO'] = (atr[-1] / close[-1] * 100) if close[-1] != 0 else 0
            
        except Exception as e:
            self.logger.warning(f"波动率指标计算失败：{str(e)}")
        
        return indicators
    
    def _calculate_volume_indicators(self, volume):
        """计算成交量指标"""
        indicators = {}
        
        try:
            # 量比指标
            if len(volume) >= 6:
                current_volume = volume[-1]
                avg_volume_5d = np.mean(volume[-6:-1])  # 过去5日平均
                indicators['VOLUME_RATIO'] = current_volume / avg_volume_5d if avg_volume_5d != 0 else 0
            
            # 成交量趋势
            if len(volume) >= 5:
                volume_recent = volume[-5:]
                volume_slope = np.polyfit(range(len(volume_recent)), volume_recent, 1)[0]
                indicators['VOLUME_TREND'] = volume_slope
            
        except Exception as e:
            self.logger.warning(f"成交量指标计算失败：{str(e)}")
        
        return indicators
    
    def calculate_bias(self, close, timeperiod=20):
        """
        计算乖离率 (BIAS)
        BIAS = (收盘价 - N日SMA) / N日SMA * 100
        """
        sma = talib.SMA(close, timeperiod=timeperiod)
        bias = ((close - sma) / sma) * 100
        return bias
    
    def analyze_ma_patterns(self, close):
        """分析均线形态"""
        patterns = {
            'trend_pattern': '',
            'cross_pattern': '',
            'position_pattern': '',
            'arrangement_pattern': '',
            'support_resistance': ''
        }
        
        # 计算均线
        ma5 = talib.SMA(close, timeperiod=5)
        ma10 = talib.SMA(close, timeperiod=10)
        ma20 = talib.SMA(close, timeperiod=20)
        ma60 = talib.SMA(close, timeperiod=60)
        
        if len(ma5) < 2 or len(ma10) < 2 or len(ma20) < 2:
            return patterns
            
        current_price = close[-1]
        current_ma5 = ma5[-1]
        current_ma10 = ma10[-1]
        current_ma20 = ma20[-1]
        current_ma60 = ma60[-1] if len(ma60) > 0 else 0
        
        prev_ma5 = ma5[-2]
        prev_ma10 = ma10[-2]
        prev_ma20 = ma20[-2]
        
        # 使用配置中的阈值
        thresholds = self.config.MA_THRESHOLDS
        
        # 1. 趋势形态分析
        patterns['trend_pattern'] = self._analyze_trend_pattern(
            current_ma5, current_ma10, current_ma20, thresholds
        )
        
        # 2. 交叉形态分析
        patterns['cross_pattern'] = self._analyze_cross_pattern(
            current_ma5, current_ma10, current_ma20,
            prev_ma5, prev_ma10, prev_ma20
        )
        
        # 3. 位置形态分析
        patterns['position_pattern'] = self._analyze_position_pattern(
            current_price, current_ma5, current_ma10, current_ma20, current_ma60, thresholds
        )
        
        # 4. 排列形态分析
        patterns['arrangement_pattern'] = self._analyze_arrangement_pattern(
            current_ma5, current_ma10, current_ma20, current_ma60
        )
        
        # 5. 支撑阻力分析
        patterns['support_resistance'] = self._analyze_support_resistance(
            current_price, current_ma5, current_ma10, current_ma20, thresholds
        )
        
        return patterns
    
    def _analyze_trend_pattern(self, ma5, ma10, ma20, thresholds):
        """分析趋势形态"""
        if ma5 > ma10 > ma20:
            # 检查发散程度
            ma_diff_5_10 = (ma5 - ma10) / ma10 * 100
            ma_diff_10_20 = (ma10 - ma20) / ma20 * 100
            
            if ma_diff_5_10 > thresholds['divergence_strong'] * 100 and ma_diff_10_20 > thresholds['divergence_strong'] * 100:
                return "多头发散"
            elif ma_diff_5_10 < thresholds['divergence_weak'] * 100 and ma_diff_10_20 < thresholds['divergence_weak'] * 100:
                return "多头收敛"
            else:
                return "多头排列"
                
        elif ma5 < ma10 < ma20:
            # 检查发散程度
            ma_diff_5_10 = abs(ma5 - ma10) / ma10 * 100
            ma_diff_10_20 = abs(ma10 - ma20) / ma20 * 100
            
            if ma_diff_5_10 > thresholds['divergence_strong'] * 100 and ma_diff_10_20 > thresholds['divergence_strong'] * 100:
                return "空头发散"
            elif ma_diff_5_10 < thresholds['divergence_weak'] * 100 and ma_diff_10_20 < thresholds['divergence_weak'] * 100:
                return "空头收敛"
            else:
                return "空头排列"
        else:
            # 检查是否为盘整形态
            ma_diff_5_10 = abs(ma5 - ma10) / ma10 * 100
            ma_diff_10_20 = abs(ma10 - ma20) / ma20 * 100
            ma_diff_5_20 = abs(ma5 - ma20) / ma20 * 100
            
            if (ma_diff_5_10 < thresholds['adhesion_loose'] * 100 and 
                ma_diff_10_20 < thresholds['adhesion_loose'] * 100 and 
                ma_diff_5_20 < thresholds['adhesion_range'] * 100):
                return "三均线粘合"
            elif ma_diff_5_10 < thresholds['adhesion_tight'] * 100:
                return "MA5与MA10粘合"
            elif ma_diff_10_20 < thresholds['adhesion_tight'] * 100:
                return "MA10与MA20粘合"
            else:
                return "均线缠绕"
    
    def _analyze_cross_pattern(self, ma5, ma10, ma20, prev_ma5, prev_ma10, prev_ma20):
        """分析交叉形态"""
        cross_signals = []
        
        # 均线交叉检测
        if ma5 > ma10 and prev_ma5 <= prev_ma10:
            cross_signals.append("MA5上穿MA10")
        if ma10 > ma20 and prev_ma10 <= prev_ma20:
            cross_signals.append("MA10上穿MA20")
            
        if ma5 < ma10 and prev_ma5 >= prev_ma10:
            cross_signals.append("MA5下穿MA10")
        if ma10 < ma20 and prev_ma10 >= prev_ma20:
            cross_signals.append("MA10下穿MA20")
            
        return "，".join(cross_signals) if cross_signals else "无交叉信号"
    
    def _analyze_position_pattern(self, price, ma5, ma10, ma20, ma60, thresholds):
        """分析位置形态"""
        position_signals = []
        support_threshold = thresholds['support_resistance']
        
        # 价格与均线关系
        for ma_value, ma_name in [(ma5, "MA5"), (ma10, "MA10"), (ma20, "MA20")]:
            if price > ma_value:
                position_signals.append(f"站上{ma_name}")
            elif abs(price - ma_value) / ma_value < support_threshold:
                position_signals.append(f"接近{ma_name}")
        
        if ma60 > 0:
            if price > ma60:
                position_signals.append("站上MA60")
            elif abs(price - ma60) / ma60 < support_threshold:
                position_signals.append("接近MA60")
        
        return "，".join(position_signals) if position_signals else "均线下方运行"
    
    def _analyze_arrangement_pattern(self, ma5, ma10, ma20, ma60):
        """分析排列形态"""
        if ma5 > ma10 > ma20:
            if ma60 > 0 and ma20 > ma60:
                return "MA5>MA10>MA20>MA60"
            else:
                return "MA5>MA10>MA20"
        elif ma5 < ma10 < ma20:
            if ma60 > 0 and ma20 < ma60:
                return "MA5<MA10<MA20<MA60"
            else:
                return "MA5<MA10<MA20"
        else:
            return "均线无规律排列"
    
    def _analyze_support_resistance(self, price, ma5, ma10, ma20, thresholds):
        """分析支撑阻力"""
        support_resistance = []
        support_threshold = thresholds['support_resistance']
        
        for ma_value, ma_name in [(ma5, "MA5"), (ma10, "MA10"), (ma20, "MA20")]:
            if abs(price - ma_value) / ma_value < support_threshold:
                support_resistance.append(f"接近{ma_name}")
        
        return "，".join(support_resistance) if support_resistance else "无接近均线"
    
    def _analyze_obv(self, obv, close):
        """分析OBV指标"""
        indicators = {}
        obv_config = self.config.OBV_CONFIG
        
        # 计算OBV基础指标
        indicators['OBV_current'] = obv[-1]
        indicators['OBV_5d_ago'] = obv[-6] if len(obv) >= 6 else obv[0]
        indicators['OBV_20d_ago'] = obv[-21] if len(obv) >= 21 else obv[0]
        
        # 计算OBV变化率
        obv_5d_change = ((obv[-1] - obv[-6]) / abs(obv[-6]) * 100) if len(obv) >= 6 and obv[-6] != 0 else 0
        obv_20d_change = ((obv[-1] - obv[-21]) / abs(obv[-21]) * 100) if len(obv) >= 21 and obv[-21] != 0 else 0
        
        indicators['OBV_5d_change'] = obv_5d_change
        indicators['OBV_20d_change'] = obv_20d_change
        
        # 计算OBV趋势（最近5日斜率）
        if len(obv) >= 5:
            recent_obv = obv[-5:]
            x = np.arange(len(recent_obv))
            slope = np.polyfit(x, recent_obv, 1)[0]
            indicators['OBV_trend'] = slope
        
        # OBV背离验证
        if len(obv) >= 20 and len(close) >= 20:
            try:
                indicators['OBV_DIVERGENCE'] = self._detect_obv_divergence(obv, close, obv_config)
            except ImportError:
                indicators['OBV_DIVERGENCE'] = "需要scipy库进行背离分析"
        
        return indicators
    
    def _detect_obv_divergence(self, obv, close, config):
        """检测OBV背离"""
        try:
            from scipy.signal import argrelextrema
            
            recent_close = close[-20:]
            recent_obv = obv[-20:]
            
            # 找局部高点和低点
            order = config['extrema_order']
            price_highs = argrelextrema(recent_close, np.greater, order=order)[0]
            price_lows = argrelextrema(recent_close, np.less, order=order)[0]
            obv_highs = argrelextrema(recent_obv, np.greater, order=order)[0]
            obv_lows = argrelextrema(recent_obv, np.less, order=order)[0]
            
            # 顶背离：价格创新高，OBV未创新高
            if len(price_highs) >= 2 and len(obv_highs) >= 2:
                latest_price_high_idx = price_highs[-1]
                prev_price_high_idx = price_highs[-2]
                
                if latest_price_high_idx < len(obv_highs) and prev_price_high_idx < len(obv_highs):
                    if (recent_close[latest_price_high_idx] > recent_close[prev_price_high_idx] and
                        recent_obv[latest_price_high_idx] < recent_obv[prev_price_high_idx]):
                        return "顶背离"
            
            # 底背离：价格创新低，OBV未创新低
            if len(price_lows) >= 2 and len(obv_lows) >= 2:
                latest_price_low_idx = price_lows[-1]
                prev_price_low_idx = price_lows[-2]
                
                if latest_price_low_idx < len(obv_lows) and prev_price_low_idx < len(obv_lows):
                    if (recent_close[latest_price_low_idx] < recent_close[prev_price_low_idx] and
                        recent_obv[latest_price_low_idx] > recent_obv[prev_price_low_idx]):
                        return "底背离"
            
            return "无明显背离"
            
        except Exception:
            return "背离检测失败"
