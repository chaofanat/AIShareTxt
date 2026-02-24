#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标数据处理模块
负责计算和处理各种技术指标数据
"""

import talib
import pandas as pd
import numpy as np
from ..core.config import IndicatorConfig as Config
from ..utils.utils import LoggerManager
import warnings
warnings.filterwarnings('ignore')


class TechnicalIndicators:
    """技术指标数据处理器"""
    
    def __init__(self):
        self.config = Config()
        self.logger = LoggerManager.get_logger('technical_indicators')
    
    def process_all_indicators(self, data):
        """
        处理所有技术指标数据

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

        # 处理各类指标
        indicators.update(self._process_moving_averages(close))
        indicators.update(self._process_ma_derived_indicators(close))
        indicators.update(self._process_volume_price_indicators(high, low, close, volume))
        indicators.update(self._process_trend_strength_indicators(high, low, close))
        indicators.update(self._process_momentum_oscillators(high, low, close))
        indicators.update(self._process_volatility_indicators(high, low, close))
        indicators.update(self._process_volume_indicators(volume))

        # 添加基础数据
        indicators['current_price'] = close[-1]
        # 将日期转换为字符串格式以便JSON序列化
        date_value = data['date'].iloc[-1]
        if hasattr(date_value, 'strftime'):
            indicators['date'] = date_value.strftime('%Y-%m-%d')
        else:
            indicators['date'] = str(date_value)

        # 添加时空维度分析数据
        spatial_temporal = self.calculate_spatial_temporal(data, close[-1])
        if spatial_temporal:
            indicators.update(spatial_temporal)

        # 添加量能环比数据
        volume_comparison = self.calculate_volume_comparison(data, volume[-1])
        if volume_comparison:
            indicators.update(volume_comparison)

        # 添加连续涨跌日统计数据
        consecutive_days = self.calculate_consecutive_days(data)
        if consecutive_days:
            indicators.update(consecutive_days)

        return indicators
    
    def _process_moving_averages(self, close):
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
    
    def _process_ma_derived_indicators(self, close):
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
    
    def _process_volume_price_indicators(self, high, low, close, volume):
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
    
    def _process_trend_strength_indicators(self, high, low, close):
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
    
    def _process_momentum_oscillators(self, high, low, close):
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
    
    def _process_volatility_indicators(self, high, low, close):
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
    
    def _process_volume_indicators(self, volume):
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
            
            # 将pandas Series转换为numpy数组以解决兼容性问题
            if hasattr(recent_close, 'values'):
                recent_close = recent_close.values
            if hasattr(recent_obv, 'values'):
                recent_obv = recent_obv.values

            # 确保有足够的数据点
            if len(recent_close) < 10 or len(recent_obv) < 10:
                return "数据不足"

            # 找局部高点和低点
            order = config['extrema_order']
            price_highs = argrelextrema(recent_close, np.greater, order=order)[0]
            price_lows = argrelextrema(recent_close, np.less, order=order)[0]
            obv_highs = argrelextrema(recent_obv, np.greater, order=order)[0]
            obv_lows = argrelextrema(recent_obv, np.less, order=order)[0]

            # 顶背离检测：价格创新高，但OBV未创新高
            if len(price_highs) >= 2:
                # 取最后两个价格高点
                latest_price_high_idx = price_highs[-1]
                prev_price_high_idx = price_highs[-2]

                # 检查价格是否创新高
                if recent_close[latest_price_high_idx] > recent_close[prev_price_high_idx]:
                    # 寻找对应时间点的OBV值进行比较
                    latest_obv_value = recent_obv[latest_price_high_idx]
                    prev_obv_value = recent_obv[prev_price_high_idx]

                    # 如果OBV没有创新高，则为顶背离
                    if latest_obv_value < prev_obv_value:
                        return "顶背离"

            # 底背离检测：价格创新低，但OBV未创新低
            if len(price_lows) >= 2:
                # 取最后两个价格低点
                latest_price_low_idx = price_lows[-1]
                prev_price_low_idx = price_lows[-2]

                # 检查价格是否创新低
                if recent_close[latest_price_low_idx] < recent_close[prev_price_low_idx]:
                    # 寻找对应时间点的OBV值进行比较
                    latest_obv_value = recent_obv[latest_price_low_idx]
                    prev_obv_value = recent_obv[prev_price_low_idx]

                    # 如果OBV没有创新低，则为底背离
                    if latest_obv_value > prev_obv_value:
                        return "底背离"

            return "无明显背离"

        except ImportError:
            return "需要scipy库进行背离分析"
        except Exception as e:
            print(f"OBV背离检测错误: {e}")
            return "背离检测失败"

    def calculate_spatial_temporal(self, df: pd.DataFrame, current_price: float) -> dict:
        """
        计算时空维度分析数据

        Args:
            df: 历史数据DataFrame
            current_price: 当前价格

        Returns:
            dict: 包含前高前低、斐波那契窗口、连续涨跌日等数据
        """
        result = {}
        config = self.config.SPATIAL_TEMPORAL_CONFIG

        try:
            # 确保有足够的数据
            lookback_days = min(config['high_low_lookback_days'], len(df) - 1)
            if lookback_days < 5:
                return result

            # 获取今日数据（最后一行）
            today_data = df.iloc[-1]
            today_high = float(today_data['high'])
            today_low = float(today_data['low'])
            today_date_str = today_data['date'].strftime('%Y-%m-%d') if hasattr(today_data['date'], 'strftime') else str(today_data['date'])

            # 获取回溯范围内的历史数据（排除今日）
            historical_data = df.iloc[-lookback_days:-1] if lookback_days > 1 else df.iloc[:-1]

            # 检测是否今日创新高/新低
            if len(historical_data) > 0:
                historical_high_max = float(historical_data['high'].max())
                historical_low_min = float(historical_data['low'].min())

                # 判断是否今日创新高
                result['is_new_high_today'] = today_high > historical_high_max
                result['is_new_low_today'] = today_low < historical_low_min

                # 前高分析 - 使用历史数据的前高
                result['recent_high_price'] = float(historical_data['high'].max())
                high_idx = historical_data['high'].idxmax()
                high_date = df.loc[high_idx, 'date'] if 'date' in df.columns else high_idx
                result['recent_high_date'] = high_date.strftime('%Y-%m-%d') if hasattr(high_date, 'strftime') else str(high_date)

                # 如果今日创新高，记录今日新高信息
                if result['is_new_high_today']:
                    result['new_high_today_price'] = today_high
                    result['new_high_today_date'] = today_date_str

                # 前低分析 - 使用历史数据的前低
                result['recent_low_price'] = float(historical_data['low'].min())
                low_idx = historical_data['low'].idxmin()
                low_date = df.loc[low_idx, 'date'] if 'date' in df.columns else low_idx
                result['recent_low_date'] = low_date.strftime('%Y-%m-%d') if hasattr(low_date, 'strftime') else str(low_date)

                # 如果今日创新低，记录今日新低信息
                if result['is_new_low_today']:
                    result['new_low_today_price'] = today_low
                    result['new_low_today_date'] = today_date_str

                # 计算距历史前高的空间
                result['space_to_high_yuan'] = round(result['recent_high_price'] - current_price, 2)
                result['space_to_high_pct'] = round((result['recent_high_price'] / current_price - 1) * 100, 2)

                # 计算距历史前低的涨幅
                result['gain_from_low_yuan'] = round(current_price - result['recent_low_price'], 2)
                result['gain_from_low_pct'] = round((current_price / result['recent_low_price'] - 1) * 100, 2)
            else:
                # 数据不足，使用包含今日的数据
                recent_data = df.iloc[-lookback_days:]

                high_idx = recent_data['high'].idxmax()
                low_idx = recent_data['low'].idxmin()

                result['recent_high_price'] = float(recent_data['high'].max())
                high_date = df.loc[high_idx, 'date'] if 'date' in df.columns else high_idx
                result['recent_high_date'] = high_date.strftime('%Y-%m-%d') if hasattr(high_date, 'strftime') else str(high_date)
                result['space_to_high_yuan'] = round(result['recent_high_price'] - current_price, 2)
                result['space_to_high_pct'] = round((result['recent_high_price'] / current_price - 1) * 100, 2)

                result['recent_low_price'] = float(recent_data['low'].min())
                low_date = df.loc[low_idx, 'date'] if 'date' in df.columns else low_idx
                result['recent_low_date'] = low_date.strftime('%Y-%m-%d') if hasattr(low_date, 'strftime') else str(low_date)
                result['gain_from_low_yuan'] = round(current_price - result['recent_low_price'], 2)
                result['gain_from_low_pct'] = round((current_price / result['recent_low_price'] - 1) * 100, 2)

                result['is_new_high_today'] = False
                result['is_new_low_today'] = False

            # 2. 斐波那契时间窗口分析
            fib_result = self._calculate_fibonacci_windows(df, current_price)
            result.update(fib_result)

        except Exception as e:
            self.logger.warning(f"时空维度分析计算失败：{str(e)}")

        return result

    def _calculate_fibonacci_windows(self, df: pd.DataFrame, current_price: float) -> dict:
        """
        计算斐波那契时间窗口

        Args:
            df: 历史数据DataFrame
            current_price: 当前价格

        Returns:
            dict: 包含斐波那契窗口分析数据
        """
        result = {}
        config = self.config.SPATIAL_TEMPORAL_CONFIG

        try:
            # 获取斐波那契数列
            fib_sequence = config['fibonacci_sequence']
            window_threshold = config['fibonacci_window_threshold']

            # 寻找趋势起点（近期重要低点）
            close_data = df['close'].values
            n = len(close_data)

            if n < 10:
                return result

            # 使用局部极值检测寻找近期低点（取最近30个交易日内）
            search_range = min(30, n // 2)
            recent_close = close_data[-search_range:]

            # 找到最低点作为起点
            min_idx_in_recent = np.argmin(recent_close)
            fib_start_idx = n - search_range + min_idx_in_recent

            fib_start_price = float(close_data[fib_start_idx])
            fib_start_date = df.loc[fib_start_idx, 'date'] if 'date' in df.columns else fib_start_idx
            fib_start_date_str = fib_start_date.strftime('%Y-%m-%d') if hasattr(fib_start_date, 'strftime') else str(fib_start_date)

            # 计算起点至今的天数
            days_from_start = n - 1 - fib_start_idx

            result['fib_start_date'] = fib_start_date_str
            result['fib_start_price'] = round(fib_start_price, 2)
            result['days_from_start'] = int(days_from_start)

            # 计算各斐波那契窗口状态
            fib_windows = {}
            nearest_window = None
            nearest_distance = float('inf')

            for fib_num in fib_sequence:
                if days_from_start >= fib_num:
                    # 已通过的窗口
                    fib_windows[f'F{fib_num}'] = 'passed'
                else:
                    # 未来的窗口
                    days_to_window = fib_num - days_from_start
                    if days_to_window <= nearest_distance:
                        nearest_distance = days_to_window
                        nearest_window = f'F{fib_num}'
                    fib_windows[f'F{fib_num}'] = 'future'

            result['fib_windows'] = fib_windows

            # 判断是否临近窗口
            is_near_window = nearest_distance <= window_threshold
            result['fib_near_window'] = bool(is_near_window)

            if nearest_window:
                result['fib_nearest_window'] = f'{nearest_window}(距{nearest_distance}日)'

        except Exception as e:
            self.logger.warning(f"斐波那契窗口计算失败：{str(e)}")

        return result

    def calculate_volume_comparison(self, df: pd.DataFrame, current_volume: float) -> dict:
        """
        计算成交量环比数据

        Args:
            df: 历史数据DataFrame
            current_volume: 当日成交量（手）

        Returns:
            dict: 包含较前日、5日均量、20日均量的环比数据
        """
        result = {}
        config = self.config.VOLUME_COMPARISON_CONFIG

        try:
            volume_data = df['volume'].values
            n = len(volume_data)

            if n < 2:
                return result

            # 当日成交量（转换为万手：1手=100股，所以先除以100再除以10000）
            result['volume_current_wan'] = round(current_volume / 100 / 10000, 1)

            # 前一日成交量
            result['volume_yesterday_wan'] = round(volume_data[-2] / 100 / 10000, 1)
            result['volume_change_yoy'] = round((current_volume / volume_data[-2] - 1) * 100, 2)

            # 5日均量
            short_period = config['short_ma_period']
            if n >= short_period + 1:
                volume_5d_avg = np.mean(volume_data[-short_period-1:-1])
                result['volume_5d_avg_wan'] = round(volume_5d_avg / 100 / 10000, 1)
                result['volume_vs_5d_avg'] = round((current_volume / volume_5d_avg - 1) * 100, 2)

            # 20日均量
            medium_period = config['medium_ma_period']
            if n >= medium_period + 1:
                volume_20d_avg = np.mean(volume_data[-medium_period-1:-1])
                result['volume_20d_avg_wan'] = round(volume_20d_avg / 100 / 10000, 1)
                result['volume_vs_20d_avg'] = round((current_volume / volume_20d_avg - 1) * 100, 2)

            # 量能位置评级
            if 'volume_5d_avg_wan' in result and result['volume_5d_avg_wan'] > 0:
                ratio = current_volume / volume_5d_avg if volume_5d_avg > 0 else 1
                if ratio >= config['high_volume_threshold']:
                    result['volume_level'] = '高位'
                elif ratio <= config['low_volume_threshold']:
                    result['volume_level'] = '低位'
                else:
                    result['volume_level'] = '正常'
            else:
                result['volume_level'] = '正常'

        except Exception as e:
            self.logger.warning(f"量能环比计算失败：{str(e)}")

        return result

    def calculate_limit_status(self, df: pd.DataFrame, stock_code: str) -> dict:
        """
        计算涨跌停状态

        Args:
            df: 历史数据DataFrame
            stock_code: 股票代码

        Returns:
            dict: 包含涨跌停状态、当日涨跌幅等数据
        """
        result = {}
        config = self.config.LIMIT_CONFIG

        try:
            if len(df) < 2:
                return result

            # 获取当日涨跌幅
            today_data = df.iloc[-1]
            pct_chg = today_data.get('pct_chg', 0)

            # 如果没有涨跌幅数据，自己计算
            if pd.isna(pct_chg) or pct_chg == 0:
                prev_close = df.iloc[-2]['close']
                current_close = today_data['close']
                if prev_close > 0:
                    pct_chg = (current_close / prev_close - 1) * 100
                else:
                    pct_chg = 0

            # 判断是否为ST股票（股票代码包含ST）
            is_st = 'ST' in stock_code

            # 判断是否为科创板/创业板（以688、300、301开头）
            is_star_board = stock_code.startswith('688') or stock_code.startswith('300') or stock_code.startswith('301')

            # 根据股票类型确定涨跌停阈值
            if is_st:
                limit_up = config['st_limit_up_threshold']
                limit_down = config['st_limit_down_threshold']
            elif is_star_board:
                limit_up = config['star_threshold']
                limit_down = -config['star_threshold']
            else:
                limit_up = config['limit_up_threshold']
                limit_down = config['limit_down_threshold']

            # 判断涨跌停状态
            if pct_chg >= limit_up:
                limit_status = '涨停'
            elif pct_chg <= limit_down:
                limit_status = '跌停'
            elif abs(pct_chg) <= config['normal_threshold']:
                limit_status = '正常'
            else:
                limit_status = '正常'

            result['limit_status'] = limit_status
            result['daily_change_pct'] = round(pct_chg, 2)
            result['is_limit_up'] = pct_chg >= limit_up
            result['is_limit_down'] = pct_chg <= limit_down
            result['is_normal'] = not result['is_limit_up'] and not result['is_limit_down']

        except Exception as e:
            self.logger.warning(f"涨跌停状态计算失败：{str(e)}")

        return result

    def calculate_turnover_and_volatility(self, df: pd.DataFrame) -> dict:
        """
        计算换手率数据和历史波动率

        Args:
            df: 历史数据DataFrame（需包含换手率列）

        Returns:
            dict: 包含换手率、换手率均值、历史波动率等数据
        """
        result = {}
        config = self.config.TURNOVER_CONFIG

        try:
            n = len(df)
            if n < 2:
                return result

            # 获取换手率数据
            if 'turnover_rate' in df.columns:
                turnover_data = df['turnover_rate'].values
            else:
                # 如果没有换手率数据，尝试从原始列名获取
                if '换手率' in df.columns:
                    turnover_data = df['换手率'].values
                else:
                    self.logger.warning("未找到换手率数据")
                    return result

            # 当日换手率
            current_turnover = float(turnover_data[-1]) if not pd.isna(turnover_data[-1]) else 0.0
            result['turnover_rate'] = round(current_turnover, 2)

            # 计算各周期换手率均值
            for period in config['ma_periods']:
                if n >= period:
                    # 取最近period天的数据（包含今日）
                    recent_data = turnover_data[-period:]
                    # 过滤NaN值
                    valid_data = recent_data[~pd.isna(recent_data)]
                    if len(valid_data) > 0:
                        avg_turnover = float(np.mean(valid_data))
                        result[f'turnover_avg_{period}d'] = round(avg_turnover, 2)
                    else:
                        result[f'turnover_avg_{period}d'] = 0.0
                else:
                    result[f'turnover_avg_{period}d'] = 0.0

            # 计算历史波动率（年化）
            volatility_period = min(config['volatility_period'], n)
            if volatility_period >= 2:
                # 获取收盘价数据
                close_data = df['close'].values

                # 计算日收益率
                returns = np.diff(np.log(close_data[-volatility_period:]))

                # 计算标准差作为波动率
                volatility_daily = np.std(returns)

                # 年化波动率（假设一年约250个交易日）
                volatility_annual = volatility_daily * np.sqrt(250) * 100

                result['volatility_annual'] = round(volatility_annual, 2)

        except Exception as e:
            self.logger.warning(f"换手率与波动率计算失败：{str(e)}")

        return result

    def calculate_consecutive_days(self, df: pd.DataFrame) -> dict:
        """
        计算连续涨跌日统计数据

        Args:
            df: 历史数据DataFrame

        Returns:
            dict: 包含连续上涨日数、连续下跌日数、期间涨跌幅等数据
        """
        result = {}

        try:
            close_data = df['close'].values
            n = len(close_data)

            if n < 2:
                return result

            # 计算每日涨跌（1为涨，-1为跌，0为平）
            changes = np.diff(close_data)

            # 统计连续上涨日数（从最新日期往前数）
            consecutive_up = 0
            consecutive_up_start_price = close_data[-1]
            for i in range(len(changes) - 1, -1, -1):
                if changes[i] > 0:
                    consecutive_up += 1
                elif changes[i] < 0:
                    break
                # 如果涨跌幅为0，继续往前数

            if consecutive_up > 0:
                consecutive_up_start_idx = n - 1 - consecutive_up
                consecutive_up_start_price = close_data[consecutive_up_start_idx] if consecutive_up_start_idx >= 0 else close_data[0]
                consecutive_up_gain = ((close_data[-1] - consecutive_up_start_price) / consecutive_up_start_price * 100) if consecutive_up_start_price > 0 else 0
                result['consecutive_up_days'] = consecutive_up
                result['consecutive_up_start_price'] = round(consecutive_up_start_price, 2)
                result['consecutive_up_gain_pct'] = round(consecutive_up_gain, 2)
            else:
                result['consecutive_up_days'] = 0
                result['consecutive_up_start_price'] = round(close_data[-1], 2)
                result['consecutive_up_gain_pct'] = 0.0

            # 统计连续下跌日数（从最新日期往前数）
            consecutive_down = 0
            for i in range(len(changes) - 1, -1, -1):
                if changes[i] < 0:
                    consecutive_down += 1
                elif changes[i] > 0:
                    break
                # 如果涨跌幅为0，继续往前数

            if consecutive_down > 0:
                consecutive_down_start_idx = n - 1 - consecutive_down
                consecutive_down_start_price = close_data[consecutive_down_start_idx] if consecutive_down_start_idx >= 0 else close_data[0]
                consecutive_down_loss = ((close_data[-1] - consecutive_down_start_price) / consecutive_down_start_price * 100) if consecutive_down_start_price > 0 else 0
                result['consecutive_down_days'] = consecutive_down
                result['consecutive_down_start_price'] = round(consecutive_down_start_price, 2)
                result['consecutive_down_loss_pct'] = round(consecutive_down_loss, 2)
            else:
                result['consecutive_down_days'] = 0
                result['consecutive_down_start_price'] = round(close_data[-1], 2)
                result['consecutive_down_loss_pct'] = 0.0

        except Exception as e:
            self.logger.warning(f"连续涨跌日统计计算失败：{str(e)}")

        return result
