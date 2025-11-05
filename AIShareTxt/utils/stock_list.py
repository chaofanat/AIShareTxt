#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
沪深300主板成分股数据获取工具

主要功能：
- 获取沪深300成分股中的主板股票（过滤掉创业板和科创板）
- 返回包含股票代码、名称、价格、涨跌幅等信息的DataFrame
- 基于东方财富行情API，实时数据

使用方法：
    from final_stock_api import get_hs300_main_board_stocks
    
    # 获取数据
    df = get_hs300_main_board_stocks()
    
    if df is not None:
        print(f"获取到 {len(df)} 只股票")
        print(df.head())
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime

class FinalStockAPI:
    def __init__(self):
        self.session = requests.Session()
        self.base_url = "https://push2.eastmoney.com/api/qt/clist/get"
        self.setup_headers()
    
    def setup_headers(self):
        """设置请求头"""
        self.session.headers.update({
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Host': 'push2.eastmoney.com',
            'Referer': 'https://quote.eastmoney.com/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
        })
    
    def fetch_stock_data(self, market_filter="m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23", page_size=50):
        """获取股票数据"""
        
        # 构建请求参数
        params = {
            'pn': '1',                                    # 页码
            'pz': str(page_size),                        # 每页数量
            'po': '1',                                   # 排序
            'np': '1',                                   # 
            'ut': 'bd1d9ddb04089700cf9c27f6f7426281',    # token
            'fltt': '2',                                 # 过滤类型
            'invt': '2',                                 # 
            'fid': 'f3',                                 # 排序字段(涨跌幅)
            'fs': market_filter,                         # 市场过滤
            'fields': 'f12,f14,f2,f3,f4,f5,f6,f7,f15,f16,f17,f18,f8,f10,f9,f23,f20,f21,f13'  # 字段
        }
        
        try:
            # 发起请求
            response = self.session.get(self.base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                response_text = response.text
                
                # 确保内容是JSON格式
                if response_text.strip().startswith('{'):
                    try:
                        data = response.json()
                        
                        # 检查数据结构
                        if 'data' in data and data['data'] and 'diff' in data['data']:
                            stock_list = data['data']['diff']
                            if stock_list:
                                return stock_list
                    
                    except json.JSONDecodeError:
                        pass
        
        except Exception:
            pass
        
        return None
    
    def get_all_stocks(self, page_size=50):
        """获取所有A股"""
        return self.fetch_stock_data("m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23", page_size)
    
    def get_hs300_stocks_all_pages(self):
        """获取全部沪深300成分股（分页获取）"""
        all_stocks = []
        page = 1
        page_size = 100  # 每页100只
        
        while True:
            # 分页获取股票数据
            stocks = self.fetch_stock_data_with_page("b:BK0500", page, page_size)
            
            if stocks:
                all_stocks.extend(stocks)
                
                # 如果获取的股票数量小于page_size，说明已到最后一页
                if len(stocks) < page_size:
                    break
                    
                page += 1
            else:
                break
        
        return all_stocks
    
    def fetch_stock_data_with_page(self, market_filter, page, page_size):
        """带分页的获取股票数据"""
        
        # 构建请求参数
        params = {
            'pn': str(page),                             # 页码
            'pz': str(page_size),                        # 每页数量
            'po': '1',                                   # 排序
            'np': '1',                                   # 
            'ut': 'bd1d9ddb04089700cf9c27f6f7426281',    # token
            'fltt': '2',                                 # 过滤类型
            'invt': '2',                                 # 
            'fid': 'f3',                                 # 排序字段(涨跌幅)
            'fs': market_filter,                         # 市场过滤
            'fields': 'f12,f14,f2,f3,f4,f5,f6,f7,f15,f16,f17,f18,f8,f10,f9,f23,f20,f21,f13'  # 字段
        }
        
        try:
            # 发起请求
            response = self.session.get(self.base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                response_text = response.text
                
                if response_text.strip().startswith('{'):
                    try:
                        data = response.json()
                        
                        # 检查数据结构
                        if 'data' in data and data['data'] and 'diff' in data['data']:
                            stock_list = data['data']['diff']
                            return stock_list
                    
                    except json.JSONDecodeError:
                        pass
        
        except Exception:
            pass
        
        return None
    
    def get_main_board_stocks(self, page_size=50):
        """获取主板股票"""
        return self.fetch_stock_data("m:0 t:6,m:1 t:2,m:1 t:23", page_size)
    
    def get_growth_stocks(self, page_size=50):
        """获取创业板股票"""
        return self.fetch_stock_data("m:0 t:80", page_size)
    
    def is_main_board_stock(self, stock_code):
        """判断是否为主板股票"""
        if not stock_code:
            return False
        
        # 上海主板：600xxx, 601xxx, 603xxx, 605xxx
        # 深圳主板：000xxx, 001xxx, 002xxx
        # 创业板：300xxx（排除）
        # 科创板：688xxx（排除）
        
        if stock_code.startswith(('600', '601', '603', '605')):  # 上海主板
            return True
        elif stock_code.startswith(('000', '001', '002')):      # 深圳主板
            return True
        else:
            return False  # 创业板(300xxx)、科创板(688xxx)等
    
    def format_dataframe(self, stock_list, filter_main_board=True):
        """格式化股票数据为DataFrame"""
        if not stock_list:
            return None
        
        formatted_data = []
        
        for stock in stock_list:
            try:
                stock_code = stock.get('f12', '')
                
                # 如果开启主板过滤，则只保留主板股票
                if filter_main_board and not self.is_main_board_stock(stock_code):
                    continue
                
                # 根据东方财富API字段映射
                formatted_stock = {
                    '股票代码': stock_code,                      # f12: 股票代码
                    '股票名称': stock.get('f14', ''),           # f14: 股票名称
                    '最新价': float(stock.get('f2', 0)) if stock.get('f2') else 0,              # f2: 最新价
                    '涨跌幅(%)': float(stock.get('f3', 0)) if stock.get('f3') else 0,           # f3: 涨跌幅
                    '涨跌额': float(stock.get('f4', 0)) if stock.get('f4') else 0,              # f4: 涨跌额
                    '成交量(手)': int(stock.get('f5', 0)) if stock.get('f5') else 0,           # f5: 成交量
                    '成交额': int(stock.get('f6', 0)) if stock.get('f6') else 0,               # f6: 成交额
                    '振幅(%)': float(stock.get('f7', 0)) if stock.get('f7') else 0,            # f7: 振幅
                    '最高价': float(stock.get('f15', 0)) if stock.get('f15') else 0,           # f15: 最高价
                    '最低价': float(stock.get('f16', 0)) if stock.get('f16') else 0,           # f16: 最低价
                    '今开': float(stock.get('f17', 0)) if stock.get('f17') else 0,             # f17: 今开
                    '昨收': float(stock.get('f18', 0)) if stock.get('f18') else 0,             # f18: 昨收
                    '换手率(%)': float(stock.get('f8', 0)) if stock.get('f8') else 0,          # f8: 换手率
                    '量比': float(stock.get('f10', 0)) if stock.get('f10') else 0,             # f10: 量比
                    '市盈率': float(stock.get('f9', 0)) if stock.get('f9') else 0,             # f9: 市盈率
                    '市净率': float(stock.get('f23', 0)) if stock.get('f23') else 0,           # f23: 市净率
                    '总市值': int(stock.get('f20', 0)) if stock.get('f20') else 0,             # f20: 总市值
                    '流通市值': int(stock.get('f21', 0)) if stock.get('f21') else 0,           # f21: 流通市值
                    '市场': '上交所' if stock.get('f13') == 1 else '深交所',                    # f13: 市场
                    '板块': self.get_board_type(stock_code),                                   # 板块类型
                    '数据时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                formatted_data.append(formatted_stock)
                
            except Exception:
                continue
        
        if formatted_data:
            df = pd.DataFrame(formatted_data)
            # 按换手率由大到小排序
            df = df.sort_values('换手率(%)', ascending=False).reset_index(drop=True)
            return df
        
        return None
    
    def get_board_type(self, stock_code):
        """获取板块类型"""
        if stock_code.startswith(('600', '601', '603', '605')):
            return '上海主板'
        elif stock_code.startswith(('000', '001', '002')):
            return '深圳主板'
        elif stock_code.startswith('300'):
            return '创业板'
        elif stock_code.startswith('688'):
            return '科创板'
        else:
            return '其他'

def get_hs300_main_board_stocks():
    """获取沪深300主板成分股数据DataFrame
    
    Returns:
        pandas.DataFrame: 沪深300主板成分股数据，包含股票代码、名称、价格等信息
        None: 获取失败时返回None
    
    使用示例:
        # 获取数据
        df = get_hs300_main_board_stocks()
        
        # 检查数据
        if df is not None:
            print(f"获取到 {len(df)} 只股票")
            print(df.head())
            
            # 保存到文件
            df.to_csv('hs300_main_board.csv', index=False, encoding='utf-8-sig')
        else:
            print("获取数据失败")
    """
    try:
        # 创建API客户端
        api = FinalStockAPI()
        
        # 获取沪深300成分股数据
        stock_list = api.get_hs300_stocks_all_pages()
        
        if stock_list:
            # 格式化数据（自动过滤掉非主板股票）
            df = api.format_dataframe(stock_list, filter_main_board=True)
            return df
        
        return None
        
    except Exception:
        return None

if __name__ == "__main__":
    # 测试示例
    df = get_hs300_main_board_stocks()
    
    if df is not None:
        print(f"成功获取 {len(df)} 只沪深300主板成分股")
        print("\n板块分布:")
        print(df['板块'].value_counts())
        
        # 保存数据
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"沪深300主板成分股_{timestamp}.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n数据已保存到: {filename}")
    else:
        print("获取数据失败")


# 向后兼容的别名
get_stock_list = get_hs300_main_board_stocks
