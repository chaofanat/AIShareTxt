from AIShareTxt.core.data_fetcher import StockDataFetcher
from datetime import datetime

print('=== 测试混合判断策略 ===')
fetcher = StockDataFetcher()

# 测试今天
today = datetime.now().date()
result = fetcher._is_trading_day(today)
print(f'今天 {today} 是否为交易日: {result}')

# 测试完整的判断
result2 = fetcher._is_trading_day_and_not_closed()
print(f'今天是否为交易日且未收盘: {result2}')