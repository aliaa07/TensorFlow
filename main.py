import pandas as pd
import pandas.core.series

from Symbol_data import SymbolData as Sd

btc = Sd("BTC-USD", 500, 3)
a = pd.DataFrame(btc.ohlc)
a.to_csv("btc.csv")#, index=False)


def bull_bear(candle: pandas.core.series.Series):
    Close = candle["Close"]
    Open = candle["Open"]
    if Close > Open:
        return 1
    if Close < Open:
        return -1
    if Close == Open:
        return 0


a = bull_bear(btc.ohlc.iloc[1])
