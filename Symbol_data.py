"""Crate your own dreams by yore own hands grate man"""
import time
import mplfinance
import numpy as np
import pandas as pd
import pandas.core.frame
import pandas.core.series
from urllib.error import URLError
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
from datetime import datetime, timedelta


class SymbolData:
    """
import matplotlib.pyplot as plt
    This class initialization ends up with returning
    a dataframe and using self.ohlc you can get to a
    pandas dataframe ready to be plotted on a chart
    or to be analyzed utilizing yor ML algorithm
    """
    ohlc = None

    def __init__(self, symbol: str, from_candle=0, number_of_years=3) -> None:
        self.ticker = symbol
        today = datetime.now()
        current_time = timedelta(hours=today.hour, minutes=today.minute, seconds=(today.second + 1), microseconds=today.microsecond)
        yesterday = today - current_time
        _3_years_ago = yesterday - timedelta(365 * number_of_years)
        try:
            url = f"https://query1.finance.yahoo.com/v7/finance/download/{self.ticker}" \
                  f"?period1={int(time.mktime(_3_years_ago.timetuple()))}&" \
                  f"period2={int(time.mktime(yesterday.timetuple()))}&" \
                  f"interval=1d&events=history&includeAdjustedClose=true"
            df = pd.read_csv(url)
            df = df.set_index(pd.DatetimeIndex(df["Date"]))
            SymbolData.ohlc = df.loc[:, ["Open", "High", "Low", "Close", "Volume"]]
            if from_candle > 0:
                SymbolData.ohlc = SymbolData.ohlc.iloc[df.count(0)["Open"] - from_candle:]
        except URLError:
            raise ConnectionError("Check name of your pair (it is mostly all in uppercase)"
                                  "if no there was no problem there then"
                                  "check your internet connection or use vpn (normally not necessary)")

    def candle_chart(self, mav=tuple(), volume=False) -> None:
        """
        plot your dataframe on a candle stick chart
        :param mav: set mav=(5, 10, ... up to 7 mavs) and see moving averages on chart
        :param volume: set volume(volume=Ture) to plot it on chart
        """
        plot_reqs = self.ohlc
        k = dict()
        if len(mav) != 0:
            k.update({"mav": mav})
        if volume:
            k.update({"volume": True})
        mplfinance.plot(plot_reqs, type="candle", **k)

    @staticmethod
    def list_of_symbols() -> list:
        """
        This func returns a list of crypto assets available,
        the list you get is a sample of the assets available for more symbols
        checkout 'https://finance.yahoo.com' and you have a lot more symbols of
        forex pairs , stocks, crypto assets , indexes and so on

        ** note: forex pairs are like: 'EURUSD=X' (i have no idea why they have =X ask
        yahoo finance developers:) )
        :return: a list of available symbols
        """
        symbols_list = [
            'BTC-USD', 'ETH-USD', 'USDT-USD', 'BNB-USD', 'USDC-USD', 'ADA-USD', 'HEX-USD', 'XRP-USD', 'DOGE-USD', 'AVAX-USD', 'MATIC-USD',
            'SHIB-USD', 'CRO-USD', 'LTC-USD', 'LINK-USD', 'TRX-USD', 'ALGO-USD', 'BCH-USD', 'FTM-USD', 'XLM-USD', 'MANA-USD', 'HBAR-USD',
            'ETC-USD', 'VET-USD', 'SAND-USD', 'XTZ-USD', 'FIL-USD', 'EGLD-USD', 'AXS-USD', 'THETA-USD', 'XMR-USD', 'MIOTA-USD', 'EOS-USD',
            'MKR-USD', 'AAVE-USD', 'CAKE-USD', 'BSV-USD', 'QNT-USD', 'NEO-USD', 'ENJ-USD', 'TUSD-USD', 'CRV-USD', 'KSM-USD', 'ZEC-USD',
            'RUNE-USD', 'BAT-USD', 'CELO-USD', 'AR-USD', 'ROSE-USD', 'LRC-USD', 'CHZ-USD', 'OMI-USD', 'DASH-USD', 'CCXX-USD', 'KDA-USD',
            'WAVES-USD', 'XEM-USD', 'TFUEL-USD', 'YFI-USD', 'DCR-USD', 'DFI-USD', 'IOTX-USD', 'RVN-USD', 'XDC-USD', 'OMG-USD', 'WAXP-USD',
            'QTUM-USD', 'ANKR-USD', 'CEL-USD', 'BNT-USD', 'SNX-USD', 'ZIL-USD', 'VLX-USD', 'GNO-USD', 'BTG-USD', 'ICX-USD', 'SC-USD',
            'SUSHI-USD', 'VGX-USD', 'KAVA-USD', 'ZRX-USD', 'ZEN-USD', 'IOST-USD', 'ONT-USD', 'SYS-USD', 'STORJ-USD', 'CKB-USD', 'HIVE-USD',
            'XWC-USD', 'UMA-USD', 'SKL-USD', 'GLM-USD', 'NU-USD', 'CELR-USD', 'DGB-USD', 'SRM-USD', 'RAY-USD', 'ANT-USD', 'SXP-USD',
            'COTI-USD', 'XCH-USD', 'RSR-USD', 'SAPP-USD', 'CTSI-USD', 'FET-USD', 'MED-USD', 'DIVI-USD', 'LSK-USD', 'DAG-USD', 'ARDR-USD',
            'TWT-USD', 'VERI-USD', 'CVC-USD', 'MAID-USD', 'SNT-USD', 'EWT-USD', 'XVG-USD', 'BCD-USD', 'NMR-USD', 'OXT-USD', 'REP-USD',
            'RLC-USD', 'VTHO-USD', 'STMX-USD', 'ACH-USD', 'NKN-USD', 'STRAX-USD', 'ARRR-USD', 'ARK-USD', 'META-USD', 'STEEM-USD', 'ETN-USD',
            'BAND-USD', 'ABBC-USD', 'FUN-USD', 'GXC-USD', 'ERG-USD', 'MTL-USD', 'DERO-USD', 'TOMO-USD', 'MLN-USD', 'XNC-USD', 'HNS-USD',
            'RBTC-USD', 'CLV-USD', 'ELA-USD', 'BAL-USD', 'IRIS-USD', 'VRA-USD', 'KIN-USD', 'WAN-USD', 'REV-USD', 'CUDOS-USD', 'TT-USD',
            'BTS-USD', 'MWC-USD', 'PHA-USD', 'ADX-USD', 'MONA-USD', 'ZNN-USD', 'KMD-USD', 'ATRI-USD', 'DMCH-USD', 'AVA-USD', 'CTXC-USD',
            'AXEL-USD', 'MARO-USD', 'NYE-USD', 'DNT-USD', 'GAS-USD', 'NRG-USD', 'FIRO-USD', 'FSN-USD', 'GRS-USD', 'SBD-USD', 'FIO-USD',
            'AION-USD', 'BTM-USD', 'XHV-USD', 'WTC-USD', 'DGD-USD', 'APL-USD', 'SCP-USD', 'NULS-USD', 'FRONT-USD', 'BCN-USD', 'CET-USD',
            'SOLVE-USD', 'BEAM-USD', 'AE-USD', 'SERO-USD', 'VSYS-USD', 'KRT-USD', 'QASH-USD', 'WOZX-USD', 'WICC-USD', 'PAC-USD', 'POA-USD',
            'NIM-USD', 'VITE-USD', 'XCP-USD', 'GO-USD', 'NMC-USD', 'NXS-USD', 'ZYN-USD', 'RDD-USD', 'PIVX-USD', 'BEPRO-USD', 'VTC-USD',
            'PART-USD', 'MHC-USD', 'LBC-USD', 'PCX-USD', 'PPT-USD', 'GAME-USD', 'OBSR-USD', 'CRU-USD', 'CUT-USD', 'GBYTE-USD', 'HC-USD',
            'PPC-USD', 'GRIN-USD', 'QRL-USD', 'RSTR-USD', 'NAS-USD', 'WABI-USD', 'AMB-USD', 'CHI-USD', 'BIP-USD', 'ZANO-USD', 'SRK-USD',
            'NAV-USD', 'FO-USD', 'NEBL-USD', 'RINGX-USD', 'FCT-USD', 'ETP-USD', 'NXT-USD', 'TRUE-USD', 'SALT-USD', 'PZM-USD', 'XSN-USD',
            'NVT-USD', 'SFT-USD', 'DCN-USD', 'ADK-USD', 'PAY-USD', 'LCC-USD', 'DMD-USD', 'DTEP-USD', 'YOYOW-USD', 'GHOST-USD', 'PI-USD',
            'PAI-USD', 'WGR-USD', 'INSTAR-USD', 'COLX-USD', 'SKY-USD', 'IDNA-USD', 'ACT-USD', 'MAN-USD', 'BHP-USD', 'GRC-USD', 'BLOCK-USD',
            'UBQ-USD', 'XMC-USD', 'NLG-USD', 'HPB-USD', 'OTO-USD', 'INT-USD', 'PLC-USD', 'MRX-USD', 'QRK-USD', 'HTML-USD', 'VIN-USD', 'ILC-USD',
            'MASS-USD', 'POLIS-USD', 'EDG-USD', 'VIA-USD', 'SMART-USD', 'VEX-USD', 'GLEEC-USD', 'AEON-USD', 'MIR-USD', 'LEDU-USD', 'FTC-USD',
            'XST-USD', 'TRTL-USD', 'BLK-USD', 'BTX-USD', 'DYN-USD', 'XDN-USD', 'OWC-USD', 'CURE-USD', 'BHD-USD', 'XRC-USD', 'XMY-USD',
            'TERA-USD', 'DIME-USD', 'BCA-USD', 'WINGS-USD', 'NYZO-USD', 'FTX-USD', 'PHR-USD', 'IOC-USD', 'CRW-USD', 'AYA-USD', 'SUB-USD',
            'TUBE-USD', 'MBC-USD', 'XLT-USD', 'NIX-USD', 'HTDF-USD', 'DDK-USD', 'MGO-USD', 'FAIR-USD', 'HYC-USD', 'BPS-USD', 'XAS-USD',
            'USNBT-USD', 'XBY-USD', 'ERK-USD', 'ATB-USD', 'FRST-USD', 'BPC-USD', 'LKK-USD', 'BONO-USD', 'ECC-USD', 'UNO-USD', 'CSC-USD',
            'MOAC-USD', 'ECA-USD', 'CLAM-USD', 'BDX-USD', 'FLASH-USD', 'ALIAS-USD', 'DACC-USD', 'SPHR-USD', 'RBY-USD', 'HNC-USD', 'MINT-USD',
            'AIB-USD', 'XUC-USD', 'CTC-USD', 'DUN-USD', 'CCA-USD', 'JDC-USD', 'DCY-USD', 'SLS-USD', 'MIDAS-USD', 'LRG-USD', 'GRN-USD', 'VBK-USD',
            'BONFIRE-USD', 'BST-USD', 'TVK-USD', 'GBPUSD=X', 'EURUSD=X', 'USDCHF=X']
        return symbols_list


def bull_bear(candle: pandas.core.series.Series) -> int:
    """
    return a number from -1 to 1 which indicates that the given candle
    is a bullish candle or a bearish one
    :param candle: one candle from Sd.ohlc dataframe
    :return: int
    """
    Close = candle["Close"]
    Open = candle["Open"]
    if Close > Open:
        return 1
    if Close < Open:
        return -1
    if Close == Open:
        return 0


def starting_point_finder(dataframe: pandas.core.frame.DataFrame) -> pandas.core.frame.DataFrame:
    """
    finds the highest or lowest (the earliest one) within the given dataframe
    :param dataframe: Sd.ohlc
    :return: a candle if no error occurs
    """
    highest_high = dataframe.loc[dataframe["High"] == dataframe.max()["High"]]
    lowest_low = dataframe.loc[dataframe["Low"] == dataframe.min()["Low"]]
    if highest_high.index[0].date() > lowest_low.index[0].date():
        return highest_high
    elif highest_high.index[0].date() < lowest_low.index[0].date():
        return lowest_low
    else:
        print("Something went wrong")
        return pd.DataFrame()


def ATR(candle: pandas.core.series.Series, dataframe: pandas.core.frame.DataFrame, __n__=30, n=30) -> float:
    """
    calculates the ATR of the given candle in the given dataframe
    :param candle: one candle from Sd.ohlc dataframe
    :param dataframe: Sd.ohlc
    :param __n__: should not be changed in no situation
    :param n: if you wanted to change the period change this n and the other __n__ at
    the same time so the func works as it was (but don't change them for god's sack)
    :return: float
    """
    if __n__ == 0:
        return 0
    else:
        high = candle["High"]
        low = candle["Low"]
        Cp = dataframe.iloc[(1 if (candle_index := dataframe.index.get_loc(candle.name)) == 0 else candle_index) - 1]["Close"]
        local_tr = max(abs(high - low), abs(high - Cp), abs(low - Cp)) / n
        return local_tr + ATR(dataframe.iloc[candle_index - 1], dataframe, __n__ - 1)


def Pivot(dataframe: pandas.core.frame.DataFrame, Order=3, **kwargs) -> pandas.core.frame.DataFrame:
    """
    finds the high and low pivots within the dataframe
    :param dataframe: a dataframe to search in
    :param Order: defines the range of candles to check in order to find the local optimum
    :param kwargs: set draw_on_chart on true to draw pivots on chart
    :return: a new dataframe with 2 new columns for high and low pivots with true and false vals
    """
    local_highs = argrelextrema(dataframe.High.values, np.greater_equal, order=Order)[0]  # order=3
    local_lows = argrelextrema(dataframe.Low.values, np.less_equal, order=Order)[0]
    dataframe["HighPivots"] = [True if i in dataframe.iloc[local_highs].index else False for i in dataframe.index]
    dataframe["LowPivots"] = [True if i in dataframe.iloc[local_lows].index else False for i in dataframe.index]
    if kwargs.get("draw_on_chart") is True:
        dataframe.High.plot()
        dataframe.Low.plot()
        dataframe.iloc[local_highs].High.plot(style="^", lw=10, color="green")
        dataframe.iloc[local_lows].Low.plot(style="v", lw=10, color="red")
        plt.show()
    return dataframe


eur = SymbolData("EURUSD=X", 500, 3)
Pivot(eur.ohlc, Order=10, draw_on_chart=True)
