import time
from urllib.error import URLError
import mplfinance
import pandas as pd
from datetime import datetime, timedelta


class SymbolData:
    ohlc = None

    def __init__(self, symbol: str, from_candle=0, number_of_years=3) -> None:
        """
        This class initialization ends up with returning
        a dataframe and using self.ohlc you can get to a
        pandas dataframe ready to be plotted on a chart
        or to be analyzed utilizing yor ML algorithm
        """
        self.ticker = symbol
        if symbol not in SymbolData.list_of_symbols():
            raise ValueError("What? I can't find that symbol, must be a shit coin isn't it?")
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
            raise ConnectionError("Check your internet connection or use vpn(normally not necessary)")

    def candle_chart(self, mav=tuple(), volume=False) -> None:
        """
        plot your dataframe on a candle stick chart you can also plot
        moving averages(using mav=(5, 10, ... up to 7 mavs)) or show
        volume(using volume=Ture) on chart
        """
        plot_reqs = self.ohlc
        k = dict()
        if len(mav) != 0:
            k.update({"mav": mav})
        if volume:
            k.update({"volume": True})
        mplfinance.plot(plot_reqs, type="candle", **k)

    @staticmethod
    def list_of_symbols():
        """Returns a list of crypto assets available"""
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
            'BONFIRE-USD', 'BST-USD']
        return symbols_list

