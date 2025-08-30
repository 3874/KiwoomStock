import sys
import time
from datetime import datetime, timedelta
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QEventLoop
import pandas as pd
import numpy as np
from pykiwoom.kiwoom import *

class KiwoomStrategy:
    def __init__(self):
        self.kiwoom = Kiwoom()
        self.kiwoom.CommConnect(block=True) # 로그인 요청, block=True로 동기처리
        print("키움증권 로그인 완료!")

        self.tr_event_loop = QEventLoop() # TR 요청 시 응답을 기다리기 위한 이벤트 루프

        self.data = {} # 종목별 과거 데이터를 저장할 딕셔너리
        self.results = [] # 조건에 맞는 종목을 저장할 리스트

        # 이벤트 슬롯 연결
        self.kiwoom.OnReceiveTrData.connect(self.on_receive_tr_data)

        # 지표 계산을 위한 기본 파라미터
        self.RSI_PERIOD = 14
        self.STOCHASTIC_K_PERIOD = 14
        self.STOCHASTIC_D_PERIOD = 3 # SlowK
        self.STOCHASTIC_SD_PERIOD = 3 # SlowD
        self.BB_PERIOD = 20
        self.BB_STD_DEV = 2

        # 최근 며칠 이내의 데이터를 체크할 것인지 (1주일 = 5거래일)
        self.RECENT_DAYS_CHECK = 5 
        # 과거 몇일치 데이터를 조회할 것인지 (지표 계산에 필요한 충분한 기간)
        self.HISTORICAL_DATA_DAYS = 200 
        
        # TR 요청 딜레이 (너무 빠르면 요청 제한에 걸림)
        self.TR_REQUEST_DELAY = 0.25 # 0.25초

    def on_receive_tr_data(self, screen_no, rqname, trcode, record_name, sPrevNext, data_len, err_code, msg, splm_msg):
        if rqname == "주식일봉차트":
            code = self.kiwoom.GetCommData(trcode, rqname, 0, "종목코드")
            df = self.kiwoom.block_request(trcode, rqname) # DataFrame으로 편리하게 가져옴
            
            # 컬럼명 정리 및 데이터 타입 변환
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj_Close'] # Adj_Close는 수정주가 반영된 현재가
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy() # 필요한 컬럼만
            df['Date'] = pd.to_datetime(df['Date'])
            df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(int)
            
            # 데이터는 최신 날짜부터 역순으로 오므로, 오래된 날짜부터 정렬
            df = df.sort_values(by='Date').reset_index(drop=True)

            self.data[code] = df
            
            # TR 요청의 마지막인지 확인 (더이상 다음 데이터가 없으면)
            if sPrevNext == "0": 
                self.tr_event_loop.exit() # 이벤트 루프 종료하여 다음 종목 처리로 넘어감

    def calculate_indicators(self, df):
        if len(df) < max(self.RSI_PERIOD, self.STOCHASTIC_K_PERIOD, self.BB_PERIOD):
            # 지표 계산에 필요한 최소 데이터 수 부족
            return df

        # --- RSI 계산 ---
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.RSI_PERIOD).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # --- Stochastics 계산 ---
        df['Lowest_Low'] = df['Low'].rolling(window=self.STOCHASTIC_K_PERIOD).min()
        df['Highest_High'] = df['High'].rolling(window=self.STOCHASTIC_K_PERIOD).max()
        df['FastK'] = ((df['Close'] - df['Lowest_Low']) / (df['Highest_High'] - df['Lowest_Low'])) * 100
        df['SlowK'] = df['FastK'].rolling(window=self.STOCHASTIC_D_PERIOD).mean() # 통상적인 %K
        df['SlowD'] = df['SlowK'].rolling(window=self.STOCHASTIC_SD_PERIOD).mean() # 통상적인 %D

        # --- Bollinger Bands 계산 ---
        df['MA'] = df['Close'].rolling(window=self.BB_PERIOD).mean()
        df['StdDev'] = df['Close'].rolling(window=self.BB_PERIOD).std()
        df['UpperBand'] = df['MA'] + (df['StdDev'] * self.BB_STD_DEV)
        df['LowerBand'] = df['MA'] - (df['StdDev'] * self.BB_STD_DEV)

        # 불필요한 임시 컬럼 제거
        df = df.drop(columns=['Lowest_Low', 'Highest_High', 'MA', 'StdDev'], errors='ignore')
        
        return df

    def apply_strategy(self, code, df):
        # 충분한 데이터가 있는지 확인 (지표 계산 후 NaN이 아닌 유효한 데이터가 최소 최근 N일치 있어야 함)
        if len(df.dropna()) < self.RECENT_DAYS_CHECK + max(self.RSI_PERIOD, self.STOCHASTIC_K_PERIOD, self.BB_PERIOD):
            # print(f"  {code}: 데이터 부족으로 전략 적용 불가.")
            return False

        # 최근 데이터 추출
        recent_data = df.iloc[-self.RECENT_DAYS_CHECK:].copy()
        current_data = df.iloc[-1] # 가장 최근 데이터 (오늘)
        prev_data = df.iloc[-2]    # 어제 데이터

        # 1. RSI 조건: 최근 N일 이내 RSI 30 이하 진입 이력
        rsi_condition = (recent_data['RSI'] < 30).any()

        # 2. Stochastics 조건: 현재 SlowK, SlowD 모두 20 이하 & SlowK가 SlowD를 상향 돌파 (골든크로스)
        stoch_cond1 = current_data['SlowK'] <= 20 and current_data['SlowD'] <= 20
        stoch_cond2 = current_data['SlowK'] > current_data['SlowD'] and prev_data['SlowK'] <= prev_data['SlowD']
        stoch_condition = stoch_cond1 and stoch_cond2

        # 3. Bollinger Bands 조건: 최근 N일 이내 하한선 터치 또는 이탈 이력 & 현재는 하한선 위로 복귀 (반등 시작)
        bb_touch_condition = (recent_data['Low'] <= recent_data['LowerBand']).any()
        bb_rebound_condition = current_data['Close'] > current_data['LowerBand']

        # 4. 가격 반등 조건: 현재 종가가 전일 종가보다 상승 (양봉 또는 상승 마감)
        # 단기 반등을 노리는 것이므로, 어제보다 오늘 종가가 상승한 것이 중요
        price_rebound_condition = current_data['Close'] > prev_data['Close']
        
        # 5. 거래량 조건 (선택 사항): 반등 시 거래량 증가 확인
        # 현재 거래량이 최근 5일 평균 거래량보다 높거나, 특정 값 이상으로 급증
        # avg_volume_recent = df['Volume'].iloc[-self.RECENT_DAYS_CHECK-1:-1].mean() # 전일~5일전 평균
        # volume_condition = current_data['Volume'] > (avg_volume_recent * 1.5) # 1.5배 이상 증가

        # 모든 조건 만족 여부
        if (rsi_condition and
            stoch_condition and
            bb_touch_condition and
            bb_rebound_condition and
            price_rebound_condition): # and volume_condition):
            
            print(f"  [발견] 종목: {code} - {self.kiwoom.GetMasterCodeName(code)}")
            print(f"    RSI({self.RSI_PERIOD}): {current_data['RSI']:.2f} (최근 {self.RECENT_DAYS_CHECK}일 이내 30 이하 이력)")
            print(f"    Stochastics({self.STOCHASTIC_K_PERIOD},{self.STOCHASTIC_D_PERIOD},{self.STOCHASTIC_SD_PERIOD}): K={current_data['SlowK']:.2f}, D={current_data['SlowD']:.2f} (골든크로스 & 20 이하)")
            print(f"    Bollinger Bands({self.BB_PERIOD},{self.BB_STD_DEV}): 하한선={current_data['LowerBand']:.2f}, 현재가={current_data['Close']:.2f} (최근 {self.RECENT_DAYS_CHECK}일 이내 하한선 터치 & 현재 하한선 위)")
            print(f"    가격 반등: 전일 {prev_data['Close']} -> 현재 {current_data['Close']}")
            print("-" * 50)
            self.results.append(code)
            return True
        return False

    def run_scan(self):
        # 시장 코드 가져오기 (코스피: 0, 코스닥: 10)
        kospi_codes = self.kiwoom.GetCodeListByMarket('0')
        kosdaq_codes = self.kiwoom.GetCodeListByMarket('10')
        all_codes = kospi_codes + kosdaq_codes
        
        print(f"총 {len(all_codes)}개의 종목 스캔 시작...")
        
        # 오늘 날짜를 YYYYMMDD 형태로 가져옴
        today = datetime.now().strftime('%Y%m%d')

        for i, code in enumerate(all_codes):
            if i % 100 == 0:
                print(f"진행 중: {i}/{len(all_codes)} 종목 처리 중...")

            # 과거 데이터 요청 (TR 요청 제한 고려하여 딜레이 부여)
            self.kiwoom.SetInputValue("종목코드", code)
            self.kiwoom.SetInputValue("기준일자", today)
            self.kiwoom.SetInputValue("수정주가구분", "1") # 1: 수정주가 반영

            self.kiwoom.CommRqData("주식일봉차트", "OPT10081", "0", "0101")
            self.tr_event_loop.exec_() # 데이터 수신 완료될 때까지 대기

            time.sleep(self.TR_REQUEST_DELAY) # 요청 딜레이

            if code in self.data:
                df = self.data[code]
                # 충분한 과거 데이터가 있어야 지표 계산 가능
                if len(df) >= self.HISTORICAL_DATA_DAYS:
                    df_with_indicators = self.calculate_indicators(df)
                    self.apply_strategy(code, df_with_indicators)
                # else:
                    # print(f"  {code}: 필요한 최소 데이터({self.HISTORICAL_DATA_DAYS}일) 부족. 현재 {len(df)}일치.")
            # else:
                # print(f"  {code}: 데이터 수신 실패 또는 데이터 없음.")

        print("\n--- 스캔 완료 ---")
        if self.results:
            print(f"총 {len(self.results)}개의 저평가 반등 가능성이 있는 종목을 찾았습니다:")
            for code in self.results:
                print(f"- {code} ({self.kiwoom.GetMasterCodeName(code)})")
        else:
            print("조건에 맞는 종목을 찾지 못했습니다.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    strategy = KiwoomStrategy()
    strategy.run_scan()
    sys.exit(app.exec_()) # PyQt 앱 종료