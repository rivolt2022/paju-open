"""
강화된 데이터 로더 - 실제 데이터를 제대로 로드하고 통합
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime


class EnhancedDataLoader:
    """강화된 데이터 로더 클래스"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
        
        self.sheets_info = self._load_sheets_info()
    
    def _load_sheets_info(self) -> Dict:
        """sheets.json 파일 로드"""
        sheets_json_path = self.data_dir / "sheets.json"
        if sheets_json_path.exists():
            with open(sheets_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def load_excel_file(self, filename: str) -> Dict[str, pd.DataFrame]:
        """엑셀 파일의 모든 시트 로드"""
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        try:
            excel_file = pd.ExcelFile(file_path)
            sheets = {}
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    sheets[sheet_name] = df
                except Exception as e:
                    print(f"경고: 시트 '{sheet_name}' 읽기 실패: {e}")
            
            return sheets
        except Exception as e:
            print(f"경고: {filename} 읽기 실패: {e}")
            return {}
    
    def load_excel_sheet(self, filename: str, sheet_name: str) -> pd.DataFrame:
        """특정 엑셀 파일의 특정 시트 로드"""
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            return df
        except Exception as e:
            print(f"경고: {filename}::{sheet_name} 읽기 실패: {e}")
            return pd.DataFrame()
    
    def load_tourist_visit_data(self) -> pd.DataFrame:
        """관광지 방문 인구 데이터 로드 (LG유플러스)"""
        df_list = []
        
        # LG유플러스 관광지별 추이 데이터
        sheet_name = "9p_10p_관광지별_추이"
        df = self.load_excel_sheet("3_관광지분석_LG유플러스.xlsx", sheet_name)
        
        if not df.empty and '방문인구(명)' in df.columns:
            df['데이터소스'] = 'LG유플러스'
            df['유형'] = '방문인구'
            
            # 날짜 생성
            if '연도' in df.columns and '월' in df.columns:
                df['date'] = pd.to_datetime(
                    df['연도'].astype(str) + '-' + 
                    df['월'].astype(str).str.zfill(2) + '-01'
                )
            df_list.append(df)
        
        if df_list:
            combined = pd.concat(df_list, ignore_index=True)
            return combined
        return pd.DataFrame()
    
    def load_tourist_revenue_data(self) -> pd.DataFrame:
        """관광지 매출 데이터 로드 (삼성카드)"""
        df_list = []
        
        # 삼성카드 관광지별 추이 데이터
        sheet_name = "9p_10p_관광지별_추이"
        df = self.load_excel_sheet("3_관광지분석_삼성카드.xlsx", sheet_name)
        
        if not df.empty:
            df['데이터소스'] = '삼성카드'
            df['유형'] = '매출'
            
            # 날짜 생성
            if '연도' in df.columns and '월' in df.columns:
                df['date'] = pd.to_datetime(
                    df['연도'].astype(str) + '-' + 
                    df['월'].astype(str).str.zfill(2) + '-01'
                )
            
            # 방문인구 추정 (매출건수 기반)
            if '매출건수(건)' in df.columns:
                df['방문인구(명)'] = df['매출건수(건)'] * 1.5  # 추정값
            elif '매출금액(원)' in df.columns:
                df['방문인구(명)'] = df['매출금액(원)'] / 40000  # 인당 평균 소비액 추정
            df_list.append(df)
        
        if df_list:
            combined = pd.concat(df_list, ignore_index=True)
            return combined
        return pd.DataFrame()
    
    def load_life_population_data(self) -> pd.DataFrame:
        """생활인구 데이터 로드"""
        df_list = []
        
        # LG유플러스 생활인구 수 변화
        sheet_name = "7p 생활인구수 변화"
        df = self.load_excel_sheet("1_생활인구분석_LG유플러스.xlsx", sheet_name)
        
        if not df.empty:
            df['데이터소스'] = 'LG유플러스'
            df['유형'] = '생활인구'
            
            # 날짜 생성
            if '기준연월' in df.columns:
                df['date'] = pd.to_datetime(
                    df['기준연월'].astype(str).str[:4] + '-' +
                    df['기준연월'].astype(str).str[4:6] + '-01'
                )
            elif '기간' in df.columns:
                df['date'] = pd.to_datetime(df['기간'].astype(str) + '-01')
            df_list.append(df)
        
        # 행정동별 생활인구 종류별 분석
        sheet_name = "8~12p 생활인구 종류별 분석"
        df2 = self.load_excel_sheet("1_생활인구분석_경기데이터드림.xlsx", sheet_name)
        
        if not df2.empty:
            df2['데이터소스'] = '경기데이터드림'
            df2['유형'] = '생활인구'
            df_list.append(df2)
        
        if df_list:
            combined = pd.concat(df_list, ignore_index=True)
            return combined
        return pd.DataFrame()
    
    def load_time_slot_population_data(self) -> pd.DataFrame:
        """시간대별 생활인구 데이터 로드"""
        df_list = []
        
        # 시간대별 생활인구 현황
        sheet_name = "31p 시간대별 생활인구 현황"
        df = self.load_excel_sheet("1_생활인구분석_LG유플러스.xlsx", sheet_name)
        
        if not df.empty:
            df['데이터소스'] = 'LG유플러스'
            df_list.append(df)
        
        # 행정동별 시간대별 생활인구
        sheet_name = "35p 시간대별 분석"
        df2 = self.load_excel_sheet("1_생활인구분석_경기데이터드림.xlsx", sheet_name)
        
        if not df2.empty:
            df2['데이터소스'] = '경기데이터드림'
            df_list.append(df2)
        
        if df_list:
            combined = pd.concat(df_list, ignore_index=True)
            return combined
        return pd.DataFrame()
    
    def load_weekend_pattern_data(self) -> pd.DataFrame:
        """주말 패턴 데이터 로드"""
        df_list = []
        
        # 요일별 생활인구 현황
        sheet_name = "29p 요일별 생활인구 현황"
        df = self.load_excel_sheet("1_생활인구분석_LG유플러스.xlsx", sheet_name)
        
        if not df.empty:
            df['데이터소스'] = 'LG유플러스'
            df_list.append(df)
        
        # 요일별 관광지 방문 패턴
        sheet_name = "15p_27p_39p_51p_63p_요일별"
        df2 = self.load_excel_sheet("3_관광지분석_LG유플러스.xlsx", sheet_name)
        
        if not df2.empty:
            df2['데이터소스'] = 'LG유플러스'
            df2['유형'] = '방문인구'
            df_list.append(df2)
        
        if df_list:
            combined = pd.concat(df_list, ignore_index=True)
            return combined
        return pd.DataFrame()
    
    def load_consumption_pattern_data(self) -> pd.DataFrame:
        """소비 패턴 데이터 로드"""
        df_list = []
        
        # 식품위생업소 소비 분석
        sheet_name = "5,11p 식품위생업소 변화"
        df = self.load_excel_sheet("5_식품위생업소소비분석_삼성카드.xlsx", sheet_name)
        
        if not df.empty:
            df['데이터소스'] = '삼성카드'
            df['유형'] = '소비'
            df_list.append(df)
        
        # 요일별 평균 매출액
        sheet_name = "15p 요일별 평균 매출액"
        df2 = self.load_excel_sheet("5_식품위생업소소비분석_삼성카드.xlsx", sheet_name)
        
        if not df2.empty:
            df2['데이터소스'] = '삼성카드'
            df2['유형'] = '소비_요일별'
            df_list.append(df2)
        
        if df_list:
            combined = pd.concat(df_list, ignore_index=True)
            return combined
        return pd.DataFrame()
    
    def load_vitality_index_data(self) -> pd.DataFrame:
        """지역활력지수 데이터 로드"""
        df_list = []
        
        # 지역활력지수
        sheet_name = "13~14p 지역활력지수"
        df = self.load_excel_sheet("6_소비경제규모추정.xlsx", sheet_name)
        
        if not df.empty:
            df['데이터소스'] = '소비경제규모추정'
            df_list.append(df)
        
        # 인구활력지수
        sheet_name = "5~7p 인구활력지수"
        df2 = self.load_excel_sheet("6_소비경제규모추정.xlsx", sheet_name)
        
        if not df2.empty:
            df2['데이터소스'] = '소비경제규모추정'
            df2['유형'] = '인구활력'
            df_list.append(df2)
        
        if df_list:
            combined = pd.concat(df_list, ignore_index=True)
            return combined
        return pd.DataFrame()
    
    def create_integrated_dataset(self) -> pd.DataFrame:
        """통합 데이터셋 생성 (모든 데이터를 결합)"""
        print("[데이터 로더] 통합 데이터셋 생성 중...")
        
        datasets = []
        
        # 1. 관광지 방문 데이터
        visit_df = self.load_tourist_visit_data()
        if not visit_df.empty:
            datasets.append(visit_df)
        
        # 2. 관광지 매출 데이터
        revenue_df = self.load_tourist_revenue_data()
        if not revenue_df.empty:
            datasets.append(revenue_df)
        
        # 3. 생활인구 데이터
        life_pop_df = self.load_life_population_data()
        if not life_pop_df.empty:
            datasets.append(life_pop_df)
        
        if datasets:
            # 공통 컬럼 기준으로 병합 시도
            combined = pd.concat(datasets, ignore_index=True, sort=False)
            
            print(f"[데이터 로더] 통합 완료: {len(combined)}행")
            return combined
        
        return pd.DataFrame()
    
    def get_cultural_spaces(self) -> List[str]:
        """문화 공간 목록 반환"""
        df = self.load_tourist_visit_data()
        if not df.empty and '관광지명' in df.columns:
            spaces = df['관광지명'].unique().tolist()
            return [s for s in spaces if pd.notna(s)]
        return ['헤이리예술마을', '파주출판단지', '교하도서관', '파주출판도시', '파주문화센터']
    
    def get_time_slot_population(self, time_slot: str) -> float:
        """특정 시간대의 평균 생활인구 반환"""
        df = self.load_time_slot_population_data()
        if df.empty:
            return 0.0
        
        # 시간대 매핑
        time_mapping = {
            'morning': ['06-09시', '09-12시'],
            'afternoon': ['12-15시', '15-18시'],
            'evening': ['18-21시', '21-24시']
        }
        
        slots = time_mapping.get(time_slot, [])
        if not slots:
            return 0.0
        
        # 해당 시간대 데이터 추출 및 평균 계산
        total = 0.0
        count = 0
        
        for slot in slots:
            if '시간대' in df.columns:
                slot_data = df[df['시간대'].str.contains(slot, na=False)]
                if '생활인구' in slot_data.columns:
                    total += slot_data['생활인구'].mean() or 0
                    count += 1
        
        return total / count if count > 0 else 500000.0  # 기본값


if __name__ == "__main__":
    loader = EnhancedDataLoader()
    
    print("=== 데이터 로드 테스트 ===")
    
    visit_df = loader.load_tourist_visit_data()
    print(f"\n관광지 방문 데이터: {len(visit_df)}행")
    if not visit_df.empty:
        print(f"컬럼: {visit_df.columns.tolist()}")
        print(f"샘플:\n{visit_df.head()}")
    
    life_pop_df = loader.load_life_population_data()
    print(f"\n생활인구 데이터: {len(life_pop_df)}행")
    if not life_pop_df.empty:
        print(f"컬럼: {life_pop_df.columns.tolist()}")
        print(f"샘플:\n{life_pop_df.head()}")
    
    print(f"\n문화 공간 목록: {loader.get_cultural_spaces()}")

