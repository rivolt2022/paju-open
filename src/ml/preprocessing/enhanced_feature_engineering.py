"""
강화된 특징 엔지니어링 - 실제 데이터 기반
생활인구, 소비 패턴, 지역활력지수 등을 활용한 특징 생성
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class EnhancedFeatureEngineer:
    """강화된 특징 엔지니어링 클래스"""
    
    def __init__(self, data_loader=None):
        self.data_loader = data_loader
        self.feature_names = []
    
    def create_temporal_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """시간적 특징 생성"""
        df = df.copy()
        
        if date_col not in df.columns:
            return df
        
        if df[date_col].dtype != 'datetime64[ns]':
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # 기본 시간 특징
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['day_of_week'] = df[date_col].dt.dayofweek  # 0=월요일
        df['day_of_year'] = df[date_col].dt.dayofyear
        df['week_of_year'] = df[date_col].dt.isocalendar().week
        df['quarter'] = df[date_col].dt.quarter
        
        # 주말/평일
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # 계절
        df['season'] = df['month'].apply(self._get_season)
        
        # 공휴일 체크 (간단화)
        df['is_holiday'] = 0  # 실제로는 한국 공휴일 데이터 필요
        
        # 월말/월초
        df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
        
        return df
    
    def _get_season(self, month: int) -> int:
        """계절 반환 (1=봄, 2=여름, 3=가을, 4=겨울)"""
        if month in [3, 4, 5]:
            return 1
        elif month in [6, 7, 8]:
            return 2
        elif month in [9, 10, 11]:
            return 3
        else:
            return 4
    
    def create_life_population_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """생활인구 관련 특징 생성"""
        df = df.copy()
        
        if not self.data_loader:
            return df
        
        try:
            # 생활인구 데이터 로드
            life_pop_df = self.data_loader.load_life_population_data()
            
            if not life_pop_df.empty and date_col in life_pop_df.columns:
                # 날짜 기준으로 생활인구 병합
                if '생활인구수' in life_pop_df.columns:
                    life_pop_grouped = life_pop_df.groupby(date_col)['생활인구수'].mean().reset_index()
                    life_pop_grouped.columns = [date_col, 'avg_life_population']
                    df = df.merge(life_pop_grouped, on=date_col, how='left')
                    
                    # 생활인구 정규화 (0-1 스케일)
                    if 'avg_life_population' in df.columns:
                        max_pop = df['avg_life_population'].max()
                        if max_pop > 0:
                            df['life_population_normalized'] = df['avg_life_population'] / max_pop
                
                # 거주/근무/방문인구 비율
                if '거주인구' in life_pop_df.columns and '근무인구' in life_pop_df.columns and '방문인구' in life_pop_df.columns:
                    life_pop_grouped = life_pop_df.groupby(date_col).agg({
                        '거주인구': 'mean',
                        '근무인구': 'mean',
                        '방문인구': 'mean'
                    }).reset_index()
                    
                    df = df.merge(life_pop_grouped, on=date_col, how='left')
                    
                    # 인구 유형 비율
                    if all(col in df.columns for col in ['거주인구', '근무인구', '방문인구']):
                        total = df['거주인구'] + df['근무인구'] + df['방문인구']
                        df['resident_ratio'] = df['거주인구'] / (total + 1e-6)
                        df['worker_ratio'] = df['근무인구'] / (total + 1e-6)
                        df['visitor_ratio'] = df['방문인구'] / (total + 1e-6)
        except Exception as e:
            print(f"생활인구 특징 생성 오류: {e}")
        
        return df
    
    def create_time_slot_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """시간대별 특징 생성"""
        df = df.copy()
        
        if not self.data_loader:
            return df
        
        try:
            # 시간대별 생활인구 데이터
            time_slot_df = self.data_loader.load_time_slot_population_data()
            
            if not time_slot_df.empty:
                # 시간대별 평균 생활인구 계산
                if '시간대' in time_slot_df.columns and '생활인구' in time_slot_df.columns:
                    time_slot_grouped = time_slot_df.groupby('시간대')['생활인구'].mean().reset_index()
                    
                    # 시간대를 인덱스로 매핑
                    time_slot_map = {}
                    for idx, row in time_slot_grouped.iterrows():
                        time_slot = str(row['시간대'])
                        if '09-12' in time_slot or '12-15' in time_slot:
                            time_slot_map['morning'] = time_slot_map.get('morning', 0) + row['생활인구']
                        elif '15-18' in time_slot:
                            time_slot_map['afternoon'] = time_slot_map.get('afternoon', 0) + row['생활인구']
                        elif '18-21' in time_slot or '21-24' in time_slot:
                            time_slot_map['evening'] = time_slot_map.get('evening', 0) + row['생활인구']
                    
                    # 월별 평균으로 사용
                    for time_slot in ['morning', 'afternoon', 'evening']:
                        avg_pop = time_slot_map.get(time_slot, 500000) / max(len([k for k in time_slot_map.keys() if time_slot in k]), 1)
                        df[f'{time_slot}_population'] = avg_pop
        except Exception as e:
            print(f"시간대 특징 생성 오류: {e}")
        
        return df
    
    def create_consumption_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """소비 패턴 특징 생성"""
        df = df.copy()
        
        if not self.data_loader:
            return df
        
        try:
            # 소비 패턴 데이터 로드
            consumption_df = self.data_loader.load_consumption_pattern_data()
            
            if not consumption_df.empty:
                # 월별 평균 소비액 계산
                if '분기별' in consumption_df.columns:
                    # 분기별 데이터를 월별로 추정
                    df['estimated_consumption'] = 800.0  # 기본값 (백만원)
                elif '매출액 평균(백만원)' in consumption_df.columns:
                    avg_consumption = consumption_df['매출액 평균(백만원)'].mean()
                    df['estimated_consumption'] = avg_consumption
        except Exception as e:
            print(f"소비 특징 생성 오류: {e}")
        
        return df
    
    def create_vitality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """지역활력지수 특징 생성"""
        df = df.copy()
        
        if not self.data_loader:
            return df
        
        try:
            # 지역활력지수 데이터 로드
            vitality_df = self.data_loader.load_vitality_index_data()
            
            if not vitality_df.empty:
                # 지역활력지수 추출
                if '지역활력지수' in vitality_df.columns:
                    avg_vitality = vitality_df['지역활력지수'].mean()
                    df['regional_vitality'] = avg_vitality
                
                if '인구활력지수' in vitality_df.columns:
                    avg_pop_vitality = vitality_df['인구활력지수'].mean()
                    df['population_vitality'] = avg_pop_vitality
                
                if '소비활력지수' in vitality_df.columns:
                    avg_cons_vitality = vitality_df['소비활력지수'].mean()
                    df['consumption_vitality'] = avg_cons_vitality
        except Exception as e:
            print(f"활력지수 특징 생성 오류: {e}")
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, value_col: str, 
                           date_col: str = 'date', lags: List[int] = [1, 7, 30]) -> pd.DataFrame:
        """시차(lag) 특징 생성"""
        df = df.copy()
        df = df.sort_values(date_col)
        
        for lag in lags:
            df[f'{value_col}_lag_{lag}'] = df[value_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, value_col: str,
                               windows: List[int] = [7, 30, 90]) -> pd.DataFrame:
        """롤링 통계 특징 생성"""
        df = df.copy()
        df = df.sort_values('date') if 'date' in df.columns else df
        
        for window in windows:
            df[f'{value_col}_rolling_mean_{window}'] = df[value_col].rolling(window=window, min_periods=1).mean()
            df[f'{value_col}_rolling_std_{window}'] = df[value_col].rolling(window=window, min_periods=1).std()
            df[f'{value_col}_rolling_max_{window}'] = df[value_col].rolling(window=window, min_periods=1).max()
            df[f'{value_col}_rolling_min_{window}'] = df[value_col].rolling(window=window, min_periods=1).min()
        
        return df
    
    def create_spatial_features(self, df: pd.DataFrame, 
                               spot_col: str = '관광지명') -> pd.DataFrame:
        """공간적 특징 생성"""
        df = df.copy()
        
        if spot_col in df.columns:
            # 관광지명 원-핫 인코딩
            spot_dummies = pd.get_dummies(df[spot_col], prefix='space')
            df = pd.concat([df, spot_dummies], axis=1)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, 
                        target_col: str = '방문인구(명)',
                        date_col: str = 'date',
                        spot_col: str = '관광지명') -> pd.DataFrame:
        """
        모든 특징 생성 및 준비
        """
        print("[특징 엔지니어링] 특징 생성 시작...")
        
        df = df.copy()
        
        # 1. 시간적 특징
        if date_col in df.columns:
            df = self.create_temporal_features(df, date_col)
        
        # 2. 생활인구 관련 특징
        if self.data_loader:
            df = self.create_life_population_features(df, date_col)
            df = self.create_time_slot_features(df, date_col)
        
        # 3. 소비 패턴 특징
        if self.data_loader:
            df = self.create_consumption_features(df, date_col)
        
        # 4. 지역활력지수 특징
        if self.data_loader:
            df = self.create_vitality_features(df)
        
        # 5. 공간적 특징
        if spot_col in df.columns:
            df = self.create_spatial_features(df, spot_col)
        
        # 6. 시차 특징
        if target_col in df.columns:
            df = self.create_lag_features(df, target_col, date_col)
            df = self.create_rolling_features(df, target_col)
        
        # 결측치 처리
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != target_col:
                df[col] = df[col].fillna(df[col].median() if df[col].median() != np.nan else 0)
        
        df = df.fillna(0)
        
        print(f"[특징 엔지니어링] 특징 생성 완료: {len(df.columns)}개 컬럼")
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame, exclude_cols: List[str] = None, target_col: str = None) -> List[str]:
        """특징 컬럼명 반환"""
        if exclude_cols is None:
            exclude_cols = ['date', '관광지명', '행정동', '데이터소스', '유형', '기준연월', '기간']
        
        # 타겟 변수 제외 (예측 시에는 사용 불가)
        if target_col:
            exclude_cols = exclude_cols + [target_col]
        
        # 타겟 변수 관련 일반적인 이름들도 제외
        target_variants = ['방문인구(명)', '방문인구', '매출기반_방문인구']
        for target_var in target_variants:
            if target_var not in exclude_cols:
                exclude_cols.append(target_var)
        
        # 숫자형 컬럼만 추출
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        return feature_cols


if __name__ == "__main__":
    from enhanced_data_loader import EnhancedDataLoader
    
    loader = EnhancedDataLoader()
    engineer = EnhancedFeatureEngineer(data_loader=loader)
    
    # 테스트 데이터 생성
    df = loader.load_tourist_visit_data()
    print(f"\n원본 데이터: {len(df)}행")
    
    if not df.empty:
        features_df = engineer.prepare_features(df, '방문인구(명)')
        feature_cols = engineer.get_feature_names(features_df)
        
        print(f"\n생성된 특징 수: {len(feature_cols)}")
        print(f"특징 목록: {feature_cols[:10]}...")  # 처음 10개만

