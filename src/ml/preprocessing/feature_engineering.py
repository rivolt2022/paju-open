"""
특징 엔지니어링 - ML 모델 입력을 위한 특징 생성
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """특징 엔지니어링 클래스"""
    
    def __init__(self):
        self.feature_names = []
    
    def create_temporal_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """
        시간적 특징 생성
        
        Args:
            df: 입력 DataFrame
            date_col: 날짜 컬럼명
            
        Returns:
            시간적 특징이 추가된 DataFrame
        """
        df = df.copy()
        
        if date_col not in df.columns:
            return df
        
        # 날짜 파싱
        if df[date_col].dtype != 'datetime64[ns]':
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # 연도, 월, 일
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['day_of_week'] = df[date_col].dt.dayofweek  # 0=월요일
        df['day_of_year'] = df[date_col].dt.dayofyear
        df['week_of_year'] = df[date_col].dt.isocalendar().week
        
        # 주말 여부
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # 계절
        df['season'] = df['month'].apply(self._get_season)
        
        # 공휴일 여부 (간단한 휴일 체크)
        df['is_holiday'] = self._check_holiday(df[date_col])
        
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
    
    def _check_holiday(self, dates: pd.Series) -> pd.Series:
        """공휴일 체크 (간단한 휴일 리스트)"""
        # 주요 공휴일 (더 정확한 휴일 체크는 별도 라이브러리 사용 권장)
        holidays = []
        for date in dates:
            # 여기서는 간단하게 주말만 체크
            holidays.append(0)
        return pd.Series(holidays, index=dates.index)
    
    def create_lag_features(self, df: pd.DataFrame, value_col: str, 
                           date_col: str = 'date', lags: List[int] = [1, 7, 30]) -> pd.DataFrame:
        """
        시차(lag) 특징 생성
        
        Args:
            df: 입력 DataFrame
            value_col: 값 컬럼명
            date_col: 날짜 컬럼명
            lags: 시차 리스트 (일 단위)
            
        Returns:
            시차 특징이 추가된 DataFrame
        """
        df = df.copy()
        df = df.sort_values(date_col)
        
        for lag in lags:
            df[f'{value_col}_lag_{lag}'] = df[value_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, value_col: str,
                               windows: List[int] = [7, 30]) -> pd.DataFrame:
        """
        롤링 통계 특징 생성
        
        Args:
            df: 입력 DataFrame
            value_col: 값 컬럼명
            windows: 윈도우 사이즈 리스트 (일 단위)
            
        Returns:
            롤링 통계 특징이 추가된 DataFrame
        """
        df = df.copy()
        
        for window in windows:
            df[f'{value_col}_rolling_mean_{window}'] = df[value_col].rolling(window=window).mean()
            df[f'{value_col}_rolling_std_{window}'] = df[value_col].rolling(window=window).std()
            df[f'{value_col}_rolling_max_{window}'] = df[value_col].rolling(window=window).max()
            df[f'{value_col}_rolling_min_{window}'] = df[value_col].rolling(window=window).min()
        
        return df
    
    def create_spatial_features(self, df: pd.DataFrame, 
                               spot_col: str = '관광지명') -> pd.DataFrame:
        """
        공간적 특징 생성 (관광지별 인코딩)
        
        Args:
            df: 입력 DataFrame
            spot_col: 관광지명 컬럼
            
        Returns:
            공간적 특징이 추가된 DataFrame
        """
        df = df.copy()
        
        # 관광지명 원-핫 인코딩
        if spot_col in df.columns:
            spot_dummies = pd.get_dummies(df[spot_col], prefix='spot')
            df = pd.concat([df, spot_dummies], axis=1)
        
        return df
    
    def create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        인구 통계 특징 생성
        
        Args:
            df: 입력 DataFrame
            
        Returns:
            인구 통계 특징이 추가된 DataFrame
        """
        df = df.copy()
        
        # 성별 비율
        if '남성' in df.columns and '여성' in df.columns:
            df['gender_ratio'] = df['남성'] / (df['여성'] + 1e-6)
        
        # 연령대 비율
        age_cols = [col for col in df.columns if '20대' in str(col) or '30대' in str(col)]
        if age_cols:
            df['age_diversity'] = df[age_cols].std(axis=1)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, 
                        target_col: str = '방문인구(명)',
                        date_col: str = 'date',
                        spot_col: str = '관광지명') -> pd.DataFrame:
        """
        모든 특징 생성 및 준비
        
        Args:
            df: 입력 DataFrame
            target_col: 예측 대상 컬럼명
            date_col: 날짜 컬럼명
            spot_col: 관광지명 컬럼명
            
        Returns:
            특징이 추가된 DataFrame
        """
        df = df.copy()
        
        # 시간적 특징
        if date_col in df.columns:
            df = self.create_temporal_features(df, date_col)
        
        # 공간적 특징
        if spot_col in df.columns:
            df = self.create_spatial_features(df, spot_col)
        
        # 시차 특징
        if target_col in df.columns:
            df = self.create_lag_features(df, target_col, date_col)
            df = self.create_rolling_features(df, target_col)
        
        # 인구 통계 특징
        df = self.create_demographic_features(df)
        
        # 결측치 처리
        df = df.fillna(df.median(numeric_only=True))
        df = df.fillna(0)
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
        """
        특징 컬럼명 반환
        
        Args:
            df: 입력 DataFrame
            exclude_cols: 제외할 컬럼명 리스트
            
        Returns:
            특징 컬럼명 리스트
        """
        if exclude_cols is None:
            exclude_cols = ['date', '관광지명', '행정동']
        
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype in [np.int64, np.float64]]
        
        return feature_cols


if __name__ == "__main__":
    # 테스트
    engineer = FeatureEngineer()
    
    # 샘플 데이터 생성
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    sample_df = pd.DataFrame({
        'date': dates,
        '관광지명': ['헤이리예술마을'] * len(dates),
        '방문인구(명)': np.random.randint(10000, 50000, len(dates))
    })
    
    # 특징 생성
    features_df = engineer.prepare_features(sample_df, '방문인구(명)')
    
    print("생성된 특징:")
    print(features_df.columns.tolist())
    print(f"\n총 특징 수: {len(engineer.get_feature_names(features_df))}")
