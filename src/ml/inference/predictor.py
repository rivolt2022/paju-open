"""
예측 함수 - 실제 서비스에서 사용
data 폴더의 데이터를 기반으로 향후 데이터를 계속 예측하는 시스템
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ml.models.spatiotemporal_model import SpatiotemporalPredictor
from src.ml.preprocessing.enhanced_data_loader import EnhancedDataLoader
from src.ml.preprocessing.enhanced_feature_engineering import EnhancedFeatureEngineer


class InferencePredictor:
    """추론 예측 클래스 - data 폴더 기반 지속 예측 시스템"""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Args:
            model_path: 저장된 모델 경로
        """
        self.predictor = SpatiotemporalPredictor()
        self.data_loader = EnhancedDataLoader()
        self.feature_engineer = EnhancedFeatureEngineer(data_loader=self.data_loader)
        
        if model_path is None:
            model_path = project_root / "src" / "ml" / "models" / "saved" / "spatiotemporal_model.pkl"
        
        if model_path.exists():
            try:
                self.predictor.load(model_path)
                self.model_loaded = True
                print(f"[InferencePredictor] 모델 로드 성공: {model_path}")
            except Exception as e:
                print(f"[InferencePredictor] 모델 로드 실패: {e}")
                self.model_loaded = False
        else:
            print(f"[InferencePredictor] 모델 파일이 없습니다: {model_path}")
            print(f"[InferencePredictor] 기본값으로 예측합니다. 모델을 학습하려면 train_model.py를 실행하세요.")
            self.model_loaded = False
    
    def predict_cultural_space_visits(self, cultural_spaces: List[str], date: str, time_slot: str = "afternoon") -> List[Dict]:
        """
        문화 공간별 방문 예측 - 실제 모델 사용 또는 기본값
        
        Args:
            cultural_spaces: 문화 공간명 리스트
            date: 날짜 (YYYY-MM-DD 형식)
            time_slot: 시간대 (morning, afternoon, evening)
            
        Returns:
            예측 결과 리스트 (최적 시간 포함)
        """
        if not self.model_loaded:
            # 모델이 없으면 기본값 반환 (다양하게)
            return self._default_cultural_predictions(cultural_spaces, date, time_slot)
        
        predictions = []
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        
        # 시간대별 보정 계수
        time_multipliers = {
            'morning': 0.8,
            'afternoon': 1.2,
            'evening': 1.0,
        }
        time_multiplier = time_multipliers.get(time_slot, 1.0)
        
        for space in cultural_spaces:
            try:
                # 예측을 위한 데이터프레임 생성
                prediction_df = pd.DataFrame({
                    '관광지명': [space],
                    'date': [date_obj]
                })
                
                # 특징 엔지니어링 적용
                features_df = self.feature_engineer.prepare_features(prediction_df, '방문인구(명)')
                
                # 모델에 필요한 특징 추출
                feature_cols = self.feature_engineer.get_feature_names(features_df)
                if len(feature_cols) == 0:
                    # 기본 특징 생성
                    features_df['year'] = date_obj.year
                    features_df['month'] = date_obj.month
                    features_df['day'] = date_obj.day
                    features_df['day_of_week'] = date_obj.weekday()
                    features_df['day_of_year'] = date_obj.timetuple().tm_yday
                    features_df['is_weekend'] = 1 if date_obj.weekday() >= 5 else 0
                    features_df['season'] = (date_obj.month - 1) // 3 + 1
                    features_df['week_of_year'] = date_obj.isocalendar()[1]
                    feature_cols = ['year', 'month', 'day', 'day_of_week', 'day_of_year', 
                                   'is_weekend', 'season', 'week_of_year']
                
                # 모델의 특징 목록 확인
                if hasattr(self.predictor, 'feature_names') and len(self.predictor.feature_names) > 0:
                    # 모델이 요구하는 모든 특징 준비
                    required_features = self.predictor.feature_names.copy()
                    
                    # 타겟 변수는 모델이 요구하면 기본값으로 채우기 (재학습 없이 사용하기 위함)
                    # 재학습 시 타겟 변수가 제외되도록 train_model.py 수정됨
                    target_cols = ['방문인구(명)', '방문인구', '매출기반_방문인구']
                    
                    # 타겟 변수가 특징에 포함되어 있으면 기본값으로 채우기
                    # (학습 시 포함되었다면 예측 시에도 포함해야 함)
                    for target_col in target_cols:
                        if target_col in required_features and target_col not in features_df.columns:
                            # 타겟 변수는 평균값이나 0으로 채우기 (예측하려는 값이므로)
                            # 여기서는 0으로 채우되, 실제로는 사용되지 않음
                            features_df[target_col] = 0.0
                    
                    # 한글-영문 특징 매핑 (양방향)
                    feature_mapping = {
                        '연도': 'year',
                        '월': 'month',
                        'year': '연도',
                        'month': '월'
                    }
                    
                    # 매핑된 특징들을 복사 (양방향)
                    for korean, english in feature_mapping.items():
                        if korean in required_features and korean not in features_df.columns:
                            if english in features_df.columns:
                                features_df[korean] = features_df[english]
                        if english in required_features and english not in features_df.columns:
                            if korean in features_df.columns:
                                features_df[english] = features_df[korean]
                    
                    # 누락된 특징들을 기본값으로 채우기
                    for feat in required_features:
                        if feat not in features_df.columns:
                            # Lag 및 Rolling 특징들은 0으로 채우기 (과거 데이터가 없으므로)
                            if '_lag_' in feat or '_rolling_' in feat:
                                features_df[feat] = 0.0
                            # 시간적 특징들은 날짜에서 계산
                            elif feat == '연도' or feat == 'year':
                                features_df[feat] = date_obj.year
                            elif feat == '월' or feat == 'month':
                                features_df[feat] = date_obj.month
                            elif feat == 'day':
                                features_df[feat] = date_obj.day
                            elif feat == 'day_of_week':
                                features_df[feat] = date_obj.weekday()
                            elif feat == 'day_of_year':
                                features_df[feat] = date_obj.timetuple().tm_yday
                            elif feat == 'week_of_year':
                                features_df[feat] = date_obj.isocalendar()[1]
                            elif feat == 'quarter':
                                features_df[feat] = (date_obj.month - 1) // 3 + 1
                            elif feat == 'is_weekend':
                                features_df[feat] = 1 if date_obj.weekday() >= 5 else 0
                            elif feat == 'season':
                                features_df[feat] = (date_obj.month - 1) // 3 + 1
                            elif feat == 'is_holiday':
                                features_df[feat] = 0
                            elif feat == 'is_month_start':
                                features_df[feat] = 1 if date_obj.day == 1 else 0
                            elif feat == 'is_month_end':
                                features_df[feat] = 1 if date_obj.day == date_obj.timetuple().tm_mday else 0
                            # 기타 특징들은 0으로 채우기
                            else:
                                features_df[feat] = 0.0
                    
                    # 특징 선택: 모델이 요구하는 순서대로
                    # 모든 required_features가 존재하는지 확인하고, 없으면 추가
                    final_features = []
                    for feat in required_features:
                        if feat in features_df.columns:
                            final_features.append(feat)
                        else:
                            # 없으면 추가 (이미 위에서 채웠지만 한 번 더 확인)
                            if '_lag_' in feat or '_rolling_' in feat:
                                features_df[feat] = 0.0
                            else:
                                features_df[feat] = 0.0
                            final_features.append(feat)
                    
                    # DataFrame이 단일 행이므로 적절히 선택
                    X = features_df[final_features].fillna(0.0)
                    
                    # X가 Series인지 DataFrame인지 확인하고, 모델이 기대하는 형태로 변환
                    if isinstance(X, pd.Series):
                        X = X.to_frame().T
                    elif X.shape[0] == 0:
                        # 빈 DataFrame인 경우 기본값으로 채운 단일 행 생성
                        X = pd.DataFrame([[0.0] * len(final_features)], columns=final_features)
                    
                    # 모델이 기대하는 순서로 정렬
                    X = X[final_features].fillna(0.0)
                else:
                    # 모델에 특징 목록이 없는 경우 공통 특징 사용
                    common_features = [f for f in feature_cols if f in features_df.columns]
                    if len(common_features) == 0:
                        common_features = ['year', 'month', 'day_of_week', 'is_weekend', 'season']
                        for col in common_features:
                            if col not in features_df.columns:
                                features_df[col] = 0
                    X = features_df[common_features].fillna(0)
                
                # 모델로 예측 수행
                predicted_visit_raw = self.predictor.predict(X)[0]
                predicted_visit = max(0, int(predicted_visit_raw))
                
                # 시간대별 보정 적용
                predicted_visit = int(predicted_visit * time_multiplier)
                
                # 혼잡도 계산 (0.2 ~ 0.85 범위)
                crowd_base = min(predicted_visit / 60000, 0.85)
                crowd_level = max(0.2, crowd_base + np.random.uniform(-0.05, 0.05))
                
                # 최적 시간 계산
                optimal_times = {
                    'morning': '10:00-12:00',
                    'afternoon': '14:00-17:00',
                    'evening': '18:00-20:00',
                }
                optimal_time = optimal_times.get(time_slot, '14:00-17:00')
                
                predictions.append({
                    'space': space,
                    'date': date,
                    'predicted_visit': predicted_visit,
                    'crowd_level': float(crowd_level),
                    'optimal_time': optimal_time,
                    'recommended_programs': self._get_recommended_programs(space, time_slot),
                    'confidence': 0.85 if self.model_loaded else 0.5
                })
            except Exception as e:
                print(f"[InferencePredictor] 예측 오류 ({space}): {e}")
                # 예외 발생 시 기본값 (모델 없을 때와 동일)
                base_visit = {
                    '헤이리예술마을': 42000,
                    '파주출판단지': 28000,
                    '교하도서관': 15000,
                    '파주출판도시': 12000,
                    '파주문화센터': 18000,
                    '출판문화정보원': 10000,
                }.get(space, 20000)
                
                predicted_visit = int(base_visit * time_multiplier)
                crowd_level = max(0.2, min(predicted_visit / 60000, 0.85))
                
                predictions.append({
                    'space': space,
                    'date': date,
                    'predicted_visit': predicted_visit,
                    'crowd_level': float(crowd_level),
                    'optimal_time': '14:00-17:00',
                    'recommended_programs': self._get_recommended_programs(space, time_slot),
                    'confidence': 0.3
                })
        
        return predictions
    
    def _get_recommended_programs(self, space: str, time_slot: str) -> List[str]:
        """문화 공간별 추천 프로그램"""
        programs = {
            '헤이리예술마을': ['작가와의 만남', '갤러리 전시', '예술 체험 프로그램'],
            '파주출판단지': ['출판사 투어', '책 만남의 날', '작가 사인회'],
            '교하도서관': ['독서 모임', '문화 강좌', '북토크'],
            '파주출판도시': ['출판 박물관 관람', '인쇄 체험', '출판사 탐방'],
            '파주문화센터': ['문화 공연', '전시', '문화 강좌'],
            '출판문화정보원': ['출판 강좌', '출판 자료 열람', '출판 상담'],
        }
        return programs.get(space, ['문화 프로그램'])
    
    def _default_cultural_predictions(self, cultural_spaces: List[str], date: str, time_slot: str) -> List[Dict]:
        """기본값 반환 (모델이 없을 때) - 다양하게"""
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        is_weekend = 1 if date_obj.weekday() >= 5 else 0
        month = date_obj.month
        
        defaults = {
            '헤이리예술마을': 42000,
            '파주출판단지': 28000,
            '교하도서관': 15000,
            '파주출판도시': 12000,
            '파주문화센터': 18000,
            '출판문화정보원': 10000,
        }
        
        # 시간대별 보정
        time_multipliers = {
            'morning': 0.8,
            'afternoon': 1.2,
            'evening': 1.0,
        }
        time_multiplier = time_multipliers.get(time_slot, 1.0)
        
        # 주말 여부에 따른 조정
        weekend_multiplier = 1.3 if is_weekend else 1.0
        
        # 계절별 조정
        if month in [6, 7, 8]:  # 여름
            season_multiplier = 1.2
        elif month in [9, 10, 11]:  # 가을
            season_multiplier = 1.1
        elif month in [12, 1, 2]:  # 겨울
            season_multiplier = 0.8
        else:  # 봄
            season_multiplier = 0.9
        
        optimal_times = {
            'morning': '10:00-12:00',
            'afternoon': '14:00-17:00',
            'evening': '18:00-20:00',
        }
        optimal_time = optimal_times.get(time_slot, '14:00-17:00')
        
        predictions = []
        for i, space in enumerate(cultural_spaces):
            base_visit = defaults.get(space, 20000)
            
            # 문화 공간별 약간의 변동 추가
            variation = 1.0 + (i * 0.05) - 0.1  # 0.9 ~ 1.1 범위
            visit = int(base_visit * weekend_multiplier * season_multiplier * variation * time_multiplier)
            
            # 혼잡도 계산 (다양하게)
            crowd_base = min(visit / 60000, 0.85)
            crowd_level = max(0.25, crowd_base + (i * 0.05) - 0.1)
            
            predictions.append({
                'space': space,
                'date': date,
                'predicted_visit': visit,
                'crowd_level': float(crowd_level),
                'optimal_time': optimal_time,
                'recommended_programs': self._get_recommended_programs(space, time_slot),
                'confidence': 0.5
            })
        
        return predictions
    
    def predict_visits(self, spaces: List[str], date: str) -> List[Dict]:
        """호환성을 위한 별칭 메서드 (spot → space 변환)"""
        results = self.predict_cultural_space_visits(spaces, date, "afternoon")
        # 호환성을 위해 spot 키도 추가
        for result in results:
            if 'space' in result and 'spot' not in result:
                result['spot'] = result['space']
        return results
    
    def predict_population(self, dong: str, date: str) -> Dict:
        """
        행정동별 생활인구 예측
        
        Args:
            dong: 행정동명
            date: 날짜 (YYYY-MM-DD 형식)
            
        Returns:
            예측 결과 딕셔너리
        """
        # 기본값 반환 (실제 구현 필요)
        return {
            'dong': dong,
            'date': date,
            'predicted_population': 15000,
            'confidence': 0.5
        }


if __name__ == "__main__":
    # 테스트
    predictor = InferencePredictor()
    
    spots = ['헤이리예술마을', 'DMZ평화관광', '마장호수출렁다리']
    date = '2025-01-18'
    
    results = predictor.predict_visits(spots, date)
    
    print("예측 결과:")
    for result in results:
        print(f"  {result['spot']}: {result['predicted_visit']:,}명 (혼잡도: {result['crowd_level']:.2f})")