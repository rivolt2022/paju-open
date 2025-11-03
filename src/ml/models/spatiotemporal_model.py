"""
시공간 예측 모델 - 관광지 방문 예측
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class SpatiotemporalPredictor:
    """시공간 예측 모델 클래스"""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Args:
            model_type: 모델 타입 ('random_forest', 'gradient_boosting')
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = []
        self.is_trained = False
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              test_size: float = 0.2, random_state: int = 42,
              use_kfold: bool = True, cv_folds: int = 5) -> Dict:
        """
        모델 학습 (k-fold 교차 검증 지원)
        
        Args:
            X: 특징 데이터
            y: 타겟 데이터
            test_size: 테스트 데이터 비율 (k-fold 사용 시 무시)
            random_state: 랜덤 시드
            use_kfold: k-fold 교차 검증 사용 여부
            cv_folds: 교차 검증 fold 수
            
        Returns:
            평가 결과 딕셔너리
        """
        # 특징 이름 저장
        self.feature_names = list(X.columns)
        
        results = {}
        
        if use_kfold and len(X) >= cv_folds:
            # k-fold 교차 검증
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            
            # 교차 검증 수행
            cv_results = cross_validate(
                self.model, X, y,
                cv=kfold,
                scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'],
                return_train_score=True,
                n_jobs=-1
            )
            
            # 결과 정리 (음수 제거 및 평균 계산)
            results['cv_mae_mean'] = -cv_results['test_neg_mean_absolute_error'].mean()
            results['cv_mae_std'] = cv_results['test_neg_mean_absolute_error'].std()
            results['cv_rmse_mean'] = np.sqrt(-cv_results['test_neg_mean_squared_error'].mean())
            results['cv_rmse_std'] = np.sqrt(cv_results['test_neg_mean_squared_error'].std())
            results['cv_r2_mean'] = cv_results['test_r2'].mean()
            results['cv_r2_std'] = cv_results['test_r2'].std()
            
            # 각 fold별 결과
            results['cv_folds'] = []
            for i in range(cv_folds):
                results['cv_folds'].append({
                    'fold': i + 1,
                    'mae': -cv_results['test_neg_mean_absolute_error'][i],
                    'rmse': np.sqrt(-cv_results['test_neg_mean_squared_error'][i]),
                    'r2': cv_results['test_r2'][i]
                })
            
            # 전체 데이터로 최종 모델 학습
            self.model.fit(X, y)
            self.is_trained = True
            
            # 최종 예측 (전체 데이터)
            y_pred = self.model.predict(X)
            results['final_mae'] = mean_absolute_error(y, y_pred)
            results['final_rmse'] = np.sqrt(mean_squared_error(y, y_pred))
            results['final_r2'] = r2_score(y, y_pred)
            
            # MAPE 계산 (0이 아닌 값에 대해서만)
            try:
                results['final_mape'] = mean_absolute_percentage_error(y, y_pred)
            except:
                # y에 0이 있는 경우 대체 계산
                non_zero_mask = y != 0
                if non_zero_mask.sum() > 0:
                    results['final_mape'] = np.mean(np.abs((y[non_zero_mask] - y_pred[non_zero_mask]) / y[non_zero_mask])) * 100
                else:
                    results['final_mape'] = 0.0
            
            results['n_features'] = len(self.feature_names)
            results['n_samples'] = len(X)
            results['cv_folds_used'] = cv_folds
            
        else:
            # 기존 방식 (train/test split)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # 모델 학습
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            # 예측
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            # 평가
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # MAPE 계산
            try:
                train_mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100
                test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
            except:
                train_mape = 0.0
                test_mape = 0.0
            
            results = {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mape': train_mape,
                'test_mape': test_mape,
                'n_features': len(self.feature_names),
                'n_train': len(X_train),
                'n_test': len(X_test),
                'use_kfold': False
            }
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        예측 수행
        
        Args:
            X: 특징 데이터
            
        Returns:
            예측 결과 배열
        """
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다. train() 메서드를 먼저 호출하세요.")
        
        return self.model.predict(X)
    
    def predict_for_spot(self, spot: str, date: str, features: pd.DataFrame) -> Dict:
        """
        특정 관광지와 날짜에 대한 예측
        
        Args:
            spot: 관광지명
            date: 날짜 (YYYY-MM-DD 형식)
            features: 특징 DataFrame
            
        Returns:
            예측 결과 딕셔너리
        """
        # 해당 관광지와 날짜에 대한 특징 추출
        spot_features = features[
            (features.get('관광지명', '') == spot) &
            (features.get('date', '').astype(str).str.startswith(date))
        ]
        
        if len(spot_features) == 0:
            # 특징이 없으면 기본값으로 예측
            return {
                'spot': spot,
                'date': date,
                'predicted_visit': 30000,  # 기본값
                'crowd_level': 0.5,
                'confidence': 0.0
            }
        
        # 예측 수행
        X = spot_features[self.feature_names]
        predicted_visit = self.model.predict(X)[0]
        
        # 혼잡도 계산 (예측값을 0-1 사이로 정규화)
        # 최대값 100000명 가정
        crowd_level = min(predicted_visit / 100000, 1.0)
        
        # 신뢰도 (간단히 0.7로 고정, 실제로는 모델 불확실성 사용)
        confidence = 0.7
        
        return {
            'spot': spot,
            'date': date,
            'predicted_visit': int(predicted_visit),
            'crowd_level': float(crowd_level),
            'confidence': confidence
        }
    
    def save(self, filepath: Path):
        """모델 저장"""
        if not self.is_trained:
            raise ValueError("학습되지 않은 모델을 저장할 수 없습니다.")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
    
    def load(self, filepath: Path):
        """모델 로드"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = True


if __name__ == "__main__":
    # 테스트 코드
    from ..preprocessing import DataLoader, FeatureEngineer
    
    print("모델 테스트 시작...")
    
    # 데이터 로드
    loader = DataLoader()
    tourist_df = loader.load_tourist_data()
    
    print(f"데이터 로드 완료: {len(tourist_df)}행")
    
    if len(tourist_df) > 0:
        # 특징 엔지니어링
        engineer = FeatureEngineer()
        features_df = engineer.prepare_features(tourist_df, '방문인구(명)')
        
        # 타겟 추출
        if '방문인구(명)' in features_df.columns:
            feature_cols = engineer.get_feature_names(features_df)
            X = features_df[feature_cols]
            y = features_df['방문인구(명)']
            
            # 모델 학습
            predictor = SpatiotemporalPredictor(model_type='random_forest')
            results = predictor.train(X, y)
            
            print("\n학습 결과:")
            print(f"테스트 MAE: {results['test_mae']:.2f}")
            print(f"테스트 RMSE: {results['test_rmse']:.2f}")
            print(f"테스트 R²: {results['test_r2']:.4f}")
