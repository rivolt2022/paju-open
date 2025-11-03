"""
통합 모델 학습 스크립트 - data 폴더의 모든 데이터를 활용하여 학습
향후 예측을 위해 지속적으로 사용 가능한 시스템
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ml.preprocessing.enhanced_data_loader import EnhancedDataLoader
from src.ml.preprocessing.enhanced_feature_engineering import EnhancedFeatureEngineer
from src.ml.models.spatiotemporal_model import SpatiotemporalPredictor


def main():
    """메인 학습 함수"""
    print("="*60)
    print("시공간 예측 모델 학습 시작 (통합 버전)")
    print("="*60)
    print(f"학습 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 데이터 로드
    print("\n[1/6] 데이터 로드 중...")
    loader = EnhancedDataLoader()
    
    # 모든 데이터 통합 로드
    print("  - 관광지 방문 데이터 로드 중...")
    visit_df = loader.load_tourist_visit_data()
    print(f"    관광지 방문 데이터: {len(visit_df)}행")
    
    print("  - 관광지 매출 데이터 로드 중...")
    revenue_df = loader.load_tourist_revenue_data()
    print(f"    관광지 매출 데이터: {len(revenue_df)}행")
    
    print("  - 생활인구 데이터 로드 중...")
    life_pop_df = loader.load_life_population_data()
    print(f"    생활인구 데이터: {len(life_pop_df)}행")
    
    print("  - 시간대별 생활인구 데이터 로드 중...")
    time_slot_df = loader.load_time_slot_population_data()
    print(f"    시간대별 생활인구 데이터: {len(time_slot_df)}행")
    
    print("  - 주말 패턴 데이터 로드 중...")
    weekend_df = loader.load_weekend_pattern_data()
    print(f"    주말 패턴 데이터: {len(weekend_df)}행")
    
    print("  - 소비 패턴 데이터 로드 중...")
    consumption_df = loader.load_consumption_pattern_data()
    print(f"    소비 패턴 데이터: {len(consumption_df)}행")
    
    print("  - 지역활력지수 데이터 로드 중...")
    vitality_df = loader.load_vitality_index_data()
    print(f"    지역활력지수 데이터: {len(vitality_df)}행")
    
    # 데이터 병합
    print("\n[2/6] 데이터 병합 중...")
    merged_df = visit_df.copy()
    
    # 관광지 매출 데이터 병합
    if not revenue_df.empty and not merged_df.empty:
        if 'date' in merged_df.columns and 'date' in revenue_df.columns:
            if '관광지명' in merged_df.columns and '관광지명' in revenue_df.columns:
                merged_df = merged_df.merge(
                    revenue_df[['관광지명', 'date', '방문인구(명)']].rename(columns={'방문인구(명)': '매출기반_방문인구'}),
                    on=['관광지명', 'date'],
                    how='outer',
                    suffixes=('', '_revenue')
                )
                # 방문인구 병합 (하나가 없으면 다른 것으로 채움)
                if '방문인구(명)' in merged_df.columns and '매출기반_방문인구' in merged_df.columns:
                    merged_df['방문인구(명)'] = merged_df['방문인구(명)'].fillna(merged_df['매출기반_방문인구'])
                elif '매출기반_방문인구' in merged_df.columns and '방문인구(명)' not in merged_df.columns:
                    merged_df['방문인구(명)'] = merged_df['매출기반_방문인구']
                
                if '매출기반_방문인구' in merged_df.columns:
                    merged_df = merged_df.drop(columns=['매출기반_방문인구'], errors='ignore')
                print(f"    병합된 데이터: {len(merged_df)}행")
    
    if merged_df.empty:
        print("경고: 데이터가 없습니다. 샘플 데이터로 대체합니다.")
        # 샘플 데이터 생성
        dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
        merged_df = pd.DataFrame({
            '연도': dates.year,
            '월': dates.month,
            '관광지명': ['헤이리예술마을'] * len(dates),
            '방문인구(명)': np.random.randint(10000, 50000, len(dates)) + 
                         (dates.month - 1) * 1000  # 계절성 추가
        })
        merged_df['date'] = dates
    
    # 날짜 컬럼 생성
    if 'date' not in merged_df.columns:
        if '연도' in merged_df.columns and '월' in merged_df.columns:
            merged_df['date'] = pd.to_datetime(
                merged_df['연도'].astype(str) + '-' + 
                merged_df['월'].astype(str).str.zfill(2) + '-01'
            )
        else:
            merged_df['date'] = pd.date_range('2023-01-01', periods=len(merged_df), freq='D')
    
    # 관광지명 컬럼이 없으면 생성
    if '관광지명' not in merged_df.columns:
        if '문화공간명' in merged_df.columns:
            merged_df['관광지명'] = merged_df['문화공간명']
        else:
            merged_df['관광지명'] = '헤이리예술마을'
    
    print(f"    최종 병합 데이터: {len(merged_df)}행")
    
    # 특징 엔지니어링
    print("\n[3/6] 특징 엔지니어링 중...")
    engineer = EnhancedFeatureEngineer(data_loader=loader)
    
    # 특징 생성
    target_col = '방문인구(명)' if '방문인구(명)' in merged_df.columns else merged_df.columns[-1]
    features_df = engineer.prepare_features(merged_df, target_col)
    
    # 특징 컬럼 추출 (타겟 변수 제외)
    feature_cols = engineer.get_feature_names(features_df, target_col=target_col)
    if len(feature_cols) == 0:
        print("경고: 특징이 없습니다. 기본 특징을 생성합니다.")
        feature_cols = ['year', 'month', 'day_of_week', 'is_weekend', 'season']
        for col in feature_cols:
            if col not in features_df.columns:
                features_df[col] = np.random.randn(len(features_df))
    
    print(f"    생성된 특징 수: {len(feature_cols)}")
    print(f"    특징 목록: {feature_cols[:10]}...")
    
    # 타겟 추출 및 결측치 처리
    if target_col not in features_df.columns:
        print(f"경고: 타겟 컬럼 '{target_col}'이 없습니다. 샘플 데이터 생성 중...")
        features_df[target_col] = np.random.randint(10000, 50000, len(features_df))
    
    # 결측치 처리
    features_df = features_df.dropna(subset=[target_col])
    X = features_df[feature_cols].fillna(0)
    y = features_df[target_col]
    
    print(f"    학습 데이터: {len(X)}행, {len(feature_cols)}개 특징")
    print(f"    타겟 데이터 범위: {y.min():.0f} ~ {y.max():.0f}")
    
    # 모델 학습 (k-fold 교차 검증 사용)
    print("\n[4/6] 모델 학습 중 (k-fold 교차 검증)...")
    predictor = SpatiotemporalPredictor(model_type='random_forest')
    results = predictor.train(X, y, use_kfold=True, cv_folds=5)
    
    print("\n학습 결과:")
    if results.get('use_kfold', True):
        print(f"  [교차 검증 결과]")
        print(f"  평균 MAE: {results['cv_mae_mean']:.2f} ± {results['cv_mae_std']:.2f}")
        print(f"  평균 RMSE: {results['cv_rmse_mean']:.2f} ± {results['cv_rmse_std']:.2f}")
        print(f"  평균 R²: {results['cv_r2_mean']:.4f} ± {results['cv_r2_std']:.4f}")
        print(f"  평균 MAPE: {results.get('final_mape', 0):.2f}%")
        print(f"\n  [최종 모델 성능]")
        print(f"  최종 MAE: {results['final_mae']:.2f}")
        print(f"  최종 RMSE: {results['final_rmse']:.2f}")
        print(f"  최종 R²: {results['final_r2']:.4f}")
        print(f"  최종 MAPE: {results.get('final_mape', 0):.2f}%")
        print(f"\n  [Fold별 상세 결과]")
        for fold_result in results.get('cv_folds', []):
            print(f"    Fold {fold_result['fold']}: MAE={fold_result['mae']:.2f}, "
                  f"RMSE={fold_result['rmse']:.2f}, R²={fold_result['r2']:.4f}")
    else:
        print(f"  학습 데이터 MAE: {results['train_mae']:.2f}")
        print(f"  테스트 데이터 MAE: {results['test_mae']:.2f}")
        print(f"  테스트 데이터 RMSE: {results['test_rmse']:.2f}")
        print(f"  테스트 데이터 R²: {results['test_r2']:.4f}")
        print(f"  테스트 데이터 MAPE: {results.get('test_mape', 0):.2f}%")
    
    # 모델 저장
    print("\n[5/6] 모델 저장 중...")
    model_dir = project_root / "src" / "ml" / "models" / "saved"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "spatiotemporal_model.pkl"
    predictor.save(model_path)
    print(f"    모델 저장 완료: {model_path}")
    
    # 학습 결과 저장
    print("\n[6/6] 학습 결과 저장 중...")
    results_dir = project_root / "src" / "output"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "training_results.json"
    
    training_results = {
        'training_date': datetime.now().isoformat(),
        'model_type': 'Random Forest',
        'validation_method': 'K-Fold Cross Validation (5 folds)',
        'results': results,
        'n_features': len(feature_cols),
        'n_samples': len(X),
        'feature_names': feature_cols,
        'data_sources': {
            'visit_data_rows': len(visit_df),
            'revenue_data_rows': len(revenue_df),
            'life_population_rows': len(life_pop_df),
            'time_slot_rows': len(time_slot_df),
            'weekend_pattern_rows': len(weekend_df),
            'consumption_pattern_rows': len(consumption_df),
            'vitality_index_rows': len(vitality_df),
        },
        'target_range': {
            'min': float(y.min()),
            'max': float(y.max()),
            'mean': float(y.mean()),
            'std': float(y.std())
        }
    }
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(training_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"    학습 결과 저장 완료: {results_path}")
    
    print("\n" + "="*60)
    print("학습 완료!")
    print("="*60)
    print(f"학습 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n모델은 {model_path}에 저장되었습니다.")
    print("이 모델을 사용하여 향후 데이터를 계속 예측할 수 있습니다.")
    print("="*60)
    
    return predictor, results


if __name__ == "__main__":
    predictor, results = main()

