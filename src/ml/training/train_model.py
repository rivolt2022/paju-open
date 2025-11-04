"""
큐레이션 중심 모델 학습 스크립트 - data 폴더의 모든 데이터를 활용하여 큐레이션 지표 학습
혼잡도 예측 대신 프로그램 추천 점수, 시간대별 적합도, 타겟 고객층 매칭 점수를 학습
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import pandas as pd
import numpy as np
import json

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ml.preprocessing.enhanced_data_loader import EnhancedDataLoader
from src.ml.preprocessing.enhanced_feature_engineering import EnhancedFeatureEngineer
from src.ml.models.spatiotemporal_model import SpatiotemporalPredictor


def calculate_program_type_score(program_type: str, business_revenue: float, keywords: List[str]) -> float:
    """프로그램 타입별 적합도 점수 계산"""
    keyword_match = sum(1 for kw in keywords if kw in str(business_revenue))
    match_score = keyword_match / len(keywords) if keywords else 0.0
    return business_revenue * match_score


def create_curation_targets(merged_df: pd.DataFrame, loader: EnhancedDataLoader) -> pd.DataFrame:
    """
    큐레이션 타겟 변수 생성
    - 프로그램 타입별 추천 점수
    - 시간대별 프로그램 적합도
    - 타겟 고객층 매칭 점수
    """
    print("  - 큐레이션 타겟 변수 생성 중...")
    
    # EDA 결과 기반 프로그램 타입 정의
    program_types = {
        '북토크': {'keywords': ['창작', '예술', '여가', '서비스', '교육'], 'optimal_times': ['오후', '저녁'], 'target_ages': ['30대', '40대', '50대']},
        '작가 사인회': {'keywords': ['소매', '창작', '예술'], 'optimal_times': ['오후', '주말'], 'target_ages': ['20대', '30대', '40대']},
        '전시회': {'keywords': ['창작', '예술', '여가'], 'optimal_times': ['전체'], 'target_ages': ['20대', '30대', '40대', '50대']},
        '문화 프로그램': {'keywords': ['창작', '예술', '교육'], 'optimal_times': ['주말', '오후'], 'target_ages': ['전체']}
    }
    
    # 업종별 매출 데이터 로드
    try:
        samsung_file = project_root / "data" / "3_관광지분석_삼성카드.xlsx"
        if samsung_file.exists():
            business_df = pd.read_excel(samsung_file, sheet_name="23p_35p_47p_59p_70p_업종")
        else:
            business_df = pd.DataFrame()
    except:
        business_df = pd.DataFrame()
    
    # 결과 DataFrame 생성
    curation_df = merged_df.copy()
    
    # 업종별 매출 데이터 기반 기본 점수 계산
    base_scores = {}
    if not business_df.empty and '중분류업종' in business_df.columns:
        for program_type, config in program_types.items():
            scores = []
            for _, row in business_df.iterrows():
                business = str(row.get('중분류업종', ''))
                revenue = float(row.get('월평균 매출금액(백만원)', 0))
                
                keyword_match = sum(1 for kw in config['keywords'] if kw in business)
                if keyword_match > 0:
                    match_score = keyword_match / len(config['keywords'])
                    scores.append(revenue * match_score)
            
            base_score = np.mean(scores) if scores else 50.0
            max_score = np.max(scores) if scores else 100.0
            base_scores[program_type] = min((base_score / max_score) * 100, 100) if max_score > 0 else 50.0
    else:
        # 기본값 (EDA 결과 기반)
        base_scores = {
            '북토크': 68.4,
            '작가 사인회': 100.0,
            '전시회': 29.1,
            '문화 프로그램': 22.2
        }
    
    # 각 프로그램 타입별 추천 점수 계산 (행별로 변동 추가)
    for program_type, config in program_types.items():
        score_col = f'{program_type}_recommendation_score'
        base_score = base_scores.get(program_type, 50.0)
        
        # 관광지별, 날짜별 변동 추가
        scores = []
        for idx, row in curation_df.iterrows():
            # 관광지명에 따른 변동
            space_name = str(row.get('관광지명', ''))
            space_multiplier = {
                '헤이리예술마을': 1.0,
                '파주출판단지': 1.2,  # 출판단지에 특화된 프로그램
                '교하도서관': 1.1,
                '파주출판도시': 1.2,
                '파주문화센터': 0.9,
                '출판문화정보원': 1.3
            }.get(space_name, 1.0)
            
            # 날짜에 따른 변동 (주말, 계절 등)
            if 'date' in row and pd.notna(row['date']):
                date_obj = pd.to_datetime(row['date'])
                is_weekend = 1 if date_obj.weekday() >= 5 else 0
                month = date_obj.month
                
                # 주말 프로그램은 주말에 점수 증가
                if '주말' in config.get('optimal_times', []):
                    weekend_multiplier = 1.2 if is_weekend else 0.9
                else:
                    weekend_multiplier = 1.0
                
                # 계절별 변동
                if month in [9, 10, 11]:  # 가을
                    season_multiplier = 1.1
                elif month in [12, 1, 2]:  # 겨울
                    season_multiplier = 0.9
                else:
                    season_multiplier = 1.0
            else:
                weekend_multiplier = 1.0
                season_multiplier = 1.0
            
            # 방문인구 기반 조정
            visit_pop = float(row.get('방문인구(명)', 0)) if pd.notna(row.get('방문인구(명)', 0)) else 30000
            visit_multiplier = min(visit_pop / 50000, 1.2)  # 0.6 ~ 1.2 범위
            
            # 최종 점수 계산 (변동 추가)
            final_score = base_score * space_multiplier * weekend_multiplier * season_multiplier * visit_multiplier
            final_score = max(0, min(100, final_score))  # 0-100 범위로 제한
            scores.append(final_score)
        
        curation_df[score_col] = scores
    
    # 시간대별 프로그램 적합도 계산
    time_slot_df = loader.load_time_slot_population_data()
    for program_type, config in program_types.items():
        time_col = f'{program_type}_time_suitability'
        optimal_times = config.get('optimal_times', [])
        
        # 기본 점수 (EDA 결과 기반)
        base_score = 75.0
        
        # 행별로 적합도 계산
        suitability_scores = []
        for idx, row in curation_df.iterrows():
            # 날짜 기반 시간대 적합도
            if 'date' in row and pd.notna(row['date']):
                date_obj = pd.to_datetime(row['date'])
                is_weekend = 1 if date_obj.weekday() >= 5 else 0
                hour = date_obj.hour if hasattr(date_obj, 'hour') else 14
                
                # 시간대 분류
                if 6 <= hour < 12:
                    time_slot = 'morning'
                elif 12 <= hour < 18:
                    time_slot = 'afternoon'
                else:
                    time_slot = 'evening'
                
                # 최적 시간대와 매칭
                if '주말' in optimal_times and is_weekend:
                    match_score = 1.2
                elif '오후' in optimal_times and time_slot == 'afternoon':
                    match_score = 1.2
                elif '저녁' in optimal_times and time_slot == 'evening':
                    match_score = 1.2
                elif '전체' in optimal_times:
                    match_score = 1.0
                else:
                    match_score = 0.8
            else:
                match_score = 1.0
            
            final_score = base_score * match_score
            final_score = max(0, min(100, final_score))  # 0-100 범위로 제한
            suitability_scores.append(final_score)
        
        curation_df[time_col] = suitability_scores
    
    # 타겟 고객층 매칭 점수 계산
    try:
        lg_file = project_root / "data" / "3_관광지분석_LG유플러스.xlsx"
        if lg_file.exists():
            demo_df = pd.read_excel(lg_file, sheet_name="성연령별_방문인구")
        else:
            demo_df = pd.DataFrame()
    except:
        demo_df = pd.DataFrame()
    
    for program_type, config in program_types.items():
        demo_col = f'{program_type}_demographic_match'
        target_ages = config.get('target_ages', [])
        
        # 기본 점수 (EDA 결과 기반)
        base_score = 70.0
        
        # 행별로 매칭 점수 계산
        match_scores = []
        for idx, row in curation_df.iterrows():
            # 관광지별로 다른 타겟 고객층 분포 가정
            space_name = str(row.get('관광지명', ''))
            
            # 관광지별 타겟 고객층 매칭 가중치
            space_match = {
                '헤이리예술마을': 1.0,
                '파주출판단지': 1.1,  # 출판 관련 프로그램에 적합
                '교하도서관': 1.15,  # 도서관 프로그램에 적합
                '파주출판도시': 1.1,
                '파주문화센터': 1.0,
                '출판문화정보원': 1.2
            }.get(space_name, 1.0)
            
            # 방문인구 기반 조정
            visit_pop = float(row.get('방문인구(명)', 0)) if pd.notna(row.get('방문인구(명)', 0)) else 30000
            visit_multiplier = min(visit_pop / 40000, 1.1)  # 방문인구가 많을수록 매칭 점수 증가
            
            # 최종 점수
            final_score = base_score * space_match * visit_multiplier
            final_score = max(0, min(100, final_score))  # 0-100 범위로 제한
            match_scores.append(final_score)
        
        curation_df[demo_col] = match_scores
    
    print(f"    생성된 큐레이션 타겟 변수: {len([c for c in curation_df.columns if '_recommendation_score' in c or '_suitability' in c or '_match' in c])}개")
    
    return curation_df


def main():
    """메인 학습 함수 - 큐레이션 모델 및 방문인구 예측 모델 학습"""
    print("="*60)
    print("모델 학습 시작 (큐레이션 + 방문인구 예측)")
    print("="*60)
    print(f"학습 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 데이터 로드
    print("\n[1/7] 데이터 로드 중...")
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
    print("\n[2/7] 데이터 병합 및 큐레이션 타겟 생성 중...")
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
                if '방문인구(명)' in merged_df.columns and '매출기반_방문인구' in merged_df.columns:
                    merged_df['방문인구(명)'] = merged_df['방문인구(명)'].fillna(merged_df['매출기반_방문인구'])
                elif '매출기반_방문인구' in merged_df.columns and '방문인구(명)' not in merged_df.columns:
                    merged_df['방문인구(명)'] = merged_df['매출기반_방문인구']
                
                if '매출기반_방문인구' in merged_df.columns:
                    merged_df = merged_df.drop(columns=['매출기반_방문인구'], errors='ignore')
                print(f"    병합된 데이터: {len(merged_df)}행")
    
    if merged_df.empty:
        print("경고: 데이터가 없습니다. 샘플 데이터로 대체합니다.")
        dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
        merged_df = pd.DataFrame({
            '연도': dates.year,
            '월': dates.month,
            '관광지명': ['헤이리예술마을'] * len(dates),
            '방문인구(명)': np.random.randint(10000, 50000, len(dates)) + 
                         (dates.month - 1) * 1000
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
    
    # 큐레이션 타겟 변수 생성
    curation_df = create_curation_targets(merged_df, loader)
    
    print(f"    최종 데이터: {len(curation_df)}행")
    
    # 특징 엔지니어링
    print("\n[3/7] 특징 엔지니어링 중...")
    engineer = EnhancedFeatureEngineer(data_loader=loader)
    
    # 특징 생성 (기존 방문인구를 특징으로 사용)
    target_col = '방문인구(명)' if '방문인구(명)' in curation_df.columns else curation_df.columns[-1]
    features_df = engineer.prepare_features(curation_df, target_col)
    
    # 큐레이션 타겟 변수들을 특징에 추가
    curation_target_cols = [c for c in curation_df.columns if '_recommendation_score' in c or '_suitability' in c or '_match' in c]
    for col in curation_target_cols:
        if col in curation_df.columns and col not in features_df.columns:
            features_df[col] = curation_df[col].values
    
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
    
    # 큐레이션 타겟 변수들 학습
    print("\n[4/7] 큐레이션 모델 학습 중...")
    
    program_types = ['북토크', '작가 사인회', '전시회', '문화 프로그램']
    trained_models = {}
    all_results = {}
    
    for program_type in program_types:
        print(f"\n  [{program_type}] 모델 학습 중...")
        
        # 타겟 변수 선택
        target_cols = [
            f'{program_type}_recommendation_score',
            f'{program_type}_time_suitability',
            f'{program_type}_demographic_match'
        ]
        
        for target_col in target_cols:
            if target_col not in features_df.columns:
                print(f"    경고: {target_col}이 없습니다. 기본값 생성 중...")
                features_df[target_col] = 70.0  # 기본값
            
            # 결측치 처리
            valid_mask = features_df[target_col].notna()
            if valid_mask.sum() == 0:
                print(f"    경고: {target_col}에 유효한 데이터가 없습니다.")
                continue
            
            X = features_df.loc[valid_mask, feature_cols].fillna(0)
            y = features_df.loc[valid_mask, target_col]
            
            if len(X) == 0:
                print(f"    경고: {target_col} 학습 데이터가 없습니다.")
                continue
            
            print(f"    학습 데이터: {len(X)}행, 타겟 범위: {y.min():.1f} ~ {y.max():.1f}")
            
            # 모델 학습
            predictor = SpatiotemporalPredictor(model_type='random_forest')
            results = predictor.train(X, y, use_kfold=True, cv_folds=5)
            
            # 모델 저장
            model_key = f'{program_type}_{target_col.split("_")[-1]}'
            trained_models[model_key] = predictor
            
            # 결과 저장
            all_results[model_key] = {
                'target': target_col,
                'results': results,
                'n_features': len(feature_cols),
                'n_samples': len(X),
                'target_range': {
                    'min': float(y.min()),
                    'max': float(y.max()),
                    'mean': float(y.mean()),
                    'std': float(y.std())
                }
            }
            
            print(f"    학습 완료 - MAE: {results.get('cv_mae_mean', results.get('final_mae', 0)):.2f}, "
                  f"R²: {results.get('cv_r2_mean', results.get('final_r2', 0)):.4f}")
    
    # 방문인구 예측 모델 학습
    print("\n[5/7] 방문인구 예측 모델 학습 중...")
    visit_model = None
    visit_results = None
    
    if '방문인구(명)' in features_df.columns:
        # 유효한 데이터만 사용
        valid_mask = features_df['방문인구(명)'].notna() & (features_df['방문인구(명)'] > 0)
        X_visit = features_df.loc[valid_mask, feature_cols].fillna(0)
        y_visit = features_df.loc[valid_mask, '방문인구(명)']
        
        if len(X_visit) > 0:
            print(f"    학습 데이터: {len(X_visit)}행, 타겟 범위: {y_visit.min():.0f} ~ {y_visit.max():.0f}명")
            
            # 모델 학습
            visit_predictor = SpatiotemporalPredictor(model_type='random_forest')
            visit_results = visit_predictor.train(X_visit, y_visit, use_kfold=True, cv_folds=5)
            visit_model = visit_predictor
            
            print(f"    학습 완료 - MAE: {visit_results.get('cv_mae_mean', visit_results.get('final_mae', 0)):.2f}, "
                  f"RMSE: {visit_results.get('cv_rmse_mean', visit_results.get('final_rmse', 0)):.2f}, "
                  f"R²: {visit_results.get('cv_r2_mean', visit_results.get('final_r2', 0)):.4f}")
        else:
            print("    경고: 방문인구 예측 학습 데이터가 없습니다.")
    else:
        print("    경고: 방문인구(명) 컬럼이 없습니다.")
    
    # 모델 저장
    print("\n[6/7] 모델 저장 중...")
    model_dir = project_root / "src" / "ml" / "models" / "saved"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 통합 모델 저장 (각 프로그램 타입별로)
    for model_key, predictor in trained_models.items():
        model_path = model_dir / f"curation_{model_key}_model.pkl"
        predictor.save(model_path)
        print(f"    모델 저장 완료: {model_path}")
    
    # 방문인구 예측 모델 저장
    if visit_model is not None:
        visit_model_path = model_dir / "spatiotemporal_model.pkl"
        visit_model.save(visit_model_path)
        print(f"    방문인구 예측 모델 저장 완료: {visit_model_path}")
    
    # 메타데이터 저장 (모든 모델 정보)
    metadata_path = model_dir / "curation_models_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            'training_date': datetime.now().isoformat(),
            'model_type': 'Random Forest (Curation Focused)',
            'validation_method': 'K-Fold Cross Validation (5 folds)',
            'models': {k: {'target': v['target'], 'n_samples': v['n_samples']} 
                     for k, v in all_results.items()},
            'feature_names': feature_cols,
            'program_types': program_types
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"    메타데이터 저장 완료: {metadata_path}")
    
    # 학습 결과 저장
    print("\n[7/7] 학습 결과 저장 중...")
    results_dir = project_root / "src" / "output"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "curation_training_results.json"
    
    training_results = {
        'training_date': datetime.now().isoformat(),
        'model_type': 'Random Forest (Curation Focused)',
        'validation_method': 'K-Fold Cross Validation (5 folds)',
        'results': all_results,
        'visit_model_results': visit_results if visit_results else None,
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'program_types': program_types,
        'data_sources': {
            'visit_data_rows': len(visit_df),
            'revenue_data_rows': len(revenue_df),
            'life_population_rows': len(life_pop_df),
            'time_slot_rows': len(time_slot_df),
            'weekend_pattern_rows': len(weekend_df),
            'consumption_pattern_rows': len(consumption_df),
            'vitality_index_rows': len(vitality_df),
        }
    }
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(training_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"    학습 결과 저장 완료: {results_path}")
    
    print("\n" + "="*60)
    print("모델 학습 완료!")
    print("="*60)
    print(f"학습 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n학습된 모델:")
    print("  [큐레이션 모델]")
    for model_key in trained_models.keys():
        print(f"    - {model_key}")
    if visit_model is not None:
        print("  [방문인구 예측 모델]")
        print("    - spatiotemporal_model.pkl (방문인구 예측)")
    print(f"\n모델들은 {model_dir}에 저장되었습니다.")
    print("이 모델들을 사용하여 큐레이션 지표와 방문인구를 예측할 수 있습니다.")
    print("="*60)
    
    return trained_models, all_results, visit_model


if __name__ == "__main__":
    trained_models, results, visit_model = main()
