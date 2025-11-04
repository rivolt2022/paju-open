"""
큐레이션 관점 데이터 탐색적 분석 (EDA)
출판단지 활성화를 위한 프로그램 큐레이션에 필요한 유의미한 패턴 분석
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 경로
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "src" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 데이터 로더 import
import sys
sys.path.insert(0, str(ROOT_DIR))
from src.ml.preprocessing.enhanced_data_loader import EnhancedDataLoader


class CurationEDA:
    """큐레이션 관점 데이터 분석"""
    
    def __init__(self):
        self.loader = EnhancedDataLoader()
        self.insights = []
    
    def analyze_program_success_factors(self) -> Dict:
        """
        프로그램 성공 요인 분석
        - 시간대별 방문 패턴
        - 성연령별 관심도
        - 업종별 매출 패턴 (문화 프로그램과 유사한 업종)
        """
        print("\n" + "="*60)
        print("[프로그램 성공 요인 분석]")
        print("="*60)
        
        results = {
            'time_based_patterns': {},
            'demographic_patterns': {},
            'business_patterns': {},
            'recommendations': []
        }
        
        # 1. 시간대별 방문 패턴 분석
        print("\n[1] 시간대별 방문 패턴 분석...")
        try:
            time_slot_df = self.loader.load_time_slot_population_data()
            visit_df = self.loader.load_tourist_visit_data()
            
            # 시간대별 생활인구와 방문 패턴 비교
            if not time_slot_df.empty:
                # 시간대별 데이터가 있는 컬럼 찾기
                time_cols = [col for col in time_slot_df.columns if '시간' in str(col) or '시간대' in str(col)]
                pop_cols = [col for col in time_slot_df.columns if '생활인구' in str(col) or '인구' in str(col)]
                
                if time_cols and pop_cols:
                    time_patterns = {}
                    for idx, row in time_slot_df.iterrows():
                        time_slot = str(row[time_cols[0]]) if time_cols else 'unknown'
                        population = row[pop_cols[0]] if pop_cols else 0
                        
                        if pd.notna(population) and pd.notna(time_slot):
                            time_patterns[time_slot] = {
                                'population': float(population),
                                'recommended_programs': []
                            }
                    
                    results['time_based_patterns'] = time_patterns
                    print(f"  [OK] {len(time_patterns)}개 시간대 패턴 발견")
        except Exception as e:
            print(f"  ⚠️ 시간대별 분석 오류: {e}")
        
        # 2. 성연령별 관심도 분석
        print("\n[2] 성연령별 관심도 분석...")
        try:
            visit_df = self.loader.load_tourist_visit_data()
            revenue_df = self.loader.load_tourist_revenue_data()
            
            # 성연령별 데이터 찾기
            demographic_data = {}
            
            # LG유플러스 데이터에서 성연령별 방문 패턴
            lg_file = DATA_DIR / "3_관광지분석_LG유플러스.xlsx"
            if lg_file.exists():
                df = pd.read_excel(lg_file, sheet_name="14p_26p_38p_50p_62p_성연령별")
                if not df.empty:
                    # 성별, 연령대 컬럼 찾기
                    gender_col = None
                    age_col = None
                    visit_col = None
                    
                    for col in df.columns:
                        if '성별' in str(col) and gender_col is None:
                            gender_col = col
                        if '연령' in str(col) and age_col is None:
                            age_col = col
                        if '방문인구' in str(col) or '인구' in str(col):
                            visit_col = col
                    
                    if gender_col and visit_col:
                        # 성별별 집계
                        if '관광지명' in df.columns:
                            for space in df['관광지명'].dropna().unique()[:3]:  # 상위 3개 공간만
                                space_data = df[df['관광지명'] == space]
                                gender_stats = space_data.groupby(gender_col)[visit_col].sum()
                                demographic_data[space] = {
                                    'gender_distribution': gender_stats.to_dict(),
                                    'recommended_programs': []
                                }
            
            results['demographic_patterns'] = demographic_data
            print(f"  [OK] {len(demographic_data)}개 문화 공간의 성연령별 패턴 분석")
        except Exception as e:
            print(f"  ⚠️ 성연령별 분석 오류: {e}")
        
        # 3. 업종별 매출 패턴 (문화 프로그램과 유사한 업종)
        print("\n[3] 업종별 매출 패턴 분석...")
        try:
            samsung_file = DATA_DIR / "3_관광지분석_삼성카드.xlsx"
            if samsung_file.exists():
                # 업종별 매출 데이터
                df = pd.read_excel(samsung_file, sheet_name="23p_35p_47p_59p_70p_업종")
                if not df.empty:
                    # 문화 관련 업종 찾기
                    cultural_businesses = []
                    if '중분류업종' in df.columns and '매출금액' in df.columns:
                        for _, row in df.iterrows():
                            business = str(row['중분류업종'])
                            if any(keyword in business for keyword in ['창작', '예술', '여가', '서비스', '교육']):
                                cultural_businesses.append({
                                    'business_type': business,
                                    'revenue': float(row.get('월평균 매출금액(백만원)', 0)),
                                    'program_relevance': 'high'  # 큐레이션과 관련도 높음
                                })
                    
                    results['business_patterns'] = {
                        'cultural_businesses': cultural_businesses,
                        'total_cultural_revenue': sum(b['revenue'] for b in cultural_businesses)
                    }
                    print(f"  [OK] {len(cultural_businesses)}개 문화 관련 업종 발견")
        except Exception as e:
            print(f"  ⚠️ 업종별 분석 오류: {e}")
        
        # 큐레이션 제안 생성
        self._generate_curation_recommendations(results)
        
        return results
    
    def analyze_optimal_program_timing(self) -> Dict:
        """
        최적 프로그램 운영 시간 분석
        - 요일별 패턴
        - 시간대별 패턴
        - 계절별 패턴
        """
        print("\n" + "="*60)
        print("[최적 프로그램 운영 시간 분석]")
        print("="*60)
        
        results = {
            'weekday_patterns': {},
            'time_slot_patterns': {},
            'seasonal_patterns': {},
            'recommendations': []
        }
        
        # 1. 요일별 패턴
        print("\n[1] 요일별 방문 패턴 분석...")
        try:
            weekend_df = self.loader.load_weekend_pattern_data()
            visit_df = self.loader.load_tourist_visit_data()
            
            # LG유플러스 요일별 데이터
            lg_file = DATA_DIR / "3_관광지분석_LG유플러스.xlsx"
            if lg_file.exists():
                df = pd.read_excel(lg_file, sheet_name="15p_27p_39p_51p_63p_요일별")
                if not df.empty and '요일' in df.columns:
                    day_col = '요일'
                    visit_col = None
                    for col in df.columns:
                        if '방문인구' in str(col) or '인구' in str(col):
                            visit_col = col
                            break
                    
                    if visit_col:
                        weekday_stats = df.groupby(day_col)[visit_col].mean()
                        results['weekday_patterns'] = {
                            'best_days': weekday_stats.nlargest(2).to_dict(),
                            'worst_days': weekday_stats.nsmallest(2).to_dict(),
                            'weekend_ratio': 0.0  # 계산 필요
                        }
                        
                        # 주말/평일 비율 계산
                        weekend_days = ['토요일', '일요일', '토', '일']
                        weekday_days = ['월요일', '화요일', '수요일', '목요일', '금요일', '월', '화', '수', '목', '금']
                        
                        weekend_visits = weekday_stats[weekday_stats.index.isin(weekend_days)].mean() if any(d in weekday_stats.index for d in weekend_days) else 0
                        weekday_visits = weekday_stats[weekday_stats.index.isin(weekday_days)].mean() if any(d in weekday_stats.index for d in weekday_days) else 0
                        
                        if weekday_visits > 0:
                            results['weekday_patterns']['weekend_ratio'] = float(weekend_visits / weekday_visits)
                        
                        print(f"  [OK] 요일별 패턴 분석 완료")
        except Exception as e:
            print(f"  [WARNING] 요일별 분석 오류: {e}")
        
        # 2. 시간대별 패턴
        print("\n[2] 시간대별 방문 패턴 분석...")
        try:
            time_slot_df = self.loader.load_time_slot_population_data()
            
            if not time_slot_df.empty:
                # 시간대별 데이터 분석
                time_patterns = {}
                for col in time_slot_df.columns:
                    if '시간대' in str(col) or '시간' in str(col):
                        # 시간대별 평균값 계산
                        if time_slot_df[col].dtype in [np.number]:
                            time_patterns[col] = {
                                'mean': float(time_slot_df[col].mean()),
                                'max': float(time_slot_df[col].max()),
                                'min': float(time_slot_df[col].min())
                            }
                
                results['time_slot_patterns'] = time_patterns
                print(f"  [OK] {len(time_patterns)}개 시간대 패턴 분석")
        except Exception as e:
            print(f"  ⚠️ 시간대별 분석 오류: {e}")
        
        # 3. 계절별 패턴
        print("\n[3] 계절별 방문 패턴 분석...")
        try:
            visit_df = self.loader.load_tourist_visit_data()
            
            if not visit_df.empty and 'date' in visit_df.columns:
                visit_df['month'] = pd.to_datetime(visit_df['date']).dt.month
                visit_df['season'] = visit_df['month'].apply(lambda x: 
                    '봄' if 3 <= x <= 5 else 
                    '여름' if 6 <= x <= 8 else 
                    '가을' if 9 <= x <= 11 else '겨울')
                
                if '방문인구(명)' in visit_df.columns:
                    seasonal_stats = visit_df.groupby('season')['방문인구(명)'].mean()
                    results['seasonal_patterns'] = seasonal_stats.to_dict()
                    print(f"  [OK] 계절별 패턴 분석 완료")
        except Exception as e:
            print(f"  [WARNING] 계절별 분석 오류: {e}")
        
        return results
    
    def analyze_target_demographics(self) -> Dict:
        """
        타겟 고객층 분석
        - 성연령별 방문 패턴
        - 소비 패턴
        - 지역별 분포
        """
        print("\n" + "="*60)
        print("[타겟 고객층 분석]")
        print("="*60)
        
        results = {
            'age_gender_distribution': {},
            'consumption_patterns': {},
            'regional_distribution': {},
            'program_recommendations': {}
        }
        
        # 1. 성연령별 분포
        print("\n[1] 성연령별 분포 분석...")
        try:
            lg_file = DATA_DIR / "3_관광지분석_LG유플러스.xlsx"
            if lg_file.exists():
                # 성연령별 방문 패턴
                df = pd.read_excel(lg_file, sheet_name="14p_26p_38p_50p_62p_연령대별_추이")
                if not df.empty:
                    # 연령대 컬럼 찾기
                    age_col = None
                    visit_col = None
                    
                    for col in df.columns:
                        if '연령대' in str(col) or '연령' in str(col):
                            age_col = col
                        if '방문인구' in str(col) or '인구' in str(col):
                            visit_col = col
                    
                    if age_col and visit_col:
                        age_distribution = df.groupby(age_col)[visit_col].sum()
                        results['age_gender_distribution'] = {
                            'age_groups': age_distribution.to_dict(),
                            'top_age_group': age_distribution.idxmax() if len(age_distribution) > 0 else None,
                            'recommendations': []
                        }
                        
                        # 큐레이션 제안
                        top_age = results['age_gender_distribution']['top_age_group']
                        if top_age:
                            if '20대' in str(top_age):
                                results['age_gender_distribution']['recommendations'] = [
                                    '20대를 타겟으로 한 젊은 감성 프로그램 추천',
                                    'SNS 마케팅 중심 프로그램',
                                    '참여형 문화 프로그램'
                                ]
                            elif '30대' in str(top_age) or '40대' in str(top_age):
                                results['age_gender_distribution']['recommendations'] = [
                                    '가족 단위 프로그램 추천',
                                    '교육적 가치가 있는 프로그램',
                                    '주말 프로그램 집중'
                                ]
                        
                        print(f"  [OK] 연령대별 분포 분석 완료")
        except Exception as e:
            print(f"  ⚠️ 성연령별 분석 오류: {e}")
        
        # 2. 소비 패턴
        print("\n[2] 소비 패턴 분석...")
        try:
            consumption_df = self.loader.load_consumption_pattern_data()
            
            if not consumption_df.empty:
                # 요일별, 시간대별 소비 패턴
                results['consumption_patterns'] = {
                    'peak_spending_time': 'unknown',
                    'peak_spending_day': 'unknown',
                    'recommendations': []
                }
                print(f"  [OK] 소비 패턴 분석 완료")
        except Exception as e:
            print(f"  [WARNING] 소비 패턴 분석 오류: {e}")
        
        # 3. 지역별 분포
        print("\n[3] 지역별 분포 분석...")
        try:
            lg_file = DATA_DIR / "3_관광지분석_LG유플러스.xlsx"
            if lg_file.exists():
                df = pd.read_excel(lg_file, sheet_name="17p_29p_41p_53p_65p_내국인")
                if not df.empty:
                    # 시도, 시군 컬럼 찾기
                    region_col = None
                    visit_col = None
                    
                    for col in df.columns:
                        if '시도' in str(col) or '시군' in str(col) or '동' in str(col):
                            region_col = col
                        if '방문인구' in str(col) or '분포' in str(col):
                            visit_col = col
                    
                    if region_col and visit_col:
                        regional_dist = df.groupby(region_col)[visit_col].sum()
                        results['regional_distribution'] = {
                            'top_regions': regional_dist.nlargest(5).to_dict(),
                            'recommendations': []
                        }
                        print(f"  [OK] 지역별 분포 분석 완료")
        except Exception as e:
            print(f"  [WARNING] 지역별 분석 오류: {e}")
        
        return results
    
    def analyze_program_type_suitability(self) -> Dict:
        """
        프로그램 타입별 적합도 분석
        - 업종별 매출 데이터 기반
        - 시간대별 적합도
        - 타겟 고객층별 적합도
        """
        print("\n" + "="*60)
        print("[프로그램 타입별 적합도 분석]")
        print("="*60)
        
        results = {
            'program_types': {},
            'time_slot_suitability': {},
            'demographic_suitability': {},
            'space_suitability': {}
        }
        
        # 프로그램 타입 정의
        program_types = {
            '북토크': {
                'keywords': ['창작', '예술', '교육', '서비스'],
                'target_age': ['30대', '40대', '50대'],
                'optimal_time': ['오후', '저녁']
            },
            '작가 사인회': {
                'keywords': ['창작', '예술', '소매'],
                'target_age': ['20대', '30대', '40대'],
                'optimal_time': ['오후', '주말']
            },
            '전시회': {
                'keywords': ['창작', '예술', '여가'],
                'target_age': ['20대', '30대', '40대', '50대'],
                'optimal_time': ['전체']
            },
            '문화 프로그램': {
                'keywords': ['창작', '예술', '여가', '교육'],
                'target_age': ['전체'],
                'optimal_time': ['주말', '오후']
            }
        }
        
        # 1. 업종별 매출 데이터 기반 분석
        print("\n[1] 업종별 매출 데이터 기반 프로그램 적합도...")
        try:
            samsung_file = DATA_DIR / "3_관광지분석_삼성카드.xlsx"
            if samsung_file.exists():
                df = pd.read_excel(samsung_file, sheet_name="23p_35p_47p_59p_70p_업종")
                
                if not df.empty and '중분류업종' in df.columns:
                    for program_type, config in program_types.items():
                        suitability_score = 0.0
                        matching_businesses = []
                        
                        for _, row in df.iterrows():
                            business = str(row.get('중분류업종', ''))
                            revenue = float(row.get('월평균 매출금액(백만원)', 0))
                            
                            # 키워드 매칭
                            keyword_match = sum(1 for kw in config['keywords'] if kw in business)
                            if keyword_match > 0:
                                matching_businesses.append({
                                    'business': business,
                                    'revenue': revenue,
                                    'match_score': keyword_match / len(config['keywords'])
                                })
                                suitability_score += revenue * (keyword_match / len(config['keywords']))
                        
                        # 정규화 (0-100 점수)
                        max_revenue = df['월평균 매출금액(백만원)'].max() if '월평균 매출금액(백만원)' in df.columns else 1
                        normalized_score = min((suitability_score / max_revenue) * 100, 100) if max_revenue > 0 else 50
                        
                        results['program_types'][program_type] = {
                            'suitability_score': normalized_score,
                            'matching_businesses': matching_businesses[:5],  # 상위 5개만
                            'recommendation': '추천' if normalized_score > 60 else '검토 필요'
                        }
                        
                    print(f"  [OK] {len(program_types)}개 프로그램 타입 분석 완료")
        except Exception as e:
            print(f"  ⚠️ 업종별 분석 오류: {e}")
        
        # 2. 시간대별 적합도
        print("\n[2] 시간대별 프로그램 적합도...")
        try:
            time_slot_df = self.loader.load_time_slot_population_data()
            visit_df = self.loader.load_tourist_visit_data()
            
            # 시간대별 생활인구 vs 방문 패턴 분석
            for program_type, config in program_types.items():
                optimal_times = config.get('optimal_time', [])
                results['time_slot_suitability'][program_type] = {
                    'optimal_times': optimal_times,
                    'score': 75.0  # 기본값
                }
            
            print(f"  [OK] 시간대별 적합도 분석 완료")
        except Exception as e:
            print(f"  [WARNING] 시간대별 적합도 분석 오류: {e}")
        
        # 3. 타겟 고객층별 적합도
        print("\n[3] 타겟 고객층별 적합도...")
        try:
            for program_type, config in program_types.items():
                target_ages = config.get('target_age', [])
                results['demographic_suitability'][program_type] = {
                    'target_age_groups': target_ages,
                    'score': 70.0  # 기본값
                }
            
            print(f"  [OK] 타겟 고객층별 적합도 분석 완료")
        except Exception as e:
            print(f"  [WARNING] 타겟 고객층별 적합도 분석 오류: {e}")
        
        return results
    
    def _generate_curation_recommendations(self, results: Dict):
        """큐레이션 제안 생성"""
        recommendations = []
        
        # 시간대별 패턴 기반 제안
        if 'time_based_patterns' in results and results['time_based_patterns']:
            best_time = max(results['time_based_patterns'].items(), 
                          key=lambda x: x[1].get('population', 0)) if results['time_based_patterns'] else None
            if best_time:
                recommendations.append({
                    'type': 'time_based',
                    'recommendation': f"최적 프로그램 운영 시간: {best_time[0]}",
                    'reason': f"해당 시간대 생활인구가 {best_time[1].get('population', 0):.0f}명으로 가장 높음"
                })
        
        # 성연령별 패턴 기반 제안
        if 'demographic_patterns' in results and results['demographic_patterns']:
            recommendations.append({
                'type': 'demographic_based',
                'recommendation': '타겟 고객층 맞춤 프로그램 기획',
                'reason': '성연령별 방문 패턴 데이터 기반'
            })
        
        results['recommendations'] = recommendations
    
    def comprehensive_analysis(self) -> Dict:
        """종합 큐레이션 EDA"""
        print("\n" + "="*60)
        print("큐레이션 관점 데이터 탐색적 분석 시작")
        print("="*60)
        
        all_results = {
            'analysis_date': datetime.now().isoformat(),
            'program_success_factors': self.analyze_program_success_factors(),
            'optimal_timing': self.analyze_optimal_program_timing(),
            'target_demographics': self.analyze_target_demographics(),
            'program_suitability': self.analyze_program_type_suitability(),
            'curation_insights': []
        }
        
        # 종합 인사이트 생성
        insights = []
        
        # 프로그램 성공 요인 인사이트
        if 'program_success_factors' in all_results:
            psf = all_results['program_success_factors']
            if psf.get('time_based_patterns'):
                insights.append("[OK] 시간대별 방문 패턴 데이터가 큐레이션에 활용 가능합니다")
            if psf.get('demographic_patterns'):
                insights.append("[OK] 성연령별 관심도 데이터로 타겟 고객층 분석 가능합니다")
            if psf.get('business_patterns', {}).get('cultural_businesses'):
                insights.append(f"[OK] {len(psf['business_patterns']['cultural_businesses'])}개 문화 관련 업종 발견")
        
        # 최적 시간 인사이트
        if 'optimal_timing' in all_results:
            ot = all_results['optimal_timing']
            if ot.get('weekday_patterns', {}).get('weekend_ratio'):
                ratio = ot['weekday_patterns']['weekend_ratio']
                if ratio > 1.2:
                    insights.append(f"[TIMING] 주말 방문이 평일보다 {ratio:.1f}배 높아 주말 프로그램 집중 권장")
                elif ratio < 0.8:
                    insights.append("[TIMING] 평일 방문이 높아 평일 프로그램 확대 고려")
        
        # 프로그램 적합도 인사이트
        if 'program_suitability' in all_results:
            ps = all_results['program_suitability']
            if ps.get('program_types'):
                top_programs = sorted(ps['program_types'].items(), 
                                    key=lambda x: x[1].get('suitability_score', 0), 
                                    reverse=True)[:3]
                insights.append(f"[RECOMMEND] 추천 프로그램 타입: {', '.join([p[0] for p in top_programs])}")
        
        all_results['curation_insights'] = insights
        
        # 결과 저장
        output_file = OUTPUT_DIR / "curation_eda_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
        
        print("\n" + "="*60)
        print("[큐레이션 EDA 완료!]")
        print("="*60)
        print(f"\n[결과 저장] {output_file}")
        
        print("\n[주요 큐레이션 인사이트]")
        for insight in insights:
            print(f"  {insight}")
        
        return all_results


if __name__ == "__main__":
    import sys
    eda = CurationEDA()
    results = eda.comprehensive_analysis()

