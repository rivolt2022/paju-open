"""
강화된 탐색적 데이터 분석 (EDA) 스크립트
- 상세한 패턴 분석
- 상관관계 분석
- 시계열 분석
- 이상치 탐지
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 없이 사용
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# 프로젝트 루트 경로
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "src" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 한글 폰트 설정 (matplotlib)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


class EnhancedEDA:
    """강화된 EDA 클래스"""
    
    def __init__(self):
        self.analysis_results = {}
        self.insights = []
    
    def load_data(self, file_path: Path, sheet_name: str) -> pd.DataFrame:
        """엑셀 파일에서 데이터 로드"""
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            return df
        except Exception as e:
            print(f"경고: {sheet_name} 읽기 실패: {e}")
            return pd.DataFrame()
    
    def analyze_correlation(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict:
        """상관관계 분석"""
        if len(numeric_cols) < 2:
            return {}
        
        corr_matrix = df[numeric_cols].corr()
        
        # 강한 상관관계 찾기 (절댓값 > 0.7)
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if abs(val) > 0.7 and not np.isnan(val):
                    strong_corr.append({
                        'col1': corr_matrix.columns[i],
                        'col2': corr_matrix.columns[j],
                        'correlation': float(val)
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'strong_correlations': strong_corr
        }
    
    def detect_outliers(self, df: pd.DataFrame, col: str) -> Dict:
        """이상치 탐지 (IQR 방법)"""
        if col not in df.columns:
            return {}
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        return {
            'outlier_count': len(outliers),
            'outlier_percentage': len(outliers) / len(df) * 100,
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'Q1': float(Q1),
            'Q3': float(Q3),
            'median': float(df[col].median())
        }
    
    def analyze_time_series_patterns(self, df: pd.DataFrame, 
                                    date_col: str, value_col: str) -> Dict:
        """시계열 패턴 분석"""
        if date_col not in df.columns or value_col not in df.columns:
            return {}
        
        # 날짜 파싱
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col, value_col])
        
        if len(df) == 0:
            return {}
        
        # 시계열 분석
        patterns = {
            'trend': 'stable',  # 증가/감소/안정
            'seasonality': False,
            'cyclical': False,
            'mean': float(df[value_col].mean()),
            'std': float(df[value_col].std()),
            'min': float(df[value_col].min()),
            'max': float(df[value_col].max()),
            'range': float(df[value_col].max() - df[value_col].min())
        }
        
        # 월별 패턴 확인 (계절성)
        df['month'] = df[date_col].dt.month
        monthly_means = df.groupby('month')[value_col].mean()
        if monthly_means.std() / monthly_means.mean() > 0.1:
            patterns['seasonality'] = True
        
        # 트렌드 분석 (선형 추세)
        df_sorted = df.sort_values(date_col)
        if len(df_sorted) > 1:
            slope = np.polyfit(range(len(df_sorted)), df_sorted[value_col].values, 1)[0]
            if abs(slope) > patterns['mean'] * 0.01:
                patterns['trend'] = 'increasing' if slope > 0 else 'decreasing'
        
        return patterns
    
    def analyze_feature_importance_candidates(self, df: pd.DataFrame, 
                                              target_col: str) -> List[Dict]:
        """특징 중요도 후보 분석 (상관관계 기반)"""
        if target_col not in df.columns:
            return []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        feature_scores = []
        for col in numeric_cols:
            try:
                corr = df[[col, target_col]].corr().iloc[0, 1]
                if not np.isnan(corr):
                    feature_scores.append({
                        'feature': col,
                        'correlation_with_target': float(abs(corr)),
                        'direction': 'positive' if corr > 0 else 'negative'
                    })
            except:
                continue
        
        # 상관관계가 높은 순으로 정렬
        feature_scores.sort(key=lambda x: x['correlation_with_target'], reverse=True)
        
        return feature_scores[:20]  # 상위 20개
    
    def generate_insights(self, analysis_results: Dict) -> List[str]:
        """분석 결과를 바탕으로 인사이트 생성"""
        insights = []
        
        # 데이터 품질 인사이트
        total_rows = sum(v.get('전체통계', {}).get('총행수', 0) 
                        for v in analysis_results.values())
        if total_rows > 9000:
            insights.append(f"📊 풍부한 데이터: 총 {total_rows:,}행의 데이터로 모델 학습 가능")
        
        # 결측치 인사이트
        missing_pct = 0
        for file_analysis in analysis_results.values():
            for sheet in file_analysis.get('시트목록', []):
                missing_data = sheet.get('결측률', {})
                if missing_data:
                    avg_missing = sum(v for v in missing_data.values() if isinstance(v, (int, float))) / len(missing_data)
                    missing_pct = max(missing_pct, avg_missing)
        
        if missing_pct < 5:
            insights.append("✅ 데이터 품질 우수: 결측치가 5% 미만으로 모델 학습에 적합")
        elif missing_pct < 10:
            insights.append("⚠️ 데이터 품질 양호: 결측치가 10% 미만으로 전처리 필요")
        else:
            insights.append("❌ 데이터 품질 개선 필요: 결측치가 10% 이상으로 전처리 필수")
        
        # 시계열 패턴 인사이트
        time_patterns = []
        for file_analysis in analysis_results.values():
            for sheet in file_analysis.get('시트목록', []):
                if '시계열_패턴' in sheet:
                    patterns = sheet['시계열_패턴']
                    if patterns.get('seasonality'):
                        time_patterns.append("계절성 패턴 발견")
                    if patterns.get('trend') != 'stable':
                        time_patterns.append(f"{patterns['trend']} 트렌드 발견")
        
        if time_patterns:
            insights.extend(time_patterns)
        
        return insights
    
    def comprehensive_analysis(self) -> Dict:
        """종합 분석 수행"""
        print("\n" + "="*60)
        print("강화된 탐색적 데이터 분석 (Enhanced EDA) 시작")
        print("="*60)
        
        # sheets.json 로드
        sheets_json_path = DATA_DIR / "sheets.json"
        if not sheets_json_path.exists():
            print("경고: sheets.json 파일을 찾을 수 없습니다.")
            return {}
        
        with open(sheets_json_path, 'r', encoding='utf-8') as f:
            sheets_info = json.load(f)
        
        all_analysis = {}
        excel_files = [
            "1_생활인구분석_LG유플러스.xlsx",
            "1_생활인구분석_경기데이터드림.xlsx",
            "3_관광지분석_LG유플러스.xlsx",
            "3_관광지분석_삼성카드.xlsx",
            "4_관광지트렌드분석_LG유플러스.xlsx",
            "4_관광지트렌드분석_삼성카드.xlsx",
            "5_식품위생업소소비분석_경기데이터드림.xlsx",
            "5_식품위생업소소비분석_삼성카드.xlsx",
            "6_소비경제규모추정.xlsx",
        ]
        
        for excel_file in excel_files:
            file_path = DATA_DIR / excel_file
            if not file_path.exists():
                continue
            
            print(f"\n분석 중: {excel_file}")
            
            file_analysis = {
                "파일명": excel_file,
                "시트목록": []
            }
            
            # 파일의 모든 시트 읽기
            excel_file_obj = pd.ExcelFile(file_path)
            for sheet_name in excel_file_obj.sheet_names:
                df = self.load_data(file_path, sheet_name)
                
                if df.empty:
                    continue
                
                print(f"  - {sheet_name}: {len(df)}행, {len(df.columns)}열")
                
                sheet_analysis = {
                    "시트명": sheet_name,
                    "행수": len(df),
                    "열수": len(df.columns),
                    "열명": df.columns.tolist(),
                    "결측률": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
                }
                
                # 숫자형 컬럼 분석
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    sheet_analysis["숫자형통계"] = df[numeric_cols].describe().to_dict()
                    
                    # 상관관계 분석
                    if len(numeric_cols) > 1:
                        corr_analysis = self.analyze_correlation(df, numeric_cols)
                        if corr_analysis:
                            sheet_analysis["상관관계"] = corr_analysis
                    
                    # 주요 컬럼에 대한 이상치 탐지
                    for col in numeric_cols[:3]:  # 처음 3개만
                        outlier_info = self.detect_outliers(df, col)
                        if outlier_info:
                            sheet_analysis.setdefault("이상치", {})[col] = outlier_info
                    
                    # 시계열 분석 (날짜 컬럼이 있는 경우)
                    date_cols = [col for col in df.columns 
                               if '날짜' in str(col) or 'date' in str(col).lower() 
                               or '월' in str(col) or '연도' in str(col)]
                    if date_cols and len(numeric_cols) > 0:
                        date_col = date_cols[0]
                        value_col = numeric_cols[0]
                        time_series = self.analyze_time_series_patterns(df, date_col, value_col)
                        if time_series:
                            sheet_analysis["시계열_패턴"] = time_series
                        
                        # 특징 중요도 후보
                        feature_candidates = self.analyze_feature_importance_candidates(df, value_col)
                        if feature_candidates:
                            sheet_analysis["특징_중요도_후보"] = feature_candidates[:10]
                
                file_analysis["시트목록"].append(sheet_analysis)
            
            all_analysis[excel_file] = file_analysis
        
        # 종합 인사이트 생성
        insights = self.generate_insights(all_analysis)
        
        result = {
            "분석일시": datetime.now().isoformat(),
            "파일별_분석": all_analysis,
            "종합_인사이트": insights
        }
        
        # 결과 저장
        output_file = OUTPUT_DIR / "enhanced_eda_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print(f"분석 완료! 결과 저장: {output_file}")
        print(f"{'='*60}")
        
        # 인사이트 출력
        print("\n📊 주요 인사이트:")
        for insight in insights:
            print(f"  {insight}")
        
        return result


if __name__ == "__main__":
    eda = EnhancedEDA()
    results = eda.comprehensive_analysis()
