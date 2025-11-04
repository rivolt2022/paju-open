"""
유의미한 ML 지표 서비스
출판단지 활성화에 도움이 되는 실제 지표들을 계산하여 반환
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ml.analytics.meaningful_metrics import MeaningfulMetricsCalculator


class MeaningfulMetricsService:
    """유의미한 지표 서비스 클래스"""
    
    def __init__(self):
        self.calculator = MeaningfulMetricsCalculator()
        self.calculator.load_all_data()
    
    def get_comprehensive_metrics(self, space_name: str = "헤이리예술마을", date: str = None) -> Dict:
        """종합 지표 반환 (날짜 파라미터 지원)"""
        # date 파라미터는 현재 구현에서는 사용하지 않지만, 향후 날짜별 계산에 활용 가능
        return self.calculator.get_comprehensive_metrics(space_name)
    
    def get_activation_scores(self, space_name: str, date: str = None) -> Dict:
        """활성화 점수 반환 (날짜 파라미터 지원)"""
        # date 파라미터는 현재 구현에서는 사용하지 않지만, 향후 날짜별 계산에 활용 가능
        return self.calculator.calculate_cultural_space_activation_score(space_name)
    
    def get_optimal_time_analysis(self, space_name: str) -> Dict:
        """최적 시간 분석 반환"""
        return self.calculator.calculate_optimal_time_analysis(space_name)
    
    def get_demographic_targeting(self, space_name: str) -> Dict:
        """타겟팅 분석 반환"""
        return self.calculator.calculate_demographic_targeting_score(space_name)
    
    def get_publishing_complex_vitality(self, date: str = None) -> Dict:
        """출판단지 활성화 지수 반환 (날짜 파라미터 지원)"""
        # date 파라미터는 현재 구현에서는 사용하지 않지만, 향후 날짜별 계산에 활용 가능
        return self.calculator.calculate_publishing_complex_vitality_index()


# 싱글톤 인스턴스
_metrics_service = None

def get_meaningful_metrics_service() -> MeaningfulMetricsService:
    """의미 있는 지표 서비스 인스턴스 반환"""
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = MeaningfulMetricsService()
    return _metrics_service

