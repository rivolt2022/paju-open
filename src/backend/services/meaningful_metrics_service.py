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
    
    def __init__(self, predictor=None, content_generator=None):
        """
        Args:
            predictor: ML 예측 모델 인스턴스
            content_generator: LLM 콘텐츠 생성기 인스턴스
        """
        self.calculator = MeaningfulMetricsCalculator()
        self.calculator.load_all_data()
        self.predictor = predictor
        self.content_generator = content_generator
    
    def set_predictor(self, predictor):
        """예측 모델 설정"""
        self.predictor = predictor
    
    def set_content_generator(self, content_generator):
        """LLM 생성기 설정"""
        self.content_generator = content_generator
    
    def get_comprehensive_metrics(self, space_name: str = "헤이리예술마을", date: str = None) -> Dict:
        """종합 지표 반환 (날짜 파라미터 지원 - ML 예측 사용)"""
        if date and self.predictor:
            # 날짜가 주어지면 ML 모델로 예측 데이터 생성
            return self.calculator.get_comprehensive_metrics_with_prediction(
                space_name, date, self.predictor, self.content_generator
            )
        else:
            # 기본 계산 (기존 방식)
            return self.calculator.get_comprehensive_metrics(space_name)
    
    def get_activation_scores(self, space_name: str, date: str = None) -> Dict:
        """활성화 점수 반환 (날짜 파라미터 지원 - ML 예측 사용)"""
        if date and self.predictor:
            # 날짜가 주어지면 ML 모델로 예측 데이터 생성
            try:
                # 방문인구 예측
                visit_prediction = None
                if hasattr(self.predictor, 'predict_cultural_space_visits'):
                    visit_results = self.predictor.predict_cultural_space_visits([space_name], date, "afternoon")
                    if visit_results:
                        visit_prediction = visit_results[0]
                
                # 큐레이션 지표 예측
                curation_metrics = None
                if hasattr(self.predictor, 'predict_curation_metrics'):
                    curation_metrics = self.predictor.predict_curation_metrics(space_name, date)
                
                return self.calculator.calculate_cultural_space_activation_score_with_prediction(
                    space_name, date, visit_prediction, curation_metrics, self.predictor, self.content_generator
                )
            except Exception as e:
                print(f"[서비스] ML 예측 오류: {e}")
                # 기본 계산으로 폴백
                return self.calculator.calculate_cultural_space_activation_score(space_name)
        else:
            # 기본 계산 (기존 방식)
            return self.calculator.calculate_cultural_space_activation_score(space_name)
    
    def get_optimal_time_analysis(self, space_name: str) -> Dict:
        """최적 시간 분석 반환"""
        return self.calculator.calculate_optimal_time_analysis(space_name)
    
    def get_demographic_targeting(self, space_name: str) -> Dict:
        """타겟팅 분석 반환"""
        return self.calculator.calculate_demographic_targeting_score(space_name)
    
    def get_publishing_complex_vitality(self, date: str = None) -> Dict:
        """출판단지 활성화 지수 반환 (날짜 파라미터 지원 - 동적 계산 + LLM 평가)"""
        if date:
            # 날짜가 주어지면 ML 예측 데이터 생성
            visit_prediction = None
            activation_scores = None
            
            try:
                if self.predictor:
                    # 방문인구 예측
                    if hasattr(self.predictor, 'predict_cultural_space_visits'):
                        visit_results = self.predictor.predict_cultural_space_visits(['헤이리예술마을'], date, "afternoon")
                        if visit_results:
                            visit_prediction = visit_results[0]
                    
                    # 활성화 점수 계산
                    activation_scores = self.get_activation_scores('헤이리예술마을', date)
            except Exception as e:
                print(f"[서비스] 활성화 지수 계산을 위한 예측 데이터 생성 오류: {e}")
            
            if self.content_generator:
                # LLM 평가 포함
                return self.calculator._calculate_publishing_vitality_with_llm(
                    date, self.predictor, self.content_generator, visit_prediction, activation_scores
                )
            else:
                # LLM 평가 없이 날짜별 계산만
                return self.calculator.calculate_publishing_complex_vitality_index(
                    date=date, visit_prediction=visit_prediction, activation_scores=activation_scores
                )
        else:
            # 기본 계산 (날짜 없음)
            return self.calculator.calculate_publishing_complex_vitality_index()


# 싱글톤 인스턴스
_metrics_service = None

def get_meaningful_metrics_service(predictor=None, content_generator=None) -> MeaningfulMetricsService:
    """의미 있는 지표 서비스 인스턴스 반환"""
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = MeaningfulMetricsService(predictor, content_generator)
    else:
        # 이미 생성된 경우 predictor와 content_generator 설정
        if predictor:
            _metrics_service.set_predictor(predictor)
        if content_generator:
            _metrics_service.set_content_generator(content_generator)
    return _metrics_service

