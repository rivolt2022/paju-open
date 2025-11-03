"""
업스테이지 LLM 통합 - 생성형 AI 콘텐츠 생성
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import json

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sample.upstage_llm_model import UpstageModel


class ContentGenerator:
    """생성형 AI 콘텐츠 생성 클래스"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: 업스테이지 API 키 (None이면 기본값 사용)
        """
        if api_key:
            self.llm = UpstageModel(api_key=api_key, max_tokens=2000)
        else:
            self.llm = UpstageModel(max_tokens=2000)
    
    def generate_journey(self, user_info: Dict, predictions: List[Dict], 
                        date: str) -> Dict:
        """
        개인화 문화 여정 생성
        
        Args:
            user_info: 사용자 정보 (age, gender, preferences, available_time)
            predictions: 예측 결과 리스트
            date: 날짜 (YYYY-MM-DD)
            
        Returns:
            생성된 문화 여정 딕셔너리
        """
        # 프롬프트 생성
        prompt = self._create_journey_prompt(user_info, predictions, date)
        
        try:
            # LLM 호출
            response = self.llm.generate_response(prompt)
            
            # 응답 파싱 (JSON 형식으로 요청)
            journey_data = self._parse_response(response)
            
            return journey_data
        except Exception as e:
            print(f"생성형 AI 호출 실패: {e}")
            # 기본값 반환
            return self._default_journey(user_info, predictions)
    
    def generate_story(self, user_info: Dict, predictions: List[Dict], 
                      date: str) -> Dict:
        """generate_journey의 별칭 (호환성)"""
        return self.generate_journey(user_info, predictions, date)
    
    def generate_course(self, user_info: Dict, predictions: List[Dict], 
                       date: str) -> Dict:
        """맞춤형 코스 생성 (generate_journey의 별칭)"""
        return self.generate_journey(user_info, predictions, date)
    
    def _create_journey_prompt(self, user_info: Dict, predictions: List[Dict], 
                              date: str) -> str:
        """문화 여정 생성 프롬프트 생성"""
        
        age = user_info.get('age', 30)
        gender = user_info.get('gender', 'female')
        preferences = ', '.join(user_info.get('preferences', ['문학', '예술']))
        available_time = user_info.get('available_time', 'afternoon')
        if isinstance(available_time, str) and '_' in available_time:
            available_time = available_time.split('_')[0]
        
        # 예측 결과 요약
        predictions_summary = "\n".join([
            f"- {p.get('space', p.get('spot', 'N/A'))}: 예상 방문 {p.get('predicted_visit', 0):,}명, 혼잡도 {p.get('crowd_level', 0):.2f}, 최적 시간: {p.get('optimal_time', 'N/A')}"
            for p in predictions
        ])
        
        # 최적 문화 공간 추천
        best_spaces = sorted(predictions, key=lambda x: x.get('crowd_level', 1))[:3]
        recommended_spaces = ', '.join([s.get('space', s.get('spot', 'N/A')) for s in best_spaces])
        
        # 추천 프로그램 정보
        programs_info = "\n".join([
            f"- {p.get('space', p.get('spot', 'N/A'))}: {', '.join(p.get('recommended_programs', ['문화 프로그램']))}"
            for p in predictions if p.get('recommended_programs')
        ])
        
        prompt = f"""당신은 파주시 출판단지 문화 콘텐츠 큐레이터입니다.

사용자 정보:
- 연령: {age}세
- 성별: {gender}
- 선호 활동: {preferences}
- 이용 가능 시간: {available_time}

예측된 문화 공간 정보 ({date}):
{predictions_summary}

추천 문화 공간: {recommended_spaces}

가능한 문화 프로그램:
{programs_info}

다음 형식으로 개인화된 문화 여정을 생성해주세요:

제목: 매력적인 문화 여정 제목 (30자 이내)
설명: 여정에 대한 간단한 설명 (100자 이내)

상세 여정:
- 시간대별로 문화 공간과 프로그램 추천
- 각 시간대마다:
  1. 문화 공간명
  2. 구체적인 프로그램명 (작가와의 만남, 북토크, 출판사 투어, 전시 등)
  3. 생활인구 패턴을 고려한 방문 이유
  4. 사용자 선호 활동과의 연계성
  5. 실제 참여 시 유용한 팁

스토리: 200자 이내의 매력적인 문화 여정 스토리텔링

JSON 형식으로 응답해주세요:
{{
  "title": "제목",
  "description": "설명",
  "journey": [
    {{
      "time": "14:00-16:00",
      "place": "문화 공간명",
      "program": "프로그램명",
      "reason": "방문 이유 (생활인구 패턴, 취향 분석 등을 고려)",
      "tip": "팁"
    }}
  ],
  "story": "스토리"
}}
"""
        
        return prompt
    
    def _parse_response(self, response: str) -> Dict:
        """LLM 응답 파싱"""
        try:
            # JSON 형식으로 응답 받기
            # 응답에서 JSON 부분만 추출
            if '{' in response and '}' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                # journey 또는 course 키 모두 처리
                if 'journey' in parsed:
                    return parsed
                elif 'course' in parsed:
                    return {
                        'title': parsed.get('title', ''),
                        'description': parsed.get('description', ''),
                        'journey': parsed.get('course', []),
                        'story': parsed.get('story', '')
                    }
                return parsed
        except Exception as e:
            print(f"JSON 파싱 실패: {e}")
        
        # 파싱 실패 시 기본값 반환
        return {
            'title': '파주 문화 여정',
            'description': '개인화된 문화 여정을 추천합니다.',
            'journey': [
                {
                    'time': '14:00-16:00',
                    'place': '헤이리예술마을',
                    'program': '작가와의 만남',
                    'reason': '오후 시간대는 비교적 한산하여 작가와의 만남에 집중할 수 있습니다.',
                    'tip': '사전 예약을 권장합니다.'
                }
            ],
            'story': '파주의 문화와 예술을 만나보세요.'
        }
    
    def _default_journey(self, user_info: Dict, predictions: List[Dict]) -> Dict:
        """기본 문화 여정 (LLM 호출 실패 시)"""
        if not predictions:
            return {
                'title': f"{user_info.get('age', 30)}세를 위한 파주 문화 여정",
                'description': '개인화된 문화 여정을 추천합니다.',
                'journey': [],
                'story': '파주시의 문화를 만나는 특별한 하루...'
            }
        
        best_space = min(predictions, key=lambda x: x.get('crowd_level', 1))
        space_name = best_space.get('space', best_space.get('spot', '문화 공간'))
        optimal_time = best_space.get('optimal_time', '14:00-17:00')
        programs = best_space.get('recommended_programs', ['문화 프로그램'])
        
        return {
            'title': f"{user_info.get('age', 30)}세를 위한 파주 문화 여정",
            'description': f"{space_name} 중심의 여유로운 문화 여정을 추천합니다.",
            'journey': [
                {
                    'time': optimal_time,
                    'place': space_name,
                    'program': programs[0] if programs else '문화 프로그램',
                    'reason': f"예상 혼잡도가 낮아({best_space.get('crowd_level', 0.5):.2f}) 편안하게 참여할 수 있습니다. 당신의 {', '.join(user_info.get('preferences', ['문화']))} 취향과 일치합니다.",
                    'tip': f'{optimal_time} 시간대 방문을 권장합니다.'
                }
            ],
            'story': '파주시의 출판단지와 문화 공간을 탐방하며 특별한 하루를 보내세요.'
        }
    
    def analyze_data(self, prompt: str, return_type: str = 'dict') -> Dict | str:
        """
        LLM 기반 데이터 분석 (채팅 응답 지원)
        
        Args:
            prompt: 분석 프롬프트
            return_type: 반환 타입 ('dict' 또는 'string')
            
        Returns:
            분석 결과 딕셔너리 또는 문자열
        """
        try:
            # LLM 호출
            response = self.llm.generate_response(prompt)
            
            # 문자열 반환 요청인 경우
            if return_type == 'string':
                # JSON이 아닌 순수 텍스트 반환
                # JSON 블록 제거
                if '{' in response and '}' in response:
                    # JSON 부분 찾기
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    
                    # JSON 앞부분 텍스트가 있으면 사용
                    before_json = response[:json_start].strip()
                    after_json = response[json_end:].strip()
                    
                    # JSON 파싱 시도 (인사이트 추출)
                    try:
                        json_str = response[json_start:json_end]
                        parsed = json.loads(json_str)
                        
                        # JSON에서 텍스트 추출
                        if 'response' in parsed:
                            return parsed['response']
                        elif 'answer' in parsed:
                            return parsed['answer']
                        elif 'insights' in parsed and parsed['insights']:
                            # 인사이트를 자연스러운 텍스트로 변환
                            return '\n\n'.join(parsed['insights'])
                        elif 'recommendations' in parsed and parsed['recommendations']:
                            return '\n\n'.join(parsed['recommendations'])
                    except:
                        pass
                    
                    # JSON 앞뒤 텍스트 결합
                    combined = before_json + '\n\n' + after_json if before_json or after_json else response
                    return combined if combined.strip() else response
                
                # JSON이 없는 경우 그대로 반환
                return response
            
            # 딕셔너리 반환 (기본 동작)
            analysis_data = self._parse_analysis_response(response)
            return analysis_data
        except Exception as e:
            print(f"LLM 분석 호출 실패: {e}")
            # 기본값 반환
            if return_type == 'string':
                return "죄송합니다. 데이터 분석 중 오류가 발생했습니다. 다시 시도해주세요."
            return {
                "insights": ["데이터 분석 중 오류가 발생했습니다."],
                "recommendations": ["데이터를 확인해주세요."],
                "trends": ["분석을 다시 시도해주세요."]
            }
    
    def _parse_analysis_response(self, response: str) -> Dict:
        """분석 응답 파싱"""
        try:
            # JSON 형식으로 응답 받기
            if '{' in response and '}' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                # 다양한 응답 형식 지원
                # 1. explain-metric 형식 (explanation, importance, interpretation, recommendation)
                if 'explanation' in parsed or 'importance' in parsed or 'interpretation' in parsed:
                    return parsed
                
                # 2. chart-insight 형식 (pattern, trend, insight, recommendation)
                if 'pattern' in parsed or 'trend' in parsed or 'insight' in parsed:
                    return parsed
                
                # 3. 기존 형식 (insights, recommendations, trends)
                if 'insights' in parsed and 'recommendations' in parsed and 'trends' in parsed:
                    return parsed
                
                # 4. summary, insights, recommendations 형식
                if 'summary' in parsed or 'insights' in parsed or 'recommendations' in parsed:
                    return parsed
                
                # 5. 그 외 모든 키가 있는 경우도 반환
                if parsed:
                    return parsed
        except Exception as e:
            print(f"JSON 파싱 실패: {e}")
            print(f"응답 원문: {response[:500]}")
        
        # 파싱 실패 시 기본값 - 하지만 빈 객체를 반환하여 호출자가 처리하도록 함
        return {}
    
    def generate_tips(self, space: str, predictions: List[Dict]) -> str:
        """문화 공간별 팁 생성"""
        space_pred = next((p for p in predictions if p.get('space') == space or p.get('spot') == space), None)
        
        if space_pred:
            crowd_level = space_pred.get('crowd_level', 0.5)
            optimal_time = space_pred.get('optimal_time', 'N/A')
            programs = space_pred.get('recommended_programs', [])
            
            if crowd_level > 0.7:
                return f"{space}은 혼잡할 예정입니다. {optimal_time} 시간대나 평일 방문을 권장합니다. 추천 프로그램: {', '.join(programs[:2]) if programs else '문화 프로그램'}"
            else:
                return f"{space}은 비교적 한산할 예정입니다. {optimal_time} 시간대에 {', '.join(programs[:2]) if programs else '문화 프로그램'} 참여를 권장합니다."
        
        return f"{space} 방문 시 날씨를 확인하고 편안한 복장을 권장합니다."


if __name__ == "__main__":
    # 테스트
    generator = ContentGenerator()
    
    user_info = {
        'age': 30,
        'gender': 'female',
        'preferences': ['문학', '예술', '독서']
    }
    
    predictions = [
        {'space': '헤이리예술마을', 'predicted_visit': 42000, 'crowd_level': 0.68, 'optimal_time': '14:00-17:00', 'recommended_programs': ['작가와의 만남', '갤러리 전시']},
        {'space': '파주출판단지', 'predicted_visit': 28000, 'crowd_level': 0.45, 'optimal_time': '14:00-16:00', 'recommended_programs': ['출판사 투어', '책 만남의 날']},
        {'space': '교하도서관', 'predicted_visit': 15000, 'crowd_level': 0.28, 'optimal_time': '10:00-12:00', 'recommended_programs': ['독서 모임', '북토크']},
    ]
    
    date = '2025-01-18'
    
    result = generator.generate_journey(user_info, predictions, date)
    
    print("생성된 문화 여정:")
    print(json.dumps(result, indent=2, ensure_ascii=False))