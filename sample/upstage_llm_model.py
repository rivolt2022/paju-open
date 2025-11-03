"""
업스테이지 Solar Pro2 모델 클래스

sample.py의 Solar Pro2 구현 부분을 참조하여 작성되었습니다.
"""

import requests
import time
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI


class UpstageModel:
    """업스테이지 Solar Pro2 모델 클래스 - chat 버전"""
    
    def __init__(self, model: str = "solar-pro2", 
                 api_base: str = "https://api.upstage.ai/v1",
                 max_tokens: int = 200, 
                 temperature: float = 0.2,
                 api_key: str = "up_mdTJEYyYmDaWpWovoACcamECy3aTM",
                 top_p: float = 0.1,
                 frequency_penalty: float = 0,
                 presence_penalty: float = 0,
                 reasoning_effort: str = "high"):
        """
        업스테이지 Solar Pro2 모델 초기화
        
        Args:
            model (str): 모델명 (기본값: "solar-pro2")
            api_base (str): API 베이스 URL (기본값: "https://api.upstage.ai/v1")
            max_tokens (int): 최대 토큰 수 (기본값: 200)
            temperature (float): 온도 설정 (기본값: 0.2)
            api_key (str): API 키
            top_p (float): nucleus sampling 파라미터 (기본값: 0.1)
            frequency_penalty (float): 빈도 페널티 (기본값: 0)
            presence_penalty (float): 존재 페널티 (기본값: 0)
            reasoning_effort (str): 추론 노력 수준 (기본값: "high")
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=30.0  # LLM API 응답을 위해 30초로 설정
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.reasoning_effort = reasoning_effort
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, prompt: str, response_format: str = None, reasoning_effort: str = "high") -> List[str]:
        """
        모델 호출 - chat 방식
        
        Args:
            prompt (str): 입력 프롬프트
            response_format (str): 응답 형식 ("json_object" 또는 "json_schema")
            reasoning_effort (str): 추론 노력 수준 ("low" 또는 "high")
            
        Returns:
            List[str]: 모델 응답 리스트
        """
        try:
            # API 파라미터 구성
            params = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "reasoning_effort": reasoning_effort
            }
            
            # 응답 형식 설정
            if response_format:
                if response_format == "json_object":
                    params["response_format"] = {"type": "json_object"}
                elif response_format == "json_schema":
                    params["response_format"] = {"type": "json_schema"}
            
            response = self.client.chat.completions.create(**params)
            
            if response and response.choices:
                return [response.choices[0].message.content]
            else:
                self.logger.warning("모델 응답이 비어있습니다.")
                return ["답변 없음"]
                
        except Exception as e:
            self.logger.error(f"모델 호출 오류: {e}")
            return ["오류 발생"]
    
    def generate_response(self, prompt: str, system_message: str = None, 
                         response_format: str = None, reasoning_effort: str = "low",
                         top_p: float = 0.1, frequency_penalty: float = 0, 
                         presence_penalty: float = 0) -> str:
        """
        시스템 메시지와 함께 응답 생성
        
        Args:
            prompt (str): 사용자 프롬프트
            system_message (str): 시스템 메시지 (선택사항)
            response_format (str): 응답 형식 ("json_object" 또는 "json_schema")
            reasoning_effort (str): 추론 노력 수준 ("low" 또는 "high")
            top_p (float): nucleus sampling 파라미터 (0.0~1.0)
            frequency_penalty (float): 빈도 페널티 (-2.0~2.0)
            presence_penalty (float): 존재 페널티 (-2.0~2.0)
            
        Returns:
            str: 모델 응답
        """
        try:
            messages = []
            
            if system_message:
                messages.append({
                    "role": "system",
                    "content": system_message
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # API 파라미터 구성
            params = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "reasoning_effort": reasoning_effort,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty
            }
            
            # 응답 형식 설정
            if response_format:
                if response_format == "json_object":
                    params["response_format"] = {"type": "json_object"}
                elif response_format == "json_schema":
                    # JSON 스키마가 제공된 경우 여기에 추가할 수 있음
                    params["response_format"] = {"type": "json_schema"}
            
            response = self.client.chat.completions.create(**params)
            
            if response and response.choices:
                return response.choices[0].message.content
            else:
                return "답변 없음"
                
        except Exception as e:
            self.logger.error(f"응답 생성 오류: {e}")
            return "오류 발생"
    
    def set_parameters(self, max_tokens: int = None, temperature: float = None, 
                      top_p: float = None, frequency_penalty: float = None, 
                      presence_penalty: float = None, reasoning_effort: str = None):
        """
        모델 파라미터 설정
        
        Args:
            max_tokens (int): 최대 토큰 수
            temperature (float): 온도 설정
            top_p (float): nucleus sampling 파라미터 (0.0~1.0)
            frequency_penalty (float): 빈도 페널티 (-2.0~2.0)
            presence_penalty (float): 존재 페널티 (-2.0~2.0)
            reasoning_effort (str): 추론 노력 수준 ("low" 또는 "high")
        """
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        if frequency_penalty is not None:
            self.frequency_penalty = frequency_penalty
        if presence_penalty is not None:
            self.presence_penalty = presence_penalty
        if reasoning_effort is not None:
            self.reasoning_effort = reasoning_effort
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보 반환
        
        Returns:
            Dict[str, Any]: 모델 정보
        """
        return {
            "model": self.model,
            "api_base": self.client.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "reasoning_effort": self.reasoning_effort
        }


class UpstageModelConfig:
    """업스테이지 모델 설정 클래스"""
    
    # 기본 설정
    DEFAULT_MODEL = "solar-pro2"
    DEFAULT_API_BASE = "https://api.upstage.ai/v1"
    DEFAULT_API_KEY = "up_mdTJEYyYmDaWpWovoACcamECy3aTM"
    
    # 토큰 설정 (제한 없음)
    SHORT_RESPONSE = 1000
    MEDIUM_RESPONSE = 4000
    LONG_RESPONSE = 8000
    UNLIMITED_RESPONSE = 32000  # 최대 토큰 제한
    
    # 온도 설정
    CREATIVE = 0.7
    BALANCED = 0.3
    PRECISE = 0.1
    
    @classmethod
    def create_model(cls, 
                    model: str = None,
                    max_tokens: int = MEDIUM_RESPONSE,
                    temperature: float = BALANCED,
                    api_key: str = None,
                    top_p: float = 0.1,
                    frequency_penalty: float = 0,
                    presence_penalty: float = 0,
                    reasoning_effort: str = "high") -> UpstageModel:
        """
        설정에 따른 모델 생성
        
        Args:
            model (str): 모델명
            max_tokens (int): 최대 토큰 수
            temperature (float): 온도 설정
            api_key (str): API 키
            top_p (float): nucleus sampling 파라미터
            frequency_penalty (float): 빈도 페널티
            presence_penalty (float): 존재 페널티
            reasoning_effort (str): 추론 노력 수준
            
        Returns:
            UpstageModel: 생성된 모델 인스턴스
        """
        return UpstageModel(
            model=model or cls.DEFAULT_MODEL,
            api_base=cls.DEFAULT_API_BASE,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key or cls.DEFAULT_API_KEY,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            reasoning_effort=reasoning_effort
        )
    
    @classmethod
    def create_precise_model(cls, api_key: str = None) -> UpstageModel:
        """정확한 답변을 위한 모델 생성"""
        return cls.create_model(
            max_tokens=cls.UNLIMITED_RESPONSE,  # 제한 없음
            temperature=cls.PRECISE,
            api_key=api_key
        )
    
    @classmethod
    def create_creative_model(cls, api_key: str = None) -> UpstageModel:
        """창의적인 답변을 위한 모델 생성"""
        return cls.create_model(
            max_tokens=cls.UNLIMITED_RESPONSE,  # 제한 없음
            temperature=cls.CREATIVE,
            api_key=api_key
        )


# 사용 예시
if __name__ == "__main__":
    # 기본 모델 생성
    model = UpstageModel()
    
    # 테스트 프롬프트
    test_prompt = "안녕하세요! 간단한 인사말을 해주세요."
    
    # 응답 생성
    response = model.generate_response(test_prompt)
    print(f"응답: {response}")
    
    # 모델 정보 출력
    info = model.get_model_info()
    print(f"모델 정보: {info}") 