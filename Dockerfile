# Python 3.11 슬림 이미지 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 도구 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 파일 복사
COPY requirements.txt ./
COPY src/backend/requirements.txt ./src/backend/requirements.txt

# Python 패키지 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r src/backend/requirements.txt && \
    pip install --no-cache-dir uvicorn[standard]

# 프로젝트 파일 복사
COPY . .

# 데이터 디렉토리와 모델 디렉토리 확인
RUN ls -la data/ && \
    ls -la src/ml/models/saved/ || echo "모델 파일이 없습니다. 학습 후 배포하세요."

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 포트 노출
EXPOSE 8000

# Health check 설정
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# 애플리케이션 실행
CMD ["python", "-m", "uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

