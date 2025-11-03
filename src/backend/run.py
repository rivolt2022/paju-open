"""
Backend 서버 실행 스크립트 (프로젝트 루트에서 실행)
이 스크립트는 프로젝트 루트에서 실행해야 합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 찾기
# run.py가 src/backend/run.py에 있으므로 프로젝트 루트는 2단계 위
backend_dir = Path(__file__).parent.resolve()
project_root = backend_dir.parent.parent.resolve()

# 프로젝트 루트를 Python 경로에 추가
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 현재 작업 디렉토리를 프로젝트 루트로 변경
os.chdir(project_root)

print(f"프로젝트 루트: {project_root}")
print(f"작업 디렉토리: {os.getcwd()}")

# main 모듈 import
from src.backend.main import app

if __name__ == "__main__":
    import uvicorn
    print(f"\n[Backend] 서버 시작...")
    print(f"[Backend] 접속 URL: http://localhost:8000")
    print(f"[Backend] API 문서: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)