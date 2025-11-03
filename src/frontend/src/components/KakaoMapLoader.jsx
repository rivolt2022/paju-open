import { useEffect, useState } from 'react'

/**
 * 카카오맵 SDK 로드 확인 컴포넌트
 */
export function useKakaoLoader() {
  const [loaded, setLoaded] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    // 카카오맵 SDK가 이미 로드되어 있는지 확인
    if (window.kakao && window.kakao.maps) {
      setLoaded(true)
      return
    }

    // SDK 로드 대기
    const checkKakao = setInterval(() => {
      if (window.kakao && window.kakao.maps) {
        setLoaded(true)
        clearInterval(checkKakao)
      }
    }, 100)

    // 10초 후 타임아웃
    const timeout = setTimeout(() => {
      clearInterval(checkKakao)
      if (!window.kakao || !window.kakao.maps) {
        setError('카카오맵 SDK를 로드할 수 없습니다. API 키를 확인하세요.')
        console.error('카카오맵 SDK 로드 실패')
      }
    }, 10000)

    return () => {
      clearInterval(checkKakao)
      clearTimeout(timeout)
    }
  }, [])

  return { loaded, error }
}
