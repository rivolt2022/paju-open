import { MdRefresh } from 'react-icons/md'
import './LoadingSpinner.css'

/**
 * 통일된 로딩 스피너 컴포넌트
 * @param {string} message - 로딩 메시지 (선택사항)
 * @param {string} size - 스피너 크기 ('small', 'medium', 'large') 기본값: 'medium'
 * @param {boolean} fullScreen - 전체 화면 오버레이 여부 (기본값: false)
 */
function LoadingSpinner({ message = '데이터를 불러오는 중...', size = 'medium', fullScreen = false }) {
  const spinnerClass = `loading-spinner loading-spinner-${size}`
  const containerClass = fullScreen ? 'loading-container-fullscreen' : 'loading-container'

  return (
    <div className={containerClass}>
      <div className={spinnerClass}>
        <MdRefresh className="spinner-icon" />
      </div>
      {message && (
        <div className="loading-message">{message}</div>
      )}
    </div>
  )
}

export default LoadingSpinner




