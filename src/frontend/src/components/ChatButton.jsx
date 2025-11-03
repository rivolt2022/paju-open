import { MdAutoAwesome, MdClose } from 'react-icons/md'
import './ChatButton.css'

function ChatButton({ onClick, unreadCount = 0, isOpen = false }) {
  const handleClick = () => {
    if (isOpen) {
      // 모달 닫기
      onClick()
    } else {
      // 모달 열기
      onClick()
    }
  }

  return (
    <button 
      className={`chat-button ${isOpen ? 'chat-button-open' : ''}`}
      onClick={handleClick} 
      title={isOpen ? "AI 분석 어시스턴트 닫기" : "AI 분석 어시스턴트 열기"}
      aria-label={isOpen ? "AI 분석 어시스턴트 닫기" : "AI 분석 어시스턴트 열기"}
    >
      <div className="chat-button-glow"></div>
      <div className="chat-button-pulse"></div>
      <div className="chat-button-icon">
        {isOpen ? <MdClose /> : <MdAutoAwesome />}
      </div>
      {unreadCount > 0 && !isOpen && (
        <span className="chat-button-badge">{unreadCount}</span>
      )}
    </button>
  )
}

export default ChatButton

