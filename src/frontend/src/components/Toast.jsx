import { useState, useEffect } from 'react'
import { MdCheckCircle, MdError, MdClose } from 'react-icons/md'
import './Toast.css'

function Toast({ message, type = 'success', duration = 3000, onClose }) {
  const [isVisible, setIsVisible] = useState(true)

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(false)
      setTimeout(() => {
        if (onClose) onClose()
      }, 300) // 애니메이션 완료 후 제거
    }, duration)

    return () => clearTimeout(timer)
  }, [duration, onClose])

  const handleClose = () => {
    setIsVisible(false)
    setTimeout(() => {
      if (onClose) onClose()
    }, 300)
  }

  return (
    <div className={`toast toast-${type} ${isVisible ? 'toast-visible' : ''}`}>
      <div className="toast-content">
        {type === 'success' && <MdCheckCircle className="toast-icon" />}
        {type === 'error' && <MdError className="toast-icon" />}
        <span className="toast-message">{message}</span>
      </div>
      <button className="toast-close" onClick={handleClose}>
        <MdClose />
      </button>
    </div>
  )
}

export default Toast



