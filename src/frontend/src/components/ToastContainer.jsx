import { createPortal } from 'react-dom'
import Toast from './Toast'

function ToastContainer({ toasts, onRemove }) {
  if (!toasts || toasts.length === 0) return null

  return createPortal(
    <div className="toast-container">
      {toasts.map(toast => (
        <Toast
          key={toast.id}
          message={toast.message}
          type={toast.type}
          duration={toast.duration}
          onClose={() => onRemove(toast.id)}
        />
      ))}
    </div>,
    document.body
  )
}

export default ToastContainer

