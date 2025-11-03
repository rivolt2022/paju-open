import { useState } from 'react'
import './MetricsGroup.css'

function MetricsGroup({ title, icon, children, defaultOpen = true, priority = 'medium' }) {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  return (
    <div className={`metrics-group metrics-group-${priority}`}>
      <div 
        className="metrics-group-header"
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="metrics-group-title">
          <span className="metrics-group-icon">{icon}</span>
          <h3 className="metrics-group-title-text">{title}</h3>
        </div>
        <button 
          className={`metrics-group-toggle ${isOpen ? 'open' : ''}`}
          aria-label={isOpen ? '접기' : '펼치기'}
        >
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <path 
              d="M5 7.5L10 12.5L15 7.5" 
              stroke="currentColor" 
              strokeWidth="2" 
              strokeLinecap="round" 
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </div>
      {isOpen && (
        <div className="metrics-group-content">
          {children}
        </div>
      )}
    </div>
  )
}

export default MetricsGroup

