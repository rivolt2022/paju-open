import { useState, useEffect } from 'react'
import { createPortal } from 'react-dom'
import { MdDescription, MdClose, MdDownload, MdShare } from 'react-icons/md'
import LLMReportCard from './LLMReportCard'
import './LLMReportModal.css'

function LLMReportModal({ isOpen, onClose, reports = [] }) {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    return () => setMounted(false)
  }, [])

  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = 'unset'
    }
    return () => {
      document.body.style.overflow = 'unset'
    }
  }, [isOpen])

  if (!mounted || !isOpen) return null

  const handleExport = () => {
    const content = reports.map(r => {
      const reportContent = typeof r.content === 'string' 
        ? r.content 
        : JSON.stringify(r.content, null, 2)
      return `=== ${r.title} ===\n${reportContent}\n\n`
    }).join('\n')
    
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `AI_리포트_${new Date().toISOString().split('T')[0]}.txt`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const handleShare = async () => {
    if (navigator.share) {
      try {
        const content = reports.map(r => r.title).join(', ')
        await navigator.share({
          title: 'AI 분석 리포트',
          text: content,
        })
      } catch (error) {
        console.log('공유 취소됨')
      }
    } else {
      navigator.clipboard.writeText(reports.map(r => r.title).join('\n'))
      alert('리포트 제목이 클립보드에 복사되었습니다.')
    }
  }

  const modalContent = (
    <div 
      className={`llm-report-modal-overlay ${isOpen ? 'open' : ''}`}
      onClick={onClose}
    >
      <div 
        className={`llm-report-modal ${isOpen ? 'open' : ''}`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="llm-report-modal-header">
          <div className="modal-header-content">
            <MdDescription className="modal-header-icon" />
            <h2 className="modal-header-title">AI 분석 리포트</h2>
            <span className="modal-header-count">({reports.length}개)</span>
          </div>
          <div className="modal-header-actions">
            {reports.length > 0 && (
              <>
                <button className="modal-action-btn" onClick={handleExport} title="내보내기">
                  <MdDownload />
                </button>
                <button className="modal-action-btn" onClick={handleShare} title="공유">
                  <MdShare />
                </button>
              </>
            )}
            <button className="modal-close-btn" onClick={onClose}>
              <MdClose />
            </button>
          </div>
        </div>

        <div className="llm-report-modal-body">
          {reports.length === 0 ? (
            <div className="empty-reports">
              <MdDescription className="empty-icon" />
              <p>표시할 리포트가 없습니다.</p>
            </div>
          ) : (
            <div className="reports-list">
              {reports.map((report, index) => (
                <LLMReportCard
                  key={index}
                  title={report.title}
                  content={report.content}
                  type={report.type || 'default'}
                  metadata={report.metadata}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )

  return createPortal(modalContent, document.body)
}

export default LLMReportModal
