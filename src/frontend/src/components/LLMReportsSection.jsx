import { useState } from 'react'
import { MdDescription, MdLightbulb, MdClose, MdFilterList } from 'react-icons/md'
import LLMReportCard from './LLMReportCard'
import './LLMReportsSection.css'

function LLMReportsSection({ reports, onClose }) {
  const [filter, setFilter] = useState('all')

  const filteredReports = filter === 'all' 
    ? reports 
    : reports.filter(r => r.type === filter)

  const reportTypes = [
    { value: 'all', label: '전체' },
    { value: 'insight', label: '인사이트' },
    { value: 'recommendation', label: '추천사항' },
    { value: 'trend', label: '트렌드' },
    { value: 'analysis', label: '분석' }
  ]

  const handleExport = (report) => {
    const content = typeof report.content === 'string' 
      ? report.content 
      : JSON.stringify(report.content, null, 2)
    
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${report.title}_${new Date().toISOString().split('T')[0]}.txt`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const handleShare = async (report) => {
    if (navigator.share) {
      try {
        const content = typeof report.content === 'string' 
          ? report.content 
          : JSON.stringify(report.content, null, 2)
        
        await navigator.share({
          title: report.title,
          text: content.substring(0, 500) + '...',
        })
      } catch (error) {
        console.log('공유 취소됨')
      }
    } else {
      // 공유 API가 없는 경우 클립보드에 복사
      const content = typeof report.content === 'string' 
        ? report.content 
        : JSON.stringify(report.content, null, 2)
      
      navigator.clipboard.writeText(`${report.title}\n\n${content}`)
      alert('내용이 클립보드에 복사되었습니다.')
    }
  }

  return (
    <div className="llm-reports-section">
      <div className="reports-header">
        <div className="reports-header-left">
          <MdDescription className="reports-icon" />
          <h2 className="reports-title">AI 리포트 모음</h2>
          <span className="reports-count">({filteredReports.length}개)</span>
        </div>
        {onClose && (
          <button className="reports-close-btn" onClick={onClose}>
            <MdClose />
          </button>
        )}
      </div>

      <div className="reports-filters">
        <MdFilterList className="filter-icon" />
        <div className="filter-buttons">
          {reportTypes.map(type => (
            <button
              key={type.value}
              className={`filter-btn ${filter === type.value ? 'active' : ''}`}
              onClick={() => setFilter(type.value)}
            >
              {type.label}
            </button>
          ))}
        </div>
      </div>

      <div className="reports-grid">
        {filteredReports.length === 0 ? (
          <div className="reports-empty">
            <MdLightbulb className="empty-icon" />
            <p>표시할 리포트가 없습니다.</p>
          </div>
        ) : (
          filteredReports.map((report, index) => (
            <LLMReportCard
              key={index}
              title={report.title}
              content={report.content}
              type={report.type || 'default'}
              metadata={report.metadata}
              onExport={() => handleExport(report)}
              onShare={() => handleShare(report)}
            />
          ))
        )}
      </div>
    </div>
  )
}

export default LLMReportsSection

