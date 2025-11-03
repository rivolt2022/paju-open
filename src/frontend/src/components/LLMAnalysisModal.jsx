import { useState, useEffect } from 'react'
import { MdSmartToy, MdLightbulb, MdGpsFixed, MdTrendingUp, MdBarChart, MdClose } from 'react-icons/md'
import './LLMAnalysisModal.css'

function LLMAnalysisModal({ isOpen, onClose, analysis, loading }) {
  const [activeTab, setActiveTab] = useState('insights')

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

  if (!isOpen) return null

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2 className="modal-title">
            <MdSmartToy className="modal-title-icon" />
            AI 기반 데이터 분석 및 추천
          </h2>
          <button className="modal-close" onClick={onClose}>
            <MdClose />
          </button>
        </div>

        <div className="modal-body">
          {loading ? (
            <div className="modal-loading">
              <div className="loading-spinner"></div>
              <p>AI가 데이터를 분석하고 있습니다...</p>
            </div>
          ) : analysis ? (
            <>
              <div className="modal-tabs">
                <button
                  className={`tab-button ${activeTab === 'insights' ? 'active' : ''}`}
                  onClick={() => setActiveTab('insights')}
                >
                  <MdLightbulb className="tab-icon" />
                  인사이트
                </button>
                <button
                  className={`tab-button ${activeTab === 'recommendations' ? 'active' : ''}`}
                  onClick={() => setActiveTab('recommendations')}
                >
                  <MdGpsFixed className="tab-icon" />
                  추천사항
                </button>
                <button
                  className={`tab-button ${activeTab === 'trends' ? 'active' : ''}`}
                  onClick={() => setActiveTab('trends')}
                >
                  <MdTrendingUp className="tab-icon" />
                  트렌드 분석
                </button>
              </div>

              <div className="modal-tab-content">
                {activeTab === 'insights' && (
                  <div className="insights-section">
                    <h3>주요 인사이트</h3>
                    <ul className="analysis-list">
                      {analysis.insights?.map((insight, index) => (
                        <li key={index}>
                          <span className="bullet">•</span>
                          <span>{insight}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {activeTab === 'recommendations' && (
                  <div className="recommendations-section">
                    <h3>추천사항</h3>
                    <ul className="analysis-list">
                      {analysis.recommendations?.map((rec, index) => (
                        <li key={index}>
                          <span className="bullet">→</span>
                          <span>{rec}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {activeTab === 'trends' && (
                  <div className="trends-section">
                    <h3>트렌드 분석</h3>
                    <ul className="analysis-list">
                      {analysis.trends?.map((trend, index) => (
                        <li key={index}>
                          <MdBarChart className="bullet-icon" />
                          <span>{trend}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="modal-empty">
              <p>분석 데이터가 없습니다.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default LLMAnalysisModal
