import { MdLightbulb, MdCheckCircle, MdTrendingUp, MdBarChart, MdDescription, MdDownload, MdShare } from 'react-icons/md'
import './LLMReportCard.css'

function LLMReportCard({ title, content, type = 'default', metadata, onExport, onShare }) {
  const getTypeConfig = () => {
    const configs = {
      insight: {
        icon: <MdLightbulb />,
        color: '#667eea',
        gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
      },
      recommendation: {
        icon: <MdCheckCircle />,
        color: '#10b981',
        gradient: 'linear-gradient(135deg, #10b981 0%, #059669 100%)'
      },
      trend: {
        icon: <MdTrendingUp />,
        color: '#f59e0b',
        gradient: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)'
      },
      analysis: {
        icon: <MdBarChart />,
        color: '#8b5cf6',
        gradient: 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)'
      },
      default: {
        icon: <MdDescription />,
        color: '#6b7280',
        gradient: 'linear-gradient(135deg, #6b7280 0%, #4b5563 100%)'
      }
    }
    return configs[type] || configs.default
  }

  const config = getTypeConfig()

  // 내용이 문자열인 경우와 객체인 경우 처리
  const renderContent = () => {
    if (typeof content === 'string') {
      return <div className="report-text" dangerouslySetInnerHTML={{ __html: content.replace(/\n/g, '<br/>') }} />
    }
    
    if (Array.isArray(content)) {
      return (
        <ul className="report-list">
          {content.map((item, index) => (
            <li key={index} className="report-list-item">
              <span className="list-bullet">{index + 1}.</span>
              <span>{item}</span>
            </li>
          ))}
        </ul>
      )
    }

    if (content && typeof content === 'object') {
      // 백엔드 응답 형식 처리 (explanation, importance, interpretation, recommendation)
      const hasExplainMetricFormat = content.explanation || content.importance || content.interpretation || content.recommendation
      const hasChartInsightFormat = content.pattern || content.trend || content.insight
      
      // 데이터가 없는 경우 처리
      if (!hasExplainMetricFormat && !hasChartInsightFormat && !content.summary && !content.insights && !content.recommendations && !content.trends && !content.text) {
        return (
          <div className="report-text" style={{ color: '#9ca3af', fontStyle: 'italic' }}>
            분석 내용이 준비되지 않았습니다.
          </div>
        )
      }
      
      return (
        <div className="report-structured">
          {content.summary && (
            <div className="report-section">
              <h4 className="section-title">요약</h4>
              <p className="section-content">{content.summary}</p>
            </div>
          )}
          
          {/* explain-metric 형식 (explanation, importance, interpretation, recommendation) */}
          {hasExplainMetricFormat && (
            <>
              {content.explanation && (
                <div className="report-section">
                  <h4 className="section-title">주요 인사이트</h4>
                  <p className="section-content">{content.explanation}</p>
                </div>
              )}
              {content.importance && (
                <div className="report-section">
                  <h4 className="section-title">중요성</h4>
                  <p className="section-content">{content.importance}</p>
                </div>
              )}
              {content.interpretation && (
                <div className="report-section">
                  <h4 className="section-title">평가</h4>
                  <p className="section-content">{content.interpretation}</p>
                </div>
              )}
              {content.recommendation && (
                <div className="report-section">
                  <h4 className="section-title">추천사항</h4>
                  <p className="section-content">{content.recommendation}</p>
                </div>
              )}
            </>
          )}
          
          {/* chart-insight 형식 (pattern, trend, insight, recommendation) */}
          {hasChartInsightFormat && (
            <>
              {content.pattern && (
                <div className="report-section">
                  <h4 className="section-title">주요 패턴</h4>
                  <p className="section-content">{content.pattern}</p>
                </div>
              )}
              {content.trend && (
                <div className="report-section">
                  <h4 className="section-title">트렌드 분석</h4>
                  <p className="section-content">{content.trend}</p>
                </div>
              )}
              {content.insight && (
                <div className="report-section">
                  <h4 className="section-title">핵심 인사이트</h4>
                  <p className="section-content">{content.insight}</p>
                </div>
              )}
              {content.recommendation && (
                <div className="report-section">
                  <h4 className="section-title">추천사항</h4>
                  <p className="section-content">{content.recommendation}</p>
                </div>
              )}
            </>
          )}
          
          {/* 기존 형식 (insights 배열, recommendations 배열, trends 배열) */}
          {!hasExplainMetricFormat && !hasChartInsightFormat && (
            <>
              {content.insights && content.insights.length > 0 && (
                <div className="report-section">
                  <h4 className="section-title">주요 인사이트</h4>
                  <ul className="report-list">
                    {content.insights.map((insight, index) => (
                      <li key={index} className="report-list-item">
                        <MdLightbulb className="list-icon" />
                        <span>{insight}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {content.recommendations && content.recommendations.length > 0 && (
                <div className="report-section">
                  <h4 className="section-title">추천사항</h4>
                  <ul className="report-list">
                    {content.recommendations.map((rec, index) => (
                      <li key={index} className="report-list-item">
                        <MdCheckCircle className="list-icon" />
                        <span>{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {content.trends && content.trends.length > 0 && (
                <div className="report-section">
                  <h4 className="section-title">트렌드 분석</h4>
                  <ul className="report-list">
                    {content.trends.map((trend, index) => (
                      <li key={index} className="report-list-item">
                        <MdTrendingUp className="list-icon" />
                        <span>{trend}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}
          
          {content.text && (
            <div className="report-text">{content.text}</div>
          )}
        </div>
      )
    }

    return <div className="report-text">{JSON.stringify(content)}</div>
  }

  return (
    <div className="llm-report-card">
      <div className="report-header" style={{ background: config.gradient }}>
        <div className="report-header-content">
          <div className="report-title-wrapper">
            <span className="report-icon">{config.icon}</span>
            <h3 className="report-title">{title}</h3>
          </div>
          {metadata && (
            <div className="report-metadata">
              {metadata.date && (
                <span className="metadata-item">{metadata.date}</span>
              )}
              {metadata.source && (
                <span className="metadata-item">{metadata.source}</span>
              )}
            </div>
          )}
        </div>
        {(onExport || onShare) && (
          <div className="report-actions">
            {onExport && (
              <button className="report-action-btn" onClick={onExport} title="내보내기">
                <MdDownload />
              </button>
            )}
            {onShare && (
              <button className="report-action-btn" onClick={onShare} title="공유">
                <MdShare />
              </button>
            )}
          </div>
        )}
      </div>
      
      <div className="report-body">
        {renderContent()}
      </div>
    </div>
  )
}

export default LLMReportCard

