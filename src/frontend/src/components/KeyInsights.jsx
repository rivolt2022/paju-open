import { useState, useEffect, useRef, useCallback } from 'react'
import { MdDescription, MdRefresh, MdLightbulb, MdTrendingUp, MdCheckCircle, MdInfo } from 'react-icons/md'
import LoadingSpinner from './LoadingSpinner'
import axios from 'axios'
import './KeyInsights.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || (import.meta.env.PROD ? '' : 'http://localhost:8000')

function KeyInsights({ predictions, statistics, date, trigger = 0 }) {
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(true) // 페이지 로드 시 로딩 표시
  const [error, setError] = useState(null)
  const timeoutRef = useRef(null)
  const lastTriggerRef = useRef(-1) // 마지막 트리거 값 추적
  // 이전 값 추적 (데이터 변경 감지용)
  const prevPredictionsRef = useRef(null)
  const prevStatisticsRef = useRef(null)
  const prevDateRef = useRef(null)

  const dateLabel = date ? new Date(date).toLocaleDateString('ko-KR', { month: 'long', day: 'numeric' }) : '오늘'

  const loadLLMReport = useCallback(async (force = false) => {
    // 중복 요청 방지 (force가 true이면 강제 실행)
    if (!force && timeoutRef.current) {
      console.log('[KeyInsights] loadLLMReport 중복 요청 방지 - 이미 로딩 중')
      return
    }
    
    // force가 true이고 이전 요청이 있으면 취소
    if (force && timeoutRef.current) {
      console.log('[KeyInsights] 이전 요청 취소 - 새 요청 시작')
      clearTimeout(timeoutRef.current)
      timeoutRef.current = null
    }
    
    console.log('[KeyInsights] loadLLMReport 함수 호출됨', { predictions: !!predictions, statistics: !!statistics, date })
    setLoading(true)
    setError(null)
    
    // 타임아웃 설정 (70초)
    timeoutRef.current = setTimeout(() => {
      console.log('[KeyInsights] 타임아웃 발생 - 기본값 사용')
      setLoading(false)
      setError('로딩 시간이 초과되었습니다.')
      // 기본 리포트 표시
      setReport({
        summary: `${dateLabel}의 데이터를 분석한 결과입니다.`,
        insights: ['데이터 분석 중입니다.'],
        recommendations: ['추천사항을 준비 중입니다.'],
        trends: ['트렌드 분석 중입니다.']
      })
    }, 70000)
    
    try {
      console.log('[KeyInsights] API 호출 시작', { 
        predictions_count: predictions?.predictions?.length || predictions?.length || 0,
        statistics_keys: Object.keys(statistics || {}),
        date 
      })
      
      // predictions 데이터 구조 처리
      let predictionsData = []
      if (Array.isArray(predictions)) {
        predictionsData = predictions
      } else if (predictions && predictions.predictions && Array.isArray(predictions.predictions)) {
        predictionsData = predictions.predictions
      } else if (predictions && typeof predictions === 'object') {
        const values = Object.values(predictions).filter(item => item && typeof item === 'object')
        if (values.length > 0) {
          predictionsData = values
        }
      }
      
      const response = await axios.post(`${API_BASE_URL}/api/analytics/llm-analysis`, {
        predictions: predictionsData,
        statistics: statistics || {},
        model_metrics: {}, // 모델 지표는 큐레이션에 불필요
        date: date || new Date().toISOString().split('T')[0]
        // trendData는 현재 사용하지 않으므로 제외
      }, {
        timeout: 120000
      })
      
      console.log('[KeyInsights] API 응답 받음', { 
        has_report: !!(response.data),
        response_data: response.data
      })
      
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
      
      if (response.data) {
        console.log('[KeyInsights] LLM 리포트 사용', response.data)
        setReport(response.data)
      } else {
        console.warn('[KeyInsights] API 응답이 없음 - 기본값 사용', response.data)
        setReport({
          summary: `${dateLabel}의 데이터를 분석한 결과입니다.`,
          insights: ['데이터 분석 중입니다.'],
          recommendations: ['추천사항을 준비 중입니다.'],
          trends: ['트렌드 분석 중입니다.']
        })
      }
    } catch (err) {
      console.error('[KeyInsights] 로드 실패:', err)
      console.error('[KeyInsights] 에러 상세:', {
        message: err.message,
        response: err.response?.data,
        status: err.response?.status
      })
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
      // 에러 발생 시에도 기본 리포트 표시
      console.warn('[KeyInsights] 에러로 인해 기본값 사용')
      setReport({
        summary: `${dateLabel}의 데이터를 분석한 결과입니다.`,
        insights: ['데이터 분석 중입니다.'],
        recommendations: ['추천사항을 준비 중입니다.'],
        trends: ['트렌드 분석 중입니다.']
      })
      setError(null)
    } finally {
      setLoading(false)
    }
  }, [predictions, statistics, date, dateLabel])

  useEffect(() => {
    console.log('[KeyInsights] useEffect 실행', { 
      hasPredictions: !!predictions, 
      hasStatistics: !!statistics,
      date,
      trigger,
      lastTrigger: lastTriggerRef.current
    })
    
    // 데이터 변경 감지 (null/undefined 안전 처리)
    const predictionsChanged = (predictions !== prevPredictionsRef.current) && 
      (predictions !== null && predictions !== undefined) &&
      (prevPredictionsRef.current === null || prevPredictionsRef.current === undefined || 
       JSON.stringify(predictions) !== JSON.stringify(prevPredictionsRef.current))
    const statisticsChanged = (statistics !== prevStatisticsRef.current) && 
      (statistics !== null && statistics !== undefined) &&
      (prevStatisticsRef.current === null || prevStatisticsRef.current === undefined || 
       JSON.stringify(statistics) !== JSON.stringify(prevStatisticsRef.current))
    const dateChanged = date !== prevDateRef.current
    const triggerChanged = trigger !== lastTriggerRef.current
    
    // 첫 로드 감지
    const isFirstLoad = prevPredictionsRef.current === null && prevStatisticsRef.current === null && prevDateRef.current === null
    
    // 로드 조건: 첫 로드 OR trigger 변경 시 API 호출
    // 첫 로드 = 페이지 로드 시 (데이터 없어도 호출 - 빈 객체로 처리)
    // trigger 변경 = 예측 실행 버튼 클릭 시 (새로운 데이터로 리포트 생성 필요)
    const shouldLoad = triggerChanged || isFirstLoad
    
    if (shouldLoad) {
      console.log('[KeyInsights] loadLLMReport 호출', {
        reason: triggerChanged ? '예측 실행 버튼 클릭' : 
                predictionsChanged || statisticsChanged ? '데이터 변경' : 
                dateChanged ? '날짜 변경' : '페이지 로드',
        trigger,
        lastTrigger: lastTriggerRef.current,
        predictionsChanged,
        statisticsChanged,
        dateChanged,
        triggerChanged,
        isFirstLoad,
        hasPredictions: !!predictions,
        hasStatistics: !!statistics
      })
      
      // trigger 변경 시 강제로 새 요청 실행
      loadLLMReport(triggerChanged)
      if (triggerChanged) {
        lastTriggerRef.current = trigger
      }
      // 이전 값 업데이트 (안전하게 처리)
      try {
        prevPredictionsRef.current = predictions ? JSON.parse(JSON.stringify(predictions)) : null
        prevStatisticsRef.current = statistics ? JSON.parse(JSON.stringify(statistics)) : null
        prevDateRef.current = date
      } catch (e) {
        console.warn('[KeyInsights] 이전 값 업데이트 실패:', e)
        prevPredictionsRef.current = predictions
        prevStatisticsRef.current = statistics
        prevDateRef.current = date
      }
    } else if (triggerChanged) {
      lastTriggerRef.current = trigger
    }
    
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [predictions, statistics, date, trigger, loadLLMReport])

  // 로딩 중일 때 로딩 표시 (활성화 분석과 동일한 방식)
  if (loading) {
    return (
      <div className="key-insights-container">
        <div className="key-insights-header">
          <span className="key-insights-icon"><MdDescription /></span>
          <span>LLM 분석 리포트</span>
        </div>
        <div className="key-insights-loading">
          <LoadingSpinner message="AI가 데이터를 분석 중입니다..." size="large" />
        </div>
      </div>
    )
  }

  // 리포트가 없으면 기본 리포트 표시
  if (!loading && !report) {
    return (
      <div className="key-insights-container">
        <div className="key-insights-header">
          <span className="key-insights-icon"><MdDescription /></span>
          <span>LLM 분석 리포트</span>
          <button 
            className="key-insights-refresh"
            onClick={loadLLMReport}
            title="새로고침"
            disabled={loading}
          >
            <MdRefresh />
          </button>
        </div>
        <div className="key-insights-report">
          <p className="key-insights-summary">데이터를 분석하여 리포트를 생성 중입니다.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="key-insights-container">
      <div className="key-insights-header">
        <span className="key-insights-icon"><MdDescription /></span>
        <span>LLM 분석 리포트</span>
        <button 
          className="key-insights-refresh"
          onClick={loadLLMReport}
          title="새로고침"
          disabled={loading}
        >
          <MdRefresh />
        </button>
      </div>
      
      <div className="key-insights-report">
        {/* 주요 인사이트 섹션 */}
        {report.insights && report.insights.length > 0 && (
          <div className="report-section">
            <div className="report-section-header">
              <MdLightbulb className="report-section-icon" />
              <h3 className="report-section-title">주요 인사이트</h3>
            </div>
            <div className="report-section-content">
              {report.insights.map((insight, index) => (
                <div key={index} className="report-item">
                  <span className="report-item-bullet">•</span>
                  <p className="report-item-text">{insight}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 실행 가능한 추천사항 섹션 */}
        {report.recommendations && report.recommendations.length > 0 && (
          <div className="report-section">
            <div className="report-section-header">
              <MdCheckCircle className="report-section-icon" />
              <h3 className="report-section-title">실행 가능한 추천사항</h3>
            </div>
            <div className="report-section-content">
              {report.recommendations.map((recommendation, index) => (
                <div key={index} className="report-item">
                  <span className="report-item-bullet">•</span>
                  <p className="report-item-text">{recommendation}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 트렌드 분석 섹션 */}
        {report.trends && report.trends.length > 0 && (
          <div className="report-section">
            <div className="report-section-header">
              <MdTrendingUp className="report-section-icon" />
              <h3 className="report-section-title">트렌드 분석 및 전망</h3>
            </div>
            <div className="report-section-content">
              {report.trends.map((trend, index) => (
                <div key={index} className="report-item">
                  <span className="report-item-bullet">•</span>
                  <p className="report-item-text">{trend}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 요약 섹션 (있는 경우) */}
        {report.summary && (
          <div className="report-section report-section-summary">
            <div className="report-section-header">
              <MdInfo className="report-section-icon" />
              <h3 className="report-section-title">종합 요약</h3>
            </div>
            <div className="report-section-content">
              <p className="report-summary-text">{report.summary}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default KeyInsights
