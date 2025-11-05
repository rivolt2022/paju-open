import { useState, useEffect, useRef, useCallback } from 'react'
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadialBarChart, RadialBar } from 'recharts'
import { MdStar, MdPeople, MdEvent, MdLocationCity, MdTrendingUp, MdLink, MdMenuBook, MdLightbulb, MdCheckCircle, MdDescription } from 'react-icons/md'
import LoadingSpinner from './LoadingSpinner'
import LLMReportModal from './LLMReportModal'
import axios from 'axios'
import './MeaningfulMetrics.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || (import.meta.env.PROD ? '' : 'http://localhost:8000')

function MeaningfulMetrics({ spaceName = "헤이리예술마을", date = null, startDate = null, endDate = null, trigger = 0 }) {
  const [metrics, setMetrics] = useState(null)
  const [activationScores, setActivationScores] = useState(null)
  const [vitality, setVitality] = useState(null)
  const [loading, setLoading] = useState(true)
  const isLoadingRef = useRef(false) // 중복 요청 방지
  const lastLoadParamsRef = useRef({ spaceName: null, date: null }) // 마지막 로드 파라미터 추적
  const lastTriggerRef = useRef(0) // 마지막 trigger 값 추적
  const hasLoadedMetricsRef = useRef(false) // loadMetrics가 실제로 호출되었는지 추적
  
  // 날짜 포맷 함수 (단일 날짜만 사용)
  const formatDateLabel = (dateValue) => {
    if (!dateValue) return ''
    const dateObj = new Date(dateValue)
    return dateObj.toLocaleDateString('ko-KR', { month: 'long', day: 'numeric', weekday: 'long' })
  }
  
  // LLM 설명 상태 관리
  const [llmInsights, setLlmInsights] = useState({})
  const [llmLoading, setLlmLoading] = useState({})
  const timeoutRefs = useRef({})
  
  // 리포트 관리
  const [reports, setReports] = useState([])
  const [showReportModal, setShowReportModal] = useState(false)

  useEffect(() => {
    // trigger 변경 감지 (예측 실행 시 갱신을 위해)
    const triggerChanged = trigger !== lastTriggerRef.current
    
    // 중복 요청 방지
    if (isLoadingRef.current) {
      console.log('[MeaningfulMetrics] 중복 요청 방지', { isLoading: isLoadingRef.current })
      return
    }
    
    // 첫 로드 감지 (이전 값이 모두 null이면 첫 로드)
    const isFirstLoad = lastLoadParamsRef.current.date === null && lastLoadParamsRef.current.spaceName === null
    
    // 날짜나 공간이 실제로 변경되었는지 확인
    const dateActuallyChanged = date && date !== lastLoadParamsRef.current.date
    const spaceActuallyChanged = spaceName && spaceName !== lastLoadParamsRef.current.spaceName
    
    // trigger 변경 OR 첫 로드 OR 날짜/공간 변경 시 API 호출
    if (!triggerChanged && !isFirstLoad && !dateActuallyChanged && !spaceActuallyChanged) {
      console.log('[MeaningfulMetrics] 변경 없음 - API 호출 스킵', { 
        triggerChanged,
        isFirstLoad,
        dateActuallyChanged,
        spaceActuallyChanged,
        date,
        lastDate: lastLoadParamsRef.current.date,
        trigger,
        lastTrigger: lastTriggerRef.current
      })
      return
    }
    
    // trigger 변경 시 (예측 실행) 날짜가 같아도 갱신
    if (triggerChanged) {
      console.log('[MeaningfulMetrics] 예측 실행으로 인한 갱신', { 
        trigger, 
        lastTrigger: lastTriggerRef.current,
        date 
      })
    }
    
    // 모든 LLM 인사이트 초기화
    setLlmInsights({})
    setLlmLoading({})
    setComprehensiveAnalysis(null) // 종합 분석도 초기화
    // 로딩 상태를 즉시 표시
    setLoading(true)
    lastLoadParamsRef.current = { spaceName, date }
    if (triggerChanged) {
      lastTriggerRef.current = trigger
    }
    hasLoadedMetricsRef.current = false // loadMetrics 호출 전 플래그 초기화
    loadMetrics()
  }, [spaceName, date, trigger])

  useEffect(() => {
    // 컴포넌트 언마운트 시 타임아웃 정리
    return () => {
      Object.values(timeoutRefs.current).forEach(ref => {
        if (ref) clearTimeout(ref)
      })
    }
  }, [])

  // 활성화 점수 LLM 인사이트 생성 (loadMetrics 호출 후에만)
  useEffect(() => {
    if (hasLoadedMetricsRef.current && activationScores?.overall && !llmLoading['activation_scores'] && !llmInsights['activation_scores']) {
      generateLLMInsight('activation_scores', '문화 공간 활성화 점수', activationScores.overall, {
        accessibility: activationScores.accessibility,
        interest: activationScores.interest,
        potential: activationScores.potential,
        utilization: activationScores.utilization,
        overall: activationScores.overall
      })
    }
  }, [activationScores?.overall])

  // 성연령별 타겟팅 LLM 인사이트 생성 (loadMetrics 호출 후에만)
  useEffect(() => {
    if (hasLoadedMetricsRef.current && metrics?.demographic_targeting && !llmLoading['demographic_targeting'] && !llmInsights['demographic_targeting']) {
      const demographicData = Object.entries(metrics.demographic_targeting.demographic_scores || {}).map(([age, scores]) => ({
        age,
        male: (scores.male * 100).toFixed(0),
        female: (scores.female * 100).toFixed(0),
        total: (scores.total * 100).toFixed(0)
      }))
      
      if (demographicData.length > 0) {
        generateChartInsight('demographic_targeting', '성연령별 타겟팅 분석', demographicData, {
          targeting_strategy: metrics.demographic_targeting.targeting_strategy
        })
      }
    }
  }, [metrics?.demographic_targeting])

  // 주말/평일 분석 LLM 인사이트 생성 (loadMetrics 호출 후에만)
  useEffect(() => {
    if (hasLoadedMetricsRef.current && metrics?.weekend_analysis && !llmLoading['weekend_analysis'] && !llmInsights['weekend_analysis']) {
      const weekendData = [
        { name: '평일', value: metrics.weekend_analysis.weekday_average },
        { name: '주말', value: metrics.weekend_analysis.weekend_average }
      ]
      
      if (weekendData.length > 0) {
        generateChartInsight('weekend_analysis', '주말/평일 방문 패턴', weekendData, {
          weekend_ratio: metrics.weekend_analysis.weekend_ratio
        })
      }
    }
  }, [metrics?.weekend_analysis])

  // 출판단지 활성화 지수 LLM 인사이트 생성 (loadMetrics 호출 후에만)
  useEffect(() => {
    if (hasLoadedMetricsRef.current && vitality?.overall_publishing_complex_vitality && !llmLoading['publishing_vitality'] && !llmInsights['publishing_vitality']) {
      generateLLMInsight('publishing_vitality', '출판단지 활성화 지수', vitality.overall_publishing_complex_vitality * 100, {
        trend: vitality.trend,
        regional_indices: vitality.regional_indices
      })
    }
  }, [vitality?.overall_publishing_complex_vitality])

  // 종합 출판단지 활성화 분석 LLM 생성
  const [comprehensiveAnalysis, setComprehensiveAnalysis] = useState(null)
  const [comprehensiveAnalysisLoading, setComprehensiveAnalysisLoading] = useState(false)

  const generateComprehensiveAnalysis = useCallback(async () => {
    console.log('[MeaningfulMetrics] generateComprehensiveAnalysis 호출됨')
    
    // 이미 로딩 중이거나 이미 생성된 경우 스킵
    if (comprehensiveAnalysisLoading || comprehensiveAnalysis) {
      console.log('[MeaningfulMetrics] 종합 분석 생성 스킵 (이미 로딩 중이거나 생성됨)')
      return
    }
    
    setComprehensiveAnalysisLoading(true)
    
    try {
      console.log('[MeaningfulMetrics] comprehensive-publishing-analysis API 호출 시작')
      const response = await axios.post(`${API_BASE_URL}/api/analytics/comprehensive-publishing-analysis`, {
        space_name: spaceName,
        date: date || undefined,
        activation_scores: activationScores,
        metrics: {
          demographic_targeting: metrics.demographic_targeting,
          weekend_analysis: metrics.weekend_analysis,
          seasonal_patterns: metrics.seasonal_patterns,
          optimal_time_analysis: metrics.optimal_time_analysis
        },
        vitality: vitality
      }, {
        timeout: 60000 // 60초 타임아웃
      })

      console.log('[MeaningfulMetrics] comprehensive-publishing-analysis API 응답 받음')
      setComprehensiveAnalysis(response.data)
    } catch (error) {
      console.error('종합 분석 생성 오류:', error)
      // 기본 분석 생성
      setComprehensiveAnalysis({
        summary: '출판단지 활성화를 위한 종합 분석을 준비 중입니다.',
        strengths: [],
        weaknesses: [],
        opportunities: [],
        recommendations: [],
        action_plan: []
      })
    } finally {
      setComprehensiveAnalysisLoading(false)
    }
  }, [spaceName, date, activationScores, metrics, vitality, comprehensiveAnalysisLoading, comprehensiveAnalysis])

  useEffect(() => {
    // loadMetrics가 호출된 후에만 종합 분석 생성 (예측 실행 시 불필요한 API 호출 방지)
    if (hasLoadedMetricsRef.current && metrics && activationScores && vitality && !comprehensiveAnalysis && !comprehensiveAnalysisLoading) {
      console.log('[MeaningfulMetrics] 종합 분석 생성 시작', { metrics: !!metrics, activationScores: !!activationScores, vitality: !!vitality })
      generateComprehensiveAnalysis()
    }
  }, [metrics, activationScores, vitality, comprehensiveAnalysis, comprehensiveAnalysisLoading, generateComprehensiveAnalysis])

  const loadMetrics = async () => {
    // 중복 요청 방지
    if (isLoadingRef.current) {
      console.log('[MeaningfulMetrics] loadMetrics 중복 요청 방지')
      return
    }
    
    isLoadingRef.current = true
    try {
      // 종합 지표 로드 (날짜 파라미터 포함)
      const metricsResponse = await axios.get(`${API_BASE_URL}/api/analytics/meaningful-metrics`, {
        params: { 
          space_name: spaceName,
          date: date || undefined
        }
      })
      setMetrics(metricsResponse.data)

      // 활성화 점수 로드 (날짜 파라미터 포함)
      const scoresResponse = await axios.get(`${API_BASE_URL}/api/analytics/activation-scores`, {
        params: { 
          space_name: spaceName,
          date: date || undefined
        }
      })
      setActivationScores(scoresResponse.data)

      // 출판단지 활성화 지수 로드 (날짜 파라미터 포함)
      const vitalityResponse = await axios.get(`${API_BASE_URL}/api/analytics/publishing-vitality`, {
        params: { 
          date: date || undefined
        }
      })
      setVitality(vitalityResponse.data)

      // loadMetrics 완료 후 플래그 설정 (이제 LLM API 호출 가능)
      hasLoadedMetricsRef.current = true

    } catch (error) {
      console.error('지표 로드 오류:', error)
    } finally {
      setLoading(false)
      isLoadingRef.current = false
    }
  }

  // LLM 기반 인사이트 생성
  const generateLLMInsight = async (metricKey, metricName, metricValue, context = {}) => {
    // 이미 로딩 중이거나 이미 생성된 경우 스킵
    if (llmLoading[metricKey] || llmInsights[metricKey]) {
      return
    }

    setLlmLoading(prev => ({ ...prev, [metricKey]: true }))

    // 타임아웃 설정 (25초 - LLM 응답 시간을 고려)
    const timeoutId = setTimeout(() => {
      setLlmLoading(prev => ({ ...prev, [metricKey]: false }))
      setLlmInsights(prev => ({
        ...prev,
        [metricKey]: {
          explanation: `${metricName}에 대한 기본 설명입니다.`,
          importance: '이 지표는 출판단지 활성화를 평가하는 중요한 요소입니다.',
          interpretation: metricValue >= 70 ? '좋음' : metricValue >= 50 ? '보통' : '나쁨',
          recommendation: '지속적으로 모니터링하여 개선 기회를 찾아보세요.'
        }
      }))
    }, 10000)

    timeoutRefs.current[metricKey] = timeoutId

    try {
      const response = await axios.post(`${API_BASE_URL}/api/llm/explain-metric`, {
        metric_name: metricName,
        metric_value: metricValue,
        metric_type: 'publishing_activation',
        context: {
          space_name: spaceName,
          ...context
        }
      }, {
        timeout: 120000  // LLM 응답을 위해 120초로 증가
      })

      if (timeoutRefs.current[metricKey]) {
        clearTimeout(timeoutRefs.current[metricKey])
        delete timeoutRefs.current[metricKey]
      }

      const insightData = response.data
      setLlmInsights(prev => ({ ...prev, [metricKey]: insightData }))
      
      // 리포트에 추가
      if (insightData) {
        console.log(`[MeaningfulMetrics] 리포트 추가: ${metricKey}`, insightData)
        setReports(prev => {
          const newReport = {
            title: `${metricName} 분석 리포트`,
            content: insightData,
            type: 'insight',
            metadata: {
              date: new Date().toISOString().split('T')[0],
              source: `출판단지 활성화 분석 - ${metricName}`
            }
          }
          // 중복 제거
          const exists = prev.some(r => r.title === newReport.title)
          if (exists) return prev
          return [...prev, newReport]
        })
      }
    } catch (error) {
      console.error(`[LLM] ${metricKey} 인사이트 생성 실패:`, error)
      // 기본값 제공
      setLlmInsights(prev => ({
        ...prev,
        [metricKey]: {
          explanation: `${metricName}에 대한 분석입니다.`,
          importance: '출판단지 활성화를 위한 중요한 지표입니다.',
          interpretation: metricValue >= 70 ? '좋음' : metricValue >= 50 ? '보통' : '나쁨',
          recommendation: '데이터 기반 의사결정에 활용하세요.'
        }
      }))
    } finally {
      setLlmLoading(prev => ({ ...prev, [metricKey]: false }))
    }
  }

  // 차트 인사이트 생성
  const generateChartInsight = async (metricKey, chartType, chartData, context = {}) => {
    if (llmLoading[metricKey] || llmInsights[metricKey]) {
      return
    }

    setLlmLoading(prev => ({ ...prev, [metricKey]: true }))

    const timeoutId = setTimeout(() => {
      setLlmLoading(prev => ({ ...prev, [metricKey]: false }))
    }, 25000)  // LLM 응답 시간을 고려하여 25초로 증가

    timeoutRefs.current[metricKey] = timeoutId

    try {
      const response = await axios.post(`${API_BASE_URL}/api/llm/chart-insight`, {
        chart_type: chartType,
        chart_data: chartData,
        context: {
          space_name: spaceName,
          date: date || undefined,
          ...context
        }
      }, {
        timeout: 120000  // LLM 응답을 위해 120초로 증가
      })

      if (timeoutRefs.current[metricKey]) {
        clearTimeout(timeoutRefs.current[metricKey])
        delete timeoutRefs.current[metricKey]
      }

      const insightData = response.data
      setLlmInsights(prev => ({ ...prev, [metricKey]: insightData }))
      
      // 리포트에 추가
      if (insightData) {
        console.log(`[MeaningfulMetrics] 차트 리포트 추가: ${metricKey}`, insightData)
        setReports(prev => {
          const newReport = {
            title: `${chartType} 분석 리포트`,
            content: insightData,
            type: 'analysis',
            metadata: {
              date: new Date().toISOString().split('T')[0],
              source: `출판단지 활성화 분석 - ${chartType}`
            }
          }
          // 중복 제거
          const exists = prev.some(r => r.title === newReport.title)
          if (exists) return prev
          return [...prev, newReport]
        })
      }
    } catch (error) {
      console.error(`[LLM] ${metricKey} 차트 인사이트 생성 실패:`, error)
    } finally {
      setLlmLoading(prev => ({ ...prev, [metricKey]: false }))
    }
  }

  if (loading) {
    return (
      <div className="meaningful-metrics">
        <div className="metrics-header">
          <div className="metrics-header-top">
            <h2 className="metrics-title">
              <MdMenuBook className="header-icon" />
              출판단지 활성화를 위한 AI 분석
            </h2>
          </div>
          <p className="metrics-subtitle">AI 문화 및 콘텐츠 서비스를 통한 지역 활성화 데이터 분석</p>
        </div>
        <div className="metrics-loading-container">
          <LoadingSpinner message="지표 데이터를 불러오는 중..." size="large" />
        </div>
      </div>
    )
  }

  if (!metrics && !activationScores) {
    return null
  }

  // 활성화 점수 차트 데이터
  const activationData = activationScores ? [
    { name: '접근성', value: activationScores.accessibility, color: '#667eea' },
    { name: '관심도', value: activationScores.interest, color: '#764ba2' },
    { name: '잠재력', value: activationScores.potential, color: '#f093fb' },
    { name: '활용도', value: activationScores.utilization, color: '#4facfe' },
    { name: '종합', value: activationScores.overall, color: '#00f2fe' }
  ] : []

  // 성연령별 타겟팅 데이터
  const demographicData = metrics?.demographic_targeting?.demographic_scores
    ? Object.entries(metrics.demographic_targeting.demographic_scores).map(([age, scores]) => ({
        age,
        male: (scores.male * 100).toFixed(0),
        female: (scores.female * 100).toFixed(0),
        total: (scores.total * 100).toFixed(0)
      }))
    : []

  // 주말/평일 분석 데이터
  const weekendData = metrics?.weekend_analysis
    ? [
        { name: '평일', value: metrics.weekend_analysis.weekday_average },
        { name: '주말', value: metrics.weekend_analysis.weekend_average }
      ]
    : []

  // 출판단지 활성화 지수 데이터
  const vitalityData = vitality?.regional_indices
    ? Object.entries(vitality.regional_indices).map(([region, data]) => ({
        region,
        vitality: (data.vitality_score * 100).toFixed(1),
        consumption: (data.consumption_score * 100).toFixed(1),
        production: (data.production_score * 100).toFixed(1),
        overall: (data.overall * 100).toFixed(1)
      }))
    : []

  // 계절별 패턴 데이터
  const seasonalData = metrics?.seasonal_patterns?.seasonal_scores
    ? Object.entries(metrics.seasonal_patterns.seasonal_scores).map(([season, data]) => ({
        season,
        visits: data.visits,
        score: (data.score * 100).toFixed(1)
      }))
    : []

  const COLORS = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe']

  // 초기 로딩 상태
  if (loading && !metrics && !activationScores && !vitality) {
    return (
      <div className="meaningful-metrics">
        <div className="metrics-header">
          <div className="metrics-header-top">
            <h2 className="metrics-title">
              <MdMenuBook className="header-icon" />
              출판단지 활성화를 위한 AI 분석
            </h2>
          </div>
          <p className="metrics-subtitle">AI 문화 및 콘텐츠 서비스를 통한 지역 활성화 데이터 분석</p>
        </div>
        <div className="metrics-loading-container">
          <LoadingSpinner message="지표 데이터를 불러오는 중..." size="large" />
        </div>
      </div>
    )
  }

  return (
    <div className="meaningful-metrics">
      <div className="metrics-header">
        <div className="metrics-header-top">
          <h2 className="metrics-title">
            <MdMenuBook className="header-icon" />
            출판단지 활성화를 위한 AI 분석
          </h2>
          {reports.length > 0 && (
            <button 
              className="reports-header-btn"
              onClick={() => setShowReportModal(true)}
            >
              <MdDescription className="reports-header-icon" />
              AI 분석 리포트
              <span className="reports-count-badge">{reports.length}</span>
            </button>
          )}
        </div>
        <p className="metrics-subtitle">AI 문화 및 콘텐츠 서비스를 통한 지역 활성화 데이터 분석</p>
      </div>

      {/* 종합 출판단지 활성화 분석 (LLM 강화) */}
      {(comprehensiveAnalysis || comprehensiveAnalysisLoading) && (
        <div className="comprehensive-analysis-card">
          <h3 className="card-title">
            <MdLightbulb className="card-title-icon" />
            AI 종합 활성화 분석
            {date && (
              <span className="date-range-label"> ({formatDateLabel(date)})</span>
            )}
          </h3>
          {comprehensiveAnalysisLoading ? (
            <div className="analysis-loading">
              <LoadingSpinner message="AI가 출판단지 활성화 데이터를 종합 분석 중입니다..." size="medium" />
            </div>
          ) : comprehensiveAnalysis ? (
            <div className="comprehensive-analysis-content">
              {/* 종합 결과 요약 */}
              <div className="comprehensive-result-summary">
                <div className="result-header">
                  <h4 className="result-title">
                    <MdLightbulb className="result-icon" />
                    종합 분석 결과
                  </h4>
                  <div className="result-score">
                    <span className="score-label">활성화 지수</span>
                    <span className="score-value">
                      {vitality?.overall_publishing_complex_vitality 
                        ? (vitality.overall_publishing_complex_vitality * 100).toFixed(1)
                        : '0.0'}
                    </span>
                    <span className="score-unit">/ 100</span>
                  </div>
                </div>
                {comprehensiveAnalysis.summary && (
                  <p className="result-summary-text">{comprehensiveAnalysis.summary}</p>
                )}
              </div>

              {/* 서술형 분석 */}
              {comprehensiveAnalysis.detailed_analysis && (
                <div className="analysis-section detailed-analysis-section">
                  <h4 className="section-title">
                    <MdDescription className="section-icon" />
                    서술형 분석
                  </h4>
                  <div className="detailed-analysis-content">
                    {comprehensiveAnalysis.detailed_analysis.map((paragraph, index) => (
                      <p key={index} className="analysis-paragraph">
                        {paragraph}
                      </p>
                    ))}
                  </div>
                </div>
              )}

              {/* 요약 (기존) */}
              {comprehensiveAnalysis.summary && !comprehensiveAnalysis.detailed_analysis && (
                <div className="analysis-section">
                  <h4 className="section-title">
                    <MdDescription className="section-icon" />
                    분석 요약
                  </h4>
                  <p className="analysis-summary">{comprehensiveAnalysis.summary}</p>
                </div>
              )}

              {/* 강점 */}
              {comprehensiveAnalysis.strengths && comprehensiveAnalysis.strengths.length > 0 && (
                <div className="analysis-section">
                  <h4 className="section-title">
                    <MdStar className="section-icon" />
                    주요 강점
                  </h4>
                  <ul className="analysis-list">
                    {comprehensiveAnalysis.strengths.map((item, index) => (
                      <li key={index}>
                        <MdCheckCircle className="list-icon success" />
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* 개선점 */}
              {comprehensiveAnalysis.weaknesses && comprehensiveAnalysis.weaknesses.length > 0 && (
                <div className="analysis-section">
                  <h4 className="section-title">
                    <MdTrendingUp className="section-icon" />
                    개선 필요 영역
                  </h4>
                  <ul className="analysis-list">
                    {comprehensiveAnalysis.weaknesses.map((item, index) => (
                      <li key={index}>
                        <MdLightbulb className="list-icon warning" />
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* 기회 */}
              {comprehensiveAnalysis.opportunities && comprehensiveAnalysis.opportunities.length > 0 && (
                <div className="analysis-section">
                  <h4 className="section-title">
                    <MdEvent className="section-icon" />
                    활성화 기회
                  </h4>
                  <ul className="analysis-list">
                    {comprehensiveAnalysis.opportunities.map((item, index) => (
                      <li key={index}>
                        <MdCheckCircle className="list-icon opportunity" />
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* 추천사항 */}
              {comprehensiveAnalysis.recommendations && comprehensiveAnalysis.recommendations.length > 0 && (
                <div className="analysis-section">
                  <h4 className="section-title">
                    <MdCheckCircle className="section-icon" />
                    실행 가능한 추천사항
                  </h4>
                  <ul className="analysis-list recommendations">
                    {comprehensiveAnalysis.recommendations.map((item, index) => (
                      <li key={index}>
                        <MdCheckCircle className="list-icon recommendation" />
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* 실행 계획 */}
              {comprehensiveAnalysis.action_plan && comprehensiveAnalysis.action_plan.length > 0 && (
                <div className="analysis-section">
                  <h4 className="section-title">
                    <MdPeople className="section-icon" />
                    단계별 실행 계획
                  </h4>
                  <ol className="action-plan-list">
                    {comprehensiveAnalysis.action_plan.map((item, index) => (
                      <li key={index}>
                        <span className="step-number">{index + 1}</span>
                        <span className="step-content">{item}</span>
                      </li>
                    ))}
                  </ol>
                </div>
              )}
            </div>
          ) : null}
        </div>
      )}

      <div className="metrics-grid">
        {/* 활성화 점수 */}
        {activationScores && (() => {
          const metricKey = 'activation_scores'
          const insight = llmInsights[metricKey]
          const isLlmLoading = llmLoading[metricKey]
          
          return (
            <div className="metric-card activation-scores">
              <h3 className="card-title">
                <MdStar className="card-title-icon" />
                문화 공간 활성화 점수
              </h3>
              <div className="score-summary">
                <div className="overall-score">
                  <span className="score-label">종합 점수</span>
                  <span className="score-value">{activationScores.overall?.toFixed(1)}</span>
                  <span className="score-unit">/ 100</span>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={activationData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="name" stroke="#64748b" />
                  <YAxis stroke="#64748b" domain={[0, 100]} />
                  <Tooltip formatter={(value) => `${value.toFixed(1)}점`} />
                  <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                    {activationData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div className="score-insight">
                {isLlmLoading ? (
                  <p className="llm-loading">
                    <MdLightbulb className="insight-icon" />
                    AI가 분석 중입니다...
                  </p>
                ) : insight ? (
                  <>
                    <p>
                      <MdLightbulb className="insight-icon" />
                      <strong>AI 분석:</strong> {insight.explanation}
                    </p>
                    <p>
                      <strong>중요도:</strong> {insight.importance}
                    </p>
                    <p>
                      <MdCheckCircle className="recommendation-icon" />
                      <strong>활용 방안:</strong> {insight.recommendation}
                    </p>
                  </>
                ) : (
                  <p>
                    <MdLightbulb className="insight-icon" />
                    <strong>인사이트:</strong> 접근성, 관심도, 잠재력, 활용도를 종합한 활성화 점수입니다. 
                    {activationScores.overall >= 70 ? ' 높은 활성화 가능성을 보입니다.' : 
                     activationScores.overall >= 50 ? ' 보통 수준의 활성화 가능성입니다.' : 
                     ' 활성화를 위한 전략 수립이 필요합니다.'}
                  </p>
                )}
              </div>
            </div>
          )
        })()}

        {/* 성연령별 타겟팅 */}
        {demographicData.length > 0 && (
          <div className="metric-card demographic-targeting">
            <h3 className="card-title">
              <MdPeople className="card-title-icon" />
              성연령별 타겟팅 분석
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={demographicData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="age" stroke="#64748b" />
                <YAxis stroke="#64748b" domain={[0, 100]} />
                <Tooltip formatter={(value) => `${value}%`} />
                <Legend />
                <Bar dataKey="male" fill="#667eea" name="남성" radius={[8, 8, 0, 0]} />
                <Bar dataKey="female" fill="#f093fb" name="여성" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            {metrics?.demographic_targeting && (() => {
              const metricKey = 'demographic_targeting'
              const insight = llmInsights[metricKey]
              const isLlmLoading = llmLoading[metricKey]
              
              return (
                <div className="targeting-recommendation">
                  {isLlmLoading ? (
                    <p className="llm-loading">
                      <MdCheckCircle className="recommendation-icon" />
                      AI가 분석 중입니다...
                    </p>
                  ) : insight ? (
                    <>
                      <p>
                        <MdLightbulb className="insight-icon" />
                        <strong>AI 분석:</strong> {insight.pattern || insight.insight}
                      </p>
                      <p>
                        <MdCheckCircle className="recommendation-icon" />
                        <strong>추천:</strong> {insight.recommendation || metrics.demographic_targeting.targeting_strategy}
                      </p>
                    </>
                  ) : (
                    <p>
                      <MdCheckCircle className="recommendation-icon" />
                      <strong>추천 타겟:</strong> {metrics.demographic_targeting.targeting_strategy}
                    </p>
                  )}
                </div>
              )
            })()}
          </div>
        )}

        {/* 주말/평일 분석 */}
        {weekendData.length > 0 && (
          <div className="metric-card weekend-analysis">
            <h3 className="card-title">
              <MdEvent className="card-title-icon" />
              주말/평일 방문 패턴
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={weekendData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="name" stroke="#64748b" />
                <YAxis stroke="#64748b" />
                <Tooltip formatter={(value) => `${value.toLocaleString()}명`} />
                <Bar dataKey="value" fill="#667eea" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            {metrics?.weekend_analysis && (() => {
              const metricKey = 'weekend_analysis'
              const insight = llmInsights[metricKey]
              const isLlmLoading = llmLoading[metricKey]
              
              return (
                <div className="weekend-insight">
                  {isLlmLoading ? (
                    <p className="llm-loading">
                      <MdLightbulb className="insight-icon" />
                      AI가 분석 중입니다...
                    </p>
                  ) : insight ? (
                    <>
                      <p>
                        <MdLightbulb className="insight-icon" />
                        <strong>AI 분석:</strong> {insight.pattern || insight.insight}
                      </p>
                      <p>
                        <strong>트렌드:</strong> {insight.trend}
                      </p>
                      <p>
                        <MdCheckCircle className="recommendation-icon" />
                        <strong>추천:</strong> {insight.recommendation || metrics.weekend_analysis.recommendation}
                      </p>
                    </>
                  ) : (
                    <>
                      <p>
                        <MdLightbulb className="insight-icon" />
                        <strong>인사이트:</strong> 주말 평균 {metrics.weekend_analysis.weekend_average.toLocaleString()}명, 
                      평일 평균 {metrics.weekend_analysis.weekday_average.toLocaleString()}명으로 
                      주말이 {(metrics.weekend_analysis.weekend_ratio * 100 - 100).toFixed(0)}% 높습니다.
                      </p>
                      <p>
                        <MdCheckCircle className="recommendation-icon" />
                        <strong>추천:</strong> {metrics.weekend_analysis.recommendation}
                      </p>
                    </>
                  )}
                </div>
              )
            })()}
          </div>
        )}

        {/* 출판단지 활성화 지수 */}
        {vitalityData.length > 0 && (
          <div className="metric-card publishing-vitality">
            <h3 className="card-title">
              <MdLocationCity className="card-title-icon" />
              출판단지 활성화 지수
            </h3>
            <div className="vitality-summary">
              <div className="overall-vitality">
                <span className="vitality-label">전체 출판단지 활성화 지수</span>
                <span className="vitality-value">
                  {(vitality.overall_publishing_complex_vitality * 100).toFixed(1)}
                </span>
                <span className="vitality-unit">/ 100</span>
              </div>
              <div className={`vitality-trend ${vitality.trend === '증가' ? 'up' : 'down'}`}>
                <MdTrendingUp className={vitality.trend === '증가' ? 'trend-icon up' : 'trend-icon down'} />
                <span>트렌드: {vitality.trend === '증가' ? '증가' : '감소'}</span>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={vitalityData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" domain={[0, 100]} stroke="#64748b" />
                <YAxis dataKey="region" type="category" stroke="#64748b" width={100} />
                <Tooltip formatter={(value) => `${value}점`} />
                <Legend />
                <Bar dataKey="vitality" fill="#667eea" name="인구활력" radius={[0, 8, 8, 0]} />
                <Bar dataKey="consumption" fill="#764ba2" name="소비활력" radius={[0, 8, 8, 0]} />
                <Bar dataKey="production" fill="#f093fb" name="생산활력" radius={[0, 8, 8, 0]} />
              </BarChart>
            </ResponsiveContainer>
            {(() => {
              const metricKey = 'publishing_vitality'
              const insight = llmInsights[metricKey]
              const isLlmLoading = llmLoading[metricKey]
              
              return vitality.recommendation && (
                <div className="vitality-recommendation">
                  {isLlmLoading ? (
                    <p className="llm-loading">
                      <MdCheckCircle className="recommendation-icon" />
                      AI가 분석 중입니다...
                    </p>
                  ) : insight ? (
                    <>
                      <p>
                        <MdLightbulb className="insight-icon" />
                        <strong>AI 분석:</strong> {insight.explanation}
                      </p>
                      <p>
                        <strong>중요도:</strong> {insight.importance}
                      </p>
                      <p>
                        <MdCheckCircle className="recommendation-icon" />
                        <strong>활용 방안:</strong> {insight.recommendation || vitality.recommendation}
                      </p>
                    </>
                  ) : (
                    <p>
                      <MdCheckCircle className="recommendation-icon" />
                      <strong>추천:</strong> {vitality.recommendation}
                    </p>
                  )}
                </div>
              )
            })()}
          </div>
        )}


        {/* 계절별 패턴 */}
        {seasonalData.length > 0 && (
          <div className="metric-card seasonal-patterns">
            <h3 className="card-title">
              <MdTrendingUp className="card-title-icon" />
              계절별 방문 패턴
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={seasonalData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="season" stroke="#64748b" />
                <YAxis stroke="#64748b" />
                <Tooltip formatter={(value, name) => {
                  if (name === 'visits') return `${value.toLocaleString()}명`
                  if (name === 'score') return `${value}점`
                  return value
                }} />
                <Legend />
                <Line type="monotone" dataKey="visits" stroke="#667eea" strokeWidth={3} name="방문 수" />
                <Line type="monotone" dataKey="score" stroke="#f093fb" strokeWidth={2} name="활성화 점수" />
              </LineChart>
            </ResponsiveContainer>
            {metrics?.seasonal_patterns && (
              <div className="seasonal-insight">
                <p>
                  <MdLightbulb className="insight-icon" />
                  <strong>인사이트:</strong> {metrics.seasonal_patterns.best_season} 시즌에 가장 높은 활성화를 보입니다.
                </p>
                <p>
                  <MdCheckCircle className="recommendation-icon" />
                  <strong>추천:</strong> {metrics.seasonal_patterns.recommendation}
                </p>
              </div>
            )}
          </div>
        )}

        {/* 생활인구 상관관계 - 큐레이션에 직접 필요하지 않으므로 제거 */}
        {/* 통계적 상관관계는 큐레이터가 직접 활용하기 어려운 기술적 지표 */}

        {/* 프로그램 준비도 */}
        {metrics?.program_readiness && (
          <div className="metric-card program-readiness">
            <h3 className="card-title">
              <MdMenuBook className="card-title-icon" />
              문화 프로그램 준비도
            </h3>
            <div className="program-readiness-grid">
              {Object.entries(metrics.program_readiness).map(([program, data]) => (
                <div key={program} className="program-item">
                  <div className="program-header">
                    <span className="program-name">{program}</span>
                    <span className={`program-score ${data.readiness_score >= 70 ? 'high' : data.readiness_score >= 50 ? 'medium' : 'low'}`}>
                      {data.readiness_score.toFixed(0)}점
                    </span>
                  </div>
                  <div className="program-factors">
                    {Object.entries(data.factors).map(([factor, value]) => (
                      <div key={factor} className="factor-item">
                        <span className="factor-name">{factor.replace('_', ' ')}</span>
                        <span className="factor-value">{(value * 100).toFixed(0)}%</span>
                      </div>
                    ))}
                  </div>
                  <div className="program-recommendation">
                    <p>{data.recommendation}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* 리포트 모달 */}
      <LLMReportModal
        isOpen={showReportModal}
        onClose={() => setShowReportModal(false)}
        reports={reports}
      />
    </div>
  )
}

export default MeaningfulMetrics

