import { useState, useEffect, useRef, useCallback } from 'react'
import { MdFlashOn, MdRefresh, MdPerson, MdCalendarToday, MdSettings, MdPalette, MdCampaign, MdGroup, MdLocationOn } from 'react-icons/md'
import LoadingSpinner from './LoadingSpinner'
import axios from 'axios'
import './ActionItems.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || (import.meta.env.PROD ? '' : 'http://localhost:8000')

function ActionItems({ predictions, statistics, date, onReportAdd }) {
  const [actionItems, setActionItems] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const timeoutRef = useRef(null)
  
  const dateLabel = date ? new Date(date).toLocaleDateString('ko-KR', { month: 'long', day: 'numeric' }) : '오늘'

  const loadActionItems = useCallback(async () => {
    console.log('[ActionItems] loadActionItems 함수 호출됨', { predictions: !!predictions, statistics: !!statistics, date })
    setLoading(true)
    setError(null)
    
    // 타임아웃 설정 (70초 - API 타임아웃보다 약간 더 길게)
    timeoutRef.current = setTimeout(() => {
      console.log('[ActionItems] 타임아웃 발생 - 기본값 사용')
      setLoading(false)
      setError('로딩 시간이 초과되었습니다.')
      // 기본 액션 아이템 표시
      setActionItems([
        {
          id: 1,
          title: '주말 프로그램 확대',
          description: '혼잡도가 높은 시간대에 특별 프로그램 운영으로 방문자 만족도 향상',
          priority: 'High',
          department: '프로그램 기획팀',
          timeline: '이번 주',
          icon: '🎯',
          impact: '높음'
        },
        {
          id: 2,
          title: `${dateLabel} 방문 혜택 마케팅`,
          description: `예상 방문자를 위한 ${dateLabel} 특가 이벤트 공지`,
          priority: 'High',
          department: '마케팅팀',
          timeline: dateLabel,
          icon: '📢',
          impact: '높음'
        }
      ])
    }, 70000)
    
    try {
      console.log('[ActionItems] API 호출 시작', { 
        predictions_count: predictions?.predictions?.length || predictions?.length || 0,
        statistics_keys: Object.keys(statistics || {}),
        date 
      })
      
      // predictions 데이터 구조 처리 (predictions.predictions 또는 predictions 자체)
      let predictionsData = []
      if (Array.isArray(predictions)) {
        predictionsData = predictions
      } else if (predictions && predictions.predictions && Array.isArray(predictions.predictions)) {
        predictionsData = predictions.predictions
      } else if (predictions && typeof predictions === 'object') {
        // 객체인 경우 배열로 변환 시도
        const values = Object.values(predictions).filter(item => item && typeof item === 'object')
        if (values.length > 0) {
          predictionsData = values
        }
      }
      
      console.log('[ActionItems] API 호출 데이터 준비', {
        originalPredictions: predictions,
        processedPredictions: predictionsData,
        predictionsCount: predictionsData.length,
        statistics: statistics,
        date
      })
      
      const response = await axios.post(`${API_BASE_URL}/api/analytics/action-items`, {
        predictions: predictionsData,
        statistics: statistics || {},
        date: date || new Date().toISOString().split('T')[0]
      }, {
        timeout: 120000  // LLM 응답을 위해 120초로 증가
      })
      
      console.log('[ActionItems] API 응답 받음', { 
        has_action_items: !!(response.data && response.data.action_items),
        action_items_count: response.data?.action_items?.length || 0,
        response_data: response.data
      })
      
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
      
      if (response.data && response.data.action_items && response.data.action_items.length > 0) {
        console.log('[ActionItems] LLM 생성 액션 아이템 사용', response.data.action_items)
        setActionItems(response.data.action_items)
        
        // 리포트에 추가
        if (onReportAdd && response.data.action_items.length > 0) {
          onReportAdd({
            title: `실행 가능한 액션 아이템 리포트 (${date})`,
            content: {
              summary: `${response.data.action_items.length}개의 실행 가능한 액션 아이템이 생성되었습니다.`,
              recommendations: response.data.action_items.map(item => 
                `[${item.priority}] ${item.title}: ${item.description} (${item.department}, ${item.timeline})`
              )
            },
            type: 'recommendation',
            metadata: {
              date: date,
              source: '액션 아이템 생성'
            }
          })
        }
      } else {
        console.warn('[ActionItems] API 응답에 action_items가 없음 - 기본값 사용', response.data)
        // 기본 액션 아이템 표시
        setActionItems([
          {
            id: 1,
            title: '주말 프로그램 확대',
            description: '혼잡도가 높은 시간대에 특별 프로그램 운영',
            priority: 'High',
            department: '프로그램 기획팀',
            timeline: '이번 주',
            icon: '🎯',
            impact: '높음'
          },
          {
            id: 2,
            title: `${dateLabel} 방문 혜택 마케팅`,
            description: `예상 방문자를 위한 ${dateLabel} 이벤트 공지`,
            priority: 'High',
            department: '마케팅팀',
            timeline: dateLabel,
            icon: '📢',
            impact: '높음'
          }
        ])
      }
    } catch (err) {
      console.error('[ActionItems] 로드 실패:', err)
      console.error('[ActionItems] 에러 상세:', {
        message: err.message,
        response: err.response?.data,
        status: err.response?.status
      })
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
      // 에러 발생 시에도 기본 액션 아이템 표시
      console.warn('[ActionItems] 에러로 인해 기본값 사용')
      setActionItems([
        {
          id: 1,
          title: '주말 프로그램 확대',
          description: '혼잡도가 높은 시간대에 특별 프로그램 운영',
          priority: 'High',
          department: '프로그램 기획팀',
          timeline: '이번 주',
          icon: '🎯',
          impact: '높음'
        },
        {
          id: 2,
          title: `${dateLabel} 방문 혜택 마케팅`,
          description: `예상 방문자를 위한 ${dateLabel} 이벤트 공지`,
          priority: 'High',
          department: '마케팅팀',
          timeline: dateLabel,
          icon: '📢',
          impact: '높음'
        },
        {
          id: 3,
          title: '혼잡도 관리 강화',
          description: '예측된 혼잡도 높은 공간에 추가 직원 배치',
          priority: 'Medium',
          department: '운영팀',
          timeline: dateLabel,
          icon: '👥',
          impact: '중간'
        }
      ])
      setError(null) // 에러를 표시하지 않고 기본값 사용
    } finally {
      setLoading(false)
    }
  }, [predictions, statistics, date])

  // 컴포넌트 마운트 여부 추적
  const isMountedRef = useRef(false)
  const hasLoadedRef = useRef(false)

  useEffect(() => {
    console.log('[ActionItems] useEffect 실행', { 
      hasPredictions: !!predictions, 
      hasStatistics: !!statistics, 
      date,
      predictionsType: typeof predictions,
      statisticsType: typeof statistics,
      predictionsValue: predictions,
      statisticsValue: statistics,
      isMounted: isMountedRef.current,
      hasLoaded: hasLoadedRef.current
    })
    
    // 마운트 상태 업데이트
    if (!isMountedRef.current) {
      isMountedRef.current = true
    }
    
    // predictions와 statistics가 null이 아니고, 의미 있는 데이터가 있는지 확인
    // 빈 객체 {}도 null로 처리
    const isEmptyObject = (obj) => {
      return obj !== null && typeof obj === 'object' && Object.keys(obj).length === 0
    }
    
    const isNullishOrEmpty = (value) => {
      if (value === null || value === undefined) return true
      if (isEmptyObject(value)) return true
      return false
    }
    
    // predictions 검사
    let hasValidPredictions = false
    if (!isNullishOrEmpty(predictions)) {
      if (Array.isArray(predictions)) {
        hasValidPredictions = predictions.length > 0
      } else if (predictions.predictions && Array.isArray(predictions.predictions)) {
        hasValidPredictions = predictions.predictions.length > 0
      } else if (typeof predictions === 'object') {
        // 객체인 경우 키가 있고 의미 있는 값이 있는지 확인
        const keys = Object.keys(predictions)
        if (keys.length > 0) {
          hasValidPredictions = keys.some(key => {
            const value = predictions[key]
            return value !== null && value !== undefined && 
              !isEmptyObject(value) &&
              (Array.isArray(value) ? value.length > 0 : typeof value === 'object')
          })
        }
      }
    }
    
    // statistics 검사 (빈 객체도 무효로 처리)
    const hasValidStatistics = !isNullishOrEmpty(statistics) && 
      Object.keys(statistics || {}).length > 0
    
    // 디버깅: predictions와 statistics의 실제 값 확인
    const predictionsKeys = predictions ? Object.keys(predictions) : []
    const statisticsKeys = statistics ? Object.keys(statistics) : []
    const predictionsIsEmptyObj = isEmptyObject(predictions)
    const statisticsIsEmptyObj = isEmptyObject(statistics)
    
    console.log('[ActionItems] 데이터 유효성 검사', {
      hasValidPredictions,
      hasValidStatistics,
      predictionsKeys,
      statisticsKeys,
      predictionsIsEmpty: predictionsIsEmptyObj,
      statisticsIsEmpty: statisticsIsEmptyObj,
      predictionsValue: predictions,
      statisticsValue: statistics,
      predictionsIsNull: predictions === null,
      statisticsIsNull: statistics === null,
      predictionsIsUndefined: predictions === undefined,
      statisticsIsUndefined: statistics === undefined,
      predictionsType: typeof predictions,
      statisticsType: typeof statistics,
      predictionsLength: Array.isArray(predictions) ? predictions.length : (predictions?.predictions?.length || 'N/A'),
      statisticsLength: Array.isArray(statistics) ? statistics.length : Object.keys(statistics || {}).length
    })
    
    // 페이지 로드 시점(마운트 후 첫 실행) 또는 데이터가 유효할 때 loadActionItems 호출
    const shouldLoad = !hasLoadedRef.current || (hasValidPredictions && hasValidStatistics)
    
    if (shouldLoad) {
      console.log('[ActionItems] loadActionItems 호출', {
        reason: !hasLoadedRef.current ? '페이지 로드 시점 (초기 로드)' : '데이터 유효 (데이터 변경)',
        hasValidPredictions,
        hasValidStatistics,
        hasLoaded: hasLoadedRef.current
      })
      loadActionItems()
      hasLoadedRef.current = true
    } else {
      console.warn('[ActionItems] loadActionItems 호출 안 함', {
        hasValidPredictions,
        hasValidStatistics,
        hasLoaded: hasLoadedRef.current,
        predictions: predictions ? '있음' : '없음',
        statistics: statistics ? '있음' : '없음'
      })
    }
    
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [predictions, statistics, date, loadActionItems])

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'High':
        return '#ef4444'
      case 'Medium':
        return '#f59e0b'
      case 'Low':
        return '#10b981'
      default:
        return '#6b7280'
    }
  }

  const getPriorityBadge = (priority) => {
    switch (priority) {
      case 'High':
        return (
          <>
            <MdFlashOn className="inline-icon" /> 긴급
          </>
        )
      case 'Medium':
        return (
          <>
            <MdFlashOn className="inline-icon" /> 중요
          </>
        )
      case 'Low':
        return (
          <>
            <MdCalendarToday className="inline-icon" /> 일반
          </>
        )
      default:
        return (
          <>
            <MdCalendarToday className="inline-icon" /> 일반
          </>
        )
    }
  }

  const getActionIcon = (iconName) => {
    const iconMap = {
      '🎯': <MdLocationOn />,
      '🎨': <MdPalette />,
      '📢': <MdCampaign />,
      '👥': <MdGroup />,
      default: <MdSettings />
    }
    return iconMap[iconName] || iconMap.default
  }

  // 로딩 중이고 액션 아이템이 없을 때만 로딩 표시 (최대 3초)
  if (loading && actionItems.length === 0) {
    return (
      <div className="action-items-container">
        <div className="action-items-header">
          <span className="action-items-icon"><MdFlashOn /></span>
          <span>당장 실행할 일</span>
        </div>
        <div className="action-items-loading">
          <LoadingSpinner message="액션 아이템 생성 중..." size="medium" />
        </div>
      </div>
    )
  }

  // 액션 아이템이 없으면 기본 액션 아이템 표시
  if (!loading && actionItems.length === 0) {
    // 기본 액션 아이템 설정
    const defaultActionItems = [
      {
        id: 1,
        title: '주말 프로그램 확대',
        description: '혼잡도가 높은 시간대에 특별 프로그램 운영으로 방문자 만족도 향상',
        priority: 'High',
        department: '프로그램 기획팀',
        timeline: '이번 주',
        icon: '🎯',
        impact: '높음'
      },
      {
        id: 2,
            title: `${dateLabel} 방문 혜택 마케팅`,
            description: `예상 방문자를 위한 ${dateLabel} 특가 이벤트 공지`,
            priority: 'High',
            department: '마케팅팀',
            timeline: dateLabel,
        icon: '📢',
        impact: '높음'
      },
      {
        id: 3,
        title: '혼잡도 관리 강화',
        description: '예측된 혼잡도 높은 공간에 추가 직원 배치 및 대기 공간 확보',
        priority: 'Medium',
        department: '운영팀',
        timeline: dateLabel,
        icon: '👥',
        impact: '중간'
      }
    ]
    
    return (
      <div className="action-items-container">
        <div className="action-items-header">
          <span className="action-items-icon"><MdFlashOn /></span>
          <span>당장 실행할 일</span>
          <button 
            className="action-items-refresh"
            onClick={loadActionItems}
            title="새로고침"
            disabled={loading}
          >
            <MdRefresh />
          </button>
        </div>
        <div className="action-items-list">
          {defaultActionItems.map((item) => (
            <div 
              key={item.id} 
              className={`action-item action-item-${item.priority?.toLowerCase() || 'medium'}`}
            >
              <div className="action-item-icon">{getActionIcon(item.icon || '🎯')}</div>
              <div className="action-item-content">
                <div className="action-item-header">
                  <h4 className="action-item-title">{item.title}</h4>
                  <span 
                    className="action-item-priority"
                    style={{ color: getPriorityColor(item.priority) }}
                  >
                    {getPriorityBadge(item.priority)}
                  </span>
                </div>
                <p className="action-item-description">{item.description}</p>
                <div className="action-item-meta">
                  <span className="action-item-department">
                    <MdPerson className="inline-icon" /> {item.department || '프로그램 기획팀'}
                  </span>
                  <span className="action-item-timeline">
                    <MdCalendarToday className="inline-icon" /> {item.timeline || '이번 주'}
                  </span>
                  {item.impact && (
                    <span className="action-item-impact">
                      <MdFlashOn className="inline-icon" /> 효과: {item.impact}
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="action-items-container">
      <div className="action-items-header">
        <span className="action-items-icon"><MdFlashOn /></span>
        <span>당장 실행할 일</span>
        <button 
          className="action-items-refresh"
          onClick={loadActionItems}
          title="새로고침"
          disabled={loading}
        >
          <MdRefresh />
        </button>
      </div>
      <div className="action-items-list">
        {actionItems.slice(0, 5).map((item) => (
          <div 
            key={item.id} 
            className={`action-item action-item-${item.priority?.toLowerCase() || 'medium'}`}
          >
            <div className="action-item-icon">{getActionIcon(item.icon || '🎯')}</div>
            <div className="action-item-content">
              <div className="action-item-header">
                <h4 className="action-item-title">{item.title}</h4>
                <span 
                  className="action-item-priority"
                  style={{ color: getPriorityColor(item.priority) }}
                >
                  {getPriorityBadge(item.priority)}
                </span>
              </div>
              <p className="action-item-description">{item.description}</p>
              <div className="action-item-meta">
                <span className="action-item-department">
                  <MdPerson className="inline-icon" /> {item.department || '프로그램 기획팀'}
                </span>
                <span className="action-item-timeline">
                  <MdCalendarToday className="inline-icon" /> {item.timeline || '이번 주'}
                </span>
                {item.impact && (
                  <span className="action-item-impact">
                    <MdFlashOn className="inline-icon" /> 효과: {item.impact}
                  </span>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default ActionItems

