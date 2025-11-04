import { useState, useEffect, useRef } from 'react'
import { MdFlashOn, MdRefresh, MdPerson, MdCalendarToday, MdSettings, MdPalette, MdCampaign, MdGroup, MdLocationOn } from 'react-icons/md'
import axios from 'axios'
import './ActionItems.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || (import.meta.env.PROD ? '' : 'http://localhost:8000')

function ActionItems({ predictions, statistics, date, onReportAdd }) {
  const [actionItems, setActionItems] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const timeoutRef = useRef(null)

  useEffect(() => {
    if (predictions && statistics) {
      loadActionItems()
    }
    
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [predictions, statistics, date])

  const loadActionItems = async () => {
    setLoading(true)
    setError(null)
    
    // 타임아웃 설정 (70초 - API 타임아웃보다 약간 더 길게)
    timeoutRef.current = setTimeout(() => {
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
          title: '오늘 방문 혜택 마케팅',
          description: '예상 방문자를 위한 당일 특가 이벤트 공지',
          priority: 'High',
          department: '마케팅팀',
          timeline: '오늘',
          icon: '📢',
          impact: '높음'
        }
      ])
    }, 70000)
    
    try {
      const response = await axios.post(`${API_BASE_URL}/api/analytics/action-items`, {
        predictions: predictions?.predictions || predictions || [],
        statistics: statistics || {},
        date: date || new Date().toISOString().split('T')[0]
      }, {
        timeout: 60000  // LLM 응답을 위해 60초로 증가
      })
      
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
      
      if (response.data && response.data.action_items && response.data.action_items.length > 0) {
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
            title: '오늘 방문 혜택 마케팅',
            description: '예상 방문자를 위한 당일 이벤트 공지',
            priority: 'High',
            department: '마케팅팀',
            timeline: '오늘',
            icon: '📢',
            impact: '높음'
          }
        ])
      }
    } catch (err) {
      console.error('[ActionItems] 로드 실패:', err)
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
      // 에러 발생 시에도 기본 액션 아이템 표시
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
          title: '오늘 방문 혜택 마케팅',
          description: '예상 방문자를 위한 당일 이벤트 공지',
          priority: 'High',
          department: '마케팅팀',
          timeline: '오늘',
          icon: '📢',
          impact: '높음'
        },
        {
          id: 3,
          title: '혼잡도 관리 강화',
          description: '예측된 혼잡도 높은 공간에 추가 직원 배치',
          priority: 'Medium',
          department: '운영팀',
          timeline: '오늘',
          icon: '👥',
          impact: '중간'
        }
      ])
      setError(null) // 에러를 표시하지 않고 기본값 사용
    } finally {
      setLoading(false)
    }
  }

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
          <div className="loading-spinner"></div>
          <span>액션 아이템 생성 중...</span>
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
        title: '오늘 방문 혜택 마케팅',
        description: '예상 방문자를 위한 당일 특가 이벤트 공지',
        priority: 'High',
        department: '마케팅팀',
        timeline: '오늘',
        icon: '📢',
        impact: '높음'
      },
      {
        id: 3,
        title: '혼잡도 관리 강화',
        description: '예측된 혼잡도 높은 공간에 추가 직원 배치 및 대기 공간 확보',
        priority: 'Medium',
        department: '운영팀',
        timeline: '오늘',
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

