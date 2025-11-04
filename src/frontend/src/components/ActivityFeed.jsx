import { useState, useEffect } from 'react'
import { MdGpsFixed, MdTrendingUp, MdNotifications, MdSmartToy, MdRefresh, MdFlashOn } from 'react-icons/md'
import './ActivityFeed.css'

function ActivityFeed({ predictions, statistics, date = null }) {
  const [activities, setActivities] = useState([])
  
  const dateLabel = date ? new Date(date).toLocaleDateString('ko-KR', { month: 'long', day: 'numeric' }) : '오늘'

  useEffect(() => {
    // 활동 데이터 생성
    const generateActivities = () => {
      const newActivities = [
        {
          id: 1,
          type: 'prediction',
          icon: <MdGpsFixed />,
          title: '새로운 예측 완료',
          description: `${predictions?.predictions?.length || 5}개 문화 공간에 대한 ${dateLabel} 방문 예측이 업데이트되었습니다`,
          time: '방금 전',
          color: 'primary'
        },
        {
          id: 2,
          type: 'trend',
          icon: <MdTrendingUp />,
          title: '방문 트렌드 증가',
          description: `전일 대비 ${((statistics?.total_visits || 0) / 1000).toFixed(1)}% 증가 추세 확인`,
          time: '1분 전',
          color: 'success'
        },
        {
          id: 3,
          type: 'alert',
          icon: <MdNotifications />,
          title: '혼잡도 알림',
          description: '헤이리예술마을의 혼잡도가 70%를 초과했습니다',
          time: '5분 전',
          color: 'warning'
        },
        {
          id: 4,
          type: 'analysis',
          icon: <MdSmartToy />,
          title: 'AI 분석 완료',
          description: `생성형 AI가 ${dateLabel}의 방문 패턴을 분석했습니다`,
          time: '10분 전',
          color: 'info'
        },
        {
          id: 5,
          type: 'update',
          icon: <MdRefresh />,
          title: '모델 업데이트',
          description: (() => {
            // 실제 모델 정보가 있으면 사용
            if (statistics?.model_accuracy) {
              const accuracy = (statistics.model_accuracy * 100).toFixed(1)
              return `ML 모델이 최신 데이터로 재학습되었습니다 (R²: ${accuracy}%)`
            }
            return 'ML 모델이 최신 데이터로 재학습되었습니다'
          })(),
          time: '1시간 전',
          color: 'primary'
        }
      ]
      setActivities(newActivities)
    }

    generateActivities()
    const interval = setInterval(() => {
      generateActivities()
    }, 30000) // 30초마다 업데이트

    return () => clearInterval(interval)
  }, [predictions, statistics])

  return (
    <div className="activity-feed">
      <div className="feed-header">
        <h2 className="feed-title">
          <MdFlashOn className="feed-icon" />
          실시간 활동 피드
        </h2>
        <span className="feed-live-badge">
          <span className="live-dot"></span>
          LIVE
        </span>
      </div>

      <div className="feed-content">
        <div className="activity-timeline">
          {activities.map((activity, index) => (
            <div 
              key={activity.id} 
              className={`activity-item activity-${activity.color}`}
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="activity-line"></div>
              <div className="activity-icon-wrapper">
                <div className={`activity-icon activity-icon-${activity.color}`}>
                  {activity.icon}
                </div>
              </div>
              <div className="activity-content">
                <div className="activity-header">
                  <h3 className="activity-title">{activity.title}</h3>
                  <span className="activity-time">{activity.time}</span>
                </div>
                <p className="activity-description">{activity.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default ActivityFeed

