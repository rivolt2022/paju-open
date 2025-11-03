import { useState } from 'react'
import './GeneratedContent.css'

function GeneratedContent({ content, loading, onGenerate }) {
  const [userInfo, setUserInfo] = useState({
    age: 30,
    gender: 'female',
    preferences: ['문학', '예술'],
  })

  const handleGenerate = () => {
    if (onGenerate) {
      onGenerate(userInfo)
    }
  }

  return (
    <div className="generated-content">
      <div className="user-input-section">
        <h3>사용자 정보 입력</h3>
        <div className="input-group">
          <label>
            연령:
            <input
              type="number"
              value={userInfo.age}
              onChange={(e) => setUserInfo({ ...userInfo, age: parseInt(e.target.value) })}
              min="10"
              max="80"
            />
          </label>
        </div>
        <div className="input-group">
          <label>
            성별:
            <select
              value={userInfo.gender}
              onChange={(e) => setUserInfo({ ...userInfo, gender: e.target.value })}
            >
              <option value="male">남성</option>
              <option value="female">여성</option>
            </select>
          </label>
        </div>
        <div className="input-group">
          <label>
            선호 활동:
            <input
              type="text"
              value={userInfo.preferences.join(', ')}
              onChange={(e) =>
                setUserInfo({
                  ...userInfo,
                  preferences: e.target.value.split(',').map((s) => s.trim()).filter(s => s),
                })
              }
              placeholder="문학, 예술, 독서"
            />
          </label>
        </div>
        <button
          className="generate-button"
          onClick={handleGenerate}
          disabled={loading}
        >
          {loading ? '생성 중...' : '문화 여정 생성'}
        </button>
      </div>

      {content && (
        <div className="content-display">
          <h3>{content.title}</h3>
          <p className="description">{content.description}</p>
          
          {(content.journey || content.course) && Array.isArray(content.journey || content.course) && (
            <div className="course-section">
              <h4>추천 문화 여정</h4>
              {(content.journey || content.course).map((item, index) => (
                <div key={index} className="course-item">
                  <div className="course-time">{item.time}</div>
                  <div className="course-place">{item.place || item.program}</div>
                  {item.program && (
                    <div className="course-program">📚 프로그램: {item.program}</div>
                  )}
                  <div className="course-reason">{item.reason}</div>
                  {item.tip && <div className="course-tip">💡 {item.tip}</div>}
                </div>
              ))}
            </div>
          )}

          {content.story && (
            <div className="story-section">
              <h4>문화 스토리</h4>
              <p>{content.story}</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default GeneratedContent