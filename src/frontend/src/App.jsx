import { useState } from 'react'
import { MdMenuBook } from 'react-icons/md'
import Dashboard from './components/Dashboard'
import './App.css'

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <div className="header-content">
          <div className="logo-section">
            <h1 className="app-title">
              <MdMenuBook className="title-icon" />
              <span className="title-text">PAJU Culture Lab</span>
            </h1>
            <p className="app-subtitle">데이터 기반 문화 콘텐츠 큐레이터 AI</p>
          </div>
          <div className="header-badge">
            <span className="badge-dot"></span>
            <span>AI Powered</span>
          </div>
        </div>
      </header>
      <main className="App-main">
        <Dashboard />
      </main>
      <footer className="App-footer">
        <p>© 2025 PAJU Culture Lab - 출판단지 지역 활성화를 위한 AI 서비스</p>
      </footer>
    </div>
  )
}

export default App