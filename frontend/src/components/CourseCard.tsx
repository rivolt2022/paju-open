import React, { useEffect, useState } from 'react'
import { api } from '../lib/api'

export const CourseCard: React.FC<{ theme: 'family' | 'couple' | 'night' | string }> = ({ theme }) => {
    const [courses, setCourses] = useState<any[]>([])

    useEffect(() => {
        api.getCourses(theme).then((r) => setCourses(r.courses || []))
    }, [theme])

    return (
        <div style={{ border: '1px solid #ddd', borderRadius: 8, padding: 12, marginBottom: 12 }}>
            <h3 style={{ marginTop: 0 }}>{theme}</h3>
            {courses.length === 0 ? (
                <div style={{ color: '#666' }}>코스를 불러오는 중이거나 데이터가 없습니다.</div>
            ) : (
                courses.map((c, idx) => (
                    <div key={idx} style={{ padding: '8px 0', borderTop: idx ? '1px dashed #eee' : 'none' }}>
                        <div style={{ fontWeight: 600 }}>경로 #{idx + 1} {typeof c.value === 'number' ? `(A=${c.value.toFixed(2)})` : ''}</div>
                        <div>{c.path.map((n: any) => n.spot_id).join(' → ')}</div>
                    </div>
                ))
            )}
        </div>
    )
}


