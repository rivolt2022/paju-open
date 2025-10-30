import axios from 'axios'

const client = axios.create({ baseURL: '' })

export const api = {
  async getSpots(params?: { time?: string }) {
    const { data } = await client.get('/api/spots', { params })
    return data
  },
  async getGraph(params?: { time?: string; limit?: number }) {
    const { data } = await client.get('/api/graph', { params })
    return data
  },
  async getCourses(theme: string, params?: { time?: string; limit?: number }) {
    const { data } = await client.get('/api/courses', { params: { theme, ...(params || {}) } })
    return data
  }
}


