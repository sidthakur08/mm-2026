import { useState, useEffect, useRef } from 'react'
import type { TeamsData, SeedsData, TeamStatsData, ModelInfoData, PredictionsData } from '@/lib/types'

type DataState<T> = {
  data: T | null
  loading: boolean
  error: string | null
}

function useJsonData<T>(url: string, lazy = false): DataState<T> & { load: () => void } {
  const [state, setState] = useState<DataState<T>>({
    data: null,
    loading: !lazy,
    error: null,
  })
  const loaded = useRef(false)

  const load = () => {
    if (loaded.current) return
    loaded.current = true
    setState((prev) => ({ ...prev, loading: true }))

    fetch(url)
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to fetch ${url}`)
        return res.json()
      })
      .then((data: T) => {
        setState({ data, loading: false, error: null })
      })
      .catch((err: Error) => {
        setState({ data: null, loading: false, error: err.message })
      })
  }

  useEffect(() => {
    if (!lazy) {
      load()
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, lazy])

  return { ...state, load }
}

export function useTeams() {
  return useJsonData<TeamsData>('/data/teams.json')
}

export function useSeeds() {
  return useJsonData<SeedsData>('/data/seeds.json')
}

export function useTeamStats() {
  return useJsonData<TeamStatsData>('/data/team_stats.json')
}

export function useModelInfo() {
  return useJsonData<ModelInfoData>('/data/model_info.json')
}

export function usePredictions(gender: 'men' | 'women') {
  return useJsonData<PredictionsData>(`/data/predictions_${gender}.json`, true)
}

export function useLiveScores() {
  const [games, setGames] = useState<unknown[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchScores = () => {
    setLoading(true)

    fetch('/api/ncaa-live-scores')
      .then((res) => {
        if (!res.ok) throw new Error('Failed to fetch scores')
        return res.json()
      })
      .then((data) => {
        const gamesList = data?.games || []
        setGames(gamesList)
        setLoading(false)
        setError(null)
      })
      .catch((err: Error) => {
        setError(err.message)
        setLoading(false)
      })
  }

  useEffect(() => {
    fetchScores()
    const interval = setInterval(fetchScores, 60000)
    return () => clearInterval(interval)
  }, [])

  return { games, loading, error, refetch: fetchScores }
}
