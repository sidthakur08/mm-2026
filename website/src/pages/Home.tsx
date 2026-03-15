import { useLiveScores } from '@/hooks/useData'
import ScoreCard from '@/components/ScoreCard'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { RefreshCw, Wifi, WifiOff, Calendar } from 'lucide-react'

interface GameData {
  game: {
    gameID: string
    away: {
      names: { short: string; full: string; char6: string }
      score: string
      seed: string
      description: string
      winner: boolean
    }
    home: {
      names: { short: string; full: string; char6: string }
      score: string
      seed: string
      description: string
      winner: boolean
    }
    gameState: string
    startTime: string
    startTimeEpoch: number
    currentPeriod: string
    contestClock: string
    finalMessage: string
    network: string
    url: string
  }
}

function categorizeGames(games: GameData[]) {
  const live: GameData[] = []
  const upcoming: GameData[] = []
  const final_: GameData[] = []

  for (const g of games) {
    const state = g.game.gameState?.toLowerCase() || ''
    const finalMsg = g.game.finalMessage?.toLowerCase() || ''

    if (finalMsg.includes('final') || state.includes('final')) {
      final_.push(g)
    } else if (
      state.includes('live') ||
      state.includes('in progress') ||
      state.includes('half') ||
      (g.game.currentPeriod && !state.includes('pre'))
    ) {
      live.push(g)
    } else {
      upcoming.push(g)
    }
  }

  return { live, upcoming, final: final_ }
}

function GameSection({
  title,
  games,
  status,
  icon,
}: {
  title: string
  games: GameData[]
  status: 'live' | 'upcoming' | 'final'
  icon: React.ReactNode
}) {
  if (games.length === 0) return null

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        {icon}
        <h2 className="text-lg font-bold text-navy-900">{title}</h2>
        <Badge variant="secondary">{games.length}</Badge>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {games.map((g) => (
          <ScoreCard
            key={g.game.gameID}
            away={g.game.away}
            home={g.game.home}
            gameState={g.game.gameState}
            currentPeriod={g.game.currentPeriod}
            contestClock={g.game.contestClock}
            finalMessage={g.game.finalMessage}
            network={g.game.network}
            startTime={g.game.startTime}
            status={status}
          />
        ))}
      </div>
    </div>
  )
}

export default function Home() {
  const { games, loading, error, refetch } = useLiveScores()
  const typedGames = games as GameData[]
  const { live, upcoming, final: finalGames } = categorizeGames(typedGames)
  const today = new Date().toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      {/* Hero */}
      <div className="text-center mb-10">
        <div className="inline-flex items-center gap-2 bg-orange-50 text-orange-700 px-4 py-1.5 rounded-full text-sm font-medium mb-4">
          <span>🏀</span>
          <span>Tournament Season 2026</span>
        </div>
        <h1 className="text-4xl sm:text-5xl font-extrabold text-navy-900 tracking-tight">
          March Madness 2026
        </h1>
        <p className="mt-3 text-lg text-gray-500 max-w-2xl mx-auto">
          ML-powered predictions for every NCAA tournament matchup.
          Live scores, bracket analysis, and model insights.
        </p>
      </div>

      {/* Live Scores Section */}
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h2 className="text-2xl font-bold text-navy-900">Today's Games</h2>
            <div className="flex items-center gap-1.5 text-xs text-gray-400">
              <Calendar className="h-3.5 w-3.5" />
              {today}
            </div>
          </div>
          <button
            onClick={refetch}
            className="flex items-center gap-2 px-3 py-1.5 text-sm text-gray-500 hover:text-navy-700 rounded-lg hover:bg-gray-100 transition-colors"
            disabled={loading}
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>

        {loading && typedGames.length === 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {[1, 2, 3].map((i) => (
              <Card key={i} className="animate-pulse">
                <CardContent className="p-6">
                  <div className="space-y-3">
                    <div className="h-4 bg-gray-200 rounded w-3/4" />
                    <div className="h-4 bg-gray-200 rounded w-1/2" />
                    <div className="h-4 bg-gray-200 rounded w-2/3" />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        {error && (
          <Card className="border-red-200 bg-red-50">
            <CardContent className="p-6 text-center">
              <WifiOff className="h-8 w-8 text-red-400 mx-auto mb-2" />
              <p className="text-red-700 font-medium">Unable to load live scores</p>
              <p className="text-sm text-red-500 mt-1">
                The NCAA scoreboard API may be temporarily unavailable.
              </p>
              <button
                onClick={refetch}
                className="mt-3 px-4 py-2 bg-red-100 text-red-700 rounded-lg text-sm font-medium hover:bg-red-200 transition-colors"
              >
                Try Again
              </button>
            </CardContent>
          </Card>
        )}

        {!loading && !error && typedGames.length === 0 && (
          <Card>
            <CardContent className="p-12 text-center">
              <div className="text-5xl mb-4">🏀</div>
              <h3 className="text-lg font-semibold text-navy-900 mb-2">
                No Games Scheduled Today
              </h3>
              <p className="text-gray-500 max-w-md mx-auto">
                Check back on game days for live scores, or explore the
                Matchups and Bracket pages to see model predictions.
              </p>
            </CardContent>
          </Card>
        )}

        {!error && (
          <>
            <GameSection
              title="Live Now"
              games={live}
              status="live"
              icon={
                <div className="flex items-center gap-1">
                  <Wifi className="h-4 w-4 text-red-500" />
                  <span className="h-2 w-2 rounded-full bg-red-500 animate-pulse-live" />
                </div>
              }
            />

            <GameSection
              title="Upcoming"
              games={upcoming}
              status="upcoming"
              icon={<Calendar className="h-5 w-5 text-blue-500" />}
            />

            <GameSection
              title="Final"
              games={finalGames}
              status="final"
              icon={<span className="text-lg">🏁</span>}
            />
          </>
        )}

        {/* Auto-refresh indicator */}
        {!error && typedGames.length > 0 && (
          <p className="text-center text-xs text-gray-400">
            Scores auto-refresh every 60 seconds
          </p>
        )}
      </div>
    </div>
  )
}
