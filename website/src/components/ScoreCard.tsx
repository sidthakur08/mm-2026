import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

interface GameTeam {
  names: { short: string; full: string; char6: string }
  score: string
  seed: string
  description: string
  winner: boolean
}

interface ScoreCardProps {
  away: GameTeam
  home: GameTeam
  gameState: string
  currentPeriod: string
  contestClock: string
  finalMessage: string
  network: string
  startTime: string
  status: 'live' | 'upcoming' | 'final'
}

function TeamRow({ team, isWinner, isFinal }: { team: GameTeam; isWinner: boolean; isFinal: boolean }) {
  return (
    <div
      className={cn(
        'flex items-center justify-between px-4 py-2.5',
        isWinner && isFinal && 'bg-green-50'
      )}
    >
      <div className="flex items-center gap-3">
        {team.seed && (
          <span className="text-xs font-semibold text-gray-400 w-5 text-right">
            {team.seed}
          </span>
        )}
        <span
          className={cn(
            'text-sm font-medium',
            isWinner && isFinal ? 'text-navy-900 font-bold' : 'text-gray-700'
          )}
        >
          {team.names?.short || team.names?.char6 || 'TBD'}
        </span>
      </div>
      <span
        className={cn(
          'text-lg font-bold tabular-nums',
          isWinner && isFinal ? 'text-navy-900' : 'text-gray-600'
        )}
      >
        {team.score || '-'}
      </span>
    </div>
  )
}

export default function ScoreCard({
  away,
  home,
  gameState,
  currentPeriod,
  contestClock,
  finalMessage,
  network,
  startTime,
  status,
}: ScoreCardProps) {
  const isFinal = status === 'final'
  const isLive = status === 'live'

  return (
    <Card className="overflow-hidden hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-100">
        <div className="flex items-center gap-2">
          {isLive && (
            <Badge variant="destructive" className="gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-white animate-pulse-live" />
              LIVE
            </Badge>
          )}
          {isFinal && (
            <Badge variant="secondary">
              {finalMessage || 'Final'}
            </Badge>
          )}
          {status === 'upcoming' && (
            <Badge variant="outline">
              {new Date(startTime).toLocaleTimeString([], {
                hour: 'numeric',
                minute: '2-digit',
              })}
            </Badge>
          )}
          {isLive && currentPeriod && (
            <span className="text-xs text-gray-500">
              {currentPeriod} {contestClock}
            </span>
          )}
        </div>
        {network && (
          <span className="text-xs text-gray-400 font-medium">{network}</span>
        )}
      </div>

      <div className="divide-y divide-gray-100">
        <TeamRow team={away} isWinner={away.winner} isFinal={isFinal} />
        <TeamRow team={home} isWinner={home.winner} isFinal={isFinal} />
      </div>

      {isLive && gameState && (
        <div className="px-4 py-1.5 bg-red-50 border-t border-red-100">
          <p className="text-xs text-red-600 font-medium text-center">{gameState}</p>
        </div>
      )}
    </Card>
  )
}
