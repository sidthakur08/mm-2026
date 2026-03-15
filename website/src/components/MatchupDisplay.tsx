import { Card, CardContent } from '@/components/ui/card'
import StatBar from './StatBar'
import { formatProbability } from '@/lib/utils'
import type { TeamStats, Gender } from '@/lib/types'

interface MatchupDisplayProps {
  teamAName: string
  teamBName: string
  teamAStats: TeamStats
  teamBStats: TeamStats
  winProbA: number
  gender: Gender
}

export default function MatchupDisplay({
  teamAName,
  teamBName,
  teamAStats,
  teamBStats,
  winProbA,
  gender,
}: MatchupDisplayProps) {
  const winProbB = 1 - winProbA
  const favoredTeam = winProbA >= winProbB ? teamAName : teamBName
  const favoredProb = Math.max(winProbA, winProbB)

  return (
    <div className="space-y-6 animate-slide-up">
      {/* Probability Display */}
      <Card className="overflow-hidden">
        <div className="bg-gradient-to-r from-navy-900 via-navy-800 to-navy-900 p-8 text-center">
          <p className="text-navy-200 text-sm font-medium mb-2 uppercase tracking-wider">
            Predicted Winner
          </p>
          <h2 className="text-3xl sm:text-4xl font-bold text-white mb-1">
            {favoredTeam}
          </h2>
          <p className="text-orange-400 text-2xl sm:text-3xl font-bold">
            {formatProbability(favoredProb)}
          </p>
        </div>

        <CardContent className="pt-6">
          {/* Probability Bar */}
          <div className="space-y-3">
            <div className="flex justify-between text-sm font-semibold">
              <span className="text-navy-700">{teamAName}</span>
              <span className="text-orange-600">{teamBName}</span>
            </div>

            <div className="relative h-8 rounded-full overflow-hidden bg-gray-100">
              <div
                className="absolute inset-y-0 left-0 bg-gradient-to-r from-navy-700 to-navy-500 rounded-l-full transition-all duration-700 ease-out flex items-center justify-end pr-2"
                style={{ width: `${winProbA * 100}%` }}
              >
                {winProbA >= 0.15 && (
                  <span className="text-xs font-bold text-white">
                    {formatProbability(winProbA)}
                  </span>
                )}
              </div>
              <div
                className="absolute inset-y-0 right-0 bg-gradient-to-l from-orange-500 to-orange-400 rounded-r-full transition-all duration-700 ease-out flex items-center justify-start pl-2"
                style={{ width: `${winProbB * 100}%` }}
              >
                {winProbB >= 0.15 && (
                  <span className="text-xs font-bold text-white">
                    {formatProbability(winProbB)}
                  </span>
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Stats Comparison */}
      <Card>
        <div className="px-6 py-4 border-b border-gray-100">
          <div className="flex items-center justify-between">
            <span className="font-bold text-navy-700">{teamAName}</span>
            <span className="text-sm font-medium text-gray-400 uppercase tracking-wider">
              Head to Head
            </span>
            <span className="font-bold text-orange-600">{teamBName}</span>
          </div>
        </div>

        <CardContent className="pt-4">
          <StatBar
            label="Win %"
            valueA={teamAStats.winPct}
            valueB={teamBStats.winPct}
            format={(v) => `${(v * 100).toFixed(1)}%`}
          />
          <StatBar
            label="PPG"
            valueA={teamAStats.ppg}
            valueB={teamBStats.ppg}
            format={(v) => v.toFixed(1)}
          />
          <StatBar
            label="Opp PPG"
            valueA={teamAStats.oppPpg}
            valueB={teamBStats.oppPpg}
            format={(v) => v.toFixed(1)}
            higherIsBetter={false}
          />
          <StatBar
            label="Off. Eff."
            valueA={teamAStats.offEfficiency}
            valueB={teamBStats.offEfficiency}
            format={(v) => v.toFixed(1)}
          />
          <StatBar
            label="Def. Eff."
            valueA={teamAStats.defEfficiency}
            valueB={teamBStats.defEfficiency}
            format={(v) => v.toFixed(1)}
            higherIsBetter={false}
          />
          <StatBar
            label="SOS"
            valueA={teamAStats.sos}
            valueB={teamBStats.sos}
            format={(v) => v.toFixed(4)}
          />
          {gender === 'men' && teamAStats.kenpomRank != null && teamBStats.kenpomRank != null && (
            <StatBar
              label="KenPom"
              valueA={teamAStats.kenpomRank}
              valueB={teamBStats.kenpomRank}
              format={(v) => `#${Math.round(v)}`}
              higherIsBetter={false}
            />
          )}
          <StatBar
            label="TS%"
            valueA={teamAStats.trueShooting}
            valueB={teamBStats.trueShooting}
            format={(v) => `${(v * 100).toFixed(1)}%`}
          />
          <StatBar
            label="TO Rate"
            valueA={teamAStats.turnoverRate}
            valueB={teamBStats.turnoverRate}
            format={(v) => `${(v * 100).toFixed(1)}%`}
            higherIsBetter={false}
          />
        </CardContent>
      </Card>
    </div>
  )
}
