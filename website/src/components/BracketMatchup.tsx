import { useState } from 'react'
import { cn, formatProbability } from '@/lib/utils'
import type { BracketMatchupData } from '@/lib/types'

interface BracketMatchupProps {
  matchup: BracketMatchupData
}

export default function BracketMatchup({ matchup }: BracketMatchupProps) {
  const [showDetails, setShowDetails] = useState(false)

  const {
    topTeamName,
    bottomTeamName,
    topSeed,
    bottomSeed,
    topWinProb,
    bottomWinProb,
    isUpset,
    topPlayIn,
    bottomPlayIn,
  } = matchup

  const topFavored = topWinProb >= 0.5
  const upsetAlert = isUpset && (topSeed < bottomSeed ? bottomWinProb > 0.4 : topWinProb > 0.4)

  return (
    <div className="relative">
      <div
        className={cn(
          'w-56 border rounded-lg overflow-hidden cursor-pointer transition-all duration-200 hover:shadow-md',
          upsetAlert
            ? 'border-orange-300 bg-orange-50/50'
            : 'border-gray-200 bg-white'
        )}
        onClick={() => setShowDetails(!showDetails)}
      >
        {/* Top Team */}
        <div
          className={cn(
            'flex items-center justify-between px-3 py-2 border-b border-gray-100',
            topFavored && 'bg-green-50/50'
          )}
        >
          <div className="flex items-center gap-2 min-w-0 flex-1">
            <span className="text-xs font-bold text-gray-400 w-5 text-center shrink-0">
              {topSeed}
            </span>
            <span
              className={cn(
                'text-xs font-medium truncate',
                topFavored ? 'text-navy-900 font-semibold' : 'text-gray-600'
              )}
              title={topPlayIn || topTeamName}
            >
              {topPlayIn || topTeamName}
            </span>
          </div>
          <span
            className={cn(
              'text-xs font-bold tabular-nums shrink-0 ml-1',
              topFavored ? 'text-green-600' : 'text-gray-400'
            )}
          >
            {formatProbability(topWinProb)}
          </span>
        </div>

        {/* Bottom Team */}
        <div
          className={cn(
            'flex items-center justify-between px-3 py-2',
            !topFavored && 'bg-green-50/50'
          )}
        >
          <div className="flex items-center gap-2 min-w-0 flex-1">
            <span className="text-xs font-bold text-gray-400 w-5 text-center shrink-0">
              {bottomSeed}
            </span>
            <span
              className={cn(
                'text-xs font-medium truncate',
                !topFavored ? 'text-navy-900 font-semibold' : 'text-gray-600'
              )}
              title={bottomPlayIn || bottomTeamName}
            >
              {bottomPlayIn || bottomTeamName}
            </span>
          </div>
          <span
            className={cn(
              'text-xs font-bold tabular-nums shrink-0 ml-1',
              !topFavored ? 'text-green-600' : 'text-gray-400'
            )}
          >
            {formatProbability(bottomWinProb)}
          </span>
        </div>

        {/* Upset indicator */}
        {upsetAlert && (
          <div className="bg-orange-100 px-3 py-1 text-center">
            <span className="text-[10px] font-bold text-orange-700 uppercase tracking-wider">
              Upset Alert
            </span>
          </div>
        )}
      </div>

      {/* Details popup */}
      {showDetails && (
        <div className="absolute z-10 top-full mt-2 left-1/2 -translate-x-1/2 w-64 bg-white rounded-lg shadow-xl border border-gray-200 p-4 animate-fade-in">
          <div className="space-y-3">
            <div className="text-center">
              <p className="text-xs text-gray-400 uppercase tracking-wider font-medium">
                Win Probability
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-navy-800">
                  ({topSeed}) {topPlayIn || topTeamName}
                </span>
                <span className={cn('text-sm font-bold', topFavored ? 'text-green-600' : 'text-gray-500')}>
                  {formatProbability(topWinProb)}
                </span>
              </div>

              <div className="relative h-3 rounded-full overflow-hidden bg-gray-100">
                <div
                  className="absolute inset-y-0 left-0 bg-navy-600 rounded-l-full transition-all duration-500"
                  style={{ width: `${topWinProb * 100}%` }}
                />
                <div
                  className="absolute inset-y-0 right-0 bg-orange-500 rounded-r-full transition-all duration-500"
                  style={{ width: `${bottomWinProb * 100}%` }}
                />
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-navy-800">
                  ({bottomSeed}) {bottomPlayIn || bottomTeamName}
                </span>
                <span className={cn('text-sm font-bold', !topFavored ? 'text-green-600' : 'text-gray-500')}>
                  {formatProbability(bottomWinProb)}
                </span>
              </div>
            </div>

            {isUpset && (
              <div className="text-center pt-1 border-t border-gray-100">
                <p className="text-xs text-orange-600 font-medium">
                  The model favors the lower-seeded team!
                </p>
              </div>
            )}
          </div>

          <button
            className="absolute top-1 right-2 text-gray-400 hover:text-gray-600 text-sm"
            onClick={(e) => {
              e.stopPropagation()
              setShowDetails(false)
            }}
          >
            x
          </button>
        </div>
      )}
    </div>
  )
}
