import { useState } from 'react'
import { cn } from '@/lib/utils'
import { getRegionName } from '@/lib/utils'
import BracketGameNode from './BracketGameNode'
import type { FullBracket, BracketGame } from '@/lib/types'
import { Trophy, ChevronLeft, ChevronRight } from 'lucide-react'

interface BracketTreeProps {
  bracket: FullBracket
}

/** Round labels for display headers */
const ROUND_NAMES: Record<number, string> = {
  1: 'Round of 64',
  2: 'Round of 32',
  3: 'Sweet 16',
  4: 'Elite 8',
  5: 'Final Four',
  6: 'Championship',
}

const SHORT_ROUND_NAMES: Record<number, string> = {
  1: 'R64',
  2: 'R32',
  3: 'S16',
  4: 'E8',
  5: 'F4',
  6: 'Final',
}

/** Return games for a region filtered by round, in order */
function getRegionRoundGames(bracket: FullBracket, region: string, round: number): BracketGame[] {
  const regionGames = bracket.regions[region] || []
  return regionGames
    .map((id) => bracket.games[id])
    .filter((g) => g.round === round)
}

// ==============================
// DESKTOP BRACKET
// ==============================

function DesktopRegionColumn({
  bracket,
  region,
  round,
  mirrored,
}: {
  bracket: FullBracket
  region: string
  round: number
  mirrored: boolean
}) {
  const games = getRegionRoundGames(bracket, region, round)

  // Vertical spacing increases with each round to align with the tree structure.
  // Each subsequent round needs to span the height of 2x the previous round's game slots.
  const gapClass =
    round === 1
      ? 'gap-1'
      : round === 2
        ? 'gap-[36px]'
        : round === 3
          ? 'gap-[108px]'
          : 'gap-0'

  const paddingClass =
    round === 1
      ? 'py-0'
      : round === 2
        ? 'py-[18px]'
        : round === 3
          ? 'py-[54px]'
          : 'py-[126px]'

  return (
    <div className={cn('flex flex-col', gapClass, paddingClass)}>
      {games.map((game) => (
        <BracketGameNode key={game.gameId} game={game} mirrored={mirrored} />
      ))}
    </div>
  )
}

function DesktopRegionHalf({
  bracket,
  regions,
  mirrored,
}: {
  bracket: FullBracket
  regions: [string, string]
  mirrored: boolean
}) {
  const rounds = [1, 2, 3, 4]
  const displayRounds = mirrored ? [...rounds].reverse() : rounds

  return (
    <div className="flex flex-col">
      {/* Two regions stacked */}
      {regions.map((region) => (
        <div key={region} className="mb-4">
          <div className={cn('mb-2', mirrored ? 'text-right' : 'text-left')}>
            <h3 className="text-sm font-bold text-navy-900">{getRegionName(region)}</h3>
            <p className="text-[10px] text-gray-400">Region {getRegionName(region).charAt(0)}</p>
          </div>

          {/* Round headers */}
          <div className={cn('flex gap-3 mb-1', mirrored && 'flex-row-reverse')}>
            {displayRounds.map((r) => (
              <div key={r} className="w-[155px] text-center">
                <span className="text-[9px] font-semibold text-gray-400 uppercase tracking-wider">
                  {SHORT_ROUND_NAMES[r]}
                </span>
              </div>
            ))}
          </div>

          {/* Game columns */}
          <div className={cn('flex gap-3 items-start', mirrored && 'flex-row-reverse')}>
            {displayRounds.map((r) => (
              <DesktopRegionColumn
                key={r}
                bracket={bracket}
                region={region}
                round={r}
                mirrored={mirrored}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

function DesktopFinalFourCenter({ bracket }: { bracket: FullBracket }) {
  const f4Games = bracket.finalFour.map((id) => bracket.games[id])
  const champGame = bracket.games[bracket.championship]

  return (
    <div className="flex flex-col items-center justify-center gap-4 px-4 min-w-[180px]">
      {/* Final Four header */}
      <div className="text-center mb-1">
        <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">
          Final Four
        </span>
      </div>

      {/* F4 Game 1 (W vs X) */}
      {f4Games[0] && <BracketGameNode game={f4Games[0]} />}

      {/* Championship */}
      <div className="text-center">
        <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">
          Championship
        </span>
      </div>
      {champGame && <BracketGameNode game={champGame} />}

      {/* Champion */}
      <div className="flex flex-col items-center gap-1 mt-1 p-3 bg-gradient-to-b from-amber-50 to-amber-100 border border-amber-300 rounded-lg shadow">
        <Trophy className="h-5 w-5 text-amber-600" />
        <span className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold">
          Champion
        </span>
        <span className="text-sm font-extrabold text-navy-900">
          ({bracket.champion.seed}) {bracket.champion.name}
        </span>
      </div>

      {/* F4 Game 2 (Y vs Z) */}
      {f4Games[1] && <BracketGameNode game={f4Games[1]} />}
    </div>
  )
}

function DesktopBracket({ bracket }: { bracket: FullBracket }) {
  return (
    <div className="overflow-x-auto bracket-scroll">
      <div className="flex items-start justify-center gap-4 min-w-[1400px] px-4">
        {/* Left side: East (top) and South (bottom) — flows left to right */}
        <DesktopRegionHalf bracket={bracket} regions={['W', 'X']} mirrored={false} />

        {/* Center: Final Four + Championship */}
        <DesktopFinalFourCenter bracket={bracket} />

        {/* Right side: West (top) and Midwest (bottom) — flows right to left, E8 closest to center */}
        <DesktopRegionHalf bracket={bracket} regions={['Z', 'Y']} mirrored={true} />
      </div>
    </div>
  )
}

// ==============================
// MOBILE BRACKET
// ==============================

function MobileRegionView({
  bracket,
  region,
}: {
  bracket: FullBracket
  region: string
}) {
  return (
    <div className="space-y-4">
      {[1, 2, 3, 4].map((round) => {
        const games = getRegionRoundGames(bracket, region, round)
        if (games.length === 0) return null
        return (
          <div key={round}>
            <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
              {ROUND_NAMES[round]}
            </h4>
            <div className="grid grid-cols-1 xs:grid-cols-2 gap-2">
              {games.map((game) => (
                <BracketGameNode key={game.gameId} game={game} />
              ))}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function MobileFinalFourView({ bracket }: { bracket: FullBracket }) {
  const f4Games = bracket.finalFour.map((id) => bracket.games[id])
  const champGame = bracket.games[bracket.championship]

  return (
    <div className="space-y-4">
      <div>
        <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
          Final Four
        </h4>
        <div className="grid grid-cols-1 xs:grid-cols-2 gap-2">
          {f4Games.map((game) => (
            <BracketGameNode key={game.gameId} game={game} />
          ))}
        </div>
      </div>

      <div>
        <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
          Championship
        </h4>
        {champGame && <BracketGameNode game={champGame} />}
      </div>

      <div className="flex flex-col items-center gap-1 p-4 bg-gradient-to-b from-amber-50 to-amber-100 border border-amber-300 rounded-lg shadow">
        <Trophy className="h-6 w-6 text-amber-600" />
        <span className="text-xs text-gray-500 uppercase tracking-wider font-semibold">
          Predicted Champion
        </span>
        <span className="text-lg font-extrabold text-navy-900">
          ({bracket.champion.seed}) {bracket.champion.name}
        </span>
      </div>
    </div>
  )
}

/** Mobile tab config: region code + display label, ordered to match ESPN bracket */
const MOBILE_TABS: { code: string; label: string; shortLabel: string }[] = [
  { code: 'W', label: 'East', shortLabel: 'East' },
  { code: 'X', label: 'South', shortLabel: 'South' },
  { code: 'Z', label: 'West', shortLabel: 'West' },
  { code: 'Y', label: 'Midwest', shortLabel: 'MW' },
  { code: 'FF', label: 'Final Four', shortLabel: 'F4' },
]

function MobileBracket({ bracket }: { bracket: FullBracket }) {
  const [activeIdx, setActiveIdx] = useState(0)
  const activeTab = MOBILE_TABS[activeIdx]

  const goNext = () => {
    if (activeIdx < MOBILE_TABS.length - 1) setActiveIdx(activeIdx + 1)
  }
  const goPrev = () => {
    if (activeIdx > 0) setActiveIdx(activeIdx - 1)
  }

  return (
    <div>
      {/* Tab bar */}
      <div className="flex items-center justify-between mb-4 bg-white rounded-lg border border-gray-200 p-1">
        <button
          onClick={goPrev}
          disabled={activeIdx === 0}
          className={cn(
            'p-1.5 rounded transition-colors',
            activeIdx === 0 ? 'text-gray-300 cursor-not-allowed' : 'text-navy-600 hover:bg-gray-100'
          )}
          aria-label="Previous region"
        >
          <ChevronLeft className="h-4 w-4" />
        </button>

        <div className="flex gap-1">
          {MOBILE_TABS.map((tab, i) => (
            <button
              key={tab.code}
              onClick={() => setActiveIdx(i)}
              className={cn(
                'px-3 py-1.5 text-xs font-semibold rounded transition-colors',
                activeIdx === i
                  ? 'bg-navy-800 text-white'
                  : 'text-gray-500 hover:bg-gray-100'
              )}
            >
              {tab.shortLabel}
            </button>
          ))}
        </div>

        <button
          onClick={goNext}
          disabled={activeIdx === MOBILE_TABS.length - 1}
          className={cn(
            'p-1.5 rounded transition-colors',
            activeIdx === MOBILE_TABS.length - 1
              ? 'text-gray-300 cursor-not-allowed'
              : 'text-navy-600 hover:bg-gray-100'
          )}
          aria-label="Next region"
        >
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>

      {/* Region label */}
      <div className="mb-3">
        <h3 className="text-lg font-bold text-navy-900">{activeTab.label}</h3>
        {activeTab.code !== 'FF' && (
          <p className="text-xs text-gray-400">Region {activeTab.label.charAt(0)}</p>
        )}
      </div>

      {/* Content */}
      {activeTab.code === 'FF' ? (
        <MobileFinalFourView bracket={bracket} />
      ) : (
        <MobileRegionView bracket={bracket} region={activeTab.code} />
      )}
    </div>
  )
}

// ==============================
// MAIN COMPONENT
// ==============================

export default function BracketTree({ bracket }: BracketTreeProps) {
  return (
    <>
      {/* Desktop: hidden on small screens */}
      <div className="hidden lg:block">
        <DesktopBracket bracket={bracket} />
      </div>

      {/* Mobile: shown on small screens */}
      <div className="lg:hidden">
        <MobileBracket bracket={bracket} />
      </div>
    </>
  )
}
