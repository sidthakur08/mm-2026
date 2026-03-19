import { cn, formatProbability } from '@/lib/utils'
import type { BracketGame } from '@/lib/types'

interface BracketGameNodeProps {
  game: BracketGame
  /** Whether games in this column flow right-to-left (right side of bracket) */
  mirrored?: boolean
}

function getGameBg(winProb: number, isUpset: boolean): string {
  if (isUpset) return 'bg-orange-50 border-orange-300'
  if (winProb > 0.75) return 'bg-green-50 border-green-300'
  if (winProb >= 0.6) return 'bg-green-50/60 border-green-200'
  return 'bg-amber-50/60 border-amber-200'
}

function TeamRow({
  name,
  seed,
  winProb,
  isWinner,
  isTop,
}: {
  name: string
  seed: number
  winProb: number
  isWinner: boolean
  isTop: boolean
}) {
  return (
    <div
      className={cn(
        'flex items-center justify-between px-2 py-1 gap-1',
        isTop && 'border-b border-gray-200/60',
        isWinner ? 'bg-white/80' : 'bg-gray-100/60'
      )}
    >
      <div className="flex items-center gap-1.5 min-w-0 flex-1">
        <span
          className={cn(
            'text-[10px] font-bold w-4 text-center shrink-0 leading-none',
            isWinner ? 'text-navy-700' : 'text-gray-400'
          )}
        >
          {seed}
        </span>
        <span
          className={cn(
            'text-[11px] truncate leading-tight',
            isWinner ? 'text-navy-900 font-bold' : 'text-gray-400 line-through decoration-gray-300'
          )}
          title={name}
        >
          {name}
        </span>
      </div>
      <span
        className={cn(
          'text-[10px] font-semibold tabular-nums shrink-0 leading-none',
          isWinner
            ? winProb > 0.75
              ? 'text-green-700'
              : winProb >= 0.6
                ? 'text-green-600'
                : 'text-amber-700'
            : 'text-gray-400'
        )}
      >
        {formatProbability(winProb)}
      </span>
    </div>
  )
}

export default function BracketGameNode({ game, mirrored }: BracketGameNodeProps) {
  const { topTeam, bottomTeam, topWinProb, predictedWinnerId, isUpset } = game

  if (!topTeam || !bottomTeam) return null

  const topWins = predictedWinnerId === topTeam.id
  const higherProb = topWins ? topWinProb : 1 - topWinProb

  return (
    <div
      className={cn(
        'w-[155px] rounded border shadow-sm overflow-hidden relative',
        getGameBg(higherProb, isUpset),
        mirrored ? 'ml-auto' : ''
      )}
    >
      <TeamRow
        name={topTeam.name}
        seed={topTeam.seed}
        winProb={topWinProb}
        isWinner={topWins}
        isTop={true}
      />
      <TeamRow
        name={bottomTeam.name}
        seed={bottomTeam.seed}
        winProb={1 - topWinProb}
        isWinner={!topWins}
        isTop={false}
      />

      {/* Upset label */}
      {isUpset && (
        <div className="bg-orange-200 text-center py-0.5">
          <span className="text-[8px] font-extrabold text-orange-800 uppercase tracking-widest">
            Upset
          </span>
        </div>
      )}
    </div>
  )
}
