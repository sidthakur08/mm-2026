import BracketMatchup from './BracketMatchup'
import type { RegionData } from '@/lib/types'

interface BracketRegionProps {
  region: RegionData
}

export default function BracketRegion({ region }: BracketRegionProps) {
  const upsetCount = region.matchups.filter((m) => m.isUpset).length

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 sm:p-6">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h3 className="text-lg font-bold text-navy-900">{region.name}</h3>
          <p className="text-xs text-gray-400">Region {region.code}</p>
        </div>
        {upsetCount > 0 && (
          <span className="text-xs font-semibold text-orange-600 bg-orange-50 px-2.5 py-1 rounded-full">
            {upsetCount} potential upset{upsetCount > 1 ? 's' : ''}
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {region.matchups.map((matchup, i) => (
          <BracketMatchup key={i} matchup={matchup} />
        ))}
      </div>
    </div>
  )
}
