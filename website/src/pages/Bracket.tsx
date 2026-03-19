import { useState, useEffect, useMemo } from 'react'
import { useTeams, useSeeds, usePredictions } from '@/hooks/useData'
import { buildFullBracket } from '@/lib/predictions'
import BracketTree from '@/components/BracketTree'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Loader2, AlertTriangle, Trophy } from 'lucide-react'
import type { Gender } from '@/lib/types'

function countUpsets(games: Record<string, { isUpset: boolean }>): number {
  return Object.values(games).filter((g) => g.isUpset).length
}

export default function Bracket() {
  const [gender, setGender] = useState<Gender>('men')

  const { data: teams, loading: teamsLoading } = useTeams()
  const { data: seeds, loading: seedsLoading } = useSeeds()
  const menPredictions = usePredictions('men')
  const womenPredictions = usePredictions('women')

  useEffect(() => {
    if (gender === 'men') {
      menPredictions.load()
    } else {
      womenPredictions.load()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gender])

  const predictions = gender === 'men' ? menPredictions : womenPredictions

  const handleGenderChange = (value: string) => {
    setGender(value as Gender)
  }

  const isLoading = teamsLoading || seedsLoading || predictions.loading
  const hasError = predictions.error

  const bracket = useMemo(() => {
    if (!teams || !seeds || !predictions.data) return null
    return buildFullBracket(seeds, predictions.data, teams, gender)
  }, [teams, seeds, predictions.data, gender])

  const totalUpsets = bracket ? countUpsets(bracket.games) : 0

  return (
    <div className="mx-auto max-w-[1800px] px-4 py-8 sm:px-6">
      <div className="text-center mb-8">
        <h1 className="text-3xl sm:text-4xl font-extrabold text-navy-900 tracking-tight">
          Tournament Bracket
        </h1>
        <p className="mt-2 text-gray-500">
          Full tournament simulation with model-predicted win probabilities
        </p>
      </div>

      {/* Gender Toggle */}
      <div className="flex justify-center mb-6">
        <Tabs value={gender} onValueChange={handleGenderChange}>
          <TabsList>
            <TabsTrigger value="men">Men&apos;s</TabsTrigger>
            <TabsTrigger value="women">Women&apos;s</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      {isLoading ? (
        <div className="flex flex-col items-center justify-center py-20">
          <Loader2 className="h-8 w-8 animate-spin text-navy-400 mb-3" />
          <p className="text-gray-500 text-sm">Loading bracket data...</p>
        </div>
      ) : hasError ? (
        <Card className="border-red-200 bg-red-50 max-w-lg mx-auto">
          <CardContent className="p-6 text-center">
            <AlertTriangle className="h-8 w-8 text-red-400 mx-auto mb-2" />
            <p className="text-red-700 font-medium">Failed to load predictions</p>
            <p className="text-sm text-red-500 mt-1">{predictions.error}</p>
          </CardContent>
        </Card>
      ) : bracket ? (
        <>
          {/* Summary badges */}
          <div className="flex flex-wrap items-center justify-center gap-3 mb-6">
            <Badge variant="success" className="text-sm px-3 py-1">
              <Trophy className="h-3.5 w-3.5 mr-1.5" />
              Predicted Champion: ({bracket.champion.seed}) {bracket.champion.name}
            </Badge>
            {totalUpsets > 0 && (
              <Badge variant="warning" className="text-sm px-3 py-1">
                <AlertTriangle className="h-3.5 w-3.5 mr-1.5" />
                {totalUpsets} predicted upset{totalUpsets > 1 ? 's' : ''}
              </Badge>
            )}
          </div>

          {/* Bracket Tree */}
          <BracketTree bracket={bracket} />

          {/* Legend */}
          <div className="mt-8 flex flex-wrap items-center justify-center gap-4 text-xs text-gray-500">
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 bg-green-50 border border-green-300 rounded" />
              <span>{'Strong favorite (>75%)'}</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 bg-green-50/60 border border-green-200 rounded" />
              <span>Moderate favorite (60-75%)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 bg-amber-50 border border-amber-200 rounded" />
              <span>Toss-up (40-60%)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 bg-orange-50 border border-orange-300 rounded" />
              <span>Upset pick</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="font-bold text-navy-900">Bold</span>
              <span>= predicted winner</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-gray-400 line-through">Dimmed</span>
              <span>= predicted loser</span>
            </div>
          </div>
        </>
      ) : null}
    </div>
  )
}
