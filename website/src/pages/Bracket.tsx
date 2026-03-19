import { useState, useEffect } from 'react'
import { useTeams, useSeeds, usePredictions } from '@/hooks/useData'
import { buildBracketData } from '@/lib/predictions'
import BracketRegion from '@/components/BracketRegion'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Loader2, AlertTriangle } from 'lucide-react'
import type { Gender } from '@/lib/types'

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

  const regions =
    teams && seeds && predictions.data
      ? buildBracketData(seeds, predictions.data, teams, gender)
      : []

  const totalUpsets = regions.reduce(
    (sum, r) => sum + r.matchups.filter((m) => m.isUpset).length,
    0
  )

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <div className="text-center mb-8">
        <h1 className="text-3xl sm:text-4xl font-extrabold text-navy-900 tracking-tight">
          Tournament Bracket
        </h1>
        <p className="mt-2 text-gray-500">
          First round matchups with model-predicted win probabilities
        </p>
      </div>

      {/* Gender Toggle */}
      <div className="flex justify-center mb-6">
        <Tabs value={gender} onValueChange={handleGenderChange}>
          <TabsList>
            <TabsTrigger value="men">Men's</TabsTrigger>
            <TabsTrigger value="women">Women's</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      {isLoading ? (
        <div className="flex flex-col items-center justify-center py-20">
          <Loader2 className="h-8 w-8 animate-spin text-navy-400 mb-3" />
          <p className="text-gray-500 text-sm">Loading bracket data...</p>
        </div>
      ) : hasError ? (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="p-6 text-center">
            <AlertTriangle className="h-8 w-8 text-red-400 mx-auto mb-2" />
            <p className="text-red-700 font-medium">Failed to load predictions</p>
            <p className="text-sm text-red-500 mt-1">{predictions.error}</p>
          </CardContent>
        </Card>
      ) : (
        <>
          {/* Summary */}
          {totalUpsets > 0 && (
            <div className="flex items-center justify-center gap-2 mb-6">
              <Badge variant="warning" className="text-sm px-3 py-1">
                <AlertTriangle className="h-3.5 w-3.5 mr-1.5" />
                {totalUpsets} potential upset{totalUpsets > 1 ? 's' : ''} in the first round
              </Badge>
            </div>
          )}

          {/* Bracket Display */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {regions.map((region) => (
              <BracketRegion key={region.code} region={region} />
            ))}
          </div>

          {/* Legend */}
          <div className="mt-8 flex flex-wrap items-center justify-center gap-6 text-sm text-gray-500">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-green-50 border border-green-200 rounded" />
              <span>Favored team</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-orange-50 border border-orange-200 rounded" />
              <span>{'Upset alert (underdog >40%)'}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-green-600 font-bold text-xs">65.2%</span>
              <span>Higher probability = more confident</span>
            </div>
          </div>

          <p className="text-center text-xs text-gray-400 mt-4">
            Click any matchup to see probability details
          </p>
        </>
      )}
    </div>
  )
}
