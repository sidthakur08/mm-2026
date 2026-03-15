import { useState, useEffect } from 'react'
import { useTeams, useTeamStats, usePredictions } from '@/hooks/useData'
import TeamSelector from '@/components/TeamSelector'
import MatchupDisplay from '@/components/MatchupDisplay'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent } from '@/components/ui/card'
import { getWinProbability } from '@/lib/utils'
import { ArrowLeftRight, Loader2 } from 'lucide-react'
import type { Gender } from '@/lib/types'

export default function Matchups() {
  const [gender, setGender] = useState<Gender>('men')
  const [teamAId, setTeamAId] = useState<string | null>(null)
  const [teamBId, setTeamBId] = useState<string | null>(null)

  const { data: teams, loading: teamsLoading } = useTeams()
  const { data: teamStats, loading: statsLoading } = useTeamStats()
  const menPredictions = usePredictions('men')
  const womenPredictions = usePredictions('women')

  // Load predictions when needed
  useEffect(() => {
    if (gender === 'men') {
      menPredictions.load()
    } else {
      womenPredictions.load()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gender])

  const predictions = gender === 'men' ? menPredictions : womenPredictions
  const genderTeams = teams?.[gender] || {}
  const genderStats = teamStats?.[gender] || {}

  const handleGenderChange = (value: string) => {
    setGender(value as Gender)
    setTeamAId(null)
    setTeamBId(null)
  }

  const handleSwap = () => {
    setTeamAId(teamBId)
    setTeamBId(teamAId)
  }

  const isLoading = teamsLoading || statsLoading || predictions.loading

  const canShowMatchup =
    teamAId &&
    teamBId &&
    teamAId !== teamBId &&
    predictions.data &&
    genderStats[teamAId] &&
    genderStats[teamBId]

  return (
    <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6 lg:px-8">
      <div className="text-center mb-8">
        <h1 className="text-3xl sm:text-4xl font-extrabold text-navy-900 tracking-tight">
          Matchup Explorer
        </h1>
        <p className="mt-2 text-gray-500">
          Pick any two teams and see head-to-head predictions
        </p>
      </div>

      {/* Gender Toggle */}
      <div className="flex justify-center mb-8">
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
          <p className="text-gray-500 text-sm">Loading team data...</p>
        </div>
      ) : (
        <>
          {/* Team Selectors */}
          <div className="flex flex-col sm:flex-row items-end gap-4 mb-8">
            <div className="flex-1 w-full">
              <TeamSelector
                teams={genderTeams}
                selectedId={teamAId}
                onSelect={setTeamAId}
                placeholder="Select first team..."
                label="Team A"
              />
            </div>

            <button
              onClick={handleSwap}
              className="flex items-center justify-center h-11 w-11 rounded-lg border border-gray-300 bg-white hover:bg-gray-50 transition-colors shrink-0"
              title="Swap teams"
            >
              <ArrowLeftRight className="h-4 w-4 text-gray-500" />
            </button>

            <div className="flex-1 w-full">
              <TeamSelector
                teams={genderTeams}
                selectedId={teamBId}
                onSelect={setTeamBId}
                placeholder="Select second team..."
                label="Team B"
              />
            </div>
          </div>

          {/* Matchup Display */}
          {canShowMatchup && predictions.data ? (
            <MatchupDisplay
              teamAName={genderTeams[teamAId] || ''}
              teamBName={genderTeams[teamBId] || ''}
              teamAStats={genderStats[teamAId]}
              teamBStats={genderStats[teamBId]}
              winProbA={getWinProbability(teamAId, teamBId, predictions.data)}
              gender={gender}
            />
          ) : (
            <Card className="border-dashed border-2">
              <CardContent className="p-12 text-center">
                <div className="text-5xl mb-4">🏀</div>
                <h3 className="text-lg font-semibold text-navy-900 mb-2">
                  Select Two Teams
                </h3>
                <p className="text-gray-500 max-w-md mx-auto">
                  Choose any two {gender === 'men' ? "men's" : "women's"} teams above to see the model's
                  head-to-head prediction and detailed stat comparison.
                </p>
              </CardContent>
            </Card>
          )}

          {teamAId && teamBId && teamAId === teamBId && (
            <Card className="border-yellow-300 bg-yellow-50">
              <CardContent className="p-6 text-center">
                <p className="text-yellow-700 font-medium">
                  Please select two different teams to compare.
                </p>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  )
}
