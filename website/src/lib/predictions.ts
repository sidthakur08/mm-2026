import type { SeedsData, PredictionsData, TeamsData, BracketMatchupData, RegionData } from './types'
import { getWinProbability, getRegionName } from './utils'

export function buildBracketData(
  seeds: SeedsData,
  predictions: PredictionsData,
  teams: TeamsData,
  gender: 'men' | 'women'
): RegionData[] {
  const genderSeeds = seeds[gender]
  const genderTeams = teams[gender]

  // Group teams by region
  const regionTeams: Record<string, { id: string; seed: number; playIn: string }[]> = {}

  for (const [teamId, seedInfo] of Object.entries(genderSeeds)) {
    const region = seedInfo.region
    if (!regionTeams[region]) {
      regionTeams[region] = []
    }
    regionTeams[region].push({
      id: teamId,
      seed: seedInfo.seed,
      playIn: seedInfo.playIn,
    })
  }

  const regions: RegionData[] = []

  for (const [regionCode, teamList] of Object.entries(regionTeams)) {
    // Sort by seed
    teamList.sort((a, b) => a.seed - b.seed)

    const matchups: BracketMatchupData[] = []

    // Build first round matchups: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
    const matchupSeeds = [
      [1, 16],
      [8, 9],
      [5, 12],
      [4, 13],
      [6, 11],
      [3, 14],
      [7, 10],
      [2, 15],
    ]

    for (const [topSeed, bottomSeed] of matchupSeeds) {
      const topTeams = teamList.filter((t) => t.seed === topSeed)
      const bottomTeams = teamList.filter((t) => t.seed === bottomSeed)

      if (topTeams.length === 0 || bottomTeams.length === 0) continue

      // Handle play-in games - use the first team (or mark as play-in)
      const topTeam = topTeams[0]
      const bottomTeam = bottomTeams[0]

      const hasTopPlayIn = topTeams.length > 1
      const hasBottomPlayIn = bottomTeams.length > 1

      const topId = topTeam.id
      const bottomId = bottomTeam.id

      const topWinProb = getWinProbability(topId, bottomId, predictions)

      const matchup: BracketMatchupData = {
        topTeamId: topId,
        bottomTeamId: bottomId,
        topTeamName: genderTeams[topId] || `Team ${topId}`,
        bottomTeamName: genderTeams[bottomId] || `Team ${bottomId}`,
        topSeed: topSeed,
        bottomSeed: bottomSeed,
        topWinProb: topWinProb,
        bottomWinProb: 1 - topWinProb,
        isUpset: topSeed < bottomSeed ? topWinProb < 0.5 : topWinProb > 0.5,
      }

      if (hasTopPlayIn) {
        matchup.topPlayIn = topTeams.map((t) => genderTeams[t.id] || `Team ${t.id}`).join(' / ')
      }
      if (hasBottomPlayIn) {
        matchup.bottomPlayIn = bottomTeams.map((t) => genderTeams[t.id] || `Team ${t.id}`).join(' / ')
      }

      matchups.push(matchup)
    }

    regions.push({
      code: regionCode,
      name: getRegionName(regionCode),
      matchups,
    })
  }

  // Sort regions by code for consistent ordering
  regions.sort((a, b) => a.code.localeCompare(b.code))

  return regions
}
