import type {
  SeedsData,
  PredictionsData,
  TeamsData,
  BracketMatchupData,
  RegionData,
  BracketGame,
  BracketTeam,
  FullBracket,
} from './types'
import { getWinProbability, getRegionName } from './utils'

/**
 * Standard NCAA bracket matchup order within each region.
 * Each pair represents [topSeed, bottomSeed] for Round of 64.
 * The R32 matchups pair adjacent R64 games: winner of game 0 vs winner of game 1, etc.
 */
const R64_MATCHUP_SEEDS: [number, number][] = [
  [1, 16],
  [8, 9],
  [5, 12],
  [4, 13],
  [6, 11],
  [3, 14],
  [7, 10],
  [2, 15],
]

/**
 * NCAA bracket region pairing for Final Four.
 * Regions W vs X and Y vs Z meet in the Final Four.
 */
const FINAL_FOUR_PAIRS: [string, string][] = [
  ['W', 'X'],
  ['Y', 'Z'],
]

function resolvePlayIn(
  teams: { id: string; seed: number; playIn: string }[],
  predictions: PredictionsData
): { id: string; seed: number } {
  if (teams.length === 1) return teams[0]
  // For play-in games, pick the team the model favors
  const a = teams[0]
  const b = teams[1]
  const aWinProb = getWinProbability(a.id, b.id, predictions)
  return aWinProb >= 0.5 ? a : b
}

/**
 * Build a full tournament bracket by simulating every round using model predictions.
 */
export function buildFullBracket(
  seeds: SeedsData,
  predictions: PredictionsData,
  teams: TeamsData,
  gender: 'men' | 'women'
): FullBracket {
  const genderSeeds = seeds[gender]
  const genderTeams = teams[gender]
  const games: Record<string, BracketGame> = {}
  const regionGameIds: Record<string, string[]> = {}

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

  // Track regional champions for Final Four
  const regionChampions: Record<string, BracketTeam> = {}

  // Process each region through rounds 1-4
  for (const [regionCode, teamList] of Object.entries(regionTeams)) {
    regionGameIds[regionCode] = []

    // --- Round of 64 ---
    const r64Winners: BracketTeam[] = []
    const r64GameIds: string[] = []

    for (let i = 0; i < R64_MATCHUP_SEEDS.length; i++) {
      const [topSeedNum, bottomSeedNum] = R64_MATCHUP_SEEDS[i]

      const topCandidates = teamList.filter((t) => t.seed === topSeedNum)
      const bottomCandidates = teamList.filter((t) => t.seed === bottomSeedNum)

      // Resolve play-in games
      const topResolved = resolvePlayIn(topCandidates, predictions)
      const bottomResolved = resolvePlayIn(bottomCandidates, predictions)

      const topTeam: BracketTeam = {
        id: topResolved.id,
        name: genderTeams[topResolved.id] || `Team ${topResolved.id}`,
        seed: topSeedNum,
      }
      const bottomTeam: BracketTeam = {
        id: bottomResolved.id,
        name: genderTeams[bottomResolved.id] || `Team ${bottomResolved.id}`,
        seed: bottomSeedNum,
      }

      const topWinProb = getWinProbability(topTeam.id, bottomTeam.id, predictions)
      const topWins = topWinProb >= 0.5
      const winnerId = topWins ? topTeam.id : bottomTeam.id
      const winnerSeed = topWins ? topTeam.seed : bottomTeam.seed
      const loserSeed = topWins ? bottomTeam.seed : topTeam.seed
      const isUpset = winnerSeed > loserSeed

      const gameId = `${regionCode}-R64-${i}`
      r64GameIds.push(gameId)

      games[gameId] = {
        gameId,
        round: 1,
        region: regionCode,
        topTeam,
        bottomTeam,
        topWinProb,
        predictedWinnerId: winnerId,
        isUpset,
      }

      const winner = topWins ? topTeam : bottomTeam
      r64Winners.push(winner)
    }

    regionGameIds[regionCode].push(...r64GameIds)

    // --- Rounds 2-4 (R32, S16, E8) ---
    let currentRoundTeams = r64Winners
    let prevGameIds = r64GameIds
    const roundLabels = ['R32', 'S16', 'E8']

    for (let roundIdx = 0; roundIdx < 3; roundIdx++) {
      const round = roundIdx + 2 // 2, 3, 4
      const label = roundLabels[roundIdx]
      const nextRoundTeams: BracketTeam[] = []
      const nextGameIds: string[] = []

      for (let i = 0; i < currentRoundTeams.length; i += 2) {
        const topTeam = currentRoundTeams[i]
        const bottomTeam = currentRoundTeams[i + 1]
        const gameIndex = i / 2

        const topWinProb = getWinProbability(topTeam.id, bottomTeam.id, predictions)
        const topWins = topWinProb >= 0.5
        const winnerId = topWins ? topTeam.id : bottomTeam.id
        const winnerSeed = topWins ? topTeam.seed : bottomTeam.seed
        const loserSeed = topWins ? bottomTeam.seed : topTeam.seed
        const isUpset = winnerSeed > loserSeed

        const gameId = `${regionCode}-${label}-${gameIndex}`
        nextGameIds.push(gameId)

        games[gameId] = {
          gameId,
          round,
          region: regionCode,
          topTeam,
          bottomTeam,
          topWinProb,
          predictedWinnerId: winnerId,
          isUpset,
        }

        // Link previous round games to this game
        const prevTop = prevGameIds[i]
        const prevBottom = prevGameIds[i + 1]
        if (prevTop && games[prevTop]) games[prevTop].nextGameId = gameId
        if (prevBottom && games[prevBottom]) games[prevBottom].nextGameId = gameId

        const winner = topWins ? topTeam : bottomTeam
        nextRoundTeams.push(winner)
      }

      regionGameIds[regionCode].push(...nextGameIds)
      currentRoundTeams = nextRoundTeams
      prevGameIds = nextGameIds
    }

    // The last team standing is the regional champion
    regionChampions[regionCode] = currentRoundTeams[0]
  }

  // --- Final Four ---
  const finalFourGameIds: string[] = []
  const finalFourWinners: BracketTeam[] = []
  const e8GameIds: Record<string, string> = {}

  // Get the E8 game IDs for each region (the last game in each region)
  for (const [regionCode, gameIds] of Object.entries(regionGameIds)) {
    e8GameIds[regionCode] = gameIds[gameIds.length - 1]
  }

  for (let i = 0; i < FINAL_FOUR_PAIRS.length; i++) {
    const [regionA, regionB] = FINAL_FOUR_PAIRS[i]
    const topTeam = regionChampions[regionA]
    const bottomTeam = regionChampions[regionB]

    if (!topTeam || !bottomTeam) continue

    const topWinProb = getWinProbability(topTeam.id, bottomTeam.id, predictions)
    const topWins = topWinProb >= 0.5
    const winnerId = topWins ? topTeam.id : bottomTeam.id
    const winnerSeed = topWins ? topTeam.seed : bottomTeam.seed
    const loserSeed = topWins ? bottomTeam.seed : topTeam.seed
    const isUpset = winnerSeed > loserSeed

    const gameId = `F4-${i}`
    finalFourGameIds.push(gameId)

    games[gameId] = {
      gameId,
      round: 5,
      topTeam,
      bottomTeam,
      topWinProb,
      predictedWinnerId: winnerId,
      isUpset,
    }

    // Link E8 games to this F4 game
    if (e8GameIds[regionA] && games[e8GameIds[regionA]]) {
      games[e8GameIds[regionA]].nextGameId = gameId
    }
    if (e8GameIds[regionB] && games[e8GameIds[regionB]]) {
      games[e8GameIds[regionB]].nextGameId = gameId
    }

    const winner = topWins ? topTeam : bottomTeam
    finalFourWinners.push(winner)
  }

  // --- Championship ---
  const champTopTeam = finalFourWinners[0]
  const champBottomTeam = finalFourWinners[1]
  const champTopWinProb = getWinProbability(champTopTeam.id, champBottomTeam.id, predictions)
  const champTopWins = champTopWinProb >= 0.5
  const champWinnerId = champTopWins ? champTopTeam.id : champBottomTeam.id
  const champWinnerSeed = champTopWins ? champTopTeam.seed : champBottomTeam.seed
  const champLoserSeed = champTopWins ? champBottomTeam.seed : champTopTeam.seed
  const champIsUpset = champWinnerSeed > champLoserSeed

  const championshipId = 'CHAMP'

  games[championshipId] = {
    gameId: championshipId,
    round: 6,
    topTeam: champTopTeam,
    bottomTeam: champBottomTeam,
    topWinProb: champTopWinProb,
    predictedWinnerId: champWinnerId,
    isUpset: champIsUpset,
  }

  // Link F4 games to championship
  for (const f4Id of finalFourGameIds) {
    games[f4Id].nextGameId = championshipId
  }

  const champion = champTopWins ? champTopTeam : champBottomTeam

  return {
    games,
    regions: regionGameIds,
    finalFour: finalFourGameIds,
    championship: championshipId,
    champion,
  }
}

// Keep legacy function for backward compatibility
export function buildBracketData(
  seeds: SeedsData,
  predictions: PredictionsData,
  teams: TeamsData,
  gender: 'men' | 'women'
): RegionData[] {
  const genderSeeds = seeds[gender]
  const genderTeams = teams[gender]

  const regionTeamsMap: Record<string, { id: string; seed: number; playIn: string }[]> = {}

  for (const [teamId, seedInfo] of Object.entries(genderSeeds)) {
    const region = seedInfo.region
    if (!regionTeamsMap[region]) {
      regionTeamsMap[region] = []
    }
    regionTeamsMap[region].push({
      id: teamId,
      seed: seedInfo.seed,
      playIn: seedInfo.playIn,
    })
  }

  const regions: RegionData[] = []

  for (const [regionCode, teamList] of Object.entries(regionTeamsMap)) {
    teamList.sort((a, b) => a.seed - b.seed)

    const matchups: BracketMatchupData[] = []

    for (const [topSeed, bottomSeed] of R64_MATCHUP_SEEDS) {
      const topTeams = teamList.filter((t) => t.seed === topSeed)
      const bottomTeams = teamList.filter((t) => t.seed === bottomSeed)

      if (topTeams.length === 0 || bottomTeams.length === 0) continue

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

  regions.sort((a, b) => a.code.localeCompare(b.code))

  return regions
}
