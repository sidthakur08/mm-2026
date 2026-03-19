export type Gender = 'men' | 'women'

export type TeamsData = {
  men: Record<string, string>
  women: Record<string, string>
}

export type PredictionsData = Record<string, number>

export type SeedInfo = {
  region: string
  seed: number
  playIn: string
  seedStr: string
}

export type SeedsData = {
  men: Record<string, SeedInfo>
  women: Record<string, SeedInfo>
}

export type TeamStats = {
  winPct: number
  sos: number
  kenpomRank?: number
  offEfficiency: number
  defEfficiency: number
  efficiencyMargin: number
  pointDiff: number
  trueShooting: number
  turnoverRate: number
  astToRatio: number
  ppg: number
  oppPpg: number
  consistency: number
}

export type TeamStatsData = {
  men: Record<string, TeamStats>
  women: Record<string, TeamStats>
}

export type FeatureImportanceItem = {
  feature: string
  importance: number
}

export type TemporalCVData = Record<string, number>

export type HoldoutResults = {
  brier: number
  accuracy: number
  logLoss: number
  games: number
}

export type ModelPerformance = {
  testBrier: number
  testLogLoss: number
  testAccuracy: number
  valBrier: number
  trainSeasons: string
}

export type StageInfo = {
  models: string[]
  weights: number[]
  features: string[]
  nFeatures: number
}

export type ModelInfoData = {
  modelType?: string
  ensemble: {
    men: {
      stage1?: StageInfo
      stage2?: StageInfo
      // Legacy single-stage fields
      models?: string[]
      weights?: number[]
      features?: string[]
      nFeatures?: number
    }
    women: {
      stage1?: StageInfo
      stage2?: StageInfo
      models?: string[]
      weights?: number[]
      features?: string[]
      nFeatures?: number
    }
  }
  performance?: {
    men: ModelPerformance
    women: ModelPerformance
  }
  holdout2025: {
    men: HoldoutResults
    women: HoldoutResults
    combined: HoldoutResults
  }
  temporalCV?: {
    men: TemporalCVData
    women: TemporalCVData
  }
  featureImportance: {
    men: FeatureImportanceItem[]
    women: FeatureImportanceItem[]
  }
  calibration?: Record<string, unknown>
  clipRange?: number[]
  productionTrainRange: string
}

export type NcaaGame = {
  game: {
    gameID: string
    away: {
      names: { short: string; full: string; char6: string }
      score: string
      seed: string
      description: string
      winner: boolean
    }
    home: {
      names: { short: string; full: string; char6: string }
      score: string
      seed: string
      description: string
      winner: boolean
    }
    gameState: string
    startTime: string
    startTimeEpoch: number
    currentPeriod: string
    contestClock: string
    finalMessage: string
    network: string
    url: string
  }
}

export type BracketMatchupData = {
  topTeamId: string
  bottomTeamId: string
  topTeamName: string
  bottomTeamName: string
  topSeed: number
  bottomSeed: number
  topWinProb: number
  bottomWinProb: number
  isUpset: boolean
  topPlayIn?: string
  bottomPlayIn?: string
}

export type RegionData = {
  code: string
  name: string
  matchups: BracketMatchupData[]
}

export type BracketTeam = {
  id: string
  name: string
  seed: number
}

export type BracketGame = {
  gameId: string
  round: number // 1=R64, 2=R32, 3=S16, 4=E8, 5=F4, 6=Championship
  region?: string // W/X/Y/Z for rounds 1-4, undefined for F4/Championship
  topTeam: BracketTeam | null
  bottomTeam: BracketTeam | null
  topWinProb: number
  predictedWinnerId: string
  isUpset: boolean
  nextGameId?: string
}

export type FullBracket = {
  games: Record<string, BracketGame>
  regions: Record<string, string[]> // region code -> game IDs for that region
  finalFour: string[] // game IDs
  championship: string // game ID
  champion: BracketTeam
}
