import { useState, useEffect, useMemo } from 'react'
import { useLiveScores, useTeams, useSeeds } from '@/hooks/useData'
import ScoreCard from '@/components/ScoreCard'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { RefreshCw, Wifi, WifiOff, Calendar } from 'lucide-react'
import { getWinProbability } from '@/lib/utils'
import type { TeamsData, PredictionsData } from '@/lib/types'

interface GameData {
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

function categorizeGames(games: GameData[]) {
  const live: GameData[] = []
  const upcoming: GameData[] = []
  const final_: GameData[] = []

  for (const g of games) {
    const state = g.game.gameState?.toLowerCase() || ''
    const finalMsg = g.game.finalMessage?.toLowerCase() || ''

    if (finalMsg.includes('final') || state.includes('final')) {
      final_.push(g)
    } else if (
      state.includes('live') ||
      state.includes('in progress') ||
      state.includes('half') ||
      (g.game.currentPeriod && !state.includes('pre'))
    ) {
      live.push(g)
    } else {
      upcoming.push(g)
    }
  }

  return { live, upcoming, final: final_ }
}

/** Normalize a team name for matching (strips punctuation, lowercases) */
function normalizeName(name: string): string {
  return name
    .toLowerCase()
    .replace(/[.'()]/g, '')
    .replace(/&/g, '&')  // normalize HTML entity just in case
    .replace(/\s+/g, ' ')
    .trim()
}

/**
 * Strip trailing state/location qualifiers from a normalized team name.
 * E.g. "st marys ca" -> "st marys", "miami oh" -> "miami oh" (kept because
 * the qualifier is meaningful for disambiguation -- handled separately).
 *
 * Common trailing qualifiers: 2-letter state codes, or known suffixes like
 * "pa", "ca", "mn", "nj", "nc", "ny", "md".
 */
const TRAILING_STATE_RE = /\s+(?:ca|pa|mn|nj|nc|ny|md|fl)$/

function stripStateQualifier(normalized: string): string {
  return normalized.replace(TRAILING_STATE_RE, '')
}

/** Build a lookup from normalized FULL name -> team ID */
function buildNameToIdMap(
  teams: TeamsData,
  gender: 'men' | 'women'
): Map<string, string> {
  const map = new Map<string, string>()
  const genderTeams = teams[gender]
  for (const [id, name] of Object.entries(genderTeams)) {
    map.set(normalizeName(name), id)
  }
  return map
}

/** Common abbreviation expansions from NCAA API names to our data format */
const ABBREVIATION_RULES: [RegExp, string][] = [
  // "Fla." / "Fla" at end or before space -> "Florida"
  [/\bFla\.?\s/g, 'Florida '],
  [/\bFla\.?$/g, 'Florida'],
  // "Conn." -> "Connecticut"
  [/\bConn\.?\s/g, 'Connecticut '],
  [/\bConn\.?$/g, 'Connecticut'],
  // "Miss." -> "Mississippi"
  [/\bMiss\.?\s/g, 'Mississippi '],
  [/\bMiss\.?$/g, 'Mississippi'],
  // "Ill." -> "Illinois"
  [/\bIll\.?\s/g, 'Illinois '],
  [/\bIll\.?$/g, 'Illinois'],
  // "Ala." -> "Alabama"
  [/\bAla\.?\s/g, 'Alabama '],
  [/\bAla\.?$/g, 'Alabama'],
  // "Tenn." -> "Tennessee"
  [/\bTenn\.?\s/g, 'Tennessee '],
  [/\bTenn\.?$/g, 'Tennessee'],
  // "Mich." -> "Michigan"
  [/\bMich\.?\s/g, 'Michigan '],
  [/\bMich\.?$/g, 'Michigan'],
]

/** Directional word contractions: NCAA uses full words, our data abbreviates */
const DIRECTIONAL_RULES: [RegExp, string][] = [
  [/^North(?:ern)?\s/, 'N '],
  [/^South(?:ern)?\s/, 'S '],
  [/^East(?:ern)?\s/, 'E '],
  [/^West(?:ern)?\s/, 'W '],
  [/^Southeast(?:ern)?\s/, 'SE '],
  [/^Central\s/, 'C '],
  [/^Mount\s/, 'Mt '],
]

/** Expand common abbreviations in NCAA API names to match our data */
function expandAbbreviations(name: string): string[] {
  const base = name.trim()
  const variants = [base]

  // "St." -> "St" (our data uses "St" not "St.")
  const stFixed = base.replace(/\bSt\.\s/g, 'St ').replace(/\bSt\.$/g, 'St')
  if (stFixed !== base) variants.push(stFixed)

  // "Saint" -> "St" (NCAA uses "Saint Mary's", our data uses "St Mary's")
  const saintFixed = base.replace(/\bSaint\s/g, 'St ').replace(/\bSaint$/g, 'St')
  if (saintFixed !== base) variants.push(saintFixed)

  // Strip all periods (e.g. "Ohio St." -> "Ohio St")
  const noDots = base.replace(/\./g, '')
  if (noDots !== base) variants.push(noDots)

  // Apply each abbreviation rule independently
  for (const [pattern, replacement] of ABBREVIATION_RULES) {
    const expanded = base.replace(pattern, replacement)
    if (expanded !== base) variants.push(expanded)
    // Also try on the no-dots version
    const expandedNoDots = noDots.replace(pattern, replacement)
    if (expandedNoDots !== noDots) variants.push(expandedNoDots)
  }

  // Directional contractions (e.g. "North Dakota St." -> "N Dakota St.")
  for (const [pattern, replacement] of DIRECTIONAL_RULES) {
    const contracted = base.replace(pattern, replacement)
    if (contracted !== base) variants.push(contracted)
    const contractedNoDots = noDots.replace(pattern, replacement)
    if (contractedNoDots !== noDots) variants.push(contractedNoDots)
    // Also combine with Saint->St fix
    if (saintFixed !== base) {
      const contractedSaint = saintFixed.replace(pattern, replacement)
      if (contractedSaint !== saintFixed) variants.push(contractedSaint)
    }
  }

  // Handle parenthetical qualifiers: "Miami (OH)" -> "Miami OH"
  const parenExpanded = base.replace(/\((\w+)\)/g, '$1')
  if (parenExpanded !== base) variants.push(parenExpanded)

  // Strip trailing " St." / " St" — handles cases where NCAA adds "St."
  // but our data omits it (e.g. "Kennesaw St." -> "Kennesaw").
  // Safe because more-specific variants (with "St") are tried first in matching.
  const noTrailingSt = base.replace(/\s+St\.?$/, '')
  if (noTrailingSt !== base && noTrailingSt.length >= 3) variants.push(noTrailingSt)

  return [...new Set(variants)]
}

/** Find our team ID for an NCAA API team name */
function findTeamId(
  ncaaName: string,
  nameMap: Map<string, string>
): string | null {
  const variants = expandAbbreviations(ncaaName)

  // 1. Exact normalized match against any variant
  for (const v of variants) {
    const norm = normalizeName(v)
    if (nameMap.has(norm)) return nameMap.get(norm)!
  }

  // 2. Match after stripping trailing state/location qualifiers from our data
  //    e.g. NCAA sends "St. Mary's" which normalizes to "st marys",
  //    our data has "St Mary's CA" which normalizes to "st marys ca",
  //    stripping qualifier gives "st marys" -> match.
  //    BUT we must be careful: "Mt St Mary's" (1291) stripped is "mt st marys"
  //    which does NOT match "st marys", so this is safe.
  for (const v of variants) {
    const norm = normalizeName(v)
    for (const [key, id] of nameMap) {
      if (stripStateQualifier(key) === norm) return id
    }
  }

  // 3. startsWith fallback: our data name starts with the NCAA normalized name.
  //    Require the shorter string (NCAA name) to be at least 5 chars AND the
  //    match must be at a word boundary (next char in the longer string is a space).
  //    This handles e.g. "South Fla" -> expand to "South Florida" already caught above,
  //    but as a safety net for other cases.
  //    IMPORTANT: only check one direction (our data starts with NCAA name),
  //    never the reverse, to avoid "texas" matching "texas a&m".
  for (const v of variants) {
    const norm = normalizeName(v)
    if (norm.length < 5) continue
    for (const [key, id] of nameMap) {
      // Our data name must start with the NCAA name AND
      // the NCAA name must end at a word boundary in our data name
      if (
        key.length > norm.length &&
        key.startsWith(norm) &&
        key[norm.length] === ' '
      ) {
        return id
      }
    }
  }

  return null
}

/** Get model prediction for a live game */
function getGamePrediction(
  game: GameData,
  nameMap: Map<string, string>,
  predictions: PredictionsData | null
): number | null {
  if (!predictions) return null
  const awayName = game.game.away.names?.short || game.game.away.names?.char6
  const homeName = game.game.home.names?.short || game.game.home.names?.char6
  if (!awayName || !homeName) return null

  const awayId = findTeamId(awayName, nameMap)
  const homeId = findTeamId(homeName, nameMap)
  if (!awayId || !homeId) return null

  const prob = getWinProbability(awayId, homeId, predictions)
  return prob === 0.5 ? null : prob // 0.5 means no prediction found
}

function GameSection({
  title,
  games,
  status,
  icon,
  nameMap,
  predictions,
}: {
  title: string
  games: GameData[]
  status: 'live' | 'upcoming' | 'final'
  icon: React.ReactNode
  nameMap: Map<string, string>
  predictions: PredictionsData | null
}) {
  if (games.length === 0) return null

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        {icon}
        <h2 className="text-lg font-bold text-navy-900">{title}</h2>
        <Badge variant="secondary">{games.length}</Badge>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {games.map((g) => (
          <ScoreCard
            key={g.game.gameID}
            away={g.game.away}
            home={g.game.home}
            gameState={g.game.gameState}
            currentPeriod={g.game.currentPeriod}
            contestClock={g.game.contestClock}
            finalMessage={g.game.finalMessage}
            network={g.game.network}
            startTime={g.game.startTime}
            status={status}
            awayWinProb={getGamePrediction(g, nameMap, predictions)}
          />
        ))}
      </div>
    </div>
  )
}

export default function Home() {
  const { games, loading, error, refetch } = useLiveScores()
  const { data: teams } = useTeams()
  useSeeds() // preload for other pages
  const [menPredData, setMenPredData] = useState<PredictionsData | null>(null)
  useEffect(() => {
    fetch('/data/predictions_men.json')
      .then((r) => r.json())
      .then((d: PredictionsData) => setMenPredData(d))
      .catch(() => {})
  }, [])

  // Build name-to-ID lookup map
  const nameMap = useMemo(() => {
    if (!teams) return new Map<string, string>()
    return buildNameToIdMap(teams, 'men')
  }, [teams])

  const predictions = menPredData

  const typedGames = games as GameData[]
  const { live, upcoming, final: finalGames } = categorizeGames(typedGames)
  const today = new Date().toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      {/* Hero */}
      <div className="text-center mb-10">
        <div className="inline-flex items-center gap-2 bg-orange-50 text-orange-700 px-4 py-1.5 rounded-full text-sm font-medium mb-4">
          <span>🏀</span>
          <span>Tournament Season 2026</span>
        </div>
        <h1 className="text-4xl sm:text-5xl font-extrabold text-navy-900 tracking-tight">
          March Madness 2026
        </h1>
        <p className="mt-3 text-lg text-gray-500 max-w-2xl mx-auto">
          ML-powered predictions for every NCAA tournament matchup.
          Live scores, bracket analysis, and model insights.
        </p>
      </div>

      {/* Live Scores Section */}
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h2 className="text-2xl font-bold text-navy-900">Today's Games</h2>
            <div className="flex items-center gap-1.5 text-xs text-gray-400">
              <Calendar className="h-3.5 w-3.5" />
              {today}
            </div>
          </div>
          <button
            onClick={refetch}
            className="flex items-center gap-2 px-3 py-1.5 text-sm text-gray-500 hover:text-navy-700 rounded-lg hover:bg-gray-100 transition-colors"
            disabled={loading}
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>

        {loading && typedGames.length === 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {[1, 2, 3].map((i) => (
              <Card key={i} className="animate-pulse">
                <CardContent className="p-6">
                  <div className="space-y-3">
                    <div className="h-4 bg-gray-200 rounded w-3/4" />
                    <div className="h-4 bg-gray-200 rounded w-1/2" />
                    <div className="h-4 bg-gray-200 rounded w-2/3" />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        {error && (
          <Card className="border-red-200 bg-red-50">
            <CardContent className="p-6 text-center">
              <WifiOff className="h-8 w-8 text-red-400 mx-auto mb-2" />
              <p className="text-red-700 font-medium">Unable to load live scores</p>
              <p className="text-sm text-red-500 mt-1">
                The NCAA scoreboard API may be temporarily unavailable.
              </p>
              <button
                onClick={refetch}
                className="mt-3 px-4 py-2 bg-red-100 text-red-700 rounded-lg text-sm font-medium hover:bg-red-200 transition-colors"
              >
                Try Again
              </button>
            </CardContent>
          </Card>
        )}

        {!loading && !error && typedGames.length === 0 && (
          <Card>
            <CardContent className="p-12 text-center">
              <div className="text-5xl mb-4">🏀</div>
              <h3 className="text-lg font-semibold text-navy-900 mb-2">
                No Games Scheduled Today
              </h3>
              <p className="text-gray-500 max-w-md mx-auto">
                Check back on game days for live scores, or explore the
                Matchups and Bracket pages to see model predictions.
              </p>
            </CardContent>
          </Card>
        )}

        {!error && (
          <>
            <GameSection
              title="Live Now"
              games={live}
              status="live"
              nameMap={nameMap}
              predictions={predictions}
              icon={
                <div className="flex items-center gap-1">
                  <Wifi className="h-4 w-4 text-red-500" />
                  <span className="h-2 w-2 rounded-full bg-red-500 animate-pulse-live" />
                </div>
              }
            />

            <GameSection
              title="Upcoming"
              games={upcoming}
              status="upcoming"
              nameMap={nameMap}
              predictions={predictions}
              icon={<Calendar className="h-5 w-5 text-blue-500" />}
            />

            <GameSection
              title="Final"
              games={finalGames}
              status="final"
              nameMap={nameMap}
              predictions={predictions}
              icon={<span className="text-lg">🏁</span>}
            />
          </>
        )}

        {/* Auto-refresh indicator */}
        {!error && typedGames.length > 0 && (
          <p className="text-center text-xs text-gray-400">
            Scores auto-refresh every 60 seconds
          </p>
        )}
      </div>
    </div>
  )
}
