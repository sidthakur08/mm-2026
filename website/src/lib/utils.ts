import { clsx } from 'clsx'
import type { ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatProbability(prob: number): string {
  return `${(prob * 100).toFixed(1)}%`
}

export function getPredictionKey(id1: string, id2: string): { key: string; isLower: boolean } {
  const num1 = parseInt(id1)
  const num2 = parseInt(id2)
  if (num1 < num2) {
    return { key: `${id1}_${id2}`, isLower: true }
  }
  return { key: `${id2}_${id1}`, isLower: false }
}

export function getWinProbability(
  teamId: string,
  opponentId: string,
  predictions: Record<string, number>
): number {
  const { key, isLower } = getPredictionKey(teamId, opponentId)
  const prob = predictions[key]
  if (prob === undefined) return 0.5
  return isLower ? prob : 1 - prob
}

export function getRegionName(code: string): string {
  const regionNames: Record<string, string> = {
    W: 'East',
    X: 'South',
    Y: 'Midwest',
    Z: 'West',
  }
  return regionNames[code] || code
}

export function getProbabilityColor(prob: number): string {
  if (prob >= 0.8) return 'text-green-600'
  if (prob >= 0.6) return 'text-green-500'
  if (prob >= 0.4) return 'text-yellow-600'
  if (prob >= 0.2) return 'text-orange-500'
  return 'text-red-500'
}

export function getProbabilityBgColor(prob: number): string {
  if (prob >= 0.8) return 'bg-green-500'
  if (prob >= 0.6) return 'bg-green-400'
  if (prob >= 0.4) return 'bg-yellow-500'
  if (prob >= 0.2) return 'bg-orange-500'
  return 'bg-red-500'
}
