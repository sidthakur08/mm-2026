import { cn } from '@/lib/utils'

interface StatBarProps {
  label: string
  valueA: number
  valueB: number
  format?: (v: number) => string
  higherIsBetter?: boolean
}

export default function StatBar({
  label,
  valueA,
  valueB,
  format = (v) => v.toFixed(1),
  higherIsBetter = true,
}: StatBarProps) {
  const aBetter = higherIsBetter ? valueA > valueB : valueA < valueB
  const bBetter = higherIsBetter ? valueB > valueA : valueB < valueA
  const equal = valueA === valueB

  return (
    <div className="flex items-center gap-3 py-2.5 border-b border-gray-100 last:border-0">
      <span
        className={cn(
          'text-sm font-semibold tabular-nums w-20 text-right',
          aBetter && !equal ? 'text-green-600' : 'text-gray-600'
        )}
      >
        {format(valueA)}
      </span>
      <div className="flex-1 text-center">
        <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
          {label}
        </span>
      </div>
      <span
        className={cn(
          'text-sm font-semibold tabular-nums w-20 text-left',
          bBetter && !equal ? 'text-green-600' : 'text-gray-600'
        )}
      >
        {format(valueB)}
      </span>
    </div>
  )
}
