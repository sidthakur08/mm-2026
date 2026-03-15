import * as React from 'react'
import { cn } from '@/lib/utils'

type BadgeVariant = 'default' | 'secondary' | 'destructive' | 'outline' | 'success' | 'warning'

interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: BadgeVariant
}

const variantStyles: Record<BadgeVariant, string> = {
  default: 'bg-navy-900 text-white',
  secondary: 'bg-gray-100 text-gray-800',
  destructive: 'bg-red-500 text-white',
  outline: 'border border-gray-300 text-gray-700',
  success: 'bg-green-100 text-green-800',
  warning: 'bg-orange-100 text-orange-800',
}

function Badge({ className, variant = 'default', ...props }: BadgeProps) {
  return (
    <div
      className={cn(
        'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold transition-colors',
        variantStyles[variant],
        className
      )}
      {...props}
    />
  )
}

export { Badge }
