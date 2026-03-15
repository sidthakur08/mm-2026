import { NavLink } from 'react-router-dom'
import { cn } from '@/lib/utils'
import { BarChart3, GitBranch, Zap, Trophy } from 'lucide-react'

const navItems = [
  { to: '/', label: 'Live Scores', icon: Zap },
  { to: '/matchups', label: 'Matchups', icon: Trophy },
  { to: '/bracket', label: 'Bracket', icon: GitBranch },
  { to: '/model', label: 'Model', icon: BarChart3 },
]

export default function Navbar() {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-navy-200/50 bg-navy-900/95 backdrop-blur supports-[backdrop-filter]:bg-navy-900/80">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <NavLink to="/" className="flex items-center gap-2.5 group">
          <span className="text-2xl" role="img" aria-label="basketball">
            🏀
          </span>
          <div className="flex flex-col">
            <span className="text-lg font-bold text-white leading-tight tracking-tight group-hover:text-orange-400 transition-colors">
              March Madness
            </span>
            <span className="text-[10px] font-semibold text-orange-400 uppercase tracking-widest leading-none">
              2026 Predictions
            </span>
          </div>
        </NavLink>

        <nav className="flex items-center gap-1">
          {navItems.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200',
                  isActive
                    ? 'bg-orange-500/20 text-orange-400'
                    : 'text-gray-300 hover:text-white hover:bg-white/10'
                )
              }
            >
              <Icon className="h-4 w-4" />
              <span className="hidden sm:inline">{label}</span>
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  )
}
