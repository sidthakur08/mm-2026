import { Outlet } from 'react-router-dom'
import Navbar from './Navbar'

export default function Layout() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1">
        <div className="page-transition">
          <Outlet />
        </div>
      </main>
      <footer className="border-t border-gray-200 bg-white">
        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-2">
            <p className="text-sm text-gray-500">
              Built with ML -- Logistic Regression + XGBoost ensemble
            </p>
            <p className="text-xs text-gray-400">
              Predictions are for entertainment purposes only. Data sourced from publicly available NCAA statistics.
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}
