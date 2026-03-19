import { lazy, Suspense } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Analytics } from '@vercel/analytics/react'
import { TooltipProvider } from '@/components/ui/tooltip'
import Layout from '@/components/Layout'
import { Loader2 } from 'lucide-react'

const Home = lazy(() => import('@/pages/Home'))
const Matchups = lazy(() => import('@/pages/Matchups'))
const Bracket = lazy(() => import('@/pages/Bracket'))
const Model = lazy(() => import('@/pages/Model'))

function PageLoader() {
  return (
    <div className="flex items-center justify-center py-20">
      <Loader2 className="h-8 w-8 animate-spin text-navy-400" />
    </div>
  )
}

function App() {
  return (
    <TooltipProvider>
      <BrowserRouter>
        <Suspense fallback={<PageLoader />}>
          <Routes>
            <Route path="/" element={<Layout />}>
              <Route index element={<Home />} />
              <Route path="matchups" element={<Matchups />} />
              <Route path="bracket" element={<Bracket />} />
              <Route path="model" element={<Model />} />
            </Route>
          </Routes>
        </Suspense>
      </BrowserRouter>
      <Analytics />
    </TooltipProvider>
  )
}

export default App
