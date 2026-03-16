import { useModelInfo } from '@/hooks/useData'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Loader2, Target, Brain, BarChart3 } from 'lucide-react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
} from 'recharts'

function StatCard({
  label,
  value,
  description,
  color = 'navy',
}: {
  label: string
  value: string
  description?: string
  color?: 'navy' | 'orange' | 'green'
}) {
  const colorClasses = {
    navy: 'from-navy-800 to-navy-900',
    orange: 'from-orange-500 to-orange-600',
    green: 'from-green-600 to-green-700',
  }

  return (
    <div className={`bg-gradient-to-br ${colorClasses[color]} rounded-xl p-5 text-white`}>
      <p className="text-sm font-medium opacity-80">{label}</p>
      <p className="text-3xl font-bold mt-1">{value}</p>
      {description && (
        <p className="text-xs opacity-60 mt-1">{description}</p>
      )}
    </div>
  )
}

export default function Model() {
  const { data: modelInfo, loading, error } = useModelInfo()

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-20">
        <Loader2 className="h-8 w-8 animate-spin text-navy-400 mb-3" />
        <p className="text-gray-500 text-sm">Loading model info...</p>
      </div>
    )
  }

  if (error || !modelInfo) {
    return (
      <div className="mx-auto max-w-4xl px-4 py-8 text-center">
        <p className="text-red-500">Failed to load model information.</p>
      </div>
    )
  }

  const { holdout2025, featureImportance, ensemble } = modelInfo

  // Feature importance data
  const menFeatures = featureImportance.men.map((f) => ({
    ...f,
    importance: Math.round(f.importance * 100),
  }))
  const womenFeatures = featureImportance.women.map((f) => ({
    ...f,
    importance: Math.round(f.importance * 100),
  }))

  const NAVY = '#1a2b5f'
  const ORANGE = '#ff7d00'

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
      <div className="text-center mb-10">
        <h1 className="text-3xl sm:text-4xl font-extrabold text-navy-900 tracking-tight">
          About the Model
        </h1>
        <p className="mt-2 text-gray-500 max-w-2xl mx-auto">
          A two-stage prediction pipeline. Stage 1 captures regular-season team quality;
          Stage 2 learns how seeds, strength of schedule, and conference context shape tournament outcomes.
        </p>
      </div>

      {/* Two-Stage Architecture */}
      <Card className="mb-8">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-navy-600" />
            <CardTitle>Two-Stage Architecture</CardTitle>
          </div>
          <CardDescription>
            Stage 1 captures regular-season quality. Stage 2 adjusts for tournament context.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Stage 1 */}
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="text-xs">Stage 1</Badge>
                <h4 className="font-semibold text-navy-800">Regular Season Model</h4>
              </div>
              <p className="text-sm text-gray-500">
                Trained on ~77K regular-season games with rolling window features (5/7/10 games).
                Produces a team-quality probability for each matchup.
              </p>
              {(['men', 'women'] as const).map((gender) => {
                const stage1 = (ensemble as any)[gender]?.stage1
                if (!stage1) return null
                return (
                  <div key={gender} className="space-y-1">
                    <p className="text-xs font-medium text-gray-500 uppercase">{gender}'s</p>
                    <div className="flex gap-2">
                      {stage1.models.map((model: string, i: number) => (
                        <div key={model} className="flex items-center gap-1">
                          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: i === 0 ? NAVY : ORANGE }} />
                          <span className="text-xs text-gray-600">
                            {model === 'LR' ? 'LR' : 'XGB'} {(stage1.weights[i] * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                      <span className="text-xs text-gray-400">({stage1.nFeatures} features)</span>
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Stage 2 */}
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="text-xs">Stage 2</Badge>
                <h4 className="font-semibold text-navy-800">Tournament Model</h4>
              </div>
              <p className="text-sm text-gray-500">
                Trained on ~2,500 historical tournament games. Uses Stage 1 probability +
                seed diff, conference, SOS, and KenPom (men's) to learn March-specific adjustments.
              </p>
              {(['men', 'women'] as const).map((gender) => {
                const stage2 = (ensemble as any)[gender]?.stage2
                if (!stage2) return null
                return (
                  <div key={gender} className="space-y-1">
                    <p className="text-xs font-medium text-gray-500 uppercase">{gender}'s</p>
                    <div className="flex gap-2">
                      {stage2.models.map((model: string, i: number) => (
                        <div key={model} className="flex items-center gap-1">
                          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: i === 0 ? NAVY : ORANGE }} />
                          <span className="text-xs text-gray-600">
                            {model === 'LR' ? 'LR' : 'XGB'} {(stage2.weights[i] * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                      <span className="text-xs text-gray-400">({stage2.nFeatures} features)</span>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          <div className="mt-6 flex items-center gap-6 justify-center">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: NAVY }} />
              <span className="text-xs text-gray-500">Logistic Regression</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: ORANGE }} />
              <span className="text-xs text-gray-500">XGBoost</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 2025 Holdout Results */}
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-4">
          <Target className="h-5 w-5 text-navy-600" />
          <h2 className="text-xl font-bold text-navy-900">2025 Holdout Performance</h2>
          <Badge variant="success">Out-of-sample</Badge>
        </div>
        <p className="text-sm text-gray-500 mb-4">
          These results are from the 2025 tournament -- data the model never saw during training.
          A coin flip would have a Brier score of 0.250.
        </p>

        <Tabs defaultValue="combined">
          <TabsList>
            <TabsTrigger value="combined">Combined</TabsTrigger>
            <TabsTrigger value="men">Men's</TabsTrigger>
            <TabsTrigger value="women">Women's</TabsTrigger>
          </TabsList>

          {(['combined', 'men', 'women'] as const).map((tab) => {
            const results = holdout2025[tab]
            return (
              <TabsContent key={tab} value={tab}>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <StatCard
                    label="Brier Score"
                    value={results.brier.toFixed(4)}
                    description={`vs. coin flip: 0.2500 (${((1 - results.brier / 0.25) * 100).toFixed(0)}% better)`}
                    color="navy"
                  />
                  <StatCard
                    label="Accuracy"
                    value={`${(results.accuracy * 100).toFixed(1)}%`}
                    description={`${results.games} games predicted`}
                    color="green"
                  />
                  <StatCard
                    label="Log Loss"
                    value={results.logLoss.toFixed(4)}
                    description="Lower is better"
                    color="orange"
                  />
                </div>
              </TabsContent>
            )
          })}
        </Tabs>
      </div>

      {/* Feature Importance */}
      <Card className="mb-8">
        <CardHeader>
          <div className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-navy-600" />
            <CardTitle>Feature Importance</CardTitle>
          </div>
          <CardDescription>
            What the model weighs most heavily when making predictions
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="men">
            <TabsList>
              <TabsTrigger value="men">Men's</TabsTrigger>
              <TabsTrigger value="women">Women's</TabsTrigger>
            </TabsList>

            <TabsContent value="men">
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={menFeatures} layout="vertical" margin={{ left: 20, right: 30 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                  <XAxis
                    type="number"
                    tickFormatter={(v: number) => `${v}%`}
                    domain={[0, 'auto']}
                  />
                  <YAxis type="category" dataKey="feature" width={140} tick={{ fontSize: 12 }} />
                  <RechartsTooltip
                    formatter={(value: number) => [`${value}%`, 'Importance']}
                    contentStyle={{ borderRadius: '8px', border: '1px solid #e5e7eb' }}
                  />
                  <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                    {menFeatures.map((_, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={index < 2 ? NAVY : index < 4 ? '#3b5896' : '#9eaacb'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </TabsContent>

            <TabsContent value="women">
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={womenFeatures} layout="vertical" margin={{ left: 20, right: 30 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                  <XAxis
                    type="number"
                    tickFormatter={(v: number) => `${v}%`}
                    domain={[0, 'auto']}
                  />
                  <YAxis type="category" dataKey="feature" width={140} tick={{ fontSize: 12 }} />
                  <RechartsTooltip
                    formatter={(value: number) => [`${value}%`, 'Importance']}
                    contentStyle={{ borderRadius: '8px', border: '1px solid #e5e7eb' }}
                  />
                  <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                    {womenFeatures.map((_, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={index < 2 ? ORANGE : index < 4 ? '#ffa726' : '#ffcc80'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Model Details */}
      <Card>
        <CardHeader>
          <CardTitle>Training Details</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 text-sm">
            <div className="space-y-3">
              <h4 className="font-semibold text-navy-800">Data</h4>
              <ul className="space-y-1.5 text-gray-600">
                <li>Production training range: {modelInfo.productionTrainRange}</li>
                <li>Features include rolling averages (5, 7, 10 game windows)</li>
                <li>Efficiency metrics, shooting stats, and strength of schedule</li>
                <li>Men's model includes KenPom rankings</li>
              </ul>
            </div>
            <div className="space-y-3">
              <h4 className="font-semibold text-navy-800">Methodology</h4>
              <ul className="space-y-1.5 text-gray-600">
                <li>Two-stage pipeline: regular season + tournament models</li>
                <li>LR + XGBoost ensembles with Optuna-tuned hyperparameters</li>
                <li>Temporal cross-validation (train on past, test on future)</li>
                <li>Probabilities clipped to [0.01, 0.99]</li>
                <li>No data leakage -- only pre-tournament stats used</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
