export default async function handler(req, res) {
  const today = new Date()
  const year = today.getFullYear()
  const month = String(today.getMonth() + 1).padStart(2, '0')
  const day = String(today.getDate()).padStart(2, '0')

  try {
    const response = await fetch(
      `https://ncaa-api.henrygd.me/scoreboard/basketball-men/d1/${year}/${month}/${day}`
    )
    const data = await response.json()
    res.setHeader('Cache-Control', 's-maxage=30')
    res.status(200).json(data)
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch scores' })
  }
}
