import {
  BitmapLayer,
  Deck,
  OrthographicView,
  ScatterplotLayer,
  TextLayer
} from 'deck.gl'
import {
  EditableGeoJsonLayer,
  ModifyMode
} from '@deck.gl-community/editable-layers'

const colors = {
  motilityCentroid: [86, 180, 233, 150],
  cellPaintingCentroid: [230, 159, 0, 150],
  motilityLandmark: [0, 180, 120, 255],
  cellPaintingLandmark: [255, 90, 90, 255],
  pending: [255, 220, 70, 255],
  active: [255, 255, 255, 255],
  text: [255, 255, 255, 255]
}

function imageCount(model) {
  return (model.get('cell_painting_image_urls') || []).length
}

function currentIndex(model) {
  const count = imageCount(model)
  const index = Number(model.get('cell_painting_index') || 0)
  if (count <= 0) return 0
  return Math.max(0, Math.min(count - 1, index))
}

function currentCellPaintingCentroids(model) {
  const all = model.get('cell_painting_centroids_by_image') || []
  return all[currentIndex(model)] || []
}

function currentMatches(model) {
  const matchesByImage = model.get('matches_by_image') || {}
  return matchesByImage[String(currentIndex(model))] || []
}

function setCurrentMatches(model, matches) {
  const matchesByImage = {...(model.get('matches_by_image') || {})}
  matchesByImage[String(currentIndex(model))] = matches
  model.set('matches_by_image', matchesByImage)
  model.save_changes()
}

function currentCellPaintingSize(model) {
  const sizes = model.get('cell_painting_sizes') || []
  return sizes[currentIndex(model)] || model.get('cell_painting_size') || [2048, 2048]
}

function pointToFeature(point, side, matchId) {
  return {
    type: 'Feature',
    properties: {
      side,
      match_id: matchId,
      point_id: point.id || '',
      label: matchId
    },
    geometry: {
      type: 'Point',
      coordinates: [Number(point.x), Number(point.y)]
    }
  }
}

function makeLandmarkFeatures(model, side) {
  const features = []
  for (const match of currentMatches(model)) {
    const key = side === 'motility' ? 'motility' : 'cell_painting'
    if (match?.[key]) {
      features.push(pointToFeature(match[key], side, match.match_id))
    }
  }
  const pending = model.get('pending_motility') || {}
  if (side === 'motility' && pending.id) {
    features.push(pointToFeature(pending, 'motility', 'pending'))
  }
  return {
    type: 'FeatureCollection',
    features
  }
}

function landmarkPoints(model, side) {
  return makeLandmarkFeatures(model, side).features.map((feature) => ({
    id: feature.properties.point_id,
    match_id: feature.properties.match_id,
    x: feature.geometry.coordinates[0],
    y: feature.geometry.coordinates[1]
  }))
}

function upsertMovedLandmark(model, side, updatedData) {
  const features = updatedData?.features || []
  const matches = [...currentMatches(model)]
  const key = side === 'motility' ? 'motility' : 'cell_painting'

  for (const feature of features) {
    const matchId = feature.properties?.match_id
    if (!matchId || matchId === 'pending') continue

    const coords = feature.geometry?.coordinates || []
    const matchIndex = matches.findIndex((match) => match.match_id === matchId)
    if (matchIndex < 0) continue

    matches[matchIndex] = {
      ...matches[matchIndex],
      [key]: {
        ...matches[matchIndex][key],
        x: Number(coords[0]),
        y: Number(coords[1])
      }
    }
  }

  setCurrentMatches(model, matches)
}

function nextMatchId(matches) {
  let maxId = 0
  for (const match of matches) {
    const raw = String(match?.match_id || '')
    const found = raw.match(/^L(\d+)$/)
    if (found) maxId = Math.max(maxId, Number(found[1]))
  }
  return `L${maxId + 1}`
}

function handleCentroidClick(model, side, info) {
  const object = info?.object
  if (!object) return

  const clicked = {
    id: String(object.id ?? `${side}_${info.index}`),
    x: Number(object.x),
    y: Number(object.y)
  }

  if (side === 'motility') {
    model.set('pending_motility', clicked)
    model.set('active_match_id', 'pending')
    model.set('status', `selected motility cell ${clicked.id}`)
    model.save_changes()
    return
  }

  const pending = model.get('pending_motility') || {}
  if (!pending.id) {
    model.set('status', 'select a motility cell first')
    model.save_changes()
    return
  }

  const matches = [...currentMatches(model)]
  const matchId = nextMatchId(matches)
  matches.push({
    match_id: matchId,
    motility: pending,
    cell_painting: clicked
  })

  const matchesByImage = {...(model.get('matches_by_image') || {})}
  matchesByImage[String(currentIndex(model))] = matches
  model.set('matches_by_image', matchesByImage)
  model.set('pending_motility', {})
  model.set('active_match_id', matchId)
  model.set('status', `added ${matchId}: ${pending.id} <-> ${clicked.id}`)
  model.save_changes()
}

function removeActiveOrLastMatch(model) {
  const matches = [...currentMatches(model)]
  if (matches.length === 0) return
  const active = model.get('active_match_id') || ''
  let next = matches
  if (active && active !== 'pending') {
    next = matches.filter((match) => match.match_id !== active)
  } else {
    next = matches.slice(0, -1)
  }
  setCurrentMatches(model, next)
  model.set('active_match_id', '')
  model.set('pending_motility', {})
  model.set('status', 'deleted match')
  model.save_changes()
}

function clearCurrentMatches(model) {
  setCurrentMatches(model, [])
  model.set('pending_motility', {})
  model.set('active_match_id', '')
  model.set('status', `cleared matches for image ${currentIndex(model)}`)
  model.save_changes()
}

function makeLayers(model, side) {
  const imageUrl = side === 'motility'
    ? model.get('motility_image_url')
    : (model.get('cell_painting_image_urls') || [])[currentIndex(model)]
  const size = side === 'motility' ? model.get('motility_size') : currentCellPaintingSize(model)
  const width = Number(size?.[0] || 1)
  const height = Number(size?.[1] || 1)
  const centroids = side === 'motility'
    ? model.get('motility_centroids') || []
    : currentCellPaintingCentroids(model)
  const pointRadius = model.get('point_radius') || 4
  const features = makeLandmarkFeatures(model, side)
  const lmPoints = landmarkPoints(model, side)
  const activeMatchId = model.get('active_match_id') || ''

  const layers = []

  if (imageUrl) {
    layers.push(new BitmapLayer({
      id: `${side}-image`,
      image: imageUrl,
      bounds: [0, height, width, 0]
    }))
  }

  layers.push(new ScatterplotLayer({
    id: `${side}-centroids`,
    data: centroids,
    pickable: true,
    getPosition: (d) => [Number(d.x), Number(d.y)],
    getRadius: pointRadius,
    radiusUnits: 'pixels',
    getFillColor: side === 'motility' ? colors.motilityCentroid : colors.cellPaintingCentroid,
    stroked: false,
    onClick: (info) => handleCentroidClick(model, side, info)
  }))

  layers.push(new EditableGeoJsonLayer({
    id: `${side}-editable-landmarks`,
    data: features,
    mode: ModifyMode,
    selectedFeatureIndexes: features.features.map((_, i) => i),
    pickable: true,
    pointRadiusMinPixels: 7,
    getFillColor: (feature) => {
      const matchId = feature.properties?.match_id
      if (matchId === 'pending') return colors.pending
      if (matchId === activeMatchId) return colors.active
      return side === 'motility' ? colors.motilityLandmark : colors.cellPaintingLandmark
    },
    getLineColor: [255, 255, 255, 220],
    onClick: (info) => {
      const matchId = info?.object?.properties?.match_id
      if (!matchId) return
      model.set('active_match_id', matchId)
      model.save_changes()
    },
    onEdit: ({updatedData}) => upsertMovedLandmark(model, side, updatedData)
  }))

  layers.push(new TextLayer({
    id: `${side}-landmark-labels`,
    data: lmPoints,
    getPosition: (d) => [d.x + 8, d.y - 8],
    getText: (d) => d.match_id,
    getSize: 13,
    getColor: colors.text,
    getTextAnchor: 'start',
    getAlignmentBaseline: 'center'
  }))

  return layers
}

function initialViewState(size) {
  const width = Number(size?.[0] || 1)
  const height = Number(size?.[1] || 1)
  return {
    target: [width / 2, height / 2, 0],
    zoom: -1,
    minZoom: -8,
    maxZoom: 8
  }
}

function makeDeck(model, container, side) {
  const size = side === 'motility' ? model.get('motility_size') : currentCellPaintingSize(model)
  return new Deck({
    parent: container,
    views: [new OrthographicView({id: side, controller: true})],
    initialViewState: initialViewState(size),
    layers: makeLayers(model, side),
    getCursor: ({isDragging, isHovering}) => {
      if (isDragging) return 'grabbing'
      if (isHovering) return 'pointer'
      return 'crosshair'
    },
    getTooltip: ({object}) => {
      if (!object) return null
      const id = object.id || object.properties?.point_id
      return id ? String(id) : null
    }
  })
}

function render({model, el}) {
  el.classList.add('cell-motility-painting-aligner')
  el.style.width = `${model.get('width') || 1100}px`

  const toolbar = document.createElement('div')
  toolbar.className = 'cell-motility-toolbar'

  const prevButton = document.createElement('button')
  prevButton.textContent = 'Prev'
  const nextButton = document.createElement('button')
  nextButton.textContent = 'Next'
  const clearPendingButton = document.createElement('button')
  clearPendingButton.textContent = 'Clear pending'
  const deleteButton = document.createElement('button')
  deleteButton.textContent = 'Delete match'
  const clearMatchesButton = document.createElement('button')
  clearMatchesButton.textContent = 'Clear image'
  const imageLabel = document.createElement('span')
  imageLabel.className = 'cell-motility-meta'
  const matchLabel = document.createElement('span')
  matchLabel.className = 'cell-motility-meta'
  const status = document.createElement('span')
  status.className = 'cell-motility-status'

  toolbar.append(
    prevButton,
    nextButton,
    clearPendingButton,
    deleteButton,
    clearMatchesButton,
    imageLabel,
    matchLabel,
    status
  )
  el.appendChild(toolbar)

  const panels = document.createElement('div')
  panels.className = 'cell-motility-panels'

  const motilityPanel = document.createElement('div')
  motilityPanel.className = 'cell-motility-panel'
  const cellPaintingPanel = document.createElement('div')
  cellPaintingPanel.className = 'cell-motility-panel'
  motilityPanel.style.height = `${model.get('height') || 560}px`
  cellPaintingPanel.style.height = `${model.get('height') || 560}px`

  const motilityLabel = document.createElement('div')
  motilityLabel.className = 'cell-motility-label'
  const cellPaintingLabel = document.createElement('div')
  cellPaintingLabel.className = 'cell-motility-label'
  motilityPanel.appendChild(motilityLabel)
  cellPaintingPanel.appendChild(cellPaintingLabel)

  panels.append(motilityPanel, cellPaintingPanel)
  el.appendChild(panels)

  const motilityDeck = makeDeck(model, motilityPanel, 'motility')
  const cellPaintingDeck = makeDeck(model, cellPaintingPanel, 'cell_painting')

  function updateLabels() {
    const count = imageCount(model)
    const index = currentIndex(model)
    const matches = currentMatches(model)
    const pending = model.get('pending_motility') || {}
    const active = model.get('active_match_id') || ''
    imageLabel.textContent = `image ${count ? index + 1 : 0} / ${count}`
    matchLabel.textContent = `${matches.length} matches${pending.id ? ' + pending' : ''}`
    status.textContent = model.get('status') || ''
    motilityLabel.textContent = model.get('motility_label') || 'MOTILITY / REFERENCE'
    cellPaintingLabel.textContent = model.get('cell_painting_label') || 'CELL PAINTING / MOVING'
    prevButton.disabled = count <= 0 || index <= 0
    nextButton.disabled = count <= 0 || index >= count - 1
    deleteButton.disabled = matches.length === 0 && active !== 'pending'
    clearMatchesButton.disabled = matches.length === 0
  }

  function updateLayers() {
    motilityDeck.setProps({layers: makeLayers(model, 'motility')})
    cellPaintingDeck.setProps({layers: makeLayers(model, 'cell_painting')})
    updateLabels()
  }

  prevButton.addEventListener('click', () => {
    const count = imageCount(model)
    const index = currentIndex(model)
    if (count === 0) return
    model.set('cell_painting_index', Math.max(0, index - 1))
    model.set('pending_motility', {})
    model.set('active_match_id', '')
    model.save_changes()
  })

  nextButton.addEventListener('click', () => {
    const count = imageCount(model)
    const index = currentIndex(model)
    if (count === 0) return
    model.set('cell_painting_index', Math.min(count - 1, index + 1))
    model.set('pending_motility', {})
    model.set('active_match_id', '')
    model.save_changes()
  })

  clearPendingButton.addEventListener('click', () => {
    model.set('pending_motility', {})
    model.set('active_match_id', '')
    model.set('status', 'cleared pending motility selection')
    model.save_changes()
  })

  deleteButton.addEventListener('click', () => removeActiveOrLastMatch(model))
  clearMatchesButton.addEventListener('click', () => clearCurrentMatches(model))

  const observed = [
    'motility_image_url',
    'cell_painting_image_urls',
    'cell_painting_index',
    'motility_size',
    'cell_painting_size',
    'cell_painting_sizes',
    'motility_centroids',
    'cell_painting_centroids_by_image',
    'matches_by_image',
    'pending_motility',
    'active_match_id',
    'status',
    'point_radius',
    'motility_label',
    'cell_painting_label'
  ]

  for (const name of observed) {
    model.on(`change:${name}`, updateLayers)
  }

  updateLayers()

  return () => {
    for (const name of observed) {
      model.off(`change:${name}`, updateLayers)
    }
    motilityDeck.finalize()
    cellPaintingDeck.finalize()
  }
}

export default {render}

