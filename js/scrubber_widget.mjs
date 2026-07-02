// deck.gl is vendored as its self-contained standalone UMD bundle
// (js/vendor/deckgl.min.cjs, deck.gl 9.2.8) and bundled in offline, the same way
// celldega/the aligner ship a prebuilt deck.gl bundle. We vendor the prebuilt
// dist rather than installing the full dependency tree because this environment's
// npm is impractically slow (~140s/package across deck.gl's hundreds of deps).
// NOTE: the vendored file MUST keep its .cjs extension — as .js, esbuild aliases
// the UMD namespace to an empty object and every deck.gl symbol becomes undefined.
import * as deckgl from './vendor/deckgl.min.cjs'
const deck = deckgl.default ?? deckgl
const {BitmapLayer, Deck, OrthographicView, PathLayer, ScatterplotLayer} = deck

// Deterministic bright-ish color per trajectory id so a cell keeps its hue.
function colorForId(id) {
  let h = (Number(id) + 1) * 2654435761 % 2 ** 32
  const r = (h & 255)
  const g = ((h >> 8) & 255)
  const b = ((h >> 16) & 255)
  // push away from dark so points read on the dark brightfield
  return [80 + (r % 176), 80 + (g % 176), 80 + (b % 176)]
}

function clampFrame(model) {
  const n = (model.get('frame_urls') || []).length
  const f = Number(model.get('current_frame') || 0)
  if (n <= 0) return 0
  return Math.max(0, Math.min(n - 1, f))
}

// Full-path geometry per trajectory is independent of the current frame, so we
// compute it once (and again only when `trajectories` changes) to keep playback
// cheap with thousands of tracks.
function buildPathCache(model) {
  const trajectories = model.get('trajectories') || []
  return trajectories.map((traj) => {
    const pts = traj.pts || []
    return {
      id: traj.id,
      color: colorForId(traj.id),
      full: pts.map((p) => [p[1], p[2]]),
      pts,
    }
  })
}

function render({model, el}) {
  el.classList.add('motility-painting')
  el.style.width = `${model.get('width') || 820}px`

  const toolbar = document.createElement('div')
  toolbar.className = 'cell-motility-toolbar'
  const playButton = document.createElement('button')
  const slider = document.createElement('input')
  slider.type = 'range'
  slider.min = '0'
  slider.step = '1'
  slider.style.flex = '1'
  const frameLabel = document.createElement('span')
  frameLabel.className = 'cell-motility-meta'
  const imageToggle = document.createElement('button')
  const bgToggle = document.createElement('button')
  const trailMeta = document.createElement('span')
  trailMeta.className = 'cell-motility-meta'
  toolbar.append(playButton, slider, frameLabel, imageToggle, bgToggle, trailMeta)
  el.appendChild(toolbar)

  const panel = document.createElement('div')
  panel.className = 'cell-motility-panel'
  panel.style.height = `${model.get('height') || 820}px`
  el.appendChild(panel)

  let pathCache = buildPathCache(model)

  function makeLayers() {
    const f = clampFrame(model)
    const size = model.get('image_size') || [2960, 2960]
    const width = Number(size[0] || 1)
    const height = Number(size[1] || 1)
    const radius = Number(model.get('point_radius') || 6)
    const fullPath = model.get('show_full_path')
    const windowLen = Number(model.get('trail_length') || 0)
    const start = fullPath ? 0 : (windowLen > 0 ? Math.max(0, f - windowLen) : 0)
    const layers = []

    // brightfield frame, stretched to native pixel bounds
    if (model.get('show_image')) {
      const url = (model.get('frame_urls') || [])[f]
      if (url) {
        layers.push(new BitmapLayer({id: 'frame-image', image: url, bounds: [0, height, width, 0]}))
      }
    }

    // every detected cell this frame, as small black dots under the tracks
    if (model.get('show_background')) {
      const bg = (model.get('background_points') || [])[f] || []
      layers.push(new ScatterplotLayer({
        id: 'background',
        data: bg,
        getPosition: (d) => d,
        getRadius: radius * 0.6,
        radiusUnits: 'common',        // world units => shrink when zoomed out
        radiusMinPixels: 0.5,
        radiusMaxPixels: 6,
        getFillColor: [0, 0, 0, 235],
        stroked: true,
        getLineColor: [235, 235, 235, 120],
        lineWidthUnits: 'pixels',
        getLineWidth: 0.4,
      }))
    }

    // tracked-cell paths: full future (faint) + past (bright), plus the head dot
    const future = []
    const past = []
    const heads = []
    for (const t of pathCache) {
      if (fullPath) {
        const futSeg = []
        const pastSeg = []
        for (const p of t.pts) {
          const pos = p[0]
          if (pos >= f) futSeg.push([p[1], p[2]])
          if (pos <= f) pastSeg.push([p[1], p[2]])
        }
        if (futSeg.length >= 2) future.push({id: t.id, color: t.color, path: futSeg})
        if (pastSeg.length >= 2) past.push({id: t.id, color: t.color, path: pastSeg})
      } else {
        const seg = []
        for (const p of t.pts) {
          if (p[0] >= start && p[0] <= f) seg.push([p[1], p[2]])
        }
        if (seg.length >= 2) past.push({id: t.id, color: t.color, path: seg})
      }
      const head = t.pts.find((p) => p[0] === f)
      if (head) heads.push({id: t.id, color: t.color, x: head[1], y: head[2]})
    }

    if (fullPath) {
      layers.push(new PathLayer({
        id: 'future', data: future,
        getPath: (d) => d.path,
        getColor: (d) => [...d.color, 70],
        getWidth: 1.2, widthUnits: 'pixels', capRounded: true, jointRounded: true,
      }))
    }
    layers.push(new PathLayer({
      id: 'past', data: past,
      getPath: (d) => d.path,
      getColor: (d) => [...d.color, 200],
      getWidth: 1.6, widthUnits: 'pixels', capRounded: true, jointRounded: true,
    }))
    layers.push(new ScatterplotLayer({
      id: 'heads', data: heads,
      pickable: true,
      getPosition: (d) => [d.x, d.y],
      getRadius: radius,
      radiusUnits: 'common',          // world units => shrink when zoomed out
      radiusMinPixels: 1.5,
      radiusMaxPixels: 14,
      getFillColor: (d) => [...d.color, 255],
      stroked: true,
      getLineColor: [20, 20, 20, 220],
      lineWidthUnits: 'pixels',
      getLineWidth: 0.5,
    }))
    return layers
  }

  function initialViewState(size) {
    const width = Number(size?.[0] || 1)
    const height = Number(size?.[1] || 1)
    return {target: [width / 2, height / 2, 0], zoom: -2, minZoom: -8, maxZoom: 8}
  }

  const deckInstance = new Deck({
    parent: panel,
    views: [new OrthographicView({id: 'scrubber', controller: true})],
    initialViewState: initialViewState(model.get('image_size')),
    layers: makeLayers(),
    getTooltip: ({object}) => (object && object.id != null ? `cell ${object.id}` : null),
  })

  let timer = null
  const frameCount = () => (model.get('frame_urls') || []).length

  function refreshControls() {
    const n = frameCount()
    const f = clampFrame(model)
    slider.max = String(Math.max(0, n - 1))
    slider.value = String(f)
    const idx = (model.get('frame_indices') || [])[f]
    frameLabel.textContent = n ? `frame ${f + 1}/${n}${idx != null ? ` (T${idx})` : ''}` : 'no frames'
    playButton.textContent = model.get('playing') ? '❚❚ Pause' : '► Play'
    imageToggle.textContent = model.get('show_image') ? 'Hide image' : 'Show image'
    bgToggle.textContent = model.get('show_background') ? 'Hide others' : 'Show others'
    const mode = model.get('show_full_path') ? 'full path' : `trail ${model.get('trail_length') || 'full'}`
    trailMeta.textContent = `${(model.get('trajectories') || []).length} tracked · ${mode}`
  }

  function updateLayers() {
    deckInstance.setProps({layers: makeLayers()})
    refreshControls()
  }

  function stopTimer() {
    if (timer != null) { clearInterval(timer); timer = null }
  }
  function startTimer() {
    stopTimer()
    const fps = Math.max(1, Number(model.get('fps') || 8))
    timer = setInterval(() => {
      const n = frameCount()
      if (n <= 0) return
      model.set('current_frame', (clampFrame(model) + 1) % n)
      model.save_changes()
    }, 1000 / fps)
  }

  playButton.addEventListener('click', () => { model.set('playing', !model.get('playing')); model.save_changes() })
  imageToggle.addEventListener('click', () => { model.set('show_image', !model.get('show_image')); model.save_changes() })
  bgToggle.addEventListener('click', () => { model.set('show_background', !model.get('show_background')); model.save_changes() })
  slider.addEventListener('input', () => { model.set('current_frame', Number(slider.value)); model.save_changes() })

  model.on('change:playing', () => {
    if (model.get('playing')) startTimer(); else stopTimer()
    refreshControls()
  })
  model.on('change:current_frame', updateLayers)
  model.on('change:trajectories', () => { pathCache = buildPathCache(model); updateLayers() })
  const simple = ['frame_urls', 'background_points', 'trail_length', 'point_radius',
                  'show_image', 'show_background', 'show_full_path', 'image_size', 'fps']
  for (const name of simple) {
    model.on(`change:${name}`, () => {
      if (name === 'fps' && model.get('playing')) startTimer()
      updateLayers()
    })
  }

  if (model.get('playing')) startTimer()
  updateLayers()

  return () => {
    stopTimer()
    model.off()
    deckInstance.finalize()
  }
}

export default {render}
