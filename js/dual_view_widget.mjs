// Two-panel QC viewer: motility (left) | stitched Cell Painting (right), sharing
// one LCI coordinate space and one synced pan/zoom. See viz/dual_view.py.
//
// deck.gl is vendored as its self-contained standalone UMD bundle
// (js/vendor/deckgl.min.cjs, deck.gl 9.2.8) and bundled in offline. The .cjs
// extension is mandatory: as .js, esbuild aliases the UMD namespace to an empty
// object and every deck.gl symbol becomes undefined (blank widget, no error).
import * as deckgl from './vendor/deckgl.min.cjs'
const deck = deckgl.default ?? deckgl
const {BitmapLayer, Deck, OrthographicView, ScatterplotLayer, TextLayer} = deck

const C_MATCHED = [64, 196, 120]     // green
const C_UNMATCHED = [235, 120, 70]   // orange
const C_SELECTED = [255, 230, 40]    // yellow highlight
const LABEL_MIN_ZOOM = -1.0          // only draw id labels once zoomed in this far

function clampFrame(model) {
  const n = (model.get('motility_frame_urls') || []).length
  const f = Number(model.get('current_frame') || 0)
  if (n <= 0) return 0
  return Math.max(0, Math.min(n - 1, f))
}

function pointColor(d, selected) {
  if (selected >= 0 && d.link_id === selected) return C_SELECTED
  return d.link_id >= 0 ? C_MATCHED : C_UNMATCHED
}

function render({model, el}) {
  el.classList.add('mcp-dual')
  el.style.width = `${model.get('width') || 1180}px`

  // ---- toolbar ----
  const toolbar = document.createElement('div')
  toolbar.className = 'mcp-toolbar'
  const slider = document.createElement('input')
  slider.type = 'range'; slider.min = '0'; slider.step = '1'; slider.style.flex = '1'
  const frameLabel = document.createElement('span'); frameLabel.className = 'mcp-meta'
  const matchedBtn = document.createElement('button')
  const unmatchedBtn = document.createElement('button')
  const cpImgBtn = document.createElement('button')
  const motImgBtn = document.createElement('button')
  const labelsBtn = document.createElement('button')
  const info = document.createElement('span'); info.className = 'mcp-meta mcp-info'
  toolbar.append(slider, frameLabel, matchedBtn, unmatchedBtn, motImgBtn, cpImgBtn, labelsBtn, info)
  el.appendChild(toolbar)

  // ---- two panels ----
  const panels = document.createElement('div')
  panels.className = 'mcp-panels'
  const leftWrap = document.createElement('div'); leftWrap.className = 'mcp-panel'
  const rightWrap = document.createElement('div'); rightWrap.className = 'mcp-panel'
  const leftTitle = document.createElement('div'); leftTitle.className = 'mcp-title'; leftTitle.textContent = 'MOTILITY (final LCI frame)'
  const rightTitle = document.createElement('div'); rightTitle.className = 'mcp-title'; rightTitle.textContent = 'CELL PAINTING (stitched → LCI)'
  const leftCanvas = document.createElement('div'); leftCanvas.className = 'mcp-canvas'
  const rightCanvas = document.createElement('div'); rightCanvas.className = 'mcp-canvas'
  leftWrap.append(leftTitle, leftCanvas); rightWrap.append(rightTitle, rightCanvas)
  panels.append(leftWrap, rightWrap)
  const ph = model.get('height') || 620
  leftCanvas.style.height = rightCanvas.style.height = `${ph}px`
  el.appendChild(panels)

  // shared view state across both decks
  const sz = model.get('motility_size') || [2960, 2960]
  let viewState = {target: [Number(sz[0]) / 2, Number(sz[1]) / 2, 0], zoom: -2.4, minZoom: -8, maxZoom: 8}
  let syncing = false

  function motilityLayers() {
    const layers = []
    const f = clampFrame(model)
    const w = Number(sz[0] || 1), h = Number(sz[1] || 1)
    if (model.get('show_motility_image')) {
      const url = (model.get('motility_frame_urls') || [])[f]
      if (url) layers.push(new BitmapLayer({id: 'mot-img', image: url, bounds: [0, h, w, 0]}))
    }
    const pts = model.get('motility_points') || []
    layers.push(pointLayer('mot-pts', pts))
    const lbl = labelLayer('mot-lbl', pts)
    if (lbl) layers.push(lbl)
    return layers
  }

  function cpLayers() {
    const layers = []
    if (model.get('show_cp_images')) {
      const op = Number(model.get('cp_opacity'))
      const bitmaps = model.get('cp_bitmaps') || []
      bitmaps.forEach((b, i) => {
        if (b && b.url && b.bounds) {
          layers.push(new BitmapLayer({id: `cp-img-${i}`, image: b.url, bounds: b.bounds, opacity: op}))
        }
      })
    }
    const pts = model.get('cp_points') || []
    layers.push(pointLayer('cp-pts', pts))
    const lbl = labelLayer('cp-lbl', pts)
    if (lbl) layers.push(lbl)
    return layers
  }

  // Labels use link_id so a matched cell shows the SAME number in both panels,
  // making cross-panel comparison direct. Only matched cells are labelled, and
  // only once zoomed in past LABEL_MIN_ZOOM (avoids clutter / keeps it fast).
  function labelLayer(id, data) {
    if (!model.get('show_labels') || viewState.zoom < LABEL_MIN_ZOOM) return null
    const showM = model.get('show_matched')
    if (!showM) return null
    const labelData = data.filter((d) => d.link_id >= 0).map((d) => ({x: d.x, y: d.y, text: String(d.link_id)}))
    return new TextLayer({
      id, data: labelData,
      getPosition: (d) => [d.x, d.y],
      getText: (d) => d.text,
      getSize: 11, sizeUnits: 'pixels',
      getColor: [245, 245, 245, 255],
      getPixelOffset: [0, -9],
      fontFamily: 'monospace', fontWeight: 700,
      background: true, getBackgroundColor: [15, 23, 42, 170], backgroundPadding: [2, 1],
      pickable: false,
    })
  }

  function pointLayer(id, data) {
    const selected = Number(model.get('selected_link_id'))
    const radius = Number(model.get('point_radius') || 6)
    const showM = model.get('show_matched'), showU = model.get('show_unmatched')
    const shown = data.filter((d) => (d.link_id >= 0 ? showM : showU))
    return new ScatterplotLayer({
      id, data: shown, pickable: true,
      getPosition: (d) => [d.x, d.y],
      getRadius: (d) => (selected >= 0 && d.link_id === selected ? radius * 1.8 : radius),
      radiusUnits: 'common', radiusMinPixels: 1.5, radiusMaxPixels: 16,
      getFillColor: (d) => [...pointColor(d, selected), 235],
      stroked: true, lineWidthUnits: 'pixels',
      getLineWidth: (d) => (selected >= 0 && d.link_id === selected ? 2 : 0.5),
      getLineColor: (d) => (selected >= 0 && d.link_id === selected ? [20, 20, 20, 255] : [20, 20, 20, 180]),
      updateTriggers: {
        getFillColor: [selected], getRadius: [selected], getLineWidth: [selected],
        getLineColor: [selected], data: [showM, showU],
      },
      onClick: ({object}) => { if (object) onPick(object); return true },
    })
  }

  function onPick(object) {
    if (object.link_id >= 0) {
      const cur = Number(model.get('selected_link_id'))
      const next = cur === object.link_id ? -1 : object.link_id
      model.set('selected_link_id', next); model.save_changes()
      info.textContent = next >= 0 ? `link ${next} selected` : ''
    } else {
      info.textContent = `unmatched: ${object.id}`
    }
  }

  const tooltip = ({object}) => {
    if (!object) return null
    const kind = object.link_id >= 0 ? `matched · link ${object.link_id}` : 'unmatched'
    return {text: `${object.id}\n${kind}`}
  }

  function makeDeck(parent, makeLayers, viewId) {
    return new Deck({
      parent,
      views: [new OrthographicView({id: viewId, controller: true})],
      viewState,
      layers: makeLayers(),
      getTooltip: tooltip,
      onViewStateChange: ({viewState: vs}) => {
        if (syncing) return
        const wasAbove = viewState.zoom >= LABEL_MIN_ZOOM
        viewState = vs
        syncing = true
        leftDeck.setProps({viewState}); rightDeck.setProps({viewState})
        syncing = false
        // labels are zoom-gated: rebuild only when we cross the threshold
        if (model.get('show_labels') && (vs.zoom >= LABEL_MIN_ZOOM) !== wasAbove) update()
      },
    })
  }

  const leftDeck = makeDeck(leftCanvas, motilityLayers, 'mot')
  const rightDeck = makeDeck(rightCanvas, cpLayers, 'cp')

  function refreshControls() {
    const n = (model.get('motility_frame_urls') || []).length
    const f = clampFrame(model)
    slider.max = String(Math.max(0, n - 1)); slider.value = String(f)
    slider.style.display = n > 1 ? '' : 'none'
    const idx = (model.get('motility_frame_indices') || [])[f]
    frameLabel.textContent = n > 1 ? `frame ${f + 1}/${n}${idx != null ? ` (T${idx})` : ''}` : ''
    matchedBtn.textContent = model.get('show_matched') ? 'Hide matched' : 'Show matched'
    unmatchedBtn.textContent = model.get('show_unmatched') ? 'Hide unmatched' : 'Show unmatched'
    cpImgBtn.textContent = model.get('show_cp_images') ? 'Hide CP tiles' : 'Show CP tiles'
    motImgBtn.textContent = model.get('show_motility_image') ? 'Hide motility img' : 'Show motility img'
    labelsBtn.textContent = model.get('show_labels') ? 'Hide ids' : 'Show ids'
    const nm = (model.get('motility_points') || []).filter((d) => d.link_id >= 0).length
    if (!info.textContent) info.textContent = `${nm} matched pairs`
  }

  function update() {
    leftDeck.setProps({layers: motilityLayers()})
    rightDeck.setProps({layers: cpLayers()})
    refreshControls()
  }

  slider.addEventListener('input', () => { model.set('current_frame', Number(slider.value)); model.save_changes() })
  matchedBtn.addEventListener('click', () => { model.set('show_matched', !model.get('show_matched')); model.save_changes() })
  unmatchedBtn.addEventListener('click', () => { model.set('show_unmatched', !model.get('show_unmatched')); model.save_changes() })
  cpImgBtn.addEventListener('click', () => { model.set('show_cp_images', !model.get('show_cp_images')); model.save_changes() })
  motImgBtn.addEventListener('click', () => { model.set('show_motility_image', !model.get('show_motility_image')); model.save_changes() })
  labelsBtn.addEventListener('click', () => { model.set('show_labels', !model.get('show_labels')); model.save_changes() })

  const watch = ['motility_frame_urls', 'motility_points', 'cp_bitmaps', 'cp_points',
                 'current_frame', 'selected_link_id', 'point_radius', 'show_matched',
                 'show_unmatched', 'show_cp_images', 'show_motility_image', 'cp_opacity', 'show_labels']
  for (const name of watch) model.on(`change:${name}`, update)

  update()

  return () => { model.off(); leftDeck.finalize(); rightDeck.finalize() }
}

export default {render}
