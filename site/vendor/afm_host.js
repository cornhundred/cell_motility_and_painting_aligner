// Minimal static host for Anywidget Front-End Modules (AFM), per the spec at
// https://anywidget.dev/en/afm -- implements just enough of the `AnyModel`
// interface for a widget's own JS to run fully client-side, with no Jupyter
// kernel, no ipywidgets/html-manager, and no CDN dependency.
//
// Why this exists: anywidget's real npm package ships pure ESM ("type":
// "module", no UMD/AMD build), so the classic ipywidgets static-embedding
// path (nbconvert's HTMLExporter, ipywidgets.embed.embed_minimal_html --
// both route through @jupyter-widgets/html-manager's requirejs/AMD loader)
// can never load it correctly, regardless of CDN reachability or version
// pinning. AFM is anywidget's own documented answer for "standalone web
// applications" -- this file is a from-scratch implementation of exactly
// the host contract it specifies, nothing borrowed from ipywidgets.

/**
 * @param {Record<string, any>} initialState
 */
export function makeStaticModel(initialState) {
  const state = { ...initialState };
  const pending = {};
  const listeners = {};

  function fire(eventName) {
    (listeners[eventName] || []).slice().forEach((cb) => cb());
  }

  return {
    get(key) {
      return key in pending ? pending[key] : state[key];
    },
    set(key, value) {
      pending[key] = value;
    },
    save_changes() {
      const changedKeys = Object.keys(pending).filter((k) => state[k] !== pending[k]);
      Object.assign(state, pending);
      for (const k of Object.keys(pending)) delete pending[k];
      changedKeys.forEach((k) => fire(`change:${k}`));
    },
    on(eventName, callback) {
      (listeners[eventName] ||= []).push(callback);
    },
    off(eventName, callback) {
      if (!eventName) {
        for (const k in listeners) listeners[k] = [];
        return;
      }
      if (!callback) {
        listeners[eventName] = [];
        return;
      }
      listeners[eventName] = (listeners[eventName] || []).filter((f) => f !== callback);
    },
    send() {
      // No host to send custom messages to -- intentionally a no-op.
    },
  };
}

// ipywidgets/anywidget traits too large for plain JSON (e.g. a parquet-encoded
// dataframe) travel as separate binary "buffers" rather than inline state --
// real hosts reinsert them as a DataView at the trait's path before the
// widget ever sees them (this is exactly the shape @jupyter-widgets/base
// uses). build_site.py's export_widget_page marks these in `state` as
// `{__buffer_b64__: "<base64>"}`; hydrateBuffers turns them into real
// DataViews so `model.get("mat_parquet")` etc. return what the widget expects.
function hydrateBuffers(state) {
  for (const key of Object.keys(state)) {
    const value = state[key];
    if (value && typeof value === "object" && typeof value.__buffer_b64__ === "string") {
      const binary = atob(value.__buffer_b64__);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      state[key] = new DataView(bytes.buffer);
    }
  }
  return state;
}

/**
 * Render an AFM module into `el`.
 * @param {{esmUrl: string, cssUrl?: string, state: Record<string, any>, el: HTMLElement}} opts
 */
export async function renderAFM({ esmUrl, cssUrl, state, el }) {
  if (cssUrl) {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = cssUrl;
    document.head.appendChild(link);
  }

  const mod = await import(esmUrl);
  let widget = mod.default;
  if (typeof widget === "function") widget = await widget();

  const model = makeStaticModel(hydrateBuffers(state));
  const controller = new AbortController();

  if (widget.initialize) {
    await widget.initialize({ model, signal: controller.signal });
  }
  await widget.render({ model, el, signal: controller.signal, host: {} });
  return { model, controller };
}
