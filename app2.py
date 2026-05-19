from flask import Flask, request, jsonify
import numpy as np
import plotly.graph_objects as go
import json

app = Flask(__name__)

# ─── Materiales predefinidos (sigma [S/m], epsilon_r, mu_r) ───────────────────
MATERIALES = {
    "Cobre":        (5.8e7,   1.0,  1.0),
    "Aluminio":     (3.5e7,   1.0,  1.0),
    "Agua de mar":  (4.0,    80.0,  1.0),
    "Tierra húmeda":(0.01,   10.0,  1.0),
    "Tierra seca":  (0.001,   4.0,  1.0),
    "Vidrio":       (1e-12,   6.0,  1.0),
    "Aire":         (1e-15,   1.0,  1.0),
    "Músculo":      (0.7,    60.0,  1.0),
}

# ─── Cálculo electromagnético ─────────────────────────────────────────────────
def calcular(frecuencias, sigma, epsilon_r, mu_r):
    e0 = 8.854187817e-12
    u0 = 4 * np.pi * 1e-7
    omega = 2 * np.pi * frecuencias
    eps   = epsilon_r * e0
    mu    = mu_r * u0
    gamma = 1j * omega * np.sqrt(mu * eps) * np.sqrt(1 - 1j * sigma / (omega * eps + 1e-30))
    alpha = np.real(gamma)
    beta  = np.imag(gamma)
    delta = np.where(alpha > 1e-20, 1.0 / alpha, np.full_like(alpha, np.nan))
    return alpha, beta, delta

# ─── HTML completo embebido ───────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Análisis EM de Materiales</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:       #040810;
    --panel:    #080f1e;
    --border:   #0d2a4a;
    --cyan:     #00e5ff;
    --orange:   #ff6d00;
    --green:    #39ff14;
    --purple:   #7c3aed;
    --text:     #b0c4de;
    --grid-line:#0d2a4a;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Share Tech Mono', monospace;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Fondo animado con rejilla */
  body::before {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background-image:
      linear-gradient(var(--grid-line) 1px, transparent 1px),
      linear-gradient(90deg, var(--grid-line) 1px, transparent 1px);
    background-size: 40px 40px;
    opacity: 0.4;
    pointer-events: none;
  }

  .wrapper { position: relative; z-index: 1; max-width: 1400px; margin: 0 auto; padding: 1.5rem; }

  /* HEADER */
  header {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
  }
  header .tag {
    display: inline-block;
    font-size: .7rem;
    letter-spacing: .2em;
    color: var(--cyan);
    border: 1px solid var(--cyan);
    padding: .2rem .8rem;
    margin-bottom: .8rem;
    text-transform: uppercase;
  }
  header h1 {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(1.4rem, 4vw, 2.6rem);
    font-weight: 900;
    color: #fff;
    text-shadow: 0 0 30px var(--cyan), 0 0 60px rgba(0,229,255,.3);
    letter-spacing: .05em;
    line-height: 1.2;
  }
  header h1 span { color: var(--cyan); }
  header p {
    margin-top: .8rem;
    font-size: .85rem;
    color: #5a7a9a;
    max-width: 700px;
    margin-inline: auto;
    line-height: 1.6;
  }

  /* LAYOUT */
  .layout { display: grid; grid-template-columns: 310px 1fr; gap: 1.5rem; }
  @media(max-width:900px){ .layout { grid-template-columns: 1fr; } }

  /* PANEL CONTROLES */
  .panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
    align-self: start;
    position: sticky;
    top: 1rem;
  }
  .panel-title {
    font-family: 'Orbitron', sans-serif;
    font-size: .75rem;
    letter-spacing: .15em;
    color: var(--cyan);
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: .75rem;
  }

  /* CAMPOS */
  .field { display: flex; flex-direction: column; gap: .4rem; }
  .field label {
    font-size: .72rem;
    letter-spacing: .08em;
    color: #5a7a9a;
    text-transform: uppercase;
  }
  .field input, .field select {
    background: #0a1628;
    border: 1px solid var(--border);
    color: var(--cyan);
    font-family: 'Share Tech Mono', monospace;
    font-size: .9rem;
    padding: .55rem .75rem;
    border-radius: 3px;
    outline: none;
    transition: border-color .2s;
  }
  .field input:focus, .field select:focus { border-color: var(--cyan); }
  .field select option { background: #0a1628; color: var(--text); }

  /* PANEL PERSONALIZADO */
  #custom-panel {
    display: none;
    flex-direction: column;
    gap: 1rem;
    padding: 1rem;
    background: #060e1c;
    border: 1px dashed var(--border);
    border-radius: 3px;
  }
  #custom-panel.visible { display: flex; }
  #custom-panel .field label { color: #7a6a4a; }

  /* SEPARADOR */
  .sep {
    font-size: .65rem;
    letter-spacing: .1em;
    color: #2a4a6a;
    text-transform: uppercase;
    text-align: center;
  }

  /* BOTÓN */
  .btn-calc {
    background: transparent;
    border: 1px solid var(--cyan);
    color: var(--cyan);
    font-family: 'Orbitron', sans-serif;
    font-size: .8rem;
    letter-spacing: .15em;
    text-transform: uppercase;
    padding: .9rem;
    border-radius: 3px;
    cursor: pointer;
    position: relative;
    overflow: hidden;
    transition: color .2s, background .2s, box-shadow .2s;
  }
  .btn-calc:hover {
    background: var(--cyan);
    color: var(--bg);
    box-shadow: 0 0 20px var(--cyan), 0 0 40px rgba(0,229,255,.3);
  }
  .btn-calc:active { transform: scale(.98); }

  /* BADGE INFO */
  #info-badge {
    display: none;
    font-size: .72rem;
    padding: .75rem;
    background: #060e1c;
    border: 1px solid var(--border);
    border-radius: 3px;
    line-height: 1.8;
    color: #5a7a9a;
  }
  #info-badge.visible { display: block; }
  #info-badge b { color: var(--cyan); }

  /* GRAFICAS */
  .graficas { display: flex; flex-direction: column; gap: 1.2rem; }
  .grafica-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
  }
  .grafica-header {
    display: flex;
    align-items: center;
    gap: .75rem;
    padding: .75rem 1rem;
    border-bottom: 1px solid var(--border);
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .dot-cyan   { background: var(--cyan);   box-shadow: 0 0 8px var(--cyan); }
  .dot-orange { background: var(--orange); box-shadow: 0 0 8px var(--orange); }
  .dot-green  { background: var(--green);  box-shadow: 0 0 8px var(--green); }
  .grafica-header span {
    font-family: 'Orbitron', sans-serif;
    font-size: .7rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #7a9ab8;
  }
  .plot-area { padding: .5rem; }

  /* PLACEHOLDER */
  .placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 280px;
    color: #1a3a5a;
    font-size: .8rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    flex-direction: column;
    gap: 1rem;
  }
  .placeholder svg { opacity: .3; }

  /* LOADER */
  #loader {
    display: none;
    position: fixed; inset: 0; z-index: 999;
    background: rgba(4,8,16,.85);
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 1.5rem;
  }
  #loader.visible { display: flex; }
  .spinner {
    width: 48px; height: 48px;
    border: 2px solid var(--border);
    border-top-color: var(--cyan);
    border-radius: 50%;
    animation: spin .8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  #loader p { font-family: 'Orbitron', sans-serif; font-size: .75rem; letter-spacing: .2em; color: var(--cyan); }

  footer {
    text-align: center;
    padding: 2rem;
    color: #1a3a5a;
    font-size: .7rem;
    letter-spacing: .1em;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
  }
</style>
</head>
<body>

<!-- Loader -->
<div id="loader">
  <div class="spinner"></div>
  <p>Calculando...</p>
</div>

<div class="wrapper">
  <header>
    <div class="tag">Electromagnetismo Aplicado</div>
    <h1>Análisis de <span>Materiales</span><br>Electromagnéticos</h1>
    <p>Calcula y grafica la constante de propagación γ = α + jβ y la profundidad
       de penetración δ en función de la frecuencia para distintos materiales.</p>
  </header>

  <div class="layout">
    <!-- Panel de controles -->
    <aside class="panel">
      <div class="panel-title">⚙ Parámetros</div>

      <div class="field">
        <label>Material</label>
        <select id="sel-material">
          <option value="Cobre">Cobre</option>
          <option value="Aluminio">Aluminio</option>
          <option value="Agua de mar">Agua de mar</option>
          <option value="Tierra húmeda">Tierra húmeda</option>
          <option value="Tierra seca">Tierra seca</option>
          <option value="Vidrio">Vidrio</option>
          <option value="Aire">Aire</option>
          <option value="Músculo">Músculo (tejido biológico)</option>
          <option value="personalizado">⚡ Personalizado...</option>
        </select>
      </div>

      <div id="custom-panel">
        <div class="field">
          <label>σ — Conductividad (S/m)</label>
          <input type="number" id="sigma" value="1" step="any" min="0">
        </div>
        <div class="field">
          <label>ε_r — Permitividad relativa</label>
          <input type="number" id="eps_r" value="1" step="any" min="1">
        </div>
        <div class="field">
          <label>μ_r — Permeabilidad relativa</label>
          <input type="number" id="mu_r" value="1" step="any" min="1">
        </div>
      </div>

      <div class="sep">── Rango de frecuencias ──</div>

      <div class="field">
        <label>F mínima (Hz)</label>
        <input type="number" id="f_min" value="1000">
      </div>
      <div class="field">
        <label>F máxima (Hz)</label>
        <input type="number" id="f_max" value="10000000000">
      </div>
      <div class="field">
        <label>Puntos de muestreo</label>
        <input type="number" id="n_pts" value="400" min="50" max="2000">
      </div>

      <button class="btn-calc" id="btn">▶ CALCULAR Y GRAFICAR</button>

      <div id="info-badge"></div>
    </aside>

    <!-- Área de gráficas -->
    <section class="graficas">
      <div class="grafica-card">
        <div class="grafica-header">
          <div class="dot dot-cyan"></div>
          <span>Constante de Atenuación — α (Np/m)</span>
        </div>
        <div class="plot-area">
          <div id="g-alpha">
            <div class="placeholder">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#00e5ff" stroke-width="1.5">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
              </svg>
              Presiona CALCULAR para ver la gráfica
            </div>
          </div>
        </div>
      </div>

      <div class="grafica-card">
        <div class="grafica-header">
          <div class="dot dot-orange"></div>
          <span>Constante de Fase — β (rad/m)</span>
        </div>
        <div class="plot-area">
          <div id="g-beta">
            <div class="placeholder">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#ff6d00" stroke-width="1.5">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
              </svg>
              Presiona CALCULAR para ver la gráfica
            </div>
          </div>
        </div>
      </div>

      <div class="grafica-card">
        <div class="grafica-header">
          <div class="dot dot-green"></div>
          <span>Profundidad de Penetración — δ (m)</span>
        </div>
        <div class="plot-area">
          <div id="g-delta">
            <div class="placeholder">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#39ff14" stroke-width="1.5">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
              </svg>
              Presiona CALCULAR para ver la gráfica
            </div>
          </div>
        </div>
      </div>
    </section>
  </div>

  <footer>
    ANÁLISIS ELECTROMAGNÉTICO · γ = α + jβ · δ = 1/α · CONSTANTE DE PROPAGACIÓN
  </footer>
</div>

<script>
const LAYOUT_BASE = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor:  'rgba(8,15,30,0.9)',
  font:  { family: "'Share Tech Mono', monospace", color: '#5a7a9a', size: 11 },
  margin: { l: 70, r: 20, t: 20, b: 60 },
  height: 300,
  xaxis: {
    type: 'log', title: 'Frecuencia (Hz)',
    gridcolor: '#0d2a4a', linecolor: '#0d2a4a',
    tickcolor: '#0d2a4a', tickfont: { color: '#5a7a9a' }
  },
  yaxis: {
    type: 'log',
    gridcolor: '#0d2a4a', linecolor: '#0d2a4a',
    tickcolor: '#0d2a4a', tickfont: { color: '#5a7a9a' }
  },
  showlegend: false,
  hoverlabel: { bgcolor: '#040810', bordercolor: '#0d2a4a', font: { family: 'Share Tech Mono', color: '#00e5ff' } }
};

const CONFIG = { responsive: true, displayModeBar: false };

function plotLine(divId, x, y, color, ytitle) {
  const trace = {
    x, y, type: 'scatter', mode: 'lines',
    line: { color, width: 2.5, shape: 'spline' },
    hovertemplate: 'f = %{x:.3e} Hz<br>' + ytitle + ' = %{y:.4e}<extra></extra>'
  };
  const layout = JSON.parse(JSON.stringify(LAYOUT_BASE));
  layout.yaxis.title = ytitle;
  layout.yaxis.title = { text: ytitle, font: { color } };
  layout.xaxis.title = { text: 'Frecuencia (Hz)', font: { color: '#5a7a9a' } };

  // Glow effect via shape
  const glow = {
    x, y, type: 'scatter', mode: 'lines',
    line: { color, width: 12, shape: 'spline' },
    opacity: 0.08, hoverinfo: 'skip'
  };
  Plotly.newPlot(divId, [glow, trace], layout, CONFIG);
}

document.getElementById('sel-material').addEventListener('change', function () {
  const cp = document.getElementById('custom-panel');
  cp.classList.toggle('visible', this.value === 'personalizado');
});

document.getElementById('btn').addEventListener('click', async () => {
  const material = document.getElementById('sel-material').value;
  const esPers   = material === 'personalizado';
  const payload  = {
    material:     esPers ? null : material,
    personalizado: esPers,
    sigma:        parseFloat(document.getElementById('sigma').value),
    epsilon_r:    parseFloat(document.getElementById('eps_r').value),
    mu_r:         parseFloat(document.getElementById('mu_r').value),
    f_min:        parseFloat(document.getElementById('f_min').value),
    f_max:        parseFloat(document.getElementById('f_max').value),
    n_pts:        parseInt(document.getElementById('n_pts').value),
  };

  document.getElementById('loader').classList.add('visible');

  try {
    const res  = await fetch('/calcular', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    plotLine('g-alpha', data.f, data.alpha, '#00e5ff', 'α (Np/m)');
    plotLine('g-beta',  data.f, data.beta,  '#ff6d00', 'β (rad/m)');
    plotLine('g-delta', data.f, data.delta, '#39ff14', 'δ (m)');

    // Badge info
    const b = document.getElementById('info-badge');
    b.classList.add('visible');
    b.innerHTML =
      `<b>Material:</b> ${data.info.material}<br>` +
      `<b>σ:</b> ${data.info.sigma.toExponential(2)} S/m<br>` +
      `<b>ε_r:</b> ${data.info.epsilon_r}<br>` +
      `<b>μ_r:</b> ${data.info.mu_r}`;
  } catch(e) {
    alert('Error al calcular: ' + e.message);
  } finally {
    document.getElementById('loader').classList.remove('visible');
  }
});
</script>
</body>
</html>
"""

# ─── RUTAS ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return HTML

@app.route("/calcular", methods=["POST"])
def calcular_ruta():
    d = request.get_json()
    if d.get("personalizado"):
        sigma     = float(d["sigma"])
        epsilon_r = float(d["epsilon_r"])
        mu_r      = float(d["mu_r"])
        nombre    = "Personalizado"
    else:
        nombre = d["material"]
        sigma, epsilon_r, mu_r = MATERIALES[nombre]

    f_min = float(d.get("f_min", 1e3))
    f_max = float(d.get("f_max", 1e10))
    n_pts = int(d.get("n_pts", 400))

    frecuencias = np.logspace(np.log10(f_min), np.log10(f_max), n_pts)
    alpha, beta, delta = calcular(frecuencias, sigma, epsilon_r, mu_r)

    def limpia(arr):
        return [None if (np.isnan(v) or np.isinf(v) or v <= 0) else float(v) for v in arr]

    return jsonify({
        "f":     frecuencias.tolist(),
        "alpha": limpia(alpha),
        "beta":  limpia(beta),
        "delta": limpia(delta),
        "info":  {"material": nombre, "sigma": sigma, "epsilon_r": epsilon_r, "mu_r": mu_r}
    })

# ─── INICIO ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  ANÁLISIS ELECTROMAGNÉTICO DE MATERIALES")
    print("  Abre tu navegador en:  http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=True)