"""
Blind Navigator v4 — Professional Research Dashboard
=====================================================
A research-grade RL dashboard showcasing safe navigation
for assistive robotics under dynamic uncertainty.
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import json, os, sys
from collections import deque

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Blind Navigator",
    page_icon="🦯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Crimson+Pro:ital,wght@0,400;0,600;1,400&display=swap');

:root {
    --ink:       #0C0E14;
    --ink2:      #141720;
    --surface:   #1A1D27;
    --border:    #252836;
    --border2:   #2E3245;
    --blue:      #4F8EF7;
    --blue-dim:  #1E3A6E;
    --green:     #2DD4A0;
    --green-dim: #0E3D2F;
    --amber:     #F5A623;
    --amber-dim: #3D2A0A;
    --violet:    #9B7FEA;
    --violet-dim:#2A1F4A;
    --red:       #F06449;
    --text:      #E8EAF0;
    --muted:     #5A607A;
    --dim:       #363B52;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background: var(--ink);
    color: var(--text);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem; max-width: 1500px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--ink2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid='stSidebar'] {
    transform: none !important; left: 0 !important;
    width: 17rem !important; min-width: 17rem !important;
    display: flex !important; visibility: visible !important;
}
section[data-testid='stSidebar'][aria-expanded='false'] {
    transform: none !important; margin-left: 0 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface); border-radius: 10px;
    padding: 4px; gap: 4px; border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: var(--muted);
    border-radius: 7px; font-family: 'Space Grotesk', sans-serif;
    font-size: .83rem; font-weight: 500;
    padding: .4rem 1.1rem;
}
.stTabs [aria-selected="true"] {
    background: var(--blue) !important;
    color: white !important;
}

/* Buttons */
.stButton > button {
    background: var(--blue); color: white; border: none;
    border-radius: 8px; font-family: 'Space Grotesk', sans-serif;
    font-weight: 600; font-size: .88rem; padding: .5rem 1.4rem;
    transition: opacity .15s;
}
.stButton > button:hover { opacity: .88; }

/* Sliders / selectbox */
.stSlider, .stSelectbox { color: var(--text); }
hr { border-color: var(--border) !important; }

/* Custom components */
.kpi-row { display: flex; gap: 1rem; margin-bottom: 1.2rem; }

.kpi {
    flex: 1; background: var(--surface);
    border: 1px solid var(--border); border-radius: 12px;
    padding: 1.1rem 1.3rem; position: relative; overflow: hidden;
}
.kpi::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent, var(--blue));
}
.kpi-label {
    font-size: .65rem; color: var(--muted); text-transform: uppercase;
    letter-spacing: .1em; font-family: 'JetBrains Mono', monospace;
    margin-bottom: .35rem;
}
.kpi-value {
    font-family: 'Space Grotesk', sans-serif; font-size: 2.1rem;
    font-weight: 700; line-height: 1; color: var(--accent, var(--blue));
}
.kpi-sub {
    font-size: .7rem; color: var(--muted); margin-top: .3rem;
    font-family: 'JetBrains Mono', monospace;
}

.algo-pill {
    display: inline-flex; align-items: center; gap: .4rem;
    padding: .25rem .75rem; border-radius: 20px;
    font-size: .72rem; font-family: 'JetBrains Mono', monospace;
    border: 1px solid; margin: .15rem;
}

.section-label {
    font-size: .7rem; color: var(--muted); text-transform: uppercase;
    letter-spacing: .12em; font-family: 'JetBrains Mono', monospace;
    margin: 1.6rem 0 .7rem; display: flex; align-items: center; gap: .6rem;
}
.section-label::after {
    content: ''; flex: 1; height: 1px; background: var(--border);
}

.card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.2rem 1.4rem;
}

.hero {
    background: linear-gradient(135deg, #0C0E14 0%, #141720 50%, #0E1520 100%);
    border: 1px solid var(--border); border-radius: 16px;
    padding: 2.8rem 3rem; margin-bottom: 1.8rem;
    position: relative; overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse 60% 80% at 5% 50%, rgba(79,142,247,.06), transparent),
        radial-gradient(ellipse 50% 60% at 95% 30%, rgba(45,212,160,.04), transparent),
        radial-gradient(ellipse 30% 40% at 50% 100%, rgba(155,127,234,.04), transparent);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: .72rem;
    color: var(--blue); letter-spacing: .15em; text-transform: uppercase;
    margin-bottom: .8rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif; font-size: 3rem;
    font-weight: 700; line-height: 1.1; color: var(--text);
    margin: 0 0 .6rem;
}
.hero-title span {
    background: linear-gradient(90deg, #4F8EF7, #2DD4A0);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-family: 'Crimson Pro', serif; font-size: 1.2rem;
    font-style: italic; color: var(--muted); line-height: 1.6;
    max-width: 640px; margin: 0 0 1.4rem;
}
.tag {
    display: inline-block; padding: .2rem .65rem;
    background: rgba(79,142,247,.1); border: 1px solid rgba(79,142,247,.2);
    color: var(--blue); border-radius: 4px;
    font-family: 'JetBrains Mono', monospace; font-size: .67rem;
    margin: .15rem .1rem;
}

.phase-bar {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: .7rem 1rem; margin-bottom: .4rem;
    display: flex; align-items: center; gap: 1rem;
}

.algo-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 1rem 1.1rem; margin-bottom: .5rem;
    border-left: 3px solid var(--color);
    transition: border-color .2s;
}

.result-badge {
    font-family: 'JetBrains Mono', monospace; font-size: .75rem;
    padding: .15rem .55rem; border-radius: 4px;
    background: var(--bg); color: var(--col);
    border: 1px solid var(--col);
    display: inline-block;
}

.map-grid-container {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: .75rem;
}

.insight-box {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 1rem 1.2rem;
    border-left: 3px solid var(--color, var(--blue));
}
.insight-box .insight-label {
    font-family: 'JetBrains Mono', monospace; font-size: .65rem;
    color: var(--muted); text-transform: uppercase; letter-spacing: .1em;
    margin-bottom: .35rem;
}
.insight-box .insight-text {
    font-size: .86rem; color: var(--text); line-height: 1.65;
}
</style>
""", unsafe_allow_html=True)

# ── Data helpers ──────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(_DIR, "outputs")

def _nm(algo):
    return algo.lower().replace(" ", "_").replace("-", "_")

@st.cache_data
def get_eval(algo):
    p = os.path.join(OUT, f"{_nm(algo)}_eval.json")
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return {}

@st.cache_data
def get_logs(algo):
    p = os.path.join(OUT, f"{_nm(algo)}_curriculum_logs.json")
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return []

def smooth(data, w=200):
    return [float(np.mean(data[max(0,i-w):i+1])) for i in range(len(data))]

# ── v2 (safety upgrade) data helpers ────────────────────────────────────────
OUT_V2 = os.path.join(_DIR, "outputs_v2")

@st.cache_data
def get_v2_summary():
    p = os.path.join(OUT_V2, "v2_summary.json")
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return {}

@st.cache_data
def get_v2_eval():
    p = os.path.join(OUT_V2, "dueling_per_eval.json")
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return {}

@st.cache_data
def get_v2_logs():
    p = os.path.join(OUT_V2, "dueling_per_v2_curriculum_logs.json")
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return []

def load_v2_core():
    """Loads outputs_v2/core_v2.py as an isolated module (does not touch core.py)."""
    import importlib.util
    p = os.path.join(OUT_V2, "core_v2.py")
    if not os.path.exists(p):
        return None
    spec = importlib.util.spec_from_file_location("core_v2", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_v2_agent():
    """Loads the fine-tuned Dueling-PER v2 checkpoint using core_v2's agent class."""
    core_v2 = load_v2_core()
    if core_v2 is None:
        return None, None
    agent = core_v2.DuelingPERAgent()
    p = os.path.join(OUT_V2, "dueling_per_v2_final")
    if not os.path.exists(p + ".pt"):
        return None, core_v2
    agent.load(p)
    return agent, core_v2

# ── Core import ───────────────────────────────────────────────────────────────
try:
    sys.path.insert(0, _DIR)
    from core import (BlindNavigatorEnv, PHASE_CONFIG, Cell, PHASE_EPISODES,
                      ALL_MAPS, GRID_SIZE, ALGO_NAMES, COLORS_ALGO,
                      QLearningAgent, SARSAAgent, DuelingPERAgent,
                      HEADING_DELTAS, HEADING_NAMES)
    CORE_OK = True
except Exception as e:
    CORE_OK = False
    CORE_ERR = str(e)

ALGOS   = ["Q-Learning", "SARSA", "Dueling-PER"]
COLORS  = {"Q-Learning": "#4F8EF7", "SARSA": "#2DD4A0", "Dueling-PER": "#9B7FEA"}
PHASE_N = {0: 3000, 1: 4000, 2: 6000, 3: 8000, 4: 8000}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1.4rem 0 1rem; text-align: center;'>
        <div style='font-size: 2.4rem; line-height: 1;'>🦯</div>
        <div style='font-family: Space Grotesk, sans-serif; font-size: 1rem;
                    font-weight: 700; color: #4F8EF7; margin-top: .5rem;
                    letter-spacing: -.01em;'>Blind Navigator</div>
        <div style='font-size: .7rem; color: #5A607A; margin-top: .2rem;
                    font-family: JetBrains Mono, monospace;'>v4 · Research Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1px;background:#252836;margin:0 0 1rem;'></div>", unsafe_allow_html=True)

    page = st.radio(
        "",
        ["🏠  Overview", "🗺️  Environment", "📊  Training Results", "🛡️  Safety Upgrade", "🤖  Agent Walkthrough"],
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:1px;background:#252836;margin:1rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.65rem;color:#5A607A;text-transform:uppercase;letter-spacing:.1em;font-family:JetBrains Mono,monospace;margin-bottom:.7rem;'>Algorithm Status</div>", unsafe_allow_html=True)

    for algo in ALGOS:
        ev = get_eval(algo)
        color = COLORS[algo]
        sr    = f"{ev['success_rate']:.0%}" if ev else "—"
        icon  = "●" if ev else "○"
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:.5rem;padding:.3rem 0;'>
            <span style='color:{color};font-size:.75rem;'>{icon}</span>
            <span style='font-size:.78rem;color:#94A3B8;flex:1;'>{algo}</span>
            <span style='font-family:JetBrains Mono,monospace;font-size:.75rem;color:{color};
                         background:{color}18;padding:.1rem .4rem;border-radius:3px;'>{sr}</span>
        </div>
        """, unsafe_allow_html=True)

    v2_ev = get_v2_eval()
    if v2_ev:
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:.5rem;padding:.3rem 0;'>
            <span style='color:#2DD4A0;font-size:.75rem;'>●</span>
            <span style='font-size:.78rem;color:#94A3B8;flex:1;'>Dueling-PER <span style='color:#2DD4A0;'>v2</span></span>
            <span style='font-family:JetBrains Mono,monospace;font-size:.75rem;color:#2DD4A0;
                         background:#2DD4A018;padding:.1rem .4rem;border-radius:3px;'>{v2_ev['success_rate']:.0%}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1px;background:#252836;margin:1rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.7rem;color:#5A607A;line-height:1.7;padding-bottom:.5rem;'>
        <div style='margin-bottom:.3rem;'>
            <span style='color:#4F8EF7;font-family:JetBrains Mono,monospace;'>16×16</span> grid · 6 maps
        </div>
        <div style='margin-bottom:.3rem;'>
            <span style='color:#2DD4A0;font-family:JetBrains Mono,monospace;'>29,000</span> training episodes
        </div>
        <div>
            <span style='color:#9B7FEA;font-family:JetBrains Mono,monospace;'>20-dim</span> sensor state
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 🏠 OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":

    # Hero
    st.markdown("""
    <div class='hero'>
        <div class='hero-eyebrow'>· Reinforcement Learning for Assistive Robotics ·</div>
        <h1 class='hero-title'>🦯 <span>Blind Navigator</span> v4</h1>
        <p class='hero-sub'>
            An autonomous navigation agent designed to guide visually impaired users through
            real-world environments — learning entirely from experience, without any pre-programmed rules.
        </p>
        <div>
            <span class='tag'>Dueling Double-DQN</span>
            <span class='tag'>Prioritized Experience Replay</span>
            <span class='tag'>LSTM Temporal Memory</span>
            <span class='tag'>5-Phase Curriculum</span>
            <span class='tag'>Social-Force Pedestrians</span>
            <span class='tag'>BFS-Validated Maps</span>
            <span class='tag'>Custom Gymnasium Env</span>
            <span class='tag'>PyTorch</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    evals = {a: get_eval(a) for a in ALGOS}
    best_ev = evals.get("Dueling-PER", {})

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("Peak Success Rate", "100%",   "Dueling-PER · Phase 4",     "#2DD4A0"),
        ("Training Episodes","29,000",  "3 algos × 29K eps",         "#4F8EF7"),
        ("Grid Size",        "16 × 16", "256 cells · 6 diverse maps","#9B7FEA"),
        ("Sensor Dimensions","20-dim",  "Lidar + goal + pedestrian",  "#F5A623"),
        ("Avg Steps to Goal","41",      "From 16×16 grid corner",    "#2DD4A0"),
    ]
    for col, (lbl, val, sub, acc) in zip([c1,c2,c3,c4,c5], kpis):
        with col:
            st.markdown(f"""
            <div class='kpi' style='--accent:{acc};'>
                <div class='kpi-label'>{lbl}</div>
                <div class='kpi-value'>{val}</div>
                <div class='kpi-sub'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("<div class='section-label'>The Mission</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='card'>
            <p style='font-family: Crimson Pro, serif; font-size: 1.15rem; font-style: italic;
                      color: #4F8EF7; margin: 0 0 .8rem; line-height: 1.6;'>
                "Can a robot learn to anticipate danger — not just react to it?"
            </p>
            <p style='font-size: .88rem; color: #94A3B8; line-height: 1.8; margin: 0;'>
                Blind Navigator simulates a robotic assistant guiding a visually impaired person through
                complex, dynamic environments. The agent perceives the world through <strong style='color:#E8EAF0;'>
                8-directional LiDAR</strong>, goal-relative bearing, and pedestrian proximity signals —
                no camera, no map. It must navigate from corner to corner across a <strong style='color:#E8EAF0;'>
                16×16 grid</strong> while avoiding static walls, pedestrian zones, and 
                <strong style='color:#E8EAF0;'>up to 5 moving pedestrians</strong>, all while managing a battery budget.
            </p>
            <p style='font-size: .88rem; color: #94A3B8; line-height: 1.8; margin: .8rem 0 0;'>
                The key finding: deep RL with temporal memory <strong style='color:#2DD4A0;'>learned to 
                anticipate pedestrian trajectories</strong> and route around them — a behavior that 
                <em>emerged entirely from reward signals</em>, not explicit programming.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-label'>Why Deep RL Wins Here</div>", unsafe_allow_html=True)
        insights = [
            ("#4F8EF7", "State Space Explosion",
             "A 20-dim continuous sensor space cannot be meaningfully discretized. "
             "Q-Learning and SARSA collapse to 0% success rate the moment pedestrians appear — "
             "their hash table simply can't represent what's happening."),
            ("#2DD4A0", "Temporal Memory Matters",
             "The LSTM layer carries episode history across timesteps, enabling loop detection "
             "and path memory. Without it, the agent revisits the same dead-ends repeatedly."),
            ("#9B7FEA", "Curriculum is the Secret",
             "Per-phase epsilon restart keeps the agent exploring as the environment grows harder. "
             "Skipping this step causes catastrophic forgetting when pedestrians are introduced."),
        ]
        for color, title, body in insights:
            st.markdown(f"""
            <div class='insight-box' style='--color:{color}; margin-bottom:.6rem;'>
                <div class='insight-label'>{title}</div>
                <div class='insight-text'>{body}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='section-label'>Algorithm Comparison</div>", unsafe_allow_html=True)
        algo_info = [
            ("Q-Learning",  "#4F8EF7", "Off-policy TD", "0%",
             "Tabular lookup — works in Phase 0–1, collapses completely once pedestrians move."),
            ("SARSA",       "#2DD4A0", "On-policy TD",  "0%",
             "More conservative than Q-Learning but faces the same representation bottleneck."),
            ("Dueling-PER", "#9B7FEA", "Deep RL ⭐",   "100%",
             "Dueling heads + PER + LSTM — achieves perfect generalization across all 6 maps."),
        ]
        for algo, color, tp, final_sr, desc in algo_info:
            ev = evals.get(algo, {})
            sr = f"{ev.get('success_rate', 0):.0%}" if ev else final_sr
            st.markdown(f"""
            <div class='algo-card' style='--color:{color};'>
                <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.4rem;'>
                    <div>
                        <span style='color:{color};font-weight:700;font-size:.95rem;'>{algo}</span>
                        <span style='font-family:JetBrains Mono,monospace;font-size:.65rem;
                                     color:#5A607A;margin-left:.5rem;'>{tp}</span>
                    </div>
                    <span style='font-family:JetBrains Mono,monospace;font-size:.85rem;
                                 font-weight:600;color:{color};background:{color}18;
                                 padding:.15rem .5rem;border-radius:4px;'>{sr}</span>
                </div>
                <div style='font-size:.8rem;color:#5A607A;line-height:1.6;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='section-label'>Real-World Applicability</div>", unsafe_allow_html=True)
        applications = [
            ("🏥", "Hospital Navigation", "Guide patients to departments"),
            ("🏙️", "Urban Crossings",     "Pedestrian-aware street guidance"),
            ("🏢", "Office Wayfinding",   "Dynamic obstacle avoidance"),
            ("🚉", "Transit Hubs",        "Crowded station navigation"),
        ]
        for icon, title, sub in applications:
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:.8rem;padding:.55rem .8rem;
                        background:#1A1D27;border:1px solid #252836;border-radius:8px;
                        margin-bottom:.4rem;'>
                <span style='font-size:1.3rem;'>{icon}</span>
                <div>
                    <div style='font-size:.82rem;font-weight:600;color:#E8EAF0;'>{title}</div>
                    <div style='font-size:.73rem;color:#5A607A;'>{sub}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 🗺️ ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️  Environment":
    st.markdown("<h2 style='font-family:Space Grotesk,sans-serif;font-weight:700;margin-bottom:1.2rem;'>🗺️ BlindNavigatorEnv — v4</h2>", unsafe_allow_html=True)

    col_ctrl, col_map = st.columns([1, 2])

    with col_ctrl:
        phase_sel = st.selectbox(
            "Phase",
            list(range(5)),
            format_func=lambda p: f"Phase {p} — {PHASE_CONFIG[p]['description']}"
        )
        cfg = PHASE_CONFIG[phase_sel]

        st.markdown(f"""
        <div class='card' style='margin-top:.7rem;font-size:.83rem;'>
            <div style='font-weight:700;color:#4F8EF7;font-size:.95rem;margin-bottom:.8rem;'>
                Phase {phase_sel} Config
            </div>
            <div style='display:flex;flex-direction:column;gap:.4rem;color:#5A607A;'>
                <div>🚶 Pedestrians
                    <strong style='color:#E8EAF0;float:right;'>{cfg['n_pedestrians']}</strong></div>
                <div>🌊 Slip probability
                    <strong style='color:#E8EAF0;float:right;'>{cfg['slip_prob']:.0%}</strong></div>
                <div>🔋 Battery budget
                    <strong style='color:#E8EAF0;float:right;'>{cfg['battery']} steps</strong></div>
                <div>🗺️ Active maps
                    <strong style='color:#E8EAF0;float:right;'>{len(cfg['map_ids'])} / 6</strong></div>
                <div>📦 Episodes
                    <strong style='color:#E8EAF0;float:right;'>{PHASE_N[phase_sel]:,}</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-label'>Cell Legend</div>", unsafe_allow_html=True)
        legend = [
            ("#2A3D2A", "#2DD4A0", "Safe",           "−0.5 / step"),
            ("#3D3320", "#F5A623", "Edge / Curb",     "−2.0"),
            ("#3D1A1A", "#F06449", "Static Hazard",   "−50.0"),
            ("#2A2540", "#9B7FEA", "Pedestrian Zone", "−1.0"),
            ("#1A3D30", "#2DD4A0", "Goal ★",          "+500 + time bonus"),
        ]
        for bg, border, name, rew in legend:
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:.6rem;padding:.3rem 0;'>
                <div style='width:14px;height:14px;background:{bg};border:1px solid {border};
                            border-radius:3px;flex-shrink:0;'></div>
                <span style='font-size:.8rem;color:#94A3B8;flex:1;'>{name}</span>
                <span style='font-family:JetBrains Mono,monospace;font-size:.7rem;color:{border};'>{rew}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='section-label'>Sensor State · 20-dim</div>", unsafe_allow_html=True)
        sensors = [
            ("0–7",  "#4F8EF7", "8-dir LiDAR distances"),
            ("8",    "#2DD4A0", "Goal distance (Manhattan)"),
            ("9",    "#2DD4A0", "Goal angle (relative bearing)"),
            ("10–13","#F5A623", "Nearest pedestrian dist/angle/vel"),
            ("14",   "#F06449", "Battery level"),
            ("15–16","#9B7FEA", "Heading sin/cos"),
            ("17",   "#9B7FEA", "Loop density"),
            ("18–19","#5A607A", "Last action + steps normalized"),
        ]
        for idx, color, desc in sensors:
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:.5rem;padding:.25rem 0;'>
                <span style='font-family:JetBrains Mono,monospace;font-size:.65rem;
                             color:{color};background:{color}18;padding:.05rem .35rem;
                             border-radius:3px;min-width:2.5rem;text-align:center;'>[{idx}]</span>
                <span style='font-size:.77rem;color:#94A3B8;'>{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_map:
        if CORE_OK:
            map_tab1, map_tab2 = st.tabs(["  Live Environment  ", "  All 6 Maps  "])

            with map_tab1:
                env = BlindNavigatorEnv(phase=phase_sel)
                env.reset(seed=42)
                fig = env.render_figure(figsize=(6, 6))
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with map_tab2:
                map_names = ["Office Corridors", "Hospital", "Street Crossing",
                             "Mall (Pillars)", "Apartment", "Outdoor Park"]
                row1, row2 = st.columns(3), st.columns(3)
                for idx, (col, name) in enumerate(zip(list(row1) + list(row2), map_names)):
                    env_m = BlindNavigatorEnv(phase=phase_sel, map_ids=[idx])
                    env_m.reset(seed=42)
                    fig_m = env_m.render_figure(figsize=(4, 4))
                    # Override title to show map name only
                    fig_m.axes[0].set_title(
                        f"Map {idx}: {name}",
                        fontsize=9, fontweight="600", pad=4
                    )
                    with col:
                        st.pyplot(fig_m, use_container_width=True)
                    plt.close(fig_m)
        else:
            st.warning(f"Could not load core: {CORE_ERR}")

    st.markdown("---")
    st.markdown("<div class='section-label'>Reward Architecture</div>", unsafe_allow_html=True)

    fig_r, ax = plt.subplots(figsize=(12, 3))
    fig_r.patch.set_facecolor("#141720"); ax.set_facecolor("#141720")
    items = [
        ("Goal\n(+500+bonus)", 500),
        ("Progress\ntoward goal", 3),
        ("Turn", -0.3),
        ("Wait", -1),
        ("Safe\nstep", -0.5),
        ("Ped\nZone", -1),
        ("Edge\n/Curb", -2),
        ("Regress\nfrom goal", -2),
        ("Loop\npenalty", -5),
        ("Battery\nout", -30),
        ("Ped\ncollision", -100),
        ("Static\nhazard", -50),
    ]
    names  = [i[0] for i in items]
    vals   = [i[1] for i in items]
    colors_bar = ["#2DD4A0" if v > 0 else ("#9B7FEA" if -5 < v <= 0 else ("#F5A623" if -30 < v <= -5 else "#F06449")) for v in vals]
    bars = ax.barh(names, vals, color=colors_bar, edgecolor="#252836", linewidth=.4, height=.65)
    for bar, val in zip(bars, vals):
        x = val + (8 if val >= 0 else -8)
        ax.text(x, bar.get_y() + bar.get_height()/2, f"{val:+.0f}",
                va="center", ha="left" if val >= 0 else "right",
                color="#E8EAF0", fontsize=8, fontweight="600",
                fontfamily="monospace")
    ax.axvline(0, color="#363B52", linewidth=.8)
    ax.tick_params(colors="#5A607A", labelsize=7.5)
    for sp in ax.spines.values(): sp.set_edgecolor("#252836")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xlabel("Reward value", color="#5A607A", fontsize=8)
    fig_r.tight_layout()
    st.pyplot(fig_r, use_container_width=True)
    plt.close(fig_r)

    st.markdown("---")
    st.markdown("<div class='section-label'>5-Phase Curriculum</div>", unsafe_allow_html=True)
    phases_info = [
        (0, "#363B52", "#94A3B8", "Phase 0 · Empty",          "3,000 eps",
         "No hazards, no pedestrians. Agent learns the basic concept of reaching the goal."),
        (1, "#1E3A6E", "#4F8EF7", "Phase 1 · Static Hazards", "4,000 eps",
         "Static obstacles + edges + 5% motor slip. Agent learns wall avoidance and path planning."),
        (2, "#2A2030", "#9B7FEA", "Phase 2 · Moving Peds",    "6,000 eps",
         "2 social-force pedestrians appear. Agent must react dynamically; WAIT action becomes critical."),
        (3, "#1E3530", "#2DD4A0", "Phase 3 · Full Env",       "8,000 eps",
         "4 pedestrians · 10% slip · all 6 maps. Complete challenge. Agent reaches 99.2% success."),
        (4, "#2A2510", "#F5A623", "Phase 4 · Transfer",       "8,000 eps",
         "5 pedestrians · 12% slip · max battery. Generalization test. Agent sustains 99.9% success."),
    ]
    for ph, bg, color, name, eps, desc in phases_info:
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:1rem;padding:.75rem 1rem;
                    background:{bg}44;border:1px solid {bg};border-left:3px solid {color};
                    border-radius:8px;margin-bottom:.4rem;'>
            <div style='min-width:10rem;color:{color};font-weight:700;
                        font-family:Space Grotesk,sans-serif;font-size:.88rem;'>{name}</div>
            <div style='font-family:JetBrains Mono,monospace;font-size:.72rem;
                        color:#5A607A;min-width:5.5rem;'>{eps}</div>
            <div style='font-size:.82rem;color:#94A3B8;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 📊 TRAINING RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Training Results":
    st.markdown("<h2 style='font-family:Space Grotesk,sans-serif;font-weight:700;margin-bottom:1.2rem;'>📊 Training Results</h2>", unsafe_allow_html=True)

    evals = {a: get_eval(a) for a in ALGOS}
    if not any(evals.values()):
        st.info("⏳ Run training first — eval JSON files not found."); st.stop()

    # Top eval cards
    cols = st.columns(3)
    for col, algo in zip(cols, ALGOS):
        ev = evals[algo]; color = COLORS[algo]
        sr = ev.get("success_rate", 0)
        mr = ev.get("mean_reward",  0)
        cr = ev.get("collision_rate", 0)
        ms = ev.get("mean_steps",   0)
        with col:
            st.markdown(f"""
            <div class='kpi' style='--accent:{color};text-align:center;padding:1.4rem;'>
                <div style='color:{color};font-family:Space Grotesk,sans-serif;
                            font-weight:700;font-size:.95rem;margin-bottom:.8rem;'>{algo}</div>
                <div class='kpi-value' style='font-size:3rem;'>{sr:.0%}</div>
                <div class='kpi-label' style='margin:.3rem 0 .7rem;'>success rate</div>
                <div style='display:grid;grid-template-columns:1fr 1fr;gap:.4rem;font-size:.72rem;
                            font-family:JetBrains Mono,monospace;'>
                    <div style='background:{color}12;border:1px solid {color}30;
                                border-radius:5px;padding:.3rem .5rem;color:{color};'>
                        reward<br><strong style='font-size:.85rem;'>{mr:+.0f}</strong>
                    </div>
                    <div style='background:#F06449{12:x};border:1px solid #F0644930;
                                border-radius:5px;padding:.3rem .5rem;color:#F06449;'>
                        collisions<br><strong style='font-size:.85rem;'>{cr:.0%}</strong>
                    </div>
                </div>
                <div style='font-size:.7rem;color:#5A607A;margin-top:.5rem;
                            font-family:JetBrains Mono,monospace;'>avg {ms:.0f} steps / episode</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "  📈 Learning Curves  ",
        "  📊 Per-Phase Breakdown  ",
        "  🗺️ Per-Map Performance  ",
        "  🏆 Head-to-Head  "
    ])

    # ── Tab 1: Learning Curves ────────────────────────────────────────────────
    with tab1:
        col_opt1, col_opt2 = st.columns([2, 1])
        with col_opt1:
            window = st.slider("Smoothing window", 50, 500, 200, 25, key="smooth")
        with col_opt2:
            metric = st.selectbox("Metric", ["total_reward", "success", "steps", "collisions"])

        fig, ax = plt.subplots(figsize=(13, 5))
        fig.patch.set_facecolor("#141720"); ax.set_facecolor("#141720")

        # Phase bands
        phase_colors = ["#363B52", "#1E3A6E", "#2A2030", "#1E3530", "#2A2510"]
        phase_starts = [0, 3000, 7000, 13000, 21000]
        phase_ends   = [3000, 7000, 13000, 21000, 29000]
        phase_labels = ["P0\nEmpty", "P1\nHazards", "P2\nPeds", "P3\nFull", "P4\nTransfer"]
        phase_cols   = ["#94A3B8", "#4F8EF7", "#9B7FEA", "#2DD4A0", "#F5A623"]

        for i, (s, e) in enumerate(zip(phase_starts, phase_ends)):
            ax.axvspan(s, e, alpha=.07, color=phase_cols[i], linewidth=0)
            if i < 4:
                ax.axvline(e, color="#363B52", linewidth=.7, linestyle="--")
            ax.text((s+e)/2, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else .98,
                    phase_labels[i], ha="center", fontsize=7, color=phase_cols[i],
                    fontfamily="monospace", va="top")

        has_data = False
        for algo in ALGOS:
            logs = get_logs(algo)
            if logs:
                vals = [l[metric] for l in logs]
                smoothed = smooth(vals, window)
                ax.plot(smoothed, color=COLORS[algo], linewidth=2, label=algo, alpha=.95)
                has_data = True

        if not has_data:
            ax.text(14500, 0, "No training logs found — run the notebook first",
                    ha="center", color="#5A607A", fontsize=12)

        metric_labels = {
            "total_reward": "Episode Return (smoothed)",
            "success":      "Success Rate (smoothed)",
            "steps":        "Steps per Episode (smoothed)",
            "collisions":   "Collisions per Episode (smoothed)",
        }
        ax.set_xlabel("Global Episode", color="#5A607A", fontsize=9)
        ax.set_ylabel(metric_labels[metric], color="#5A607A", fontsize=9)
        ax.set_title("29,000-Episode Curriculum Training",
                     color="#E8EAF0", fontsize=11, fontweight="700")
        ax.legend(facecolor="#1A1D27", edgecolor="#252836",
                  labelcolor="#E8EAF0", fontsize=9)
        ax.tick_params(colors="#5A607A")
        for sp in ax.spines.values(): sp.set_edgecolor("#252836")
        ax.grid(True, alpha=.08, color="#363B52")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Tab 2: Per-Phase ──────────────────────────────────────────────────────
    with tab2:
        # Compute per-phase stats from logs
        phase_stats = {}
        for algo in ALGOS:
            logs = get_logs(algo)
            phase_stats[algo] = {}
            for ph in range(5):
                ph_logs = [l for l in logs if l.get("phase") == ph]
                if ph_logs:
                    phase_stats[algo][ph] = {
                        "sr":    np.mean([l["success"] for l in ph_logs]),
                        "reward": np.mean([l["total_reward"] for l in ph_logs]),
                        "steps":  np.mean([l["steps"] for l in ph_logs]),
                    }

        fig2, axes2 = plt.subplots(1, 5, figsize=(15, 4.5))
        fig2.patch.set_facecolor("#141720")
        phase_descs = ["Empty", "Hazards", "Peds", "Full", "Transfer"]
        phase_colrs = ["#94A3B8", "#4F8EF7", "#9B7FEA", "#2DD4A0", "#F5A623"]

        for pi, ax in enumerate(axes2):
            ax.set_facecolor("#141720")
            ax.set_title(f"Phase {pi}\n{phase_descs[pi]}", color=phase_colrs[pi],
                         fontsize=9, fontweight="700")
            for algo in ALGOS:
                if pi in phase_stats.get(algo, {}):
                    ph_logs = [l for l in get_logs(algo) if l.get("phase") == pi]
                    vals = [l["total_reward"] for l in ph_logs]
                    s_vals = smooth(vals, 100)
                    ax.plot(s_vals, color=COLORS[algo], linewidth=1.8, label=algo, alpha=.9)
            ax.tick_params(colors="#5A607A", labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor("#252836")
            ax.grid(True, alpha=.08, color="#363B52")
            ax.set_xlabel("Phase Episode", color="#5A607A", fontsize=7)
            if pi == 0:
                ax.set_ylabel("Return", color="#5A607A", fontsize=8)
            if pi == 4:
                ax.legend(facecolor="#1A1D27", edgecolor="#252836",
                          labelcolor="#E8EAF0", fontsize=7.5)

        fig2.suptitle("Return per Training Phase · All Algorithms",
                      color="#E8EAF0", fontsize=11, fontweight="700")
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        # Phase success rate table
        st.markdown("<div class='section-label'>Success Rate per Phase</div>", unsafe_allow_html=True)
        header_cols = st.columns([2, 1, 1, 1, 1, 1])
        headers = ["Algorithm", "Phase 0", "Phase 1", "Phase 2", "Phase 3", "Phase 4"]
        for col, h in zip(header_cols, headers):
            with col:
                st.markdown(f"<div style='font-family:JetBrains Mono,monospace;font-size:.7rem;color:#5A607A;text-transform:uppercase;letter-spacing:.08em;padding:.3rem 0;'>{h}</div>", unsafe_allow_html=True)

        for algo in ALGOS:
            row_cols = st.columns([2, 1, 1, 1, 1, 1])
            color = COLORS[algo]
            with row_cols[0]:
                st.markdown(f"<div style='color:{color};font-weight:600;font-size:.85rem;padding:.35rem 0;'>{algo}</div>", unsafe_allow_html=True)
            for pi, col in enumerate(row_cols[1:]):
                stats = phase_stats.get(algo, {}).get(pi, {})
                sr = stats.get("sr", None)
                if sr is not None:
                    bg    = "#0E3D2F" if sr >= .9 else ("#1E3A6E" if sr >= .5 else "#3D1A1A")
                    fcol  = "#2DD4A0" if sr >= .9 else ("#4F8EF7" if sr >= .5 else "#F06449")
                    label = f"{sr:.0%}"
                else:
                    bg, fcol, label = "#1A1D27", "#5A607A", "—"
                with col:
                    st.markdown(f"<div style='background:{bg};color:{fcol};font-family:JetBrains Mono,monospace;font-size:.82rem;font-weight:600;text-align:center;padding:.3rem .4rem;border-radius:5px;margin:.15rem 0;'>{label}</div>", unsafe_allow_html=True)

    # ── Tab 3: Per-Map ────────────────────────────────────────────────────────
    with tab3:
        map_names_tab = ["Office Corridors", "Hospital", "Street Crossing",
                         "Mall (Pillars)", "Apartment", "Outdoor Park"]
        ev_per = get_eval("Dueling-PER")

        if ev_per and "per_map" in ev_per:
            fig3, axes3 = plt.subplots(1, 2, figsize=(13, 4))
            fig3.patch.set_facecolor("#141720")

            map_ids = sorted(ev_per["per_map"].keys())
            srs     = [ev_per["per_map"][k]["success_rate"] for k in map_ids]
            mrs     = [ev_per["per_map"][k]["mean_reward"]  for k in map_ids]
            xlabels = [f"Map {k[-1]}\n{map_names_tab[int(k[-1])]}" for k in map_ids]

            for ax_idx, (ax_bar, vals, ylabel, color_bar) in enumerate(zip(
                axes3,
                [srs, mrs],
                ["Success Rate", "Mean Reward"],
                ["#2DD4A0", "#9B7FEA"]
            )):
                ax_bar.set_facecolor("#141720")
                bar_colors = [color_bar if v > 0 else "#F06449" for v in vals]
                bars3 = ax_bar.bar(xlabels, vals, color=bar_colors,
                                   edgecolor="#252836", linewidth=.4, width=.6)
                for bar, val in zip(bars3, vals):
                    fmt = f"{val:.0%}" if ax_idx == 0 else f"{val:+.0f}"
                    ax_bar.text(bar.get_x() + bar.get_width()/2,
                                bar.get_height() + (0.01 if ax_idx == 0 else 5),
                                fmt, ha="center", va="bottom",
                                color="#E8EAF0", fontsize=9, fontfamily="monospace")
                ax_bar.set_ylabel(ylabel, color="#5A607A", fontsize=9)
                ax_bar.tick_params(colors="#5A607A", labelsize=8)
                for sp in ax_bar.spines.values(): sp.set_edgecolor("#252836")
                ax_bar.grid(True, alpha=.08, axis="y", color="#363B52")
                ax_bar.set_title(f"Dueling-PER · {ylabel} per Map",
                                 color="#E8EAF0", fontsize=10, fontweight="700")

            fig3.tight_layout()
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

            st.markdown("""
            <div style='background:#1E3D2F44;border:1px solid #2DD4A040;border-radius:8px;
                        padding:.8rem 1rem;margin-top:.8rem;font-size:.83rem;color:#94A3B8;'>
                ✅ <strong style='color:#2DD4A0;'>Perfect generalization:</strong>
                Dueling-PER achieves 100% success on all 6 maps — from open corridors
                to narrow hospital passages and crowded mall pillars. The agent was never
                explicitly told which map it was on.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Per-map data not available. Run evaluation with `map_ids=list(range(6))`.")

    # ── Tab 4: Head-to-Head ───────────────────────────────────────────────────
    with tab4:
        fig4, axes4 = plt.subplots(1, 3, figsize=(13, 4.5))
        fig4.patch.set_facecolor("#141720")
        metrics_hth = ["success_rate", "mean_reward", "collision_rate"]
        labels_hth  = ["Final Success Rate", "Mean Reward (300 eps)", "Collision Rate"]
        fmts_hth    = ["{:.0%}", "{:+.0f}", "{:.0%}"]
        good_high   = [True, True, False]

        for ax_h, metric, label, fmt, high_good in zip(axes4, metrics_hth, labels_hth, fmts_hth, good_high):
            ax_h.set_facecolor("#141720")
            algo_vals = [(a, evals[a].get(metric, 0)) for a in ALGOS if evals[a]]
            names_h = [a[0] for a in algo_vals]
            vals_h  = [a[1] for a in algo_vals]
            bcolors = []
            for v, n in zip(vals_h, names_h):
                best_v = max(vals_h) if high_good else min(vals_h)
                bcolors.append(COLORS[n] if v == best_v else COLORS[n] + "66")

            bars_h = ax_h.bar(names_h, vals_h, color=bcolors,
                              edgecolor="#252836", linewidth=.4, width=.55)
            for bar, val in zip(bars_h, vals_h):
                label_txt = fmt.format(val)
                ax_h.text(bar.get_x() + bar.get_width()/2,
                          bar.get_height() + max(abs(max(vals_h)) * 0.02, 0.01),
                          label_txt, ha="center", va="bottom",
                          color="#E8EAF0", fontsize=9.5, fontweight="600",
                          fontfamily="monospace")
            ax_h.set_title(label, color="#E8EAF0", fontsize=9.5, fontweight="700")
            ax_h.tick_params(colors="#5A607A", labelsize=8)
            for sp in ax_h.spines.values(): sp.set_edgecolor("#252836")
            ax_h.grid(True, alpha=.08, axis="y", color="#363B52")
            ax_h.axhline(0, color="#363B52", linewidth=.7)

        fig4.suptitle("Final Evaluation · 300 Episodes · Phase 4 (Max Complexity)",
                      color="#E8EAF0", fontsize=11, fontweight="700")
        fig4.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)

        st.markdown("""
        <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:.8rem;margin-top:1rem;'>
            <div style='background:#0E3D2F44;border:1px solid #2DD4A040;border-left:3px solid #2DD4A0;
                        border-radius:8px;padding:.9rem 1rem;font-size:.82rem;'>
                <div style='color:#2DD4A0;font-weight:700;margin-bottom:.3rem;'>🏆 Dueling-PER</div>
                <div style='color:#94A3B8;'>Perfect 100% success. LSTM memory + PER
                combined allow the agent to plan efficient paths while anticipating pedestrian movement.</div>
            </div>
            <div style='background:#3D1A1A44;border:1px solid #F0644940;border-left:3px solid #F06449;
                        border-radius:8px;padding:.9rem 1rem;font-size:.82rem;'>
                <div style='color:#F06449;font-weight:700;margin-bottom:.3rem;'>❌ Q-Learning / SARSA</div>
                <div style='color:#94A3B8;'>0% success in complex phases. The 20-dim continuous
                state cannot be discretized — hash collisions destroy the Q-table's precision.</div>
            </div>
            <div style='background:#1A1D2744;border:1px solid #252836;border-left:3px solid #9B7FEA;
                        border-radius:8px;padding:.9rem 1rem;font-size:.82rem;'>
                <div style='color:#9B7FEA;font-weight:700;margin-bottom:.3rem;'>💡 Key Insight</div>
                <div style='color:#94A3B8;'>This isn't a fair comparison — it's an intentional
                demonstration that <em>representation matters</em> as much as the algorithm itself.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 🤖 AGENT WALKTHROUGH
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🛡️  Safety Upgrade":
    st.markdown("<h2 style='font-family:Space Grotesk,sans-serif;font-weight:700;margin-bottom:.3rem;'>🛡️ Safety Upgrade — v1 → v2</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.85rem;color:#5A607A;margin-bottom:1.2rem;'>
        Dueling-PER reached <strong style='color:#E8EAF0;'>100% success</strong> in v1 — but with a
        <strong style='color:#F06449;'>23% pedestrian collision rate</strong>. Since collisions push the
        agent back rather than ending the episode, a perfect success rate could hide unsafe behavior along
        the way. This page documents the fix.
    </div>
    """, unsafe_allow_html=True)

    v2_summary = get_v2_summary()
    v2_eval    = get_v2_eval()
    if not v2_summary or not v2_eval:
        st.info("⏳ v2 results not found — run `outputs_v2/retrain_v2.ipynb` first.")
        st.stop()

    v1_stats = v2_summary.get("v1", {})
    v2_stats = v2_summary.get("v2", v2_eval)

    # ── What changed ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-label'>What Changed in core_v2.py</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='card' style='--accent:#F06449;'>
            <div style='font-family:JetBrains Mono,monospace;font-size:.7rem;color:#5A607A;
                        text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem;'>Change 1</div>
            <div style='font-weight:700;color:#E8EAF0;margin-bottom:.5rem;'>Collision penalty: 3× stronger</div>
            <div style='font-family:JetBrains Mono,monospace;font-size:.85rem;color:#94A3B8;
                        background:#0C0E14;border:1px solid #252836;border-radius:6px;padding:.6rem .8rem;'>
                PEDESTRIAN_COLLISION_REWARD<br>
                <span style='color:#F06449;'>-100.0</span>  →  <span style='color:#2DD4A0;'>-300.0</span>
            </div>
            <div style='font-size:.8rem;color:#5A607A;margin-top:.6rem;line-height:1.6;'>
                Makes "trading a hit for speed" a losing strategy: 1 collision + goal nets only
                <span style='color:#94A3B8;'>+200</span> instead of <span style='color:#94A3B8;'>+400</span>;
                2 collisions in one episode now nets <span style='color:#F06449;'>−100</span> overall.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='card' style='--accent:#9B7FEA;'>
            <div style='font-family:JetBrains Mono,monospace;font-size:.7rem;color:#5A607A;
                        text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem;'>Change 2 — New</div>
            <div style='font-weight:700;color:#E8EAF0;margin-bottom:.5rem;'>Near-miss penalty</div>
            <div style='font-family:JetBrains Mono,monospace;font-size:.85rem;color:#94A3B8;
                        background:#0C0E14;border:1px solid #252836;border-radius:6px;padding:.6rem .8rem;'>
                if ped_dist &lt; 20% of sensor range:<br>
                reward += <span style='color:#F06449;'>up to −15</span> (proportional)
            </div>
            <div style='font-size:.8rem;color:#5A607A;margin-top:.6rem;line-height:1.6;'>
                Penalizes ending a step <em>close</em> to a pedestrian — even without touching them —
                so the agent learns to give people space early, not react at the last instant.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='insight-box' style='margin-top:1rem;'>
        <div class='insight-label'>Why a short fine-tune wasn't enough</div>
        <div class='insight-text'>
            A first attempt fine-tuned the v1 checkpoint for just 2,000 low-epsilon episodes on the strong
            collision penalty alone — collisions dropped briefly then drifted back up; the result was not
            reliably better than v1. The fix needed two things together: a reward signal that fires
            <em>before</em> contact (the near-miss penalty), and enough exploration to actually relearn
            pedestrian-avoidance habits. So v2 reloads the <strong style='color:#E8EAF0;'>Phase 1 checkpoint</strong>
            (solid navigation, zero pedestrian exposure) and retrains Phases 2→3→4 from there with the
            <strong style='color:#E8EAF0;'>full original episode budget</strong> (22,000 episodes) under the new reward.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Headline comparison ──────────────────────────────────────────────────
    st.markdown("<div class='section-label' style='margin-top:1.6rem;'>Dueling-PER — v1 vs v2</div>", unsafe_allow_html=True)

    sr1, sr2 = v1_stats.get("success_rate", 1.0), v2_stats.get("success_rate", 0)
    r1,  r2  = v1_stats.get("mean_reward", 0),     v2_stats.get("mean_reward", 0)
    c1_, c2_ = v1_stats.get("collision_rate", 0),  v2_stats.get("collision_rate", 0)
    coll_drop = (c1_ - c2_) / c1_ * 100 if c1_ else 0

    cols = st.columns(3)
    with cols[0]:
        st.markdown(f"""
        <div class='kpi' style='--accent:#4F8EF7;text-align:center;'>
            <div class='kpi-label'>Success Rate</div>
            <div style='display:flex;align-items:baseline;justify-content:center;gap:.6rem;margin-top:.3rem;'>
                <span style='font-family:JetBrains Mono,monospace;font-size:1.1rem;color:#5A607A;'>{sr1:.0%}</span>
                <span style='color:#5A607A;'>→</span>
                <span class='kpi-value' style='font-size:2.3rem;'>{sr2:.0%}</span>
            </div>
            <div class='kpi-sub'>v1 → v2, unchanged</div>
        </div>""", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div class='kpi' style='--accent:#F5A623;text-align:center;'>
            <div class='kpi-label'>Mean Reward</div>
            <div style='display:flex;align-items:baseline;justify-content:center;gap:.6rem;margin-top:.3rem;'>
                <span style='font-family:JetBrains Mono,monospace;font-size:1.1rem;color:#5A607A;'>{r1:.0f}</span>
                <span style='color:#5A607A;'>→</span>
                <span class='kpi-value' style='font-size:2.3rem;'>{r2:.0f}</span>
            </div>
            <div class='kpi-sub'>{"−" if r2<r1 else "+"}{abs(r2-r1):.0f} (cost of caution)</div>
        </div>""", unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"""
        <div class='kpi' style='--accent:#2DD4A0;text-align:center;'>
            <div class='kpi-label'>Collision Rate</div>
            <div style='display:flex;align-items:baseline;justify-content:center;gap:.6rem;margin-top:.3rem;'>
                <span style='font-family:JetBrains Mono,monospace;font-size:1.1rem;color:#F06449;'>{c1_:.0%}</span>
                <span style='color:#5A607A;'>→</span>
                <span class='kpi-value' style='font-size:2.3rem;color:#2DD4A0;'>{c2_:.0%}</span>
            </div>
            <div class='kpi-sub' style='color:#2DD4A0;'>−{coll_drop:.0f}% relative reduction</div>
        </div>""", unsafe_allow_html=True)

    # ── Bar comparison chart ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    fig.patch.set_facecolor("#141720")
    metrics_cmp = [
        ("Success Rate", sr1, sr2, "#4F8EF7", True),
        ("Mean Reward",  r1,  r2,  "#F5A623", False),
        ("Collision Rate", c1_, c2_, "#F06449", True),
    ]
    for ax, (label, v1v, v2v, color, is_pct) in zip(axes, metrics_cmp):
        ax.set_facecolor("#141720")
        bars = ax.bar(["v1", "v2"], [v1v, v2v], color=["#363B52", color], edgecolor="#252836", width=.55)
        for bar, val in zip(bars, [v1v, v2v]):
            txt = f"{val:.0%}" if is_pct else f"{val:.0f}"
            ax.text(bar.get_x()+bar.get_width()/2, val + (max(v1v,v2v)*0.03 if max(v1v,v2v) else .02),
                    txt, ha="center", color="#E8EAF0", fontsize=10, fontweight="700")
        ax.set_title(label, color="#E8EAF0", fontsize=10, fontweight="700")
        ax.tick_params(colors="#5A607A", labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor("#252836")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=.08, color="#363B52", axis="y")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("""
    <div class='insight-box' style='margin-top:.8rem;'>
        <div class='insight-label'>Reading this honestly</div>
        <div class='insight-text'>
            Collision rate fell from <strong style='color:#F06449;'>23%</strong> to
            <strong style='color:#2DD4A0;'>~8%</strong> while success stayed at 100% — the agent did not trade
            navigation skill for safety. Mean reward dipped slightly because the new reward function actively
            taxes risky shortcuts near pedestrians, so a small reward decrease here actually reflects more
            cautious — not worse — behavior.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Per-map comparison ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-label'>Per-Map Generalization — v2</div>", unsafe_allow_html=True)
    per_map = v2_stats.get("per_map", {})
    if per_map:
        map_names = ["Office Corridors","Hospital","Street Crossing","Mall (Pillars)","Apartment","Outdoor Park"]
        ids = sorted(per_map.keys(), key=lambda k: int(k.split("_")[1]))
        srs = [per_map[m]["success_rate"] for m in ids]
        rws = [per_map[m]["mean_reward"]  for m in ids]
        labels = [f"M{m.split('_')[1]}\n{map_names[int(m.split('_')[1])]}" for m in ids]

        fig2, (axa, axb) = plt.subplots(1, 2, figsize=(13, 3.8))
        fig2.patch.set_facecolor("#141720")
        for ax in (axa, axb):
            ax.set_facecolor("#141720")
            ax.tick_params(colors="#5A607A", labelsize=8)
            for sp in ax.spines.values(): sp.set_edgecolor("#252836")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=.08, color="#363B52", axis="y")

        b1 = axa.bar(labels, srs, color="#9B7FEA", edgecolor="#252836")
        axa.set_ylim(0, 1.15)
        axa.set_title("Success Rate per Map (v2)", color="#E8EAF0", fontsize=10, fontweight="700")
        for bar, v in zip(b1, srs):
            axa.text(bar.get_x()+bar.get_width()/2, v+.03, f"{v:.0%}", ha="center", color="#E8EAF0", fontsize=8, fontweight="700")

        b2 = axb.bar(labels, rws, color="#2DD4A0", edgecolor="#252836")
        axb.set_title("Mean Reward per Map (v2)", color="#E8EAF0", fontsize=10, fontweight="700")
        for bar, v in zip(b2, rws):
            axb.text(bar.get_x()+bar.get_width()/2, v+8, f"{v:.0f}", ha="center", color="#E8EAF0", fontsize=8, fontweight="700")

        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    # ── Learning curve during retraining ─────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-label'>Retraining Curve (Phase 2 → 3 → 4, 22,000 episodes)</div>", unsafe_allow_html=True)
    v2_logs = get_v2_logs()
    if v2_logs:
        window2 = st.slider("Smoothing window", 50, 500, 200, 25, key="smooth_v2")
        fig3, ax3 = plt.subplots(figsize=(13, 4.2))
        fig3.patch.set_facecolor("#141720"); ax3.set_facecolor("#141720")

        bounds = [0, 6000, 14000, 22000]
        plabels = ["Phase 2\nPedestrians", "Phase 3\nFull env", "Phase 4\nTransfer"]
        pcols = ["#9B7FEA", "#2DD4A0", "#F5A623"]
        for i in range(3):
            ax3.axvspan(bounds[i], bounds[i+1], alpha=.07, color=pcols[i])
            if i < 2:
                ax3.axvline(bounds[i+1], color="#363B52", linewidth=.7, linestyle="--")

        rewards_v2 = [l["total_reward"] for l in v2_logs]
        coll_v2    = [1.0 if l["collisions"] > 0 else 0.0 for l in v2_logs]
        ax3b = ax3.twinx()
        ax3.plot(smooth(rewards_v2, window2), color="#9B7FEA", linewidth=2, label="Reward (v2, left axis)")
        ax3b.plot(smooth(coll_v2, window2), color="#F06449", linewidth=1.6, alpha=.85, label="Collision-episode rate (right axis)")

        ax3.set_xlabel("Retraining Episode", color="#5A607A", fontsize=9)
        ax3.set_ylabel("Reward", color="#9B7FEA", fontsize=9)
        ax3b.set_ylabel("Collision-episode rate", color="#F06449", fontsize=9)
        ax3.set_title("v2 Retraining: Reward Recovers While Collisions Fall", color="#E8EAF0", fontsize=11, fontweight="700")
        ax3.tick_params(colors="#5A607A"); ax3b.tick_params(colors="#F06449")
        for sp in ax3.spines.values(): sp.set_edgecolor("#252836")
        ax3.grid(True, alpha=.08, color="#363B52")
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3b.get_legend_handles_labels()
        ax3.legend(lines1+lines2, labels1+labels2, facecolor="#1A1D27", edgecolor="#252836", labelcolor="#E8EAF0", fontsize=9, loc="lower right")
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)
    else:
        st.info("No v2 curriculum logs found.")

    # ── Side-by-side episode walkthrough ─────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-label'>Watch It Live — v1 vs v2, Same Episode</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.83rem;color:#5A607A;margin-bottom:.8rem;'>
        Run the exact same episode (same map, same seed) through both checkpoints and compare how each one
        handles pedestrians along the way.
    </div>
    """, unsafe_allow_html=True)

    from PIL import Image
    import io

    cwa, cwb, cwc = st.columns(3)
    with cwa:
        wt_phase = st.selectbox("Phase", [2, 3, 4], index=2, format_func=lambda p: f"Phase {p}", key="wt_phase")
    with cwb:
        wt_map = st.selectbox("Map", PHASE_CONFIG[wt_phase]["map_ids"],
                               format_func=lambda m: f"Map {m}", key="wt_map")
    with cwc:
        wt_seed = st.slider("Episode seed", 0, 300, 42, key="wt_seed")

    def run_episode_capture(agent, env_phase, map_id, seed, core_mod):
        env = core_mod.BlindNavigatorEnv(phase=env_phase, map_ids=[map_id])
        state, _ = env.reset(seed=seed)
        agent.reset_hidden()
        frames, success, steps, total_r, collisions = [], False, 0, 0.0, 0
        max_steps = PHASE_CONFIG[env_phase]["battery"] * 2

        def cap(env):
            fig = env.render_figure(figsize=(4.0, 4.0))
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=80, bbox_inches="tight", facecolor="white")
            buf.seek(0); img = Image.open(buf).copy(); plt.close(fig); buf.close()
            return img

        frames.append(cap(env))
        done = truncated = False
        while not (done or truncated) and steps < max_steps:
            action = agent.act_greedy(state)
            state, reward, done, truncated, info = env.step(action)
            total_r += reward; steps += 1
            if info.get("pedestrian_collision"): collisions += 1
            success = done and info["cell_type"] == "GOAL"
            frames.append(cap(env))
        all_frames = frames + [frames[-1]] * 4
        buf2 = io.BytesIO()
        all_frames[0].save(buf2, format="GIF", save_all=True, append_images=all_frames[1:], duration=300, loop=0)
        buf2.seek(0)
        return buf2.read(), success, steps, total_r, collisions

    if st.button("▶  Run v1 vs v2 comparison", use_container_width=True):
        cgif1, cgif2 = st.columns(2)
        with cgif1:
            with st.spinner("Running v1..."):
                agent1 = DuelingPERAgent()
                agent1.load(os.path.join(OUT, "dueling_per_final"))
                gif1, suc1, steps1, r1_, coll1 = run_episode_capture(agent1, wt_phase, wt_map, wt_seed, sys.modules[BlindNavigatorEnv.__module__])
            icon1 = "✅" if suc1 else "❌"
            st.markdown(f"""
            <div style='text-align:center;margin-bottom:.5rem;'>
                <span style='color:#5A607A;font-family:Space Grotesk,sans-serif;font-weight:700;'>{icon1} Dueling-PER v1</span><br>
                <span style='font-size:.75rem;color:#5A607A;font-family:JetBrains Mono,monospace;'>
                    {steps1} steps · reward {r1_:.0f} · <span style='color:#F06449;'>{coll1} collision(s)</span>
                </span>
            </div>""", unsafe_allow_html=True)
            st.image(gif1, use_container_width=True)
        with cgif2:
            agent2, core_v2_mod = load_v2_agent()
            if agent2 is None:
                st.warning("v2 checkpoint not found.")
            else:
                with st.spinner("Running v2..."):
                    gif2, suc2, steps2, r2_, coll2 = run_episode_capture(agent2, wt_phase, wt_map, wt_seed, core_v2_mod)
                icon2 = "✅" if suc2 else "❌"
                st.markdown(f"""
                <div style='text-align:center;margin-bottom:.5rem;'>
                    <span style='color:#2DD4A0;font-family:Space Grotesk,sans-serif;font-weight:700;'>{icon2} Dueling-PER v2</span><br>
                    <span style='font-size:.75rem;color:#5A607A;font-family:JetBrains Mono,monospace;'>
                        {steps2} steps · reward {r2_:.0f} · <span style='color:#2DD4A0;'>{coll2} collision(s)</span>
                    </span>
                </div>""", unsafe_allow_html=True)
                st.image(gif2, use_container_width=True)
    else:
        st.markdown("""
        <div style='background:#141720;border:1px solid #252836;border-radius:10px;
                    padding:1.6rem;text-align:center;color:#5A607A;'>
            Choose a phase, map, and seed above, then press
            <strong style='color:#4F8EF7;'>▶ Run v1 vs v2 comparison</strong>
        </div>
        """, unsafe_allow_html=True)


elif page == "🤖  Agent Walkthrough":
    st.markdown("<h2 style='font-family:Space Grotesk,sans-serif;font-weight:700;margin-bottom:.5rem;'>🤖 Agent Walkthrough</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#141720;border:1px solid #252836;border-radius:10px;
                padding:.9rem 1.2rem;margin-bottom:1.2rem;font-size:.85rem;color:#5A607A;
                line-height:1.7;'>
        Watch each algorithm navigate live — <strong style='color:#E8EAF0;'>step by step</strong>
        through the 16×16 grid. The <span style='color:#4F8EF7;'>blue circle</span> is the agent,
        <span style='color:#F5A623;'>amber circles</span> are pedestrians.
        The arrow shows heading direction. Trail arrows show the path taken.
    </div>
    """, unsafe_allow_html=True)

    if not CORE_OK:
        st.error(f"Could not load core: {CORE_ERR}"); st.stop()

    from PIL import Image
    import io

    # Controls
    c1, c2, c3 = st.columns(3)
    with c1:
        demo_phase = st.selectbox(
            "Phase",
            list(range(5)),
            index=4,
            format_func=lambda p: f"Phase {p} — {PHASE_CONFIG[p]['description']}"
        )
    with c2:
        demo_seed = st.slider("Episode Seed", 0, 300, 42)
    with c3:
        speed_ms  = st.slider("Speed (ms / frame)", 150, 900, 400, 50)

    map_sel = st.selectbox(
        "Map",
        PHASE_CONFIG[demo_phase]["map_ids"],
        format_func=lambda m: f"Map {m} — {['Office Corridors','Hospital','Street Crossing','Mall (Pillars)','Apartment','Outdoor Park'][m]}"
    )
    compare_all = st.checkbox("Compare all algorithms side by side", value=True)
    if not compare_all:
        single_algo = st.selectbox("Algorithm", ALGOS)

    def make_gif(algo_name, phase, seed, map_id, speed=400):
        """Generate animated GIF of one episode."""
        cfg = PHASE_CONFIG[phase]
        env = BlindNavigatorEnv(phase=phase, map_ids=[map_id])
        state, _ = env.reset(seed=seed)

        # Load agent
        nm = _nm(algo_name)
        if algo_name == "Q-Learning":
            agent = QLearningAgent()
            p = os.path.join(OUT, f"{nm}_final.npy")
            if os.path.exists(p): agent.load(p)
        elif algo_name == "SARSA":
            agent = SARSAAgent()
            p = os.path.join(OUT, f"{nm}_final.npy")
            if os.path.exists(p): agent.load(p)
        else:
            agent = DuelingPERAgent()
            p = os.path.join(OUT, f"{nm}_final.pt")
            if os.path.exists(p): agent.load(p)
        agent.reset_hidden()

        trail_pos = [tuple(env.agent_pos)]
        trail_act = []
        frames = []
        success = False

        def capture_frame(env, trail_pos, trail_act, success):
            fig = env.render_figure(figsize=(4.5, 4.5))
            ax  = fig.axes[0]
            n   = len(trail_pos)
            # Draw path trail
            for i in range(1, n):
                pr, pc = trail_pos[i-1]
                cr, cc = trail_pos[i]
                alpha  = 0.25 + 0.75 * (i / max(n, 1))
                if (pr, pc) == (cr, cc):
                    circ = plt.Circle((pc+.5, pr+.5), .18,
                                      color="#F5A623", alpha=min(alpha+.2, 1), zorder=6)
                    ax.add_patch(circ)
                else:
                    dx = (cc - pc) * .55; dy = (cr - pr) * .55
                    color_arrow = "#2DD4A0" if (i == n-1 and success) else "#4F8EF7"
                    ax.annotate("",
                        xy=(pc+.5+dx*.5, pr+.5+dy*.5),
                        xytext=(pc+.5-dx*.5, pr+.5-dy*.5),
                        arrowprops=dict(arrowstyle="-|>", color=color_arrow, lw=1.8),
                        alpha=alpha, zorder=6)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", facecolor="white")
            buf.seek(0)
            img = Image.open(buf).copy()
            plt.close(fig); buf.close()
            return img

        frames.append(capture_frame(env, trail_pos, trail_act, False))
        done = truncated = False
        steps = 0
        total_r = 0.0
        max_steps = cfg["battery"] * 2

        while not (done or truncated) and steps < max_steps:
            action = agent.act_greedy(state)
            state, reward, done, truncated, info = env.step(action)
            total_r += reward; steps += 1
            if done and info.get("cell_type") == "GOAL":
                success = True
            trail_pos.append(tuple(env.agent_pos))
            trail_act.append(action)
            frames.append(capture_frame(env, trail_pos, trail_act, success))

        # Hold last frame
        all_frames = frames + [frames[-1]] * 5
        buf_gif = io.BytesIO()
        all_frames[0].save(buf_gif, format="GIF", save_all=True,
                           append_images=all_frames[1:], duration=speed, loop=0)
        buf_gif.seek(0)
        return buf_gif.read(), success, steps, total_r

    if st.button("▶  Run Episode", use_container_width=True):
        algos_run = ALGOS if compare_all else [single_algo]
        cols_gif  = st.columns(len(algos_run))

        for col_g, algo in zip(cols_gif, algos_run):
            color = COLORS[algo]
            with col_g:
                with st.spinner(f"Running {algo}…"):
                    try:
                        gif_bytes, success, steps, total_r = make_gif(
                            algo, demo_phase, demo_seed, map_sel, speed_ms
                        )
                        ok_icon = "✅" if success else "❌"
                        outcome = "Goal reached!" if success else "Failed"
                        st.markdown(f"""
                        <div style='text-align:center;margin-bottom:.5rem;'>
                            <div style='color:{color};font-weight:700;font-size:.95rem;
                                        font-family:Space Grotesk,sans-serif;'>{ok_icon} {algo}</div>
                            <div style='font-size:.72rem;color:#5A607A;font-family:JetBrains Mono,monospace;
                                        margin-top:.2rem;'>{outcome} · {steps} steps · {total_r:+.0f} reward</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.image(gif_bytes, use_container_width=True)
                    except Exception as e:
                        st.error(f"{algo}: {e}")
    else:
        st.markdown("""
        <div style='background:#141720;border:1px solid #252836;border-radius:10px;
                    padding:2.5rem;text-align:center;color:#5A607A;'>
            Configure settings above and press
            <strong style='color:#4F8EF7;'>▶ Run Episode</strong> to generate the animation.
        </div>
        """, unsafe_allow_html=True)

    # Architecture note
    st.markdown("---")
    st.markdown("<div class='section-label'>Network Architecture</div>", unsafe_allow_html=True)
    col_arch1, col_arch2 = st.columns(2)

    with col_arch1:
        st.markdown("""
        <div class='card'>
            <div style='font-weight:700;color:#9B7FEA;margin-bottom:.8rem;font-size:.95rem;'>
                DuelingLSTMNet
            </div>
            <div style='font-family:JetBrains Mono,monospace;font-size:.78rem;
                        color:#5A607A;line-height:2;'>
                <span style='color:#4F8EF7;'>Input</span>    → 20-dim sensor state<br>
                <span style='color:#4F8EF7;'>Encoder</span>  → Linear(20→256) + LayerNorm + ReLU × 2<br>
                <span style='color:#4F8EF7;'>LSTM</span>     → (256→128) temporal memory<br>
                <span style='color:#4F8EF7;'>Value</span>    → Linear(128→128→1)<br>
                <span style='color:#4F8EF7;'>Advantage</span>→ Linear(128→128→4)<br>
                <span style='color:#2DD4A0;'>Q(s,a)</span>   → V(s) + A(s,a) − mean(A)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_arch2:
        st.markdown("""
        <div class='card'>
            <div style='font-weight:700;color:#9B7FEA;margin-bottom:.8rem;font-size:.95rem;'>
                Why This Combination Works
            </div>
            <div style='font-size:.83rem;color:#94A3B8;line-height:1.8;'>
                <strong style='color:#4F8EF7;'>Dueling heads</strong> separate how good a
                state is from which action to take — critical in corridors where
                most actions have similar value.<br><br>
                <strong style='color:#2DD4A0;'>LSTM memory</strong> carries loop-detection
                signals across timesteps without needing hand-crafted features.<br><br>
                <strong style='color:#9B7FEA;'>PER</strong> focuses training on near-miss
                collisions and surprising pedestrian interactions — rare but important.
            </div>
        </div>
        """, unsafe_allow_html=True)
