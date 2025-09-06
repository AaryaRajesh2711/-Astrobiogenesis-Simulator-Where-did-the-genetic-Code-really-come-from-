# app.py
"""
DNA/RNA Kinetics & Population Simulator — Full app with:
- Planet presets & minerals
- Arrhenius hydrolysis + pH factor
- UV damage model
- Stochastic replication with per-base errors
- Population engine with carrying capacity
- Metrics: Shannon entropy, master-sequence frequency, diversity
- Timeline mode: simulate very long real-time spans (years -> seconds compressed)
- Single Plotly animation (Master vs Others)
- AI Hypothesis Generator summarizing likelihood RNA/DNA persist / life plausibility

Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import math
import random
from collections import Counter
from typing import List

# ---------- Constants ----------
R = 8.31446261815324  # J/(mol*K) universal gas constant

# ---------- Helper functions ----------
def random_sequence(length: int, nucleic_type: str) -> str:
    alphabet = ["A","U","C","G"] if nucleic_type == "RNA" else ["A","T","C","G"]
    return "".join(random.choice(alphabet) for _ in range(length))

def seq_mutate(seq: str, epsilon: float, nucleic_type: str) -> str:
    alphabet = ["A","U","C","G"] if nucleic_type == "RNA" else ["A","T","C","G"]
    lst = list(seq)
    for i, base in enumerate(lst):
        if random.random() < epsilon:
            choices = [b for b in alphabet if b != base]
            lst[i] = random.choice(choices)
    return "".join(lst)

def arrhenius_rate(A: float, Ea: float, temp_c: float) -> float:
    T = temp_c + 273.15
    try:
        return A * math.exp(-Ea / (R * T))
    except OverflowError:
        return 0.0

def pH_factor(pH_val: float, sensitivity: float) -> float:
    return 1.0 + sensitivity * abs(pH_val - 7.0)

def uv_rate(flux: float, cross_section: float) -> float:
    return flux * cross_section

def half_life_from_k(k: float) -> float:
    if k <= 0:
        return float('inf')
    return math.log(2) / k

def readable_time_seconds(s: float) -> str:
    if s == float('inf'):
        return "infinite"
    if s <= 1:
        return f"{s:.3g} s"
    mins = s / 60.0
    if mins < 60:
        return f"{mins:.2f} min"
    hrs = mins / 60.0
    if hrs < 24:
        return f"{hrs:.2f} hr"
    days = hrs / 24.0
    if days < 365:
        return f"{days:.2f} days"
    years = days / 365.25
    if years < 1e3:
        return f"{years:.2f} years"
    if years < 1e6:
        return f"{years/1e3:.2f} thousand years"
    if years < 1e9:
        return f"{years/1e6:.2f} million years"
    return f"{years/1e9:.2f} billion years"

def shannon_entropy_from_counts(counts: List[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent

# ---------- Planet presets mapping ----------
PLANET_PRESETS = {
    "Mars surface":         {"temp": -60, "pH": 7.0, "mineral":"Basalt", "uv": 2.0, "energy":"Cosmic radiation / occasional lightning"},
    "Europa ocean":         {"temp": -20, "pH": 8.0, "mineral":"Silica", "uv": 0.1, "energy":"Tidal heating, hydrothermal vents"},
    "Titan lake":           {"temp": -180, "pH": 7.0, "mineral":"None", "uv": 0.2, "energy":"Cosmic rays, photochemistry"},
    "Hot spring":           {"temp": 90, "pH": 6.5, "mineral":"Montmorillonite clay", "uv": 0.5, "energy":"Geothermal vents"},
    "Deep space":           {"temp": -270, "pH": 7.0, "mineral":"None", "uv": 0.0, "energy":"Cosmic rays"},
    "Venus surface":        {"temp": 460, "pH": 1.0, "mineral":"Iron-sulfur surface", "uv": 3.0, "energy":"Solar heating"},
    "Enceladus geyser":     {"temp": -20, "pH": 9.0, "mineral":"Silica", "uv": 0.05, "energy":"Tidal heating / vents"},
    "Io volcanic region":   {"temp": 1200, "pH": 7.0, "mineral":"Basalt", "uv": 1.5, "energy":"Intense volcanism"},
    "Subglacial lake":      {"temp": 0, "pH": 7.5, "mineral":"Silica", "uv": 0.0, "energy":"Geothermal"},
    "Asteroid surface":     {"temp": -50, "pH": 7.0, "mineral":"Iron-sulfur surface", "uv": 5.0, "energy":"Cosmic rays & solar UV"}
}

MINERAL_EFFECTS = {
    "None": 1.0,
    "Montmorillonite clay": 0.6,   # stabilizes RNA & helps polymerization in hypotheses
    "Iron-sulfur surface": 1.3,    # catalyzes reactions but can increase degradation
    "Silica": 0.9,
    "Basalt": 1.1
}

# ---------- Streamlit UI ----------
st.set_page_config(page_title="DNA/RNA Kinetics & Timeline Simulator", layout="wide")
st.title("DNA/RNA Kinetics, Evolution & Timeline Simulator")

st.markdown("""
**Overview & instructions (short)**  
This app simulates nucleic-acid survival and evolution under selected planetary conditions.
Before inputs: each factor is explained.

- **Hydrolysis (Arrhenius + pH)**: temperature-dependent cleavage rate `k_arr = A * exp(-Ea/(R*T))`. pH moves stability away from neutral via multiplicative factor.  
- **UV damage**: linear model `k_uv = uv_flux * cross_section`.  
- **Mineral surfaces**: modify effective hydrolysis via `mineral_factor` (stabilize or destabilize).  
- **Replication & mutation**: stochastic offspring (Poisson) with per-base error ε.  
- **Population engine**: survival → offspring → sample to carrying capacity K (weighted by fitness).  
- **Timeline mode**: compresses a real-world span (years) into simulation steps (each step uses dt = total_years_seconds / steps). Use with caution—large dt tends to produce rapid decay but allows “billions-of-years-in-seconds” visualization.  
- **AI Hypothesis Generator**: gives a short, plain-language hypothesis about RNA/DNA persistence & the plausibility of life originating under chosen settings.
""")

# ---------- Left column: explanation before inputs ----------
with st.expander("Explain model components (click to expand)", expanded=False):
    st.markdown("""
**Model components (more detail)**

- **Arrhenius hydrolysis**: we choose a pre-exponential factor `A` and activation energy `Ea`. Typical Ea values for backbone cleavage are high (tens of kJ→hundreds of kJ / mol); increasing temperature strongly increases rate.  
- **pH impact**: extremes (acidic/basic) catalyze hydrolysis; modeled simply as `1 + sensitivity * |pH - 7|`.  
- **UV damage**: number of damaging events scales with incident flux × cross-section. This is a simplified proxy.  
- **Mineral surfaces**: clays (montmorillonite) are often hypothesized to stabilize and catalyze polymerization; iron-sulfur surfaces can catalyze chemistry but also increase degradation.  
- **Replication & mutation**: replication is stochastic and sequence errors accumulate according to per-base error ε.  
- **Population engine**: we include selection via a simple fitness proportional to match fraction to the provided "master sequence".
""")

# ---------- Sidebar: Inputs ----------
st.sidebar.header("Environment & simulation controls")

# Planet preset selection (hook up presets)
planet_choice = st.sidebar.selectbox("Planet preset", list(PLANET_PRESETS.keys()), index=0)
preset = PLANET_PRESETS[planet_choice]

# Show preset quick info and allow applying defaults
with st.sidebar.expander("Preset details", expanded=False):
    st.write(f"Preset: **{planet_choice}**")
    st.write(f"- Typical temp: {preset['temp']} °C")
    st.write(f"- Typical pH: {preset['pH']}")
    st.write(f"- Mineral: {preset['mineral']}")
    st.write(f"- Typical UV hint: {preset['uv']} (relative)")
    st.write(f"- Energy sources: {preset['energy']}")

apply_preset = st.sidebar.button("Apply preset defaults")

# Nucleic acid type and sequence
nucleic = st.sidebar.selectbox("Nucleic acid type", ["RNA","DNA"], index=0)
sequence_len = st.sidebar.number_input("Sequence length (bases)", min_value=6, max_value=400, value=50, step=1)
master_seq_input = st.sidebar.text_input("Master sequence (blank → random)","")

# Environmental variables (can come from preset or manual)
if apply_preset:
    temp_default = preset["temp"]
    pH_default = preset["pH"]
    uv_default = float(preset["uv"])
    mineral_default = preset["mineral"]
else:
    # sensible defaults
    temp_default = 25
    pH_default = 7.0
    uv_default = 0.5
    mineral_default = "None"

temp_c = st.sidebar.slider("Temperature (°C)", -270, 1500, int(temp_default))
pH = st.sidebar.slider("pH", 1.0, 14.0, float(pH_default), step=0.1)

# Energy / UV / mineral choices
energy_choice = st.sidebar.selectbox("Energy source (informational)", ["None","Lightning","Geothermal vents","Cosmic radiation","Solar UV","Tidal heating","Volcanism"], index=0)
uv_flux = st.sidebar.slider("UV flux (relative)", 0.0, 10.0, float(uv_default), step=0.1)
uv_cross_section = st.sidebar.slider("UV cross-section (relative)", 0.0, 5.0, 0.02, step=0.01)

mineral_choice = st.sidebar.selectbox("Mineral surface", list(MINERAL_EFFECTS.keys()), index=list(MINERAL_EFFECTS.keys()).index(mineral_default) if mineral_default in MINERAL_EFFECTS else 0)

# Hydrolysis (Arrhenius) params
st.sidebar.subheader("Hydrolysis (Arrhenius) parameters")
A_default = 1e13
Ea_default_rna = 9e4
Ea_default_dna = 1.0e5
A = st.sidebar.number_input("Pre-exponential A (s⁻¹)", value=float(A_default), format="%.3g")
Ea = st.sidebar.number_input("Activation energy Ea (J/mol)", value=float(Ea_default_rna if nucleic=="RNA" else Ea_default_dna), format="%.1f")
pH_sensitivity = st.sidebar.slider("pH sensitivity (per pH unit)", 0.0, 1.0, 0.2, step=0.01)

# Replication & population
st.sidebar.subheader("Replication & population")
replication_rate = st.sidebar.slider("Base replication rate λ (offspring per parent per step)", 0.0, 10.0, 1.0, step=0.1)
per_base_error = st.sidebar.slider("Per-base replication error ε", min_value=1e-6, max_value=0.5, value=0.001, step=1e-5, format="%.6f")
carrying_capacity = st.sidebar.number_input("Carrying capacity K", min_value=10, max_value=10000, value=1000, step=10)

# Timeline mode vs interactive
st.sidebar.subheader("Time & mode")
mode = st.sidebar.selectbox("Mode", ["Interactive (hours/days per step)", "Timeline mode (years compressed to steps)"])
if mode.startswith("Interactive"):
    dt_seconds = st.sidebar.number_input("Seconds per simulation step (dt)", min_value=1.0, max_value=86400.0, value=3600.0, step=1.0)
    time_steps = st.sidebar.number_input("Simulation steps", min_value=5, max_value=5000, value=500, step=5)
else:
    total_years = st.sidebar.number_input("Total simulated real time (years)", min_value=1.0, max_value=1e10, value=1e9, step=1.0)
    time_steps = st.sidebar.number_input("Simulation steps", min_value=5, max_value=2000, value=500, step=5)
    # dt_seconds will be computed later as total_years_seconds / steps

seed = st.sidebar.number_input("Random seed", value=1234, step=1)
run_button = st.sidebar.button("Run simulation")

# ---------- Run simulation ----------
if run_button:
    st.write("## Running simulation — summary of inputs")
    st.write(f"- Planet preset: **{planet_choice}** (applied defaults: {apply_preset})")
    st.write(f"- Nucleic acid: **{nucleic}**, Sequence length: **{sequence_len}**")
    st.write(f"- Temperature: **{temp_c} °C**, pH: **{pH}**, Mineral: **{mineral_choice}**, UV flux: **{uv_flux}** (cross-section {uv_cross_section})")
    st.write(f"- Replication rate λ: **{replication_rate}**, per-base error ε: **{per_base_error:.6f}**, K: **{carrying_capacity}**")
    st.write(f"- Time mode: **{mode}**, Steps: **{time_steps}**")

    # seed
    random.seed(int(seed))
    np.random.seed(int(seed))

    # Prepare master sequence
    if master_seq_input.strip() == "":
        master_seq = random_sequence(sequence_len, nucleic)
        st.info(f"No master sequence provided → generated random master: `{master_seq}`")
    else:
        s = master_seq_input.strip().upper()
        if nucleic == "RNA":
            s = s.replace("T", "U")
            alphabet = set(["A","U","C","G"])
        else:
            s = s.replace("U", "T")
            alphabet = set(["A","T","C","G"])
        sanitized = "".join(ch if ch in alphabet else random.choice(list(alphabet)) for ch in s)
        if len(sanitized) < sequence_len:
            sanitized += random_sequence(sequence_len - len(sanitized), nucleic)
        master_seq = sanitized[:sequence_len]
        st.success(f"Using master sequence: `{master_seq}`")

    # Compute kinetic rates
    k_arr = arrhenius_rate(A, Ea, temp_c)     # s^-1
    k_ph_mult = pH_factor(pH, pH_sensitivity)
    mineral_factor = MINERAL_EFFECTS.get(mineral_choice, 1.0)
    k_hydro = k_arr * k_ph_mult * mineral_factor
    k_uv = uv_rate(uv_flux, uv_cross_section)

    # Determine dt (seconds per simulation step)
    if mode.startswith("Interactive"):
        dt = float(dt_seconds)
        total_real_seconds = dt * float(time_steps)
        st.write(f"- Each simulation step = **{dt:.1f} s** (≈ {dt/3600.0:.3f} hr). Total simulated time ≈ **{readable_time_seconds(total_real_seconds)}**.")
    else:
        total_years_val = float(total_years)
        total_real_seconds = total_years_val * 365.25 * 24 * 3600.0
        dt = total_real_seconds / float(time_steps)
        st.write(f"- Timeline mode: total real span = **{total_years_val:.3g} years** → dt per step = **{readable_time_seconds(dt)}**.")

    # Display kinetics summary & half-lives
    t_half_hydro = half_life_from_k(k_hydro)
    t_half_uv = half_life_from_k(k_uv)
    st.write("### Kinetics summary (model-derived)")
    st.write(f"- Arrhenius k_arr = **{k_arr:.3e} s⁻¹**")
    st.write(f"- pH multiplicative factor = **{k_ph_mult:.3f}**")
    st.write(f"- Mineral factor = **{mineral_factor:.3f}**")
    st.write(f"- Effective hydrolysis k_hydro = **{k_hydro:.3e} s⁻¹** → half-life ≈ **{readable_time_seconds(t_half_hydro)}**")
    st.write(f"- UV damage k_uv = **{k_uv:.3e} s⁻¹** → half-life ≈ **{readable_time_seconds(t_half_uv)}**")
    st.info("Reminder: these are order-of-magnitude model outputs — calibrate A/Ea/pH for literature matching.")

    # Initialize population
    initial_pop = min(int(carrying_capacity), 200)  # start with up to 200 individuals of master to seed
    population = [master_seq] * initial_pop

    # History arrays
    steps = int(time_steps)
    hist_pop_size = np.zeros(steps, dtype=int)
    hist_master_frac = np.zeros(steps, dtype=float)
    hist_entropy = np.zeros(steps, dtype=float)
    hist_unique = np.zeros(steps, dtype=int)

    # Simulation loop
    K = int(carrying_capacity)
    selection_strength = 5.0  # tuneable selection multiplier

    for step in range(steps):
        # 1) Survival via first-order rates over dt
        total_k = k_hydro + k_uv
        survival_prob = math.exp(-total_k * dt)
        survivors = [s for s in population if random.random() < survival_prob]
        population = survivors

        # 2) Replication
        offspring = []
        for seq in population:
            matches = sum(1 for a,b in zip(seq, master_seq) if a == b)
            match_frac = matches / sequence_len
            fitness = 1.0 + selection_strength * match_frac
            mean_off = replication_rate * fitness
            n_off = np.random.poisson(mean_off) if mean_off > 0 else 0
            for _ in range(n_off):
                child = seq_mutate(seq, per_base_error, nucleic)
                offspring.append(child)

        # 3) Combine candidate pool
        candidate_pool = population + offspring
        if len(candidate_pool) == 0:
            # extinction
            st.warning(f"Population extinct at step {step}. Ending early.")
            hist_pop_size[step:] = 0
            hist_master_frac[step:] = 0.0
            hist_entropy[step:] = 0.0
            hist_unique[step:] = 0
            break

        # 4) Apply carrying capacity selection (weighted by fitness)
        if len(candidate_pool) <= K:
            new_population = candidate_pool.copy()
        else:
            weights = []
            for seq in candidate_pool:
                matches = sum(1 for a,b in zip(seq, master_seq) if a == b)
                match_frac = matches / sequence_len
                fitness = 1.0 + selection_strength * match_frac
                weights.append(fitness)
            weights = np.array(weights, dtype=float)
            probs = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
            chosen_idx = np.random.choice(len(candidate_pool), size=K, replace=True, p=probs)
            new_population = [candidate_pool[i] for i in chosen_idx]

        population = new_population

        # 5) Metrics
        hist_pop_size[step] = len(population)
        counter = Counter(population)
        master_count = counter.get(master_seq, 0)
        hist_master_frac[step] = master_count / len(population) if len(population) > 0 else 0.0
        hist_entropy[step] = shannon_entropy_from_counts(list(counter.values()))
        hist_unique[step] = len(counter)

    # ---------- Prepare animation data ----------
    # We'll create frames where each frame draws the history up to that step
    animation_rows = []
    for f in range(len(hist_pop_size)):
        for j in range(f+1):
            animation_rows.append({
                "Frame": f,
                "Step": j,
                "MasterFrac": hist_master_frac[j],
                "OthersFrac": max(0.0, 1.0 - hist_master_frac[j]),
                "Population": hist_pop_size[j],
                "Entropy": hist_entropy[j]
            })
    df_anim = pd.DataFrame(animation_rows)

    # Build frames
    frames = []
    max_frame = df_anim["Frame"].max() if not df_anim.empty else 0
    for f in range(max_frame + 1):
        sub = df_anim[df_anim["Frame"] == f]
        x = sub["Step"].tolist()
        y_master = sub["MasterFrac"].tolist()
        y_others = sub["OthersFrac"].tolist()
        frame = go.Frame(
            data=[
                go.Scatter(x=x, y=y_master, mode="lines+markers", name="Master fraction"),
                go.Scatter(x=x, y=y_others, mode="lines+markers", name="Others fraction")
            ],
            name=str(f)
        )
        frames.append(frame)

    fig = go.Figure(
        data=[
            go.Scatter(x=[], y=[], mode="lines+markers", name="Master fraction"),
            go.Scatter(x=[], y=[], mode="lines+markers", name="Others fraction")
        ],
        layout=go.Layout(
            title=f"Population composition animation — {len(hist_pop_size)} steps",
            xaxis=dict(title="Simulation step", range=[0, max(1, len(hist_pop_size)-1)]),
            yaxis=dict(title="Fraction of population", range=[0,1]),
            updatemenus=[{
                "type":"buttons",
                "buttons":[
                    {"label":"Play","method":"animate","args":[None, {"frame":{"duration": max(50, int(2000/len(hist_pop_size))) , "redraw":True}, "fromcurrent":True, "transition":{"duration":0}}]},
                    {"label":"Pause","method":"animate","args":[[None], {"frame":{"duration":0, "redraw":False}, "mode":"immediate", "transition":{"duration":0}}]}
                ],
                "direction":"left",
                "pad":{"r":10,"t":10},
                "showactive":True,
                "x":0.1,
                "y":1.15,
                "xanchor":"right",
                "yanchor":"top"
            }]
        ),
        frames=frames
    )

    fig.update_traces(marker=dict(size=6))
    # slider
    steps_slider = []
    for k in range(max_frame + 1):
        step = {
            "args": [[str(k)], {"frame": {"duration": 0, "redraw": True}, "mode":"immediate", "transition": {"duration": 0}}],
            "label": str(k),
            "method": "animate"
        }
        steps_slider.append(step)
    sliders = [{
        "currentvalue": {"prefix": "Frame: ", "visible": True},
        "pad": {"b": 10, "t": 50},
        "steps": steps_slider
    }]
    fig.update_layout(sliders=sliders)

    # ---------- Display results ----------
    st.write("## Results & Diagnostics")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Final population size", int(hist_pop_size[-1]) if len(hist_pop_size)>0 else 0)
        st.metric("Final master fraction", f"{hist_master_frac[-1]:.4f}" if len(hist_master_frac)>0 else "0.0000")
    with c2:
        st.metric("Final Shannon entropy (bits)", f"{hist_entropy[-1]:.3f}" if len(hist_entropy)>0 else "0.000")
        st.metric("Final unique genotypes", int(hist_unique[-1]) if len(hist_unique)>0 else 0)

    st.write("### Animated population composition (Master vs Others)")
    st.plotly_chart(fig, use_container_width=True)

    # Static plots
    df_metrics = pd.DataFrame({
        "Step": np.arange(len(hist_pop_size)),
        "Population": hist_pop_size,
        "MasterFraction": hist_master_frac,
        "ShannonEntropy": hist_entropy,
        "UniqueGenotypes": hist_unique
    })

    st.write("### Additional static plots")
    fig_pop = px.line(df_metrics, x="Step", y="Population", title="Population size over time")
    st.plotly_chart(fig_pop, use_container_width=True)
    fig_entropy = px.line(df_metrics, x="Step", y="ShannonEntropy", title="Shannon entropy over time (bits)")
    st.plotly_chart(fig_entropy, use_container_width=True)
    fig_master = px.line(df_metrics, x="Step", y="MasterFraction", title="Master-sequence frequency over time")
    st.plotly_chart(fig_master, use_container_width=True)

    # Top genotypes table
    final_counts = Counter(population)
    top = final_counts.most_common(30)
    df_top = pd.DataFrame(top, columns=["Sequence","Count"])
    st.write("### Top genotypes at final step (top 30)")
    st.dataframe(df_top)

    # Kinetics validation summary
    st.write("### Kinetics validation (qualitative)")
    expected_range = "RNA: minutes→days (aqueous; hot & extreme pH speeds hydrolysis)" if nucleic=="RNA" else "DNA: hours→years (often more stable than RNA)"
    st.write(f"- Expected qualitative half-life: **{expected_range}**")
    st.write(f"- Model-derived hydrolysis half-life: **{readable_time_seconds(t_half_hydro)}**")
    st.write(f"- Model-derived UV half-life: **{readable_time_seconds(t_half_uv)}**")
    st.info("If model results seem unrealistic, adjust A, Ea, pH sensitivity, mineral factor, UV flux, or dt/timeline span to calibrate.")

    # ---------- AI Hypothesis Generator ----------
    st.write("## AI Hypothesis Generator — short hypothesis (automatically generated)")
    # Build a compact hypothesis string based on settings
    def generate_hypothesis():
        drivers = []
        # temperature
        if temp_c >= 80:
            drivers.append("high temperature (accelerates hydrolysis)")
        elif temp_c <= -50:
            drivers.append("very low temperature (slows hydrolysis, but may limit liquid water)")
        else:
            drivers.append("moderate temperature")
        # pH
        if abs(pH-7.0) > 2.0:
            drivers.append("extreme pH (destabilizing)")
        else:
            drivers.append("near-neutral pH (more stable)")
        # mineral
        if mineral_choice == "Montmorillonite clay":
            drivers.append("clay surfaces that can stabilize and catalyze polymerization")
        elif mineral_choice == "Iron-sulfur surface":
            drivers.append("iron-sulfur surfaces (catalytic but potentially harsh)")
        elif mineral_choice == "Silica":
            drivers.append("silica surfaces (moderate stabilization)")
        # UV/energy
        if uv_flux * uv_cross_section > 1.0:
            drivers.append("high UV flux (causes photodamage)")
        elif uv_flux * uv_cross_section > 0.1:
            drivers.append("moderate UV")
        else:
            drivers.append("low UV exposure")

        # population outcomes
        final_master_frac = hist_master_frac[-1] if len(hist_master_frac)>0 else 0.0
        diversity = hist_unique[-1] if len(hist_unique)>0 else 0
        if final_master_frac > 0.5:
            outcome = "Master-like sequences remain dominant, suggesting conditions favor sequence persistence and selection."
        elif diversity > K * 0.2:
            outcome = "High diversity and low dominance suggest that error-prone replication prevents stable master-sequence persistence."
        elif final_master_frac == 0:
            outcome = "Population went extinct under these conditions—persistence unlikely."
        else:
            outcome = "Some persistence but not strong dominance; conditions may allow transient survival but not reliable long-term maintenance."

        # timeline comment
        if mode.startswith("Timeline"):
            timeline_comment = f"Simulated over {total_years_val:.3g} years compressed into {steps} steps; dt ≈ {readable_time_seconds(dt)} per step."
        else:
            total_seconds = dt * steps
            timeline_comment = f"Simulated ~{readable_time_seconds(total_seconds)} total real time across {steps} steps."

        hypothesis = (
            f"Under the chosen settings (planet: {planet_choice}; temp {temp_c}°C; pH {pH}; mineral {mineral_choice}; UV flux {uv_flux}), "
            f"the primary drivers are {', '.join(drivers[:3])}. {outcome} {timeline_comment} "
            "Interpretation: this model is simplified — favorable persistence typically requires liquid water, moderate temperature/pH, and stabilizing surfaces (e.g., clays). "
            "If you want a deeper hypothesis, try varying mineral to 'Montmorillonite clay', lower UV, or reduce per-base error ε and re-run."
        )
        return hypothesis

    hypo = generate_hypothesis()
    st.write(hypo)

    st.success("Simulation finished. You can tweak parameters and re-run. Use timeline mode with care — dt becomes very large and leads to rapid decay in many parameter regimes.")
else:
    st.write("Adjust settings on the left and press **Run simulation**. Use 'Apply preset defaults' to quickly load planet presets.")


