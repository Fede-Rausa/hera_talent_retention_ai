import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re

st.set_page_config(
    page_title="Profilo Dipendente",
    page_icon="👤",
    layout="wide"
)

# CSS minimale - solo lo stretto necessario
st.markdown("""
<style>
    /* Solo gli stili essenziali che Streamlit non fornisce nativamente */
    .info-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #e0dcd5;
    }
    .info-label {
        font-size: 11px;
        font-weight: 600;
        color: #8a8070;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .info-value {
        font-size: 16px;
        font-weight: 500;
        color: #1a1a2e;
    }
</style>
""", unsafe_allow_html=True)

# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('df_base_10300.csv', sep=';')
    df['ID_PERSONA'] = df['ID_PERSONA'].astype(str)
    return df

try:
    df0 = load_data()
except FileNotFoundError:
    st.error("❌ File `df_base_10300.csv` non trovato.")
    st.stop()


# Inizializza il session state
if 'selected_id' not in st.session_state:
    st.session_state.selected_id = None

# Lista ordinata
items = df0['ID_PERSONA'].tolist()
sorted_items = sorted(items)

# Calcola l'indice corrente
if st.session_state.selected_id in sorted_items:
    current_index = sorted_items.index(st.session_state.selected_id)
else:
    current_index = 0


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Seleziona Dipendente")


    # Nella Pagina 1
    selected_id = st.selectbox(
        "ID Persona", 
        sorted_items, 
        index=current_index,
        key='person_select_page1'  # Key diversa per ogni pagina
    )


    # Salva l'ID nel session state
    st.session_state.selected_id = selected_id

    st.divider()
    st.header("📋 Campi da visualizzare")
    selected_cols = st.multiselect(
        "Colonne",
        options=df0.columns.tolist(),
        default=[c for c in [
            'Età', 'Sesso', 'Titolo Massimo di Istruzione', 'Stato Civile', 'Data Assunzione',
            'Società', 'TI / NTI', 'CCNL', 'Qualifica', 'Ruolo', 'Ral', 'Provincia sede di lavoro',
            'valutazione performance 2022', 'valutazione performance 2023',
            'valutazione performance 2024', 'valutazione performance 2025'
        ] if c in df0.columns]
    )

# ── Employee data ─────────────────────────────────────────────────────────────
row = df0[df0['ID_PERSONA'] == selected_id].iloc[0]

def get_value(col):
    v = row.get(col)
    if pd.isna(v) or v in ['', 'nan', 'None']:
        return None
    return str(v)

def col_value(col):
    """Return value or None if missing."""
    if col not in row.index:
        return None
    v = row[col]
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    return None if s.lower() in ('nan', '', 'none') else s

# ── Hero Section (semplificata) ──────────────────────────────────────────────
st.title(f"👤 {selected_id}")
st.caption("Profilo Dipendente")

# Badges con componenti nativi
badges = []
for field in ['Ruolo', 'Società', 'Generazione', 'TI / NTI']:
    if val := get_value(field):
        badges.append(f"`{val}`")
if badges:
    st.markdown(" ".join(badges))

st.divider()





# ── Funzione helper per mostrare le card ──────────────────────────────────────
def show_card(label, value):
    if value:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-label">{label}</div>
            <div class="info-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

def show_section(title, columns):
    available = [(col, get_value(col)) for col in columns if get_value(col) and col in selected_cols]
    if available:
        st.subheader(title)
        cols = st.columns(min(len(available), 3))
        for i, (col, val) in enumerate(available):
            with cols[i % 3]:
                show_card(col, val)



# ── Sezioni principali ────────────────────────────────────────────────────────
show_section("👤 Anagrafica", [
    'Età', 'Sesso', 'Titolo Massimo di Istruzione', 'Stato Civile', 'Generazione', 'Provincia sede di lavoro'
])

show_section("📄 Contratto", [
    'Data Assunzione', 'Anzianità aziendale', 'Società', 'TI / NTI', 'CCNL', 'Qualifica'
])


# ── RAL ───────────────────────────────────────────────────────────────────────
if 'Ral' in selected_cols and (ral := get_value('Ral')):
    st.subheader("RAL")
    try:
        ral_formatted = f"€ {float(ral):,.0f}".replace(',', '.')
    except:
        ral_formatted = ral


    cols = st.columns(2)

    with cols[0]:
    #st.write("RAL Annua effettiva: ", ral_formatted)
        show_card("RAL Annua effettiva", ral_formatted)

    if 'prediction_dict' in st.session_state:
        if selected_id in st.session_state.prediction_dict:
            prediction = st.session_state.prediction_dict[selected_id]
            with cols[1]:
                show_card("RAL predetta", f"€ {float(prediction):,.0f}".replace(',', '.'))

show_section("🏢 Organizzazione", [
    'Ruolo', 'Responsabile ', 'Sottogruppo', 'Unità Organizzativa'
])



# ── Leadership & Performance Chart (versione originale) ───────────────────────
LETTER_TO_NUM = {'B': 1, 'M': 2, 'A': 3}
NUM_TO_LABEL = {1: 'B', 2: 'M', 3: 'A'}

perf_years = [2022, 2023, 2024, 2025]

def extract_leadership_performance(row):
    leadership = {}
    performance = {}
    for year in perf_years:
        col = f'valutazione performance {year}'
        if col not in row.index or col not in selected_cols:
            continue
        raw = row[col]
        if pd.isna(raw):
            continue
        code = str(raw).strip().upper()
        if len(code) != 2 or code[0] not in LETTER_TO_NUM or code[1] not in LETTER_TO_NUM:
            continue
        leadership[year] = LETTER_TO_NUM[code[0]]
        performance[year] = LETTER_TO_NUM[code[1]]
    return leadership, performance

leadership_data, performance_data = extract_leadership_performance(row)

if leadership_data or performance_data:
    st.subheader("🧭 Leadership & Performance nel Tempo")
    
    fig = go.Figure()
    
    if leadership_data:
        l_years = sorted(leadership_data.keys())
        l_values = [leadership_data[y] for y in l_years]
        fig.add_trace(go.Scatter(
            x=l_years, y=l_values,
            mode='lines+markers+text',
            name='Leadership',
            text=[NUM_TO_LABEL[v] for v in l_values],
            textposition='top center',
            line=dict(color='#e63946', width=3),
            marker=dict(size=10, color='#e63946')
        ))
    
    if performance_data:
        p_years = sorted(performance_data.keys())
        p_values = [performance_data[y] for y in p_years]
        fig.add_trace(go.Scatter(
            x=p_years, y=p_values,
            mode='lines+markers+text',
            name='Performance',
            text=[NUM_TO_LABEL[v] for v in p_values],
            textposition='bottom center',
            line=dict(color='#0f3460', width=3, dash='dot'),
            marker=dict(size=10, color='#0f3460')
        ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(
            tickmode='array',
            tickvals=sorted(set(list(leadership_data.keys()) + list(performance_data.keys()))),
            gridcolor='rgba(212,201,184,0.5)'
        ),
        yaxis=dict(
            range=[0.5, 3.5],
            tickmode='array',
            tickvals=[1, 2, 3],
            ticktext=['B (Basso)', 'M (Medio)', 'A (Alto)'],
            gridcolor='rgba(212,201,184,0.5)'
        ),
        plot_bgcolor='rgba(245,242,238,0.6)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Straordinari ──────────────────────────────────────────────────────────────
stra_data = {}
for year in [2022, 2023, 2024, 2025]:
    col = f'Ore straoardinario {year}'
    if col in selected_cols and (val := get_value(col)):
        if float(val) > 0:
            stra_data[year] = float(val)

if stra_data:
    st.subheader("⏱ Ore Straordinario")
    fig = go.Figure(data=[go.Bar(
        x=list(stra_data.keys()), 
        y=list(stra_data.values()),
        text=[f'{v:.0f}h' for v in stra_data.values()],
        textposition='outside'
    )])
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)


# ── Render Colloquio ──────────────────────────────────────────────────────────
colloquio_cols = [
    ('Situazione Attuale - RISPOSTA', 'Situazione Attuale - VALUTAZ'),
    ('Motivazione - RISPOSTA', 'Motivazione - VALUTAZ'),
    ('Aspettative Crescita - RISPOSTA', 'Aspettative Crescita - VALUTAZ'),
]

for resp_col, val_col in colloquio_cols:

    substr = resp_col.replace(' - RISPOSTA', '')

    resp = get_value(resp_col)
    val = get_value(val_col)

    if resp or val:
        st.subheader(f"🗣 {substr}")

    if resp:
        st.write(resp)
    if val:
        st.write(f'valutazione: {str(int(float(val)))}')


altre_voci = ['PROPOSTE', 'NOTE']

for v in altre_voci:
    val = get_value(v)
    if val:
        st.subheader(v)
        st.write(val)


#     if resp or val:
#         with st.expander(resp_col.replace(' - RISPOSTA', ''), expanded=True):
#             if resp:
#                 st.write(resp)
#             if val:
#                 st.write(f"**Valutazione:** {val}")

# # Verifica se almeno un campo colloquio è selezionato
# has_colloquio = any(
#     col1 in selected_cols or col2 in selected_cols 
#     for col1, col2 in colloquio_cols
# )

# st.write(str(has_colloquio))
# st.write(col_value('Situazione Attuale - RISPOSTA'))

# if has_colloquio:
#     st.subheader("🗣 Colloquio")
    
#     # Data colloquio
#     if 'DATA COLLOQUIO' in selected_cols and col_value('DATA COLLOQUIO'):
#         st.caption(f"📅 Data colloquio: **{col_value('DATA COLLOQUIO')}**")

#     for resp_col, val_col in colloquio_cols:
#         resp = col_value(resp_col) if resp_col in selected_cols else None
#         val = col_value(val_col) if val_col in selected_cols else None

#         if resp or val:
#             with st.expander(resp_col.replace(' - RISPOSTA', ''), expanded=True):
#                 if resp:
#                     st.write(resp)
#                 if val:
#                     st.write(f"**Valutazione:** {val}")
