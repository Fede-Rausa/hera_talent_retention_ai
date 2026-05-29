import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import unicodedata
import re
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Survival Analysis — Cox PH",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Stile custom ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
}
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

.metric-card {
    background: #f8f7f4;
    border-left: 4px solid #2d6a4f;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
}
.metric-card .label { font-size: 0.78rem; color: #666; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-card .value { font-size: 1.6rem; font-weight: 500; color: #1a1a1a; }

.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem;
    color: #1a1a1a;
    border-bottom: 2px solid #2d6a4f;
    padding-bottom: 0.3rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Survival Analysis — Modello di Cox")
st.markdown("**Proportional Hazards**: stima della durata attesa del contratto di lavoro per ogni dipendente.")

# ── Sidebar: caricamento dati ─────────────────────────────────────────────────
# with st.sidebar:
#     st.header("📂 Dati")
#     file_base    = st.file_uploader("Dataset base (tutti i dipendenti) — CSV", type=["csv"])
#     file_dimiss  = st.file_uploader("Dataset dimissioni — Excel", type=["xlsx"])
#     sep = st.selectbox("Separatore CSV", [";", ",", "\t"], index=0)



# ── Funzioni di supporto ──────────────────────────────────────────────────────

def clean_col_name(col: str) -> str:
    col = col.replace(' ', '_').replace('-', '_').replace('.', '_')
    col = unicodedata.normalize('NFKD', col).encode('ASCII', 'ignore').decode('ASCII')
    col = ''.join(c for c in col if c.isalnum() or c == '_')
    return col


def prepara_dati(df, categoriche, continue_cols):
    colonne_da_usare = categoriche + continue_cols + ['DURATA', 'ID_PERSONA']
    colonne_disponibili = [c for c in colonne_da_usare if c in df.columns]
    df_work = df[colonne_disponibili].copy()

    name_mapping = {col: clean_col_name(col) for col in df_work.columns}
    df_work = df_work.rename(columns=name_mapping)

    categoriche_clean   = [name_mapping[c] for c in categoriche   if c in name_mapping]
    continue_cols_clean = [name_mapping[c] for c in continue_cols if c in name_mapping]
    anz_col = name_mapping.get('DURATA', 'DURATA')
    id_col  = name_mapping.get('ID_PERSONA', 'ID_PERSONA')

    df_clean = df_work.copy()

    X_dummies = pd.get_dummies(df_clean[categoriche_clean], drop_first=True)

    if continue_cols_clean:
        X_cont = df_clean[continue_cols_clean].copy()
        for col in continue_cols_clean:
            X_cont[col] = pd.to_numeric(X_cont[col], errors='coerce').astype(float)
        X = pd.concat([X_dummies, X_cont], axis=1)
    else:
        X = X_dummies

    Y   = pd.to_numeric(df_clean[anz_col], errors='coerce')
    ids = df_clean[id_col]
    return X, Y, ids, df_clean, categoriche_clean


def build_df1(all_df_raw, main_df_raw):
    all_df = all_df_raw.copy()
    main_df = main_df_raw.copy()

    all_df['ID_PERSONA'] = all_df['ID_PERSONA'].astype(str)
    main_df['ID_PERSONA'] = main_df['ID_PERSONA'].astype(str)

    all_df.rename(columns={
        'Età': 'Eta', 'Sesso': 'SESSO',
        'Titolo Massimo di Istruzione': 'TITOLO_DI_STUDIO',
        'Qualifica': 'QUALIFICA', 'Generazione': 'GENERAZIONE',
        'Anzianità aziendale': 'DURATA'
    }, inplace=True)

    main_df.rename(columns={
        'ETA_x': 'Eta', 'Ral di uscita_x': 'Ral'
    }, inplace=True)

    to_keep = ['ID_PERSONA', 'Eta', 'SESSO', 'TITOLO_DI_STUDIO',
               'QUALIFICA', 'Ral', 'GENERAZIONE', 'CCNL', 'DURATA']

    cols_all  = [c for c in to_keep if c in all_df.columns]
    cols_main = [c for c in to_keep if c in main_df.columns]

    all_df0  = all_df[cols_all].copy();  all_df0['STATUS'] = 0
    main_df0 = main_df[cols_main].copy(); main_df0['STATUS'] = 1

    return pd.concat([main_df0, all_df0], ignore_index=True)


def parse_dummy_name(dummy_col: str, categoriche_clean: list) -> tuple[str, str] | None:
    """Restituisce (nome_variabile_originale, valore_categoria) dal nome dummy."""
    for cat in sorted(categoriche_clean, key=len, reverse=True):
        prefix = cat + "_"
        if dummy_col.startswith(prefix):
            return cat, dummy_col[len(prefix):]
    return None, dummy_col


def hr_barplot(hr_subset: pd.DataFrame, title: str):
    """Barplot orizzontale degli hazard ratio per una variabile categorica."""
    fig, ax = plt.subplots(figsize=(7, max(2, 0.5 * len(hr_subset) + 1)))

    labels = hr_subset['categoria'].tolist()
    values = hr_subset['HR'].tolist()

    colors = ['#c1121f' if v > 1 else '#2d6a4f' for v in values]
    bars = ax.barh(labels, values, color=colors, height=0.55, zorder=3)

    ax.axvline(x=1, color='#333', linewidth=1.4, linestyle='--', zorder=4)

    for bar, val in zip(bars, values):
        offset = 0.02
        ax.text(val + offset if val >= 1 else val - offset,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va='center',
                ha='left' if val >= 1 else 'right',
                fontsize=9, color='#333')

    ax.set_xlabel("Hazard Ratio", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlim(left=max(0, min(values) * 0.85),
                right=max(values) * 1.15)
    ax.grid(axis='x', linestyle=':', alpha=0.5, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    ref_patch = mpatches.Patch(color='#888', label='Categoria di riferimento = 1.0 (linea tratteggiata)')
    ax.legend(handles=[ref_patch], fontsize=8, loc='lower right')

    fig.tight_layout()
    return fig


# ── Main logic ────────────────────────────────────────────────────────────────



# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_and_build():
    df00 = pd.read_csv('df_base_10300.csv', sep=';')
    df00['ID_PERSONA'] = df00['ID_PERSONA'].astype(str)
    df11 = pd.read_excel('Dati_dimissioni_con_cluster.xlsx')
    df11['ID_PERSONA'] = df11['ID_PERSONA'].astype(str)
    df = build_df1(df00, df11)
    df = df[df['DURATA'] > 0]   # rimuovi durata 0 o mancante
    return df


@st.cache_data
def fit_model(df, categoriche_sel, continue_cols_model):
    from lifelines import CoxPHFitter
    categoriche_sel = list(categoriche_sel)        # riconverti da tuple
    continue_cols_model = list(continue_cols_model)  # riconverti da tuple
    X, Y, ids, df_clean, categoriche_clean = prepara_dati(df, categoriche_sel, continue_cols_model)
    df3 = X.copy()
    df3['DURATA'] = Y.values
    df3_fit = df3.dropna()
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df3_fit, duration_col='DURATA', event_col='STATUS')
    return cph, X, Y, ids, df3, df3_fit, categoriche_clean


try:
    df1 = load_and_build()
except FileNotFoundError:
    st.error("❌ File non trovato.")
    st.stop()

st.success(f"Dataset caricato: **{len(df1):,}** osservazioni ({df1['STATUS'].sum():,} eventi, {(df1['STATUS']==0).sum():,} censurati)")

# ── Selezione variabili ───────────────────────────────────────────────────────
st.markdown('<div class="section-title">⚙️ Selezione variabili</div>', unsafe_allow_html=True)

col_var1, col_var2 = st.columns(2)

CATEGORICHE_DEFAULT = ['SESSO', 'TITOLO_DI_STUDIO', 'QUALIFICA', 'GENERAZIONE', 'CCNL']
CONTINUE_DEFAULT    = ['Eta', 'Ral']

# Solo variabili presenti nel dataset
cat_disponibili  = [c for c in CATEGORICHE_DEFAULT if c in df1.columns]
cont_disponibili = [c for c in CONTINUE_DEFAULT    if c in df1.columns]

with col_var1:
    categoriche_sel = st.multiselect(
        "Variabili categoriche",
        options=cat_disponibili,
        default=cat_disponibili,
        help="Verranno trasformate in dummy (drop_first=True)"
    )

with col_var2:
    continue_sel = st.multiselect(
        "Variabili continue",
        options=cont_disponibili,
        default=cont_disponibili,
    )

# STATUS è sempre inclusa come continua (0/1)
continue_cols_model = continue_sel + ['STATUS']

if not categoriche_sel and not continue_sel:
    st.warning("Seleziona almeno una variabile.")
    st.stop()

# ── Fit del modello ───────────────────────────────────────────────────────────
with st.spinner("Fitting del modello Cox PH..."):
    try:
        cph, X, Y, ids, df3, df3_fit, categoriche_clean = fit_model(
            df1,
            tuple(categoriche_sel),        # le liste non sono hashable per la cache
            tuple(continue_cols_model)
        )
        fit_ok = True
    except Exception as e:
        st.error(f"Errore nel fitting: {e}")
        fit_ok = False

if not fit_ok:
    st.stop()

# ── Metriche riassuntive ──────────────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Riepilogo modello</div>', unsafe_allow_html=True)

concordance = cph.concordance_index_
n_eventi    = int(df3_fit['STATUS'].sum())
n_tot       = len(df3_fit)

m1, m2, m3 = st.columns(3)
for col_m, label, val in [
    (m1, "Osservazioni (fit)", f"{n_tot:,}"),
    (m2, "Eventi osservati",   f"{n_eventi:,}"),
    (m3, "Concordance Index",  f"{concordance:.4f}"),
]:
    col_m.markdown(
        f'<div class="metric-card"><div class="label">{label}</div><div class="value">{val}</div></div>',
        unsafe_allow_html=True
    )

# ── Tabella Hazard Ratios ─────────────────────────────────────────────────────
st.markdown('<div class="section-title">📋 Hazard Ratios delle covariate</div>', unsafe_allow_html=True)

hr_df = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].copy()
hr_df.columns = ['Hazard Ratio', 'HR CI 2.5%', 'HR CI 97.5%', 'p-value']
hr_df = hr_df.reset_index().rename(columns={'index': 'Covariata'})
hr_df['Significativo'] = hr_df['p-value'].apply(lambda p: '✅' if p < 0.05 else '—')

st.dataframe(
    hr_df.style
        .format({'Hazard Ratio': '{:.4f}', 'HR CI 2.5%': '{:.4f}',
                 'HR CI 97.5%': '{:.4f}', 'p-value': '{:.4f}'})
        .background_gradient(subset=['Hazard Ratio'], cmap='RdYlGn_r', vmin=0.5, vmax=2.0),
    use_container_width=True,
    hide_index=True
)

# ── Barplot HR per variabile categorica ──────────────────────────────────────
st.markdown('<div class="section-title">📊 Hazard Ratios per variabile categorica</div>', unsafe_allow_html=True)

hr_series = cph.hazard_ratios_

# Raggruppa le dummy per variabile originale
from collections import defaultdict
gruppi = defaultdict(list)
for coef_name in hr_series.index:
    var, cat = parse_dummy_name(coef_name, categoriche_clean)
    if var:
        gruppi[var].append({'dummy': coef_name, 'categoria': cat, 'HR': hr_series[coef_name]})

if gruppi:
    n_cols_plot = min(2, len(gruppi))
    plot_cols = st.columns(n_cols_plot)
    for i, (var_name, entries) in enumerate(gruppi.items()):
        hr_var = pd.DataFrame(entries)
        # Aggiungi riga di riferimento (categoria omessa = HR 1.0)
        ref_row = pd.DataFrame([{'dummy': f'{var_name}_REF', 'categoria': '(riferimento)', 'HR': 1.0}])
        hr_var = pd.concat([ref_row, hr_var], ignore_index=True)

        with plot_cols[i % n_cols_plot]:
            fig = hr_barplot(hr_var, f"{var_name}")
            st.pyplot(fig)
            plt.close(fig)
else:
    st.info("Nessuna variabile categorica selezionata per i grafici.")

# ── Previsioni per ogni osservazione ─────────────────────────────────────────
st.markdown('<div class="section-title">🔮 Previsioni individuali</div>', unsafe_allow_html=True)

with st.spinner("Calcolo previsioni..."):
    df3_pred = df3.copy()

    # predict_median richiede le stesse colonne usate in fit (esclusa DURATA)
    feature_cols = [c for c in df3_fit.columns if c != 'DURATA']
    df3_pred_feat = df3_pred[feature_cols].copy()

    median_pred    = cph.predict_median(df3_pred_feat)
    partial_hazard = cph.predict_partial_hazard(df3_pred_feat)

    result_df = pd.DataFrame({
        'ID_PERSONA':           ids.values,
        'DURATA_osservata':     Y.values,
        'STATUS':               df3_pred['STATUS'].values,
        'Mediana_stimata':      median_pred.values,
        'Tempo_residuo_stimato': (median_pred.values - Y.values).clip(min=0),
        'Partial_hazard':       partial_hazard.values,
    })

    status_label = {0: '⬜ Censurato', 1: '🔴 Evento'}
    result_df['Tipo'] = result_df['STATUS'].map(status_label)

# Filtro per tipo
tipo_filter = st.radio(
    "Mostra",
    ["Tutti", "Solo censurati", "Solo eventi"],
    horizontal=True
)
if tipo_filter == "Solo censurati":
    result_show = result_df[result_df['STATUS'] == 0]
elif tipo_filter == "Solo eventi":
    result_show = result_df[result_df['STATUS'] == 1]
else:
    result_show = result_df

st.dataframe(
    result_show[['ID_PERSONA', 'Tipo', 'DURATA_osservata',
                  'Mediana_stimata', 'Tempo_residuo_stimato', 'Partial_hazard']]
    .style.format({
        'DURATA_osservata':      '{:.2f}',
        'Mediana_stimata':       '{:.2f}',
        'Tempo_residuo_stimato': '{:.2f}',
        'Partial_hazard':        '{:.4f}',
    }).background_gradient(subset=['Partial_hazard'], cmap='Oranges'),
    use_container_width=True,
    hide_index=True
)

# Download
csv_out = result_df.to_csv(index=False).encode('utf-8')
st.download_button(
    "⬇️ Scarica previsioni CSV",
    data=csv_out,
    file_name="survival_predictions.csv",
    mime="text/csv"
)
