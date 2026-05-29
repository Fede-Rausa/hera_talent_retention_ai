import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
import re

def estrai_lp(row):
    PERF_COLS = {
        2022: 'valutazione performance 2022',
        2023: 'valutazione performance 2023',
        2024: 'valutazione performance 2024',
        2025: 'valutazione performance 2025',
        2026: 'valutazione performance 2026',
    }

    LETTER_TO_NUM = {'B': 1, 'M': 2, 'A': 3}
    NUM_TO_LABEL = {1: 'B', 2: 'M', 3: 'A'}

    def extract_leadership_performance(row):
        leadership = {}
        performance = {}

        for year, col in PERF_COLS.items():
            if col not in row.index:
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

    return extract_leadership_performance(row)

def descrivi_dipendente(id, df, to_keep):
    row = df[df['ID_PERSONA'] == id]
    if len(row) == 0:
        return "Nessun dato trovato per questo dipendente."
    
    des = ''
    bool_lp = []
    
    # Prima verifica quali colonne contengono valutazioni performance
    for n in to_keep:
        bool_lp.append('valutazione perf' in n)

    # Aggiungi le descrizioni delle colonne (escludendo le valutazioni performance)
    for n in to_keep:
        if 'valutazione perf' not in n:
            if n != 'Ral':
                val = row[n].values[0] if len(row) > 0 else None
                if val is not None and str(val) != 'nan':
                    des += n + ': ' + str(val) + '\n\n'
            else:
                val = row[n].values[0] if len(row) > 0 else None
                if val is not None and str(val) != 'nan':
                    des += n + ': ' + str(int(float(val))) + '\n\n'
                if 'prediction_dict' in st.session_state:
                    if selected_id in st.session_state.prediction_dict:
                        prediction = st.session_state.prediction_dict[selected_id]
                        des += 'RAL prevista da modello di ML naive: ' + str(int(float(prediction))) + '\n\n'

    # Aggiungi le valutazioni leadership e performance
    if any(bool_lp) and len(row) > 0:
        lead, perf = estrai_lp(row.iloc[0])
        if lead:
            des += 'Valutazione Leadership del dipendente da parte del suo manager (in scala 1-3): ' + str(lead) + '\n'
        if perf:
            des += 'Valutazione Performance del dipendente da parte del suo manager (in scala 1-3): ' + str(perf) + '\n'
    return des

st.set_page_config(page_title="AI agent")
st.title("HR talent retention agent")

# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('streamlit_dashboard\\df_base_10300.csv', sep=';')
    df['ID_PERSONA'] = df['ID_PERSONA'].astype(str)
    return df

try:
    df0 = load_data()
except FileNotFoundError:
    st.error("❌ File `df_base_10300.csv` non trovato.")
    st.stop()

system_prompts = {
    'system0': """
    Sei un HR. Ti viene fornita la descrizione di un dipendente. 
    Devi indicare se il dipendente mostra una propensione a dare le dimissioni.
    Devi dire se il dipendente resterà in azienda:
    - per meno di un anno
    - per almeno 1 anno
    - per almeno 2 anni
    - per almeno 5 anni
    - per almeno 10 anni
    - fino al pensionamento

    Devi provare a prevedere per quanto tempo ancora resterà in azienda, indicando una durata precisa (non importa se sbagli).
    Devi dare una risposta precisa, non una risposta prudente (sono pochi i dipendenti che danno le dimissioni, 
    e a te ne viene mostrato uno qualsiasi).
    Devi dare una risposta esaustiva e sintetica.
    """,
    'system1': "Sei un assistente HR. Aiuta a rispondere alle domande sui dipendenti.",
    'system2': """
    Sei il dipendente di cui ricevi la descrizione. 
    Fornisci la risposta alla domanda: Pensi di dare le dimissioni? Se si tra quanto tempo? Se no per quali ragioni?
    """,
    'system3': """Sei un HR manager. Aiuta a rispondere alle domande sui dipendenti.""",
    'system4': """
    dal punto di vista del dipendente che ti viene descritto, elenca la lista:
    - delle ragioni per cui potresti avere desiderio di lasciare l'azienda
    - delle ragioni per cui potresti avere desiderio di restare in azienda
    Infine prendi una decisione (puoi dire che non la lascerai adesso, ma che potresti lasciarla in futuro, o che la lascerai sicuramente in futuro) 
    e motiva la tua decisione.""",
    'system5': """
    Sei il dipendente di cui ricevi la descrizione.
    """,
    'custom': ''
}

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
    
    selected_id = st.selectbox(
        "ID Persona",
        sorted_items, 
        index=current_index,
        key='person_select_page2'
    )
    
    st.divider()
    st.header("📋 Dati del dipendente da dare al llm")
    selected_cols = st.multiselect(
        "Colonne",
        options=df0.columns.tolist(),
        default=[c for c in [
            'Età', 'Sesso', 'Titolo Massimo di Istruzione', 'Stato Civile', 'Data Assunzione',
            'Società', 'TI / NTI', 'CCNL', 'Qualifica', 'Ruolo', 'Ral',
            'valutazione performance 2022', 'valutazione performance 2023',
            'valutazione performance 2024', 'valutazione performance 2025'
        ] if c in df0.columns]
    )
    
    to_keep = selected_cols

if 'setup' not in st.session_state:
    models_dict = {
        'novita': {'models':['meta-llama/Llama-3.1-8B-Instruct', 'meta-llama/Llama-3.2-1B-Instruct', 'Sao10K/L3-8B-Stheno-v3.2', 'NousResearch/Hermes-2-Pro-Llama-3-8B'], 'nparams': ['8B', '1B', '8B', '8B']},
        'together': {'models':['Qwen/Qwen2.5-7B-Instruct', 'EssentialAI/rnj-1-instruct'], 'nparams': ['8B', '8B']},
        'cerebras': {'models':['meta-llama/Llama-3.1-8B-Instruct'], 'nparams': ['8B']},
        'nscale': {'models':['Qwen/Qwen3-8B', 'Qwen/Qwen3-4B-Instruct-2507', 
                             'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 'Qwen/Qwen2.5-Coder-3B-Instruct', 'Qwen/Qwen2.5-Coder-7B-Instruct',
                             'Qwen/Qwen3-4B-Thinking-2507', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'], 
                   'nparams': ['8B', '4B', '1.5B', '3B', '7B', '4B', '7B']},
        'featherless-ai': {'models':['ishaanxgupta/gemma-2-2bit-quantised'], 'nparams': ['3B']},
        'hf-inference': {'models':['katanemo/Arch-Router-1.5B'], 'nparams': ['1.5B']},
        'publicai': {'models':['allenai/Olmo-3-7B-Instruct', 'swiss-ai/Apertus-8B-Instruct-2509'], 'nparams': ['0.5B', '8B']},
        'cohere': {'models':['CohereLabs/c4ai-command-r7b-12-2024', 'CohereLabs/tiny-aya-fire', 'CohereLabs/tiny-aya-water', 'CohereLabs/tiny-aya-global', 'CohereLabs/tiny-aya-earth'],
                   'nparams': ['7B', '3B', '3B', '3B', '3B']}
    }
    
    all_models = []
    all_providers = []
    for provider, info in models_dict.items():
        all_models.extend(info['models'])
        all_providers.extend([provider] * len(info['models']))
    
    st.session_state.all_models = all_models
    st.session_state.all_providers = all_providers
    st.session_state.models_dict = models_dict
    
    nop = np.array([1, 3, 7, 7, 13, 13, 30, 30, 34, 34, 70, 70])
    enjoule = np.array([5, 15, 30, 60, 60, 120, 150, 300, 150, 300, 300, 700])
    
    x = nop
    y = enjoule
    
    beta = np.cov(x, y)[0, 1] / np.var(x)
    inter = np.mean(y) - beta * np.mean(x)
    
    def predict_mj(nop):
        return max(beta * nop + inter, 0.01)
    
    st.session_state.predict_mj = predict_mj
    st.session_state.setup = True

mymodel = st.selectbox('Select model', options=sorted(st.session_state.all_models), index=4)
myprovider = st.session_state.all_providers[st.session_state.all_models.index(mymodel)]
nop = st.session_state.models_dict[myprovider]['nparams'][st.session_state.models_dict[myprovider]['models'].index(mymodel)]

st.markdown(f"""
            - **Model:** {mymodel.split('/')[1]}
            - **Inference provider:** {myprovider}
            - **Model author:** {mymodel.split('/')[0]}
            - **Model parameters:** {nop}
""")

max_tok = st.slider('max tokens', min_value=1, max_value=2000, value=500, help='''maximum number of tokens the llm will generate to answer''')
temp_param = st.slider('temperature', min_value=0.0, max_value=1.5, step=0.01, value=0.3,
                        help='''The llm doesn\'t select the most likely token, but uses the token probabilities it has predicted
                        to sample the next token. The temperature is a parameter that controls how much words can be sampled.
                        The higher the temperature the higher the randomness and the creativity of the answer.
                        A lower temperature makes the answers more deterministic and repeatable.
                        A 0 temperature implies that the llm selects always the most likely token.
                        A too high temperature often leads to nonsense answers.''')

my_system_prompt = st.selectbox('Select system prompt', options=list(system_prompts.keys()), index=0)

if my_system_prompt == "custom":
    sysp = st.text_area("Enter your custom system prompt", value="", height=200)
else:
    sysp = system_prompts[my_system_prompt]

# Genera la descrizione del dipendente
des = descrivi_dipendente(selected_id, df0, to_keep)

# Combina system prompt e descrizione
sysp0 = sysp + '\n\n**Descrizione del dipendente:**\n\n' + des


st.markdown(sysp)
# Visualizzazione solo per debug (opzionale)
with st.expander("Visualizza dati dipendente"):
    st.markdown(des)

# Inizializza o aggiorna i messaggi solo se necessario
if "messages" not in st.session_state:
    st.session_state.messages = []

# Gestione del system prompt nella conversazione
def update_system_message():
    # Rimuovi eventuali vecchi messaggi di sistema
    st.session_state.messages = [msg for msg in st.session_state.messages if msg["role"] != "system"]
    # Inserisci il nuovo system prompt all'inizio
    st.session_state.messages.insert(0, {'role': 'system', 'content': sysp0})

if 'old_sysp' not in st.session_state or st.session_state.old_sysp != sysp0:
    update_system_message()
    st.session_state.old_sysp = sysp0

if 'old_model' not in st.session_state:
    st.session_state.old_model = mymodel
else:
    if st.session_state.old_model != mymodel:
        # Cambiato modello: reset conversazione ma mantieni solo system prompt
        st.session_state.messages = [msg for msg in st.session_state.messages if msg["role"] == "system"]
        st.session_state.old_model = mymodel

# Visualizza la cronologia della chat (escludendo il system prompt)
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if st.button("Reset conversation"):
    # Reset: mantieni solo il system prompt
    st.session_state.messages = [{'role': 'system', 'content': sysp0}]
    st.rerun()

my_key = st.secrets['HF_TOKEN']


# Funzione per ottenere risposta dal modello
def get_llm_response(messages):
    try:
        client = InferenceClient(
            provider=myprovider,
            api_key=my_key
        )
        
        # Prepara i messaggi nel formato corretto (assicurati che non ci siano messaggi vuoti)
        clean_messages = []
        for msg in messages:
            if msg["content"] and str(msg["content"]).strip():  # Evita messaggi vuoti
                clean_messages.append({
                    "role": msg["role"],
                    "content": str(msg["content"])
                })
        
        # Aggiungi un messaggio user di default se necessario (per alcuni provider)
        if len(clean_messages) == 1 and clean_messages[0]["role"] == "system":
            clean_messages.append({
                "role": "user",
                "content": "Rispondi alla domanda in base al contesto fornito."
            })
        
        response_stream = client.chat.completions.create(
            model=mymodel,
            messages=clean_messages,
            max_tokens=max_tok,
            temperature=temp_param,
            stream=True,
        )
        
        reply = st.write_stream(
            chunk.choices[0].delta.content or ""
            for chunk in response_stream
            if chunk.choices and chunk.choices[0].delta.content
        )
        
        return reply if reply else "Spiacenti, non ho ricevuto una risposta valida."
    
    except Exception as e:
        st.error(f"Errore: {str(e)}")
        return None

# if st.button("get answer"):
#     if len(st.session_state.messages) == 1 and st.session_state.messages[0]["role"] == "system":
#         # Solo system prompt, aggiungi un messaggio user generico
#         with st.chat_message("user"):
#             st.markdown("Cosa puoi dirmi su questo dipendente?")
#         st.session_state.messages.append({"role": "user", "content": "Cosa puoi dirmi su questo dipendente?"})
    
#     with st.chat_message("assistant"):
#         reply = get_llm_response(st.session_state.messages)
#         if reply:
#             st.session_state.messages.append({"role": "assistant", "content": reply})

if prompt := st.chat_input("Say something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        reply = get_llm_response(st.session_state.messages)
        if reply:
            st.session_state.messages.append({"role": "assistant", "content": reply})