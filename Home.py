
import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
import re
import numpy as np
import json


# 1. Funzione con cache per evitare letture ripetute dal disco
@st.cache_data
def load_large_dataset():
    # Sostituisci con il tuo file (CSV, Parquet, database, ecc.)
    # Consiglio: il formato .parquet è MOLTO più veloce del .csv per grossi dataset
    main_df = pd.read_excel("Dati_dimissioni_con_cluster.xlsx") 
    return main_df


st.set_page_config(
    page_title="Hello Hera",
    page_icon="👋",
)


st.write("# Hera talent retention")

st.write('Insights e strumenti per analizzare il rischio di abbandono')

if "dataset" not in st.session_state:
    with st.spinner("Caricamento del grosso dataset in corso..."):
        st.session_state["dataset"] = load_large_dataset()
        st.session_state['clu_des'] = json.load(open('descrizioni.json', 'r'))
    st.success("Dataset caricato con successo!")

