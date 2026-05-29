
import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
import re
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


st.set_page_config(page_title="Report degli abbandoni")

st.title('Report degli abbandoni')

# Controllo di sicurezza: se l'utente è andato direttamente a questa pagina
if "dataset" not in st.session_state:
    st.warning("Il dataset non è ancora stato caricato. Torna alla Home Page per inizializzarlo.")
    
    # Opzionale: puoi decidere di caricarlo al volo anche qui se manca
    # @st.cache_data
    # def load_large_dataset(): ...
    # st.session_state["dataset"] = load_large_dataset()
    
else:
    # 3. Recupera il dataset dalla memoria condivisa
    df = st.session_state["dataset"]
    
    # Usa il dataset normalmente per grafici o tabelle
    st.subheader("Filtro")
    filtro = st.selectbox("Seleziona cluster", ['all'] + [str(i) for i in range(df['cluster_id'].nunique())])
    
    #if st.button("Applica filtro"):
    if filtro != 'all':
        df = df[df['cluster_id'] == int(filtro)]
        clu_des = st.session_state["clu_des"]

        chiavi = list(item['cluster_id'] for item in clu_des)
        id = chiavi.index(int(filtro))
        descrizione = clu_des[id]['descrizione']

        # --- OPERAZIONE DI PULIZIA ENCODING ---
        try:
            # Converte la stringa in byte usando Latin-1 e la ridecodifica correttamente in UTF-8
            descrizione_pulita = descrizione.encode('latin1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            # Fallback di sicurezza: se la stringa era già corretta, la tiene così com'è
            descrizione_pulita = descrizione

        st.markdown(f"Descrizione del cluster {filtro}: {descrizione}")

    st.write(f'Number of samples: {df.shape[0]}')

    st.subheader("Seleziona grafici")

    grafici = st.selectbox("dati da visualizzare", ['anagrafiche', 'valutazioni questionari', 'macrogruppi', 'macro famiglie',
                                                    'durata e sopravvivenza', 'condizioni contrattuali', 'cluster_id'])


    if grafici == 'condizioni contrattuali':
        with st.container(border=True):
            # Preparazione dati per il CCNL
            tab_ccnl = df['CCNL'].value_counts().reset_index()
            tab_ccnl.columns = ['CCNL', 'Conteggio']
            
            # Grafico a torta piena interattivo
            fig_ccnl = px.pie(
                tab_ccnl, 
                names='CCNL', 
                values='Conteggio', 
                title='Distribuzione dei CCNL'
            )
            
            fig_ccnl.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            
            # Mostriamo il primo grafico
            st.plotly_chart(fig_ccnl, use_container_width=True)


        # --- SECONDO GRAFICO: ISTOGRAMMA RAL (Sotto) ---
        with st.container(border=True):
            # Selezione dati per la RAL
            ral_uscita = df['Ral di uscita_x']
            
            # Istogramma interattivo
            fig_ral = px.histogram(
                ral_uscita,
                x='Ral di uscita_x',
                nbins=20,
                title='Distribuzione della RAL di uscita',
                color_discrete_sequence=['gold'] # Mantiene il colore oro originale
            )
            
            # Ottimizzazione layout e titoli assi
            fig_ral.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="RAL di uscita",
                yaxis_title="Frequenza",
                showlegend=False
            )
            
            # Mostriamo il secondo grafico sotto al primo
            st.plotly_chart(fig_ral, use_container_width=True)



    if grafici == 'durata e sopravvivenza':

        # Selezione dei dati
        eta = df['ETA_x']

        nbins = st.slider(label='n of bins', min_value=5, max_value=100, value=20)

        # Creazione dell'istogramma interattivo
        fig_eta = px.histogram(
            eta,
            x='ETA_x',
            nbins=nbins, # Sostituisce 'bins=20' di matplotlib
            title="Distribuzione dell'età",
            color_discrete_sequence=['skyblue'] # Mantiene il tuo colore originale
        )

        # Ottimizzazione del layout e aggiunta dei bordi grigi alle barre
        fig_eta.update_traces(
            marker_line_color='grey', # Sostituisce 'edgecolor=grey'
            marker_line_width=1
        )

        fig_eta.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Età",
            yaxis_title="Frequenza",
            showlegend=False # Nasconde la legenda visto che c'è una sola serie di dati
        )

        # Visualizzazione in Streamlit
        st.plotly_chart(fig_eta, use_container_width=True)


        nbins2 = st.slider(label='n of bins_', min_value=5, max_value=100, value=20)


        # Selezione dei dati
        durata = df['DURATA']

        # Creazione dell'istogramma interattivo
        fig_durata = px.histogram(
            durata,
            x='DURATA',
            nbins=nbins2,
            title="Distribuzione dell'età aziendale",
            color_discrete_sequence=['springgreen'] # Mantiene il colore originale
        )

        # Aggiunta dei bordi grigi alle barre
        fig_durata.update_traces(
            marker_line_color='grey',
            marker_line_width=1
        )

        # Ottimizzazione del layout e dei titoli degli assi
        fig_durata.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Durata",
            yaxis_title="Frequenza",
            showlegend=False
        )

        # Visualizzazione in Streamlit
        st.plotly_chart(fig_durata, use_container_width=True)

    if grafici == 'valutazioni questionari':

        st.write('red: bad, blue: good')
        # Definiamo la lista delle colonne
        colonne_conti = [
            'Opportunità professionale in altra azienda',
            'Maggiore allineamento rispetto a competenze e interessi',
            'Maggiore impatto sul risultato', 
            'Nuove sfide',
            'Migliore offerta di Compensation e/o Benefit',
            'Migliore bilanciamento vita privata/lavoro', 
            'Altro'
        ]

        # Elaborazione dati: somma, ordinamento e reset dell'indice per Plotly
        conteggi = df[colonne_conti].sum(axis=0).sort_values(ascending=True).reset_index()
        conteggi.columns = ['Motivazione', 'Conteggio']

        # Creazione del grafico interattivo
        fig_dimissioni = px.bar(
            conteggi,
            x='Conteggio',
            y='Motivazione',
            orientation='h',
            title='Motivazioni della Scelta di dare le dimissioni',
            color='Conteggio',                           # Applica il gradiente in base al valore
            color_continuous_scale=px.colors.sequential.Bluered  # Sostituisce il coolwarm di matplotlib
        )

        # Ottimizzazione del layout
        fig_dimissioni.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Numero di risposte",
            yaxis_title=None,
            coloraxis_showscale=False # Nasconde la barra della legenda del colore (opzionale, rende il grafico più pulito)
        )

        # Visualizzazione in Streamlit (puoi metterlo dentro una colonna o lasciarlo a tutta pagina)
        st.plotly_chart(fig_dimissioni, use_container_width=True)



        # Definiamo la lista delle colonne per le medie
        colonne_medie = [
            'Livello di autonomia',
            'Opportunità di proporre nuove idee',
            'Sfide connesse al ruolo',
            'Relazione con il team',
            'Relazione con il management',
            'Ambiente fisico di lavoro',
            'Compensation e benefit',
            'Bilanciamento lavoro-vita privata',
            'Cultura e valori aziendali'
        ]

        # Elaborazione dati: media, ordinamento decrescente e reset dell'indice
        medie = df[colonne_medie].mean(axis=0).sort_values(ascending=False).reset_index()
        medie.columns = ['Fattore', 'Media']

        # Creazione del grafico interattivo
        fig_medie = px.bar(
            medie,
            x='Media',
            y='Fattore',
            orientation='h',
            title='Valutazioni medie nelle exit interviews',
            color='Media',
            # 'Bluered_r' inverte la scala: i valori alti saranno blu e i bassi rossi (come il tuo coolwarm.reversed())
            color_continuous_scale=px.colors.sequential.Bluered_r 
        )

        # Ottimizzazione del layout
        fig_medie.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Punteggio Medio",
            yaxis_title=None,
            coloraxis_showscale=False, # Nasconde la barra del gradiente per pulizia visiva
            yaxis={'categoryorder':'total ascending'} # Mantiene l'ordine corretto dall'alto verso il basso
        )

        # Visualizzazione in Streamlit
        st.plotly_chart(fig_medie, use_container_width=True)



    if grafici == 'macrogruppi':

                    # Elaborazione dati
                    colonne_macrogruppo = [c for c in df.columns if c.startswith('MACROGRUPPO')]
                    main_df_macrogruppo = df[colonne_macrogruppo].sum(axis=0).sort_values(ascending=True).reset_index()
                    main_df_macrogruppo.columns = ['Macro gruppo', 'Valore']
                    
                    # Creazione grafico a barre orizzontali (orientation='h')
                    fig_macro_g = px.bar(
                        main_df_macrogruppo,
                        x='Valore',
                        y='Macro gruppo',
                        orientation='h',
                        title='Distribuzione nei macro gruppi'
                    )
                    
                    # Ottimizzazione del layout e dei margini
                    fig_macro_g.update_layout(
                        margin=dict(l=2, r=2, t=40, b=2),
                        xaxis_title="Somma dei valori",
                        yaxis_title=None # Rimuove l'etichetta dell'asse Y superflua
                    )
                    
                    # Mostriamo il grafico
                    st.plotly_chart(fig_macro_g, use_container_width=True)

    if grafici == 'macro famiglie':
            # --- SECONDO GRAFICO: MACRO FAMIGLIE (Colonna 2) ---

                    colonne_macro = [c for c in df.columns if c.startswith('MACRO_F')]
                    main_df_macro = df[colonne_macro].sum(axis=0).sort_values(ascending=True).reset_index()
                    main_df_macro.columns = ['Macro famiglia', 'Valore']
                    
                    # Creazione grafico a barre orizzontali
                    fig_macro_f = px.bar(
                        main_df_macro,
                        x='Valore',
                        y='Macro famiglia',
                        orientation='h',
                        title='Distribuzione nelle macro famiglie'
                    )
                    
                    # Ottimizzazione del layout e dei margini
                    fig_macro_f.update_layout(
                        margin=dict(l=2, r=2, t=40, b=2),
                        xaxis_title="Somma dei valori",
                        yaxis_title=None
                    )
                    
                    # Mostriamo il grafico
                    st.plotly_chart(fig_macro_f, use_container_width=True)


    if grafici == 'cluster_id':
        colonna_df = 'cluster_id'
        with st.container(border=True):
            # Calcoliamo le frequenze e resettiamo l'indice per Plotly
            tab = df[colonna_df].value_counts().reset_index()
            tab.columns = [colonna_df, 'Conteggio']
            
            # Creiamo il grafico a torta interattivo con Plotly
            fig = px.pie(
                tab, 
                names=colonna_df, 
                values='Conteggio', 
                title=f"Distribuzione {colonna_df.replace('_', ' ').title()}",
                #hole=0.3 # Opzionale: trasforma la torta in una "Donut chart" molto moderna
            )
            
            # Rende il grafico responsive per adattarsi perfettamente alla colonna
            fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            
            # Mostriamo il grafico interattivo
            st.plotly_chart(fig, use_container_width=True)        



    if grafici == 'anagrafiche':
        colonne0 = ['SESSO', 'TITOLO_DI_STUDIO', 'GENERAZIONE']
        
        N_COLONNE_GRIGLIA = 2
        
        # Cicliamo sulle colonne a gruppi di 2
        for i in range(0, len(colonne0), N_COLONNE_GRIGLIA):
            
            with st.container():
                cols = st.columns(N_COLONNE_GRIGLIA)
                
                # Gestiamo la prima colonna
                if i < len(colonne0):
                    colonna_df = colonne0[i]
                    with cols[0]:
                        with st.container(border=True):
                            # Calcoliamo le frequenze e resettiamo l'indice per Plotly
                            tab = df[colonna_df].value_counts().reset_index()
                            tab.columns = [colonna_df, 'Conteggio']
                            
                            # Creiamo il grafico a torta interattivo con Plotly
                            fig = px.pie(
                                tab, 
                                names=colonna_df, 
                                values='Conteggio', 
                                title=f"Distribuzione {colonna_df.replace('_', ' ').title()}",
                                #hole=0.3 # Opzionale: trasforma la torta in una "Donut chart" molto moderna
                            )
                            
                            # Rende il grafico responsive per adattarsi perfettamente alla colonna
                            fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                            
                            # Mostriamo il grafico interattivo
                            st.plotly_chart(fig, use_container_width=True)
                
                # Gestiamo la seconda colonna
                if i + 1 < len(colonne0):
                    colonna_df = colonne0[i + 1]
                    with cols[1]:
                        with st.container(border=True):
                            tab = df[colonna_df].value_counts().reset_index()
                            tab.columns = [colonna_df, 'Conteggio']
                            
                            fig = px.pie(
                                tab, 
                                names=colonna_df, 
                                values='Conteggio', 
                                title=f"Distribuzione {colonna_df.replace('_', ' ').title()}",
                                #hole=0.3
                            )
                            fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                            
                            st.plotly_chart(fig, use_container_width=True)



