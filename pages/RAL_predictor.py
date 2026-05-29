import pandas as pd
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn import tree
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="Previsione della RAL")

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

st.title("Previsione della RAL")
st.write("In questa pagina puoi addestrare un modello di regressione per prevedere la RAL (Retribuzione Annua Lorda) dei dipendenti. Seleziona le variabili da includere nel modello e clicca su 'Addestra Modello' per vedere i risultati.")
# ── Sidebar per i controlli ─────────────────────────────────────────────────
st.sidebar.header("⚙️ Configurazione Modello")

# Selezione del modello
modello_scelto = st.sidebar.selectbox(
    "Seleziona il modello",
    ["Albero di Decisione", "Regressione Lineare"]
)

# Selezione variabili categoriche
st.sidebar.subheader("Variabili Categoriche")
variabili_categoriche = ['Qualifica', 'Generazione', 'CCNL', 'Sesso', 'Stato Civile', 
                         'Titolo Massimo di Istruzione','Tutele crescenti', 'Categoria Protetta']
categoriche_selezionate = st.sidebar.multiselect(
    "Seleziona variabili categoriche",
    variabili_categoriche,
    default=variabili_categoriche[:5]
)

# Selezione variabili continue
st.sidebar.subheader("Variabili Continue")
variabili_continue = ['Età', 'Anzianità aziendale']
continue_selezionate = st.sidebar.multiselect(
    "Seleziona variabili continue",
    variabili_continue,
    default=variabili_continue
)

# Parametri specifici per albero di decisione
if modello_scelto == "Albero di Decisione":
    st.sidebar.subheader("Parametri Albero")
    max_depth = st.sidebar.slider("Profondità massima", 1, 20, 3)
    min_samples_leaf = st.sidebar.slider("Minimo campioni per foglia", 5, 500, 100)

# ── Preparazione dati ─────────────────────────────────────────────────────
@st.cache_data
def prepara_dati(df, categoriche, continue_cols):
    # Seleziona le colonne da usare
    colonne_da_usare = categoriche + continue_cols + ['Ral', 'ID_PERSONA']
    
    # Verifica che tutte le colonne esistano
    colonne_disponibili = [col for col in colonne_da_usare if col in df.columns]
    if len(colonne_disponibili) < len(colonne_da_usare):
        st.warning(f"Colonne non trovate: {set(colonne_da_usare) - set(colonne_disponibili)}")
    
    df_work = df[colonne_disponibili].copy()
    
    # Crea un dizionario per mappare i nomi originali ai nomi puliti
    name_mapping = {}
    for col in df_work.columns:
        # Pulisci il nome della colonna
        clean_name = col
        # Sostituisci spazi e trattini con underscore
        clean_name = clean_name.replace(' ', '_').replace('-', '_')
        # Rimuovi caratteri speciali (opzionale, mantieni lettere accentate)
        import unicodedata
        # Normalizza le lettere accentate (es: à -> a)
        clean_name = unicodedata.normalize('NFKD', clean_name).encode('ASCII', 'ignore').decode('ASCII')
        # Rimuovi eventuali altri caratteri non alfanumerici (mantieni underscore)
        clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '_')
        name_mapping[col] = clean_name
    
    # Rinomina le colonne
    df_work = df_work.rename(columns=name_mapping)
    
    # Aggiorna i nomi delle variabili selezionate per riflettere le modifiche
    categoriche_clean = [name_mapping[col] for col in categoriche if col in name_mapping]
    continue_cols_clean = [name_mapping[col] for col in continue_cols if col in name_mapping]
    
    # Gestisci il caso in cui 'Ral' e 'ID_PERSONA' possano essere stati modificati
    ral_col = name_mapping.get('Ral', 'Ral')
    id_col = name_mapping.get('ID_PERSONA', 'ID_PERSONA')
    
    # Rimuovi righe con NaN nelle variabili selezionate o nella RAL
    colonne_per_check = categoriche_clean + continue_cols_clean + [ral_col]
    df_clean = df_work.dropna(subset=colonne_per_check)
    
    if len(df_clean) == 0:
        st.error("❌ Nessuna riga valida dopo la rimozione dei NaN!")
        return None, None, None, None
    
    # Crea variabili dummy per le categoriche (usa i nomi puliti)
    X_dummies = pd.get_dummies(df_clean[categoriche_clean], drop_first=True)
    
    # Aggiungi variabili continue
    if continue_cols_clean:
        X_continue = df_clean[continue_cols_clean]
        # Assicurati che le colonne continue siano numeriche
        for col in continue_cols_clean:
            X_continue[col] = pd.to_numeric(X_continue[col], errors='coerce')
            X_continue[col] = X_continue[col].astype(float)  # Rimuovi righe con NaN nelle continue
        X = pd.concat([X_dummies, X_continue], axis=1)
    else:
        X = X_dummies
    
    # Assicurati che Y sia numerica
    Y = pd.to_numeric(df_clean[ral_col], errors='coerce')
    
    # Rimuovi eventuali righe dove Y è NaN (dovrebbe essere già gestito da dropna)
    valid_idx = ~Y.isna()
    X = X[valid_idx]
    Y = Y[valid_idx]
    ids = df_clean[valid_idx][id_col]
    df_clean = df_clean[valid_idx]
    
    return X, Y, ids, df_clean


train_ratio = st.slider("Train/Test Split", 0.0, 1.0, 0.8, help='Seleziona la proporzione di dati da utilizzare per il training.')

# ── Addestramento e valutazione ───────────────────────────────────────────
if st.sidebar.button("🔄 Addestra Modello", type="primary"):
    if not categoriche_selezionate and not continue_selezionate:
        st.error("❌ Seleziona almeno una variabile per il modello!")
        st.stop()
    
    with st.spinner("Addestramento in corso..."):
        # Prepara i dati
        X, Y, ids, df_clean = prepara_dati(
            df0, 
            categoriche_selezionate, 
            continue_selezionate
        )
        
        if len(X) == 0:
            st.error("❌ Nessun dato valido dopo la rimozione dei NaN!")
            st.stop()
        
        # Split train/test (80/20)
        np.random.seed(42)
        n = len(X)
        id_train = np.random.choice(n, size=int(train_ratio*n), replace=False)
        id_test = np.setdiff1d(np.arange(n), id_train)
        
        X_train = X.iloc[id_train]
        Y_train = Y.iloc[id_train]
        X_test = X.iloc[id_test]
        Y_test = Y.iloc[id_test]
        
        # Addestra modello
        if modello_scelto == "Regressione Lineare":
            model = LinearRegression()
            model.fit(X_train, Y_train)
            
            # Calcola previsioni
            Y_train_pred = model.predict(X_train)
            Y_test_pred = model.predict(X_test)
            
            # Calcola metriche
            train_mse = root_mean_squared_error(Y_train, Y_train_pred)
            train_mae = mean_absolute_error(Y_train, Y_train_pred)
            test_mse = root_mean_squared_error(Y_test, Y_test_pred)
            test_mae = mean_absolute_error(Y_test, Y_test_pred)
            
            # Salva nel session_state
            st.session_state.ral_model = model
            st.session_state.model_type = "Regressione Lineare"
            st.session_state.X_columns = X.columns
            st.session_state.ids = ids
            st.session_state.df_clean = df_clean
            st.session_state.X_train = X_train
            st.session_state.Y_train = Y_train
            st.session_state.model_trained = True
            
        else:  # Albero di Decisione
            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            model.fit(X_train, Y_train)
            
            # Calcola previsioni
            Y_train_pred = model.predict(X_train)
            Y_test_pred = model.predict(X_test)
            
            # Calcola metriche
            train_mse = root_mean_squared_error(Y_train, Y_train_pred)
            train_mae = mean_absolute_error(Y_train, Y_train_pred)
            test_mse = root_mean_squared_error(Y_test, Y_test_pred)
            test_mae = mean_absolute_error(Y_test, Y_test_pred)
            
            # Salva nel session_state
            st.session_state.ral_model = model
            st.session_state.model_type = "Albero di Decisione"
            st.session_state.X_columns = X.columns
            st.session_state.ids = ids
            st.session_state.df_clean = df_clean
            st.session_state.X_train = X_train
            st.session_state.Y_train = Y_train
            st.session_state.model_trained = True
            
            # Salva i parametri per visualizzazione albero
            st.session_state.max_depth = max_depth
            st.session_state.min_samples_leaf = min_samples_leaf
        
        # Salva metriche
        st.session_state.train_mse = train_mse
        st.session_state.train_mae = train_mae
        st.session_state.test_mse = test_mse
        st.session_state.test_mae = test_mae
        
        # Crea dizionario previsioni per ID_PERSONA
        predictions = model.predict(X)
        prediction_dict = dict(zip(ids, predictions))
        st.session_state.prediction_dict = prediction_dict
        
        st.success("✅ Modello addestrato con successo!")

# ── Visualizzazione risultati ─────────────────────────────────────────────
if st.session_state.get("model_trained", False):
    st.header("📊 Risultati del Modello")
    
    # Metriche
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Performance Training")
        st.metric("RMSE", f"{st.session_state.train_mse:.2f}", help='''il RMSE (root mean squared error) è la radice quadrata della media degli errori quadratici. Indica quanto le previsioni si discostano dai valori reali.''')
        st.metric("MAE", f"{st.session_state.train_mae:.2f}", help='''il MAE (mean absolute error) è la media degli errori assoluti. Indica quanto, in media, le previsioni si discostano dai valori reali.''')

    with col2:
        st.subheader("📉 Performance Test")
        st.metric("RMSE", f"{st.session_state.test_mse:.2f}", help='''il RMSE (root mean squared error) è la radice quadrata della media degli errori quadratici. Indica quanto le previsioni si discostano dai valori reali.''')
        st.metric("MAE", f"{st.session_state.test_mae:.2f}", help='''il MAE (mean absolute error) è la media degli errori assoluti. Indica quanto, in media, le previsioni si discostano dai valori reali.''')

    # Visualizzazione specifica per modello
    if st.session_state.model_type == "Regressione Lineare":
        st.subheader("📐 Coefficienti della Regressione" , help='''
        I coefficienti della regressione rappresentano l'impatto di ciascuna variabile indipendente sulla variabile dipendente (la Ral).
        Ad esempio, se le variabili di input sono sesso_M (che è 1 se la persona è maschio, 0 altrimenti) e l'età (che è un numero positivo),
        la ral stimata si calcolerà come ral = coefficiente_intercetta + coefficiente_sesso_M * sesso_M + coefficiente_età * età
        I t-test, gli standard error e i p-value sono solamente indicatori statistici che ci dicono se i coefficienti stimati sono significativamente diversi da zero.
        ''')
        
        # Calcola p-value
        model = st.session_state.ral_model
        X_train = st.session_state.X_train
        Y_train = st.session_state.Y_train
        
        # Aggiungi intercetta
        #X_with_const = np.column_stack([np.ones(len(X_train)), X_train])
        X_with_const = np.column_stack([np.ones(len(X_train)), X_train.astype(float)])

        # Calcola errori standard e p-value
        mse = st.session_state.train_mse
        var_beta = mse * np.linalg.inv(np.dot(X_with_const.T, X_with_const)).diagonal()
        std_errors = np.sqrt(var_beta)
        t_stats = np.append(model.intercept_, model.coef_) / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), len(X_train) - len(X_train.columns) - 1))
        
        # Crea tabella coefficienti
        coef_df = pd.DataFrame({
            'Variabile': ['Intercetta'] + list(st.session_state.X_columns),
            'Coefficiente': [model.intercept_] + list(model.coef_),
            'Std Error': std_errors,
            't-stat': t_stats,
            'p-value': p_values
        })
        
        st.dataframe(coef_df.round(4), use_container_width=True)
        
    else:  # Albero di Decisione
        st.subheader("🌳 Visualizzazione Albero di Decisione", help='''
Per interpretare i risultati dell'albero, si parte dalla "radice" in alto, e si scende verso il basso, lungo i rami,
per arrivare fino alle foglie. In ogni nodo dell'albero, sono presenti tre informazioni:
                     - una condizione da rispettare. Se la condizione è vera, per un individuio, si procede verso il nodo inferiore di sinistra, altrimenti verso quello di destra
                     - il numero di osservazioni nel nodo. Nella radice ci sono tutti gli individui presenti nei dati. Man mano che si scende, questo numero può solo diminuire.
                     - il valore medio della variabile da prevedere (in questo caso la ral) degli individui che appratengono a un nodo.
                     ''')
        
        # Crea grafico dell'albero con matplotlib
        fig, ax = plt.subplots(figsize=(20, 10))
        tree.plot_tree(
            st.session_state.ral_model,
            feature_names=st.session_state.X_columns,
            filled=True,
            rounded=True,
            impurity=False,
            fontsize=8,
            ax=ax
        )
        st.pyplot(fig)
    
    # Preview delle previsioni
    st.subheader("🔍 Anteprima Previsioni")
    preview_df = st.session_state.df_clean[['ID_PERSONA', 'Ral']].copy()
    preview_df['Ral_Previsione'] = preview_df['ID_PERSONA'].map(st.session_state.prediction_dict)
    preview_df['Differenza_Ral'] = preview_df['Ral'] - preview_df['Ral_Previsione']
    st.dataframe(preview_df, use_container_width=True)
    
    # Download delle previsioni
    all_predictions_df = pd.DataFrame({
        'ID_PERSONA': list(st.session_state.prediction_dict.keys()),
        'Ral_Previsione': list(st.session_state.prediction_dict.values())
    })
    
    csv = all_predictions_df.to_csv(index=False)
    st.download_button(
        label="📥 Download previsioni complete",
        data=csv,
        file_name="previsioni_ral.csv",
        mime="text/csv"
    )

else:
    st.info("👈 Configura il modello nella sidebar e clicca su 'Addestra Modello'")
