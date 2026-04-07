import streamlit as st
import plotly.graph_objects as go
from transformers import pipeline
import time

# Configuração da página
st.set_page_config(page_title="MindBalance", page_icon="🌀")

# CSS para manter o estilo vinho/rosa
st.markdown("""
    <style>
    .stApp { background-color: #831843; color: white; }
    .stButton>button { background-color: white; color: #831843; border-radius: 20px; }
    </style>
    """, unsafe_allow_html=True)

# Motor de IA (Cache para não travar o site)
@st.cache_resource
def carregar_analisador():
    try: return pipeline("sentiment-analysis", model="pysentimiento/bertweet-pt-sentiment")
    except: return None

analisador = carregar_analisador()

# Dados e Perguntas
OPCOES = ["Nunca", "Raramente", "Às vezes", "Frequentemente", "Sempre"]
PESOS = {"Nunca": 0.2, "Raramente": 0.5, "Às vezes": 1.2, "Frequentemente": 2.2, "Sempre": 3.5}

perguntas = [
    ("Sinto que minhas emoções mudam bruscamente?", "Instabilidade"),
    ("Tenho dificuldade em me concentrar?", "Instabilidade"),
    ("Sinto um cansaço que não passa?", "Exaustão"),
    ("Pequenos problemas parecem catástrofes?", "Reatividade"),
    ("Sinto que as pessoas não entendem minha dor?", "Isolamento")
]

st.title("🌀 MindBalance: Sondagem")

# Formulário
with st.form("quiz"):
    nome = st.text_input("Nome Completo")
    idade = st.number_input("Idade", min_value=0)
    serie = st.text_input("Série / Ano")
    
    respostas = []
    for q, cat in perguntas:
        respostas.append((st.radio(q, OPCOES, horizontal=True), cat))
    
    livre = st.text_area("Se pudesse tirar um peso das costas hoje, o que seria?")
    
    # Início do tempo
    if 't_inicio' not in st.session_state:
        st.session_state.t_inicio = time.time()
        
    enviado = st.form_submit_button("GERAR DIAGNÓSTICO")

if enviado and nome:
    tempo = (time.time() - st.session_state.t_inicio) / 60
    categorias = {"Instabilidade": 0, "Exaustão": 0, "Isolamento": 0, "Reatividade": 0}
    
    for resp, cat in respostas:
        categorias[cat] += PESOS[resp]
    
    st.success(f"Análise Concluída para {nome}!")
    
    # Gráfico
    fig = go.Figure(data=[go.Pie(labels=list(categorias.keys()), values=list(categorias.values()), hole=.4)])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
    st.plotly_chart(fig)
    
    st.write(f"Tempo de reflexão: {tempo:.2f} minutos.")
