import gradio as gr
import plotly.graph_objects as go
from transformers import pipeline
import torch
import time

# --- 1. MOTOR DE IA ---
device = 0 if torch.cuda.is_available() else -1
try:
    analisador = pipeline("sentiment-analysis", 
                          model="pysentimiento/bertweet-pt-sentiment", 
                          device=device)
except:
    analisador = None

# --- 2. CONFIGURAÇÃO PSICOMÉTRICA ---
OPCOES = ["Nunca", "Raramente", "Às vezes", "Frequentemente", "Sempre"]
PESOS_OPCOES = {"Nunca": 0.2, "Raramente": 0.5, "Às vezes": 1.2, "Frequentemente": 2.2, "Sempre": 3.5}

perguntas_objetivas = [
    {"q": "Sinto que minhas emoções mudam bruscamente sem um motivo real?", "cat": "Instabilidade"},
    {"q": "Tenho dificuldade em me concentrar porque meus pensamentos me atropelam?", "cat": "Desequilíbrio Mental"},
    {"q": "Sinto um cansaço que não passa, mesmo após dormir?", "cat": "Exaustão"},
    {"q": "Pequenos problemas do dia a dia parecem catástrofes insuportáveis?", "cat": "Reatividade"},
    {"q": "Sinto que as pessoas ao meu redor não entendem minha dor?", "cat": "Isolamento"},
    {"q": "Tenho episódios de choro ou raiva que não consigo conter?", "cat": "Instabilidade"},
    {"q": "Sinto que perdi o interesse por coisas que antes me davam alegria?", "cat": "Exaustão"},
    {"q": "Meu corpo manifesta tensões, como dores ou aperto no peito?", "cat": "Desequilíbrio Físico"},
    {"q": "Evito interações sociais porque não tenho energia para fingir estar bem?", "cat": "Isolamento"},
    {"q": "Sinto que estou falhando em áreas importantes da minha vida?", "cat": "Crise"},
    {"q": "O futuro me parece uma parede intransponível de preocupações?", "cat": "Crise"},
    {"q": "Sinto que não tenho mais controle sobre o rumo da minha vida?", "cat": "Desequilíbrio Mental"},
    {"q": "Minha paciência com os outros está no limite absoluto?", "cat": "Reatividade"},
    {"q": "Sinto uma vontade profunda de me afastar de tudo e de todos?", "cat": "Isolamento"},
    {"q": "Tenho a sensação de que algo ruim está prestes a acontecer?", "cat": "Instabilidade"}
]

pergunta_vortice = "Se você pudesse tirar das suas costas o peso que mais te faz sofrer hoje, o que você jogaria fora e como se sentiria depois disso?"

# --- 3. LÓGICA DE PROCESSAMENTO ---

def validar_pg1(nome, idade, serie):
    if not nome.strip() or not serie.strip() or idade <= 0:
        return gr.update(visible=True, value="### ⚠️ Preencha todos os dados corretamente."), gr.update(visible=True), gr.update(visible=False)
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

def processar_resultado(nome, idade, serie, tempo_in, resposta_livre, *respostas_obj):
    tempo_total = (time.time() - tempo_in) / 60
    categorias = {"Instabilidade": 0, "Exaustão": 0, "Isolamento": 0, "Reatividade": 0, "Crise": 0}
    
    for i, resp in enumerate(respostas_obj):
        cat = perguntas_objetivas[i]["cat"]
        ponto = PESOS_OPCOES.get(resp, 0)
        if cat in ["Desequilíbrio Mental", "Desequilíbrio Físico"]:
            categorias["Instabilidade"] += ponto
        else:
            categorias[cat] += ponto

    impacto_ia = 0
    if analisador and resposta_livre:
        try:
            res_ia = analisador(resposta_livre[:512])[0]
            if res_ia['label'] == 'NEG': impacto_ia = 5.0
        except: pass

    cat_dominante = max(categorias, key=categorias.get)
    
    if tempo_total <= 2.5:
        status_tempo = "um processamento rápido e objetivo, indicando uma percepção imediata do seu estado atual"
    elif tempo_total <= 5:
        status_tempo = "um tempo de resposta preocupante, que sugere uma carga de dúvida ou dificuldade em lidar com os sentimentos acessados"
    else:
        status_tempo = "um sinal de alerta importante, onde a demora excessiva pode indicar resistência emocional ou um grande cansaço mental"

    if idade < 20: status_vulnerabilidade = f"considerando a sua fase de desenvolvimento ({serie}), as pressões do dia a dia podem estar aumentando a sua {cat_dominante.lower()}"
    elif idade < 40: status_vulnerabilidade = f"aos {int(idade)} anos, as cobranças da vida adulta explicam por que a {cat_dominante.lower()} apareceu com tanto destaque"
    else: status_vulnerabilidade = f"nesta etapa da vida, os resultados mostram que o desequilíbrio em {cat_dominante.lower()} pede uma atenção maior com sua rotina"

    analise_final = f"""
    ### Análise Psicométrica: {nome}
    
    A partir dos dados coletados, nota-se que {status_vulnerabilidade}. O seu desabafo revelou uma carga emocional de {'alta densidade' if impacto_ia > 0 else 'estabilidade moderada'}, o que influencia diretamente o seu nível de bem-estar geral. 
    
    Sobre o seu comportamento durante este teste, o tempo de {tempo_total:.1f} minutos é considerado {status_tempo}. Este fator é um quesito de avaliação central, pois revela o esforço necessário para traduzir o que se sente em palavras. Em resumo, os dados indicam que focar no cuidado da sua {cat_dominante.lower()} é o caminho mais importante agora para você recuperar seu equilíbrio emocional.

    ---
    ### 📖 Entenda os resultados do gráfico:
    *   **Instabilidade:** Oscilações de humor ou sinais físicos de ansiedade (como aperto no peito e pensamentos acelerados).
    *   **Exaustão:** Esgotamento profundo onde o descanso comum não parece ser suficiente para recuperar as energias.
    *   **Isolamento:** Tendência de se afastar socialmente para evitar o esforço de interagir ou "fingir estar bem".
    *   **Reatividade:** Quando a paciência está no limite e pequenos problemas geram reações emocionais muito fortes.
    *   **Crise:** Sensação de perda de controle sobre o rumo da vida ou preocupação excessiva com o futuro.
    """

    fig = go.Figure(data=[go.Pie(
        labels=list(categorias.keys()),
        values=list(categorias.values()),
        hole=.4,
        marker_colors=['#4c0519', '#831843', '#be185d', '#db2777', '#f472b6']
    )])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=350, showlegend=True)

    return analise_final, fig, gr.update(visible=True)

# --- 4. INTERFACE ---
css = """
.gradio-container { background-color: #831843 !important; }
.rose-card { background: rgba(255,255,255,0.08) !important; border: 1px solid white !important; border-radius: 15px; padding: 25px; margin-bottom: 20px; }
span, label, p, h1, h2, h3 { color: white !important; font-family: 'Inter', sans-serif; }
textarea, input { background-color: #fff1f2 !important; color: #831843 !important; }
.btn-white { background-color: white !important; color: #831843 !important; font-weight: bold !important; border-radius: 50px !important; }

input[type='radio']:checked {
    background-color: #2563eb !important;
    border-color: #ffffff !important;
}
:root {
    --radio-dot-color: #2563eb !important;
}
"""

with gr.Blocks(css=css) as demo:
    t_start = gr.State(0.0)
    gr.Markdown("# 🌀 MindBalance: Sondagem de Instabilidade")

    with gr.Column(visible=True) as pg1:
        with gr.Column(elem_classes="rose-card"):
            gr.Markdown("### 👤 1. Seus Dados")
            in_nome = gr.Textbox(label="Nome Completo")
            with gr.Row():
                in_idade = gr.Number(label="Idade", value=0)
                in_serie = gr.Textbox(label="Série / Ano")
            btn_pg2 = gr.Button("INICIAR", elem_classes="btn-white")
            err_msg = gr.Markdown(visible=False)

    with gr.Column(visible=False) as pg2:
        with gr.Column(elem_classes="rose-card"):
            gr.Markdown("### 🧠 2. Como você tem se sentido?")
            list_obj = [gr.Radio(choices=OPCOES, label=item["q"]) for item in perguntas_objetivas]
            
            gr.Markdown("---")
            in_livre = gr.TextArea(label=pergunta_vortice, placeholder="Escreva aqui...")
            btn_fim = gr.Button("VER MEU RESULTADO", elem_classes="btn-white")

        with gr.Column(visible=False, elem_classes="rose-card") as res_area:
            out_txt = gr.Markdown()
            out_plot = gr.Plot()

    btn_pg2.click(validar_pg1, [in_nome, in_idade, in_serie], [err_msg, pg1, pg2]).then(
        lambda: time.time(), None, t_start
    )

    btn_fim.click(
        processar_resultado,
        [in_nome, in_idade, in_serie, t_start, in_livre] + list_obj,
        [out_txt, out_plot, res_area]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
