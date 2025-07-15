import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
import pickle
import os
import urllib.request
from skimage.io import imsave
from joblib import Parallel, delayed
import time
import pandas as pd
from fpdf import FPDF
import datetime
from sklearn.model_selection import StratifiedShuffleSplit

DOG_IMAGE_URL = "https://static.vecteezy.com/ti/vetor-gratis/p1/6720668-cara-de-cachorro-logo-gratis-vetor.jpg"  # Imagem livre de direitos
DOG_BG_COLOR = "#003366"
DOG_ACCENT_COLOR = "#bfa77a"
DOG_TITLE_COLOR = "#7a5c2e"
DOG_EMOJI = "🐶"

# Funções utilitárias
@st.cache_data
def download_cifar10():
    if not os.path.exists('cifar-10-batches-py'):
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        urllib.request.urlretrieve(url, "cifar-10-python.tar.gz")
        import tarfile
        with tarfile.open("cifar-10-python.tar.gz", "r:gz") as tar:
            tar.extractall()
        os.remove("cifar-10-python.tar.gz")

@st.cache_data
def load_cifar10():
    download_cifar10()
    x_train, y_train = [], []
    for i in range(1, 6):
        with open(f'cifar-10-batches-py/data_batch_{i}', 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            x_train.extend(batch[b'data'])
            y_train.extend(batch[b'labels'])
    with open('cifar-10-batches-py/test_batch', 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        x_test = batch[b'data']
        y_test = batch[b'labels']
    x_train = np.array(x_train).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_test = np.array(x_test).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return (x_train, y_train), (x_test, y_test)

def preparar_dados(x, y, max_por_classe=None):
    x_cachorro = x[y == 5]
    x_outros = x[y != 5]
    if max_por_classe is not None:
        x_cachorro = x_cachorro[:max_por_classe]
        x_outros = x_outros[:max_por_classe]
    y_cachorro = np.ones(len(x_cachorro))
    y_outros = np.zeros(len(x_outros))
    x_total = np.concatenate([x_cachorro, x_outros])
    y_total = np.concatenate([y_cachorro, y_outros])
    return x_total, y_total

def extrair_hog_features(images, upscale_size=(96, 96)):
    loading_html = """
    <style>
    .custom-loader {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      width: 100%;
      margin-top: 20px;
    }
    .bar-container {
      width: 60%;
      background: #e0eaff;
      border-radius: 18px;
      box-shadow: 0 2px 12px #00336622;
      padding: 24px 0 18px 0;
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    .animated-bar {
      width: 90%;
      height: 22px;
      border-radius: 11px;
      background: linear-gradient(270deg, #b3d1ff, #003366, #b3d1ff);
      background-size: 400% 400%;
      animation: gradientMove 2s linear infinite;
      margin-bottom: 12px;
      position: relative;
      overflow: hidden;
    }
    @keyframes gradientMove {
      0% {background-position: 0% 50%;}
      100% {background-position: 100% 50%;}
    }
    .loader-text {
      text-align: center;
      font-size: 1.15em;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #003366;
      font-weight: 600;
    }
    .loader-text span {
      font-size: 1.5em;
      margin-right: 8px;
    }
    </style>
    <div class='custom-loader'>
      <div class='bar-container'>
        <div class='animated-bar'></div>
        <div class='loader-text'><span>🐶</span>Extraindo HOG de todas as imagens...</div>
      </div>
    </div>
    """
    loading_placeholder = st.empty()
    loading_placeholder.markdown(loading_html, unsafe_allow_html=True)
    def process_img(img):
        gray = rgb2gray(img)
        gray_up = resize(gray, upscale_size, anti_aliasing=True)
        return hog(gray_up, pixels_per_cell=(6, 6), cells_per_block=(3, 3), feature_vector=True)
    features = Parallel(n_jobs=-1, prefer="threads")(delayed(process_img)(img) for img in images)
    loading_placeholder.empty()
    return np.array(features)

def pipeline(descritor, classificador, upscale_size, n_examples, c_svm, max_por_classe):
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    x_total, y_total = preparar_dados(np.concatenate([x_train, x_test]),
                                      np.concatenate([y_train, y_test]),
                                      max_por_classe=max_por_classe)
    if descritor == 'HOG':
        x_features = extrair_hog_features(x_total, upscale_size)
    else:
        st.error('Descritor não implementado!')
        return None
    scaler = StandardScaler()
    x_features = scaler.fit_transform(x_features)
    # Split estratificado para garantir ambas as classes no teste
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(x_features, y_total):
        X_train, X_test = x_features[train_idx], x_features[test_idx]
        y_train_bin, y_test_bin = y_total[train_idx], y_total[test_idx]
        imgs_train, imgs_test = x_total[train_idx], x_total[test_idx]
    if classificador == 'SVM Linear':
        clf = SVC(kernel='linear', max_iter=50000, probability=True, random_state=42, verbose=True, C=c_svm)
    else:
        st.error('Classificador não implementado!')
        return None
    clf.fit(X_train, y_train_bin)
    y_pred = clf.predict(X_test)
    return clf, imgs_test, y_test_bin, y_pred, classification_report(y_test_bin, y_pred, target_names=["Não-cachorro", "Cachorro"], output_dict=True), n_examples

def mostrar_resultados_separados(imgs, y_true, y_pred, n=3):
    # Classes: 1 = cachorro, 0 = não-cachorro
    acertos_cachorro = np.where((y_true == 1) & (y_pred == 1))[0]
    acertos_nao_cachorro = np.where((y_true == 0) & (y_pred == 0))[0]
    falsos_positivos = np.where((y_true == 0) & (y_pred == 1))[0]  # não-cachorro previsto como cachorro
    falsos_negativos = np.where((y_true == 1) & (y_pred == 0))[0]  # cachorro previsto como não-cachorro

    st.markdown(f"## Exemplos de acertos - Cachorro (Verdadeiro=1, Previsto=1)")
    cols = st.columns(n)
    for idx, i in enumerate(acertos_cachorro[:n]):
        with cols[idx]:
            st.image(resize(imgs[i], (128, 128)), caption=f"Verdadeiro: 1 | Previsto: 1", use_container_width=True)

    st.markdown(f"## Exemplos de acertos - Não-cachorro (Verdadeiro=0, Previsto=0)")
    cols = st.columns(n)
    for idx, i in enumerate(acertos_nao_cachorro[:n]):
        with cols[idx]:
            st.image(resize(imgs[i], (128, 128)), caption=f"Verdadeiro: 0 | Previsto: 0", use_container_width=True)

    st.markdown(f"## Falsos positivos - Não-cachorro previsto como cachorro (Verdadeiro=0, Previsto=1)")
    cols = st.columns(n)
    for idx, i in enumerate(falsos_positivos[:n]):
        with cols[idx]:
            st.image(resize(imgs[i], (128, 128)), caption=f"Verdadeiro: 0 | Previsto: 1", use_container_width=True)

    st.markdown(f"## Falsos negativos - Cachorro previsto como não-cachorro (Verdadeiro=1, Previsto=0)")
    cols = st.columns(n)
    for idx, i in enumerate(falsos_negativos[:n]):
        with cols[idx]:
            st.image(resize(imgs[i], (128, 128)), caption=f"Verdadeiro: 1 | Previsto: 0", use_container_width=True)

    return acertos_cachorro[:n], acertos_nao_cachorro[:n], falsos_positivos[:n], falsos_negativos[:n]

def exportar_exemplos(imgs, y_true, y_pred, idxs, idxs2, n=3, pasta='resultados', tipo='acertos_cachorro'):
    pasta_tipo = os.path.join(pasta, tipo)
    os.makedirs(pasta_tipo, exist_ok=True)
    registros = []
    for j, i in enumerate(idxs):
        img = (resize(imgs[i], (128, 128)) * 255).astype(np.uint8)
        nome = f"{tipo}{j+1}_V{int(y_true[i])}_P{int(y_pred[i])}.png"
        imsave(os.path.join(pasta_tipo, nome), img)
        registros.append({"arquivo": nome, "verdadeiro": int(y_true[i]), "previsto": int(y_pred[i])})
    for j, i in enumerate(idxs2):
        img = (resize(imgs[i], (128, 128)) * 255).astype(np.uint8)
        nome = f"{tipo}_erro{j+1}_V{int(y_true[i])}_P{int(y_pred[i])}.png"
        imsave(os.path.join(pasta_tipo, nome), img)
        registros.append({"arquivo": nome, "verdadeiro": int(y_true[i]), "previsto": int(y_pred[i])})
    # Salvar CSV com os resultados
    df = pd.DataFrame(registros)
    df.to_csv(os.path.join(pasta_tipo, f"{tipo}_resultados.csv"), index=False)

def exportar_relatorio_pdf(parametros, metrics, exemplos_imgs, exemplos_legendas, pasta='resultados', nome_pdf='relatorio.pdf'):
    os.makedirs(pasta, exist_ok=True)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Relatório de Detecção de Cachorros - VCAP', ln=True, align='C')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Data: {datetime.datetime.now().strftime("%d/%m/%Y %H:%M")}', ln=True)
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Parâmetros do Pipeline:', ln=True)
    pdf.set_font('Arial', '', 12)
    for k, v in parametros.items():
        pdf.cell(0, 8, f'- {k}: {v}', ln=True)
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Métricas de Avaliação:', ln=True)
    pdf.set_font('Arial', '', 12)
    for classe in ['0', '1']:
        if classe in metrics:
            m = metrics[classe]
            pdf.cell(0, 8, f"Classe {classe} - Precisão: {m['precision']:.2f} | Recall: {m['recall']:.2f} | F1: {m['f1-score']:.2f}", ln=True)
    pdf.cell(0, 8, f"Acurácia: {metrics['accuracy']:.2f}", ln=True)
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Exemplos:', ln=True)
    pdf.set_font('Arial', '', 12)
    for img_path, legenda in zip(exemplos_imgs, exemplos_legendas):
        try:
            pdf.image(img_path, w=40)
        except:
            pass
        pdf.multi_cell(0, 8, legenda)
        pdf.ln(2)
    pdf.output(os.path.join(pasta, nome_pdf))

# --- APP STREAMLIT ---
st.set_page_config(page_title=f"{DOG_EMOJI} VCAP - Detecção de Cachorros", layout="wide", page_icon=DOG_EMOJI)
st.markdown(f"""
    <div style='background:{DOG_BG_COLOR};padding:18px 0 8px 0;border-radius:12px;margin-bottom:18px;text-align:center;'>
        <img src='{DOG_IMAGE_URL}' alt='Cachorro' style='max-width:120px;border-radius:50%;border:4px solid {DOG_ACCENT_COLOR};box-shadow:0 2px 8px #0002;margin-bottom:8px;'>
        <div class="title-bar">
            <h1>🐶 Trabalho 01 - VCAP<br>Detecção de Cachorros com Pipeline Clássico</h1>
        </div>
    </div>
""", unsafe_allow_html=True)
st.markdown("""
Este aplicativo permite parametrizar e rodar um pipeline clássico de visão computacional para detecção de cachorros no CIFAR-10.
""")

with st.sidebar:
    st.header("Parâmetros do Pipeline")
    descritor = st.selectbox("Descritor", ["HOG"], help="Por enquanto, apenas HOG está implementado.")
    classificador = st.selectbox("Classificador", ["SVM Linear"], help="Por enquanto, apenas SVM Linear está implementado.")
    upscale_size = st.slider("Tamanho para aumento de resolução", 32, 128, 96, step=16)
    n_examples = st.slider("Quantidade de exemplos de acertos/erros", 1, 6, 3)
    c_svm = st.slider("C (Regularização do SVM)", 0.01, 10.0, 1.0, step=0.01)
    max_por_classe = st.slider("Máximo de exemplos por classe (para acelerar)", 50, 5000, 200, step=50)
    rodar = st.button("Rodar Pipeline")

if rodar:
    with st.spinner("Executando pipeline, aguarde alguns instantes..."):
        result = pipeline(descritor, classificador, (upscale_size, upscale_size), n_examples, c_svm, max_por_classe)
    if result:
        clf, X_test_imgs, y_test_bin, y_pred, metrics, n_examples = result
        st.subheader("Métricas de Avaliação")
        st.write("Acurácia:", f"{metrics['accuracy']:.2f}")
        # Mostrar apenas as classes 0 e 1 na tabela
        tabela_metricas = {k: {m: f"{v[m]:.2f}" for m in ['precision','recall','f1-score']} for k,v in metrics.items() if k in ['0','1']}
        st.table(tabela_metricas)
        acertos_cachorro, acertos_nao_cachorro, falsos_positivos, falsos_negativos = mostrar_resultados_separados(X_test_imgs, y_test_bin, y_pred, n=n_examples)
        st.subheader("Relatório Completo (JSON)")
        import json
        st.json(metrics, expanded=True)
        # Calcular métricas principais em porcentagem
        def minval(val):
            return val if val > 0 else 0.1
        prec0 = minval(metrics['0']['precision']*100) if '0' in metrics else 0.1
        rec0 = minval(metrics['0']['recall']*100) if '0' in metrics else 0.1
        f10 = minval(metrics['0']['f1-score']*100) if '0' in metrics else 0.1
        prec1 = minval(metrics['1']['precision']*100) if '1' in metrics else 0.1
        rec1 = minval(metrics['1']['recall']*100) if '1' in metrics else 0.1
        f11 = minval(metrics['1']['f1-score']*100) if '1' in metrics else 0.1
        acc = minval(metrics['accuracy']*100) if 'accuracy' in metrics else 0.1
        macro = metrics['macro avg'] if 'macro avg' in metrics else {'precision':0,'recall':0,'f1-score':0,'support':0}
        weighted = metrics['weighted avg'] if 'weighted avg' in metrics else {'precision':0,'recall':0,'f1-score':0,'support':0}
        support0 = metrics['0']['support'] if '0' in metrics else 0
        support1 = metrics['1']['support'] if '1' in metrics else 0
        # Bloco de métricas principais em cards
        st.markdown(f"""
        <div style='display:flex;flex-wrap:wrap;gap:18px;justify-content:center;margin:18px 0 8px 0;'>
          <div style='background:#e3f2fd;border-radius:10px;padding:12px 22px;min-width:170px;box-shadow:0 2px 8px #00336611;border:2px solid #90caf9;'>
            <div style='font-size:1.13em;font-weight:600;color:#607d8b;margin-bottom:2px;'>🐾 Não-cachorro</div>
            <div style='color:#333;font-size:1.04em;'>Precisão: <b>{prec0:.1f}%</b></div>
            <div style='color:#333;font-size:1.04em;'>Recall: <b>{rec0:.1f}%</b></div>
            <div style='color:#333;font-size:1.04em;'>F1: <b>{f10:.1f}%</b></div>
            <div style='color:#333;font-size:0.98em;'>Support: <b>{support0}</b></div>
          </div>
          <div style='background:#fff3e0;border-radius:10px;padding:12px 22px;min-width:170px;box-shadow:0 2px 8px #00336611;border:2px solid #ffb74d;'>
            <div style='font-size:1.13em;font-weight:600;color:#795548;margin-bottom:2px;'>🐶 Cachorro</div>
            <div style='color:#333;font-size:1.04em;'>Precisão: <b>{prec1:.1f}%</b></div>
            <div style='color:#333;font-size:1.04em;'>Recall: <b>{rec1:.1f}%</b></div>
            <div style='color:#333;font-size:1.04em;'>F1: <b>{f11:.1f}%</b></div>
            <div style='color:#333;font-size:0.98em;'>Support: <b>{support1}</b></div>
          </div>
          <div style='background:#f1f8e9;border-radius:10px;padding:12px 22px;min-width:170px;box-shadow:0 2px 8px #00336611;border:2px solid #aed581;display:flex;flex-direction:column;justify-content:center;'>
            <div style='font-size:1.13em;font-weight:600;color:#33691e;margin-bottom:2px;'>🎯 Acurácia</div>
            <div style='color:#333;font-size:1.18em;font-weight:700;'>{acc:.1f}%</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        # Bloco de médias macro/weighted
        st.markdown(f"""
        <div style='display:flex;flex-wrap:wrap;gap:18px;justify-content:center;margin-bottom:8px;'>
          <div style='background:#f3e5f5;border-radius:10px;padding:10px 18px;min-width:170px;box-shadow:0 2px 8px #00336611;border:2px solid #ce93d8;'>
            <div style='font-size:1.08em;font-weight:600;color:#6a1b9a;margin-bottom:2px;'>Macro avg</div>
            <div style='color:#333;font-size:0.99em;'>Precisão: <b>{(macro['precision']*100):.1f}%</b></div>
            <div style='color:#333;font-size:0.99em;'>Recall: <b>{(macro['recall']*100):.1f}%</b></div>
            <div style='color:#333;font-size:0.99em;'>F1: <b>{(macro['f1-score']*100):.1f}%</b></div>
            <div style='color:#333;font-size:0.95em;'>Support: <b>{macro['support']}</b></div>
          </div>
          <div style='background:#e8f5e9;border-radius:10px;padding:10px 18px;min-width:170px;box-shadow:0 2px 8px #00336611;border:2px solid #81c784;'>
            <div style='font-size:1.08em;font-weight:600;color:#388e3c;margin-bottom:2px;'>Weighted avg</div>
            <div style='color:#333;font-size:0.99em;'>Precisão: <b>{(weighted['precision']*100):.1f}%</b></div>
            <div style='color:#333;font-size:0.99em;'>Recall: <b>{(weighted['recall']*100):.1f}%</b></div>
            <div style='color:#333;font-size:0.99em;'>F1: <b>{(weighted['f1-score']*100):.1f}%</b></div>
            <div style='color:#333;font-size:0.95em;'>Support: <b>{weighted['support']}</b></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        # Legenda explicativa compacta
        st.markdown(f"""
        <div style='background:#eaf6fb;border:2px solid #90caf9;border-radius:10px;padding:8px 14px 6px 14px;margin-top:8px;box-shadow:0 2px 8px #00336611;max-width:600px;margin-left:auto;margin-right:auto;'>
        <ul style='font-size:0.98em;line-height:1.35;margin-bottom:0;'>
        <li><b>precision</b>: <span style='color:black;'>Proporção de positivos previstos que são realmente positivos</span></li>
        <li><b>recall</b>: <span style='color:black;'>Proporção de positivos reais que foram corretamente identificados</span></li>
        <li><b>f1-score</b>: <span style='color:black;'>Média harmônica entre precisão e recall</span></li>
        <li><b>support</b>: <span style='color:black;'>Número de amostras de cada classe</span></li>
        <li><b>macro avg</b>: <span style='color:black;'>Média das métricas entre as classes (sem ponderação)</span></li>
        <li><b>weighted avg</b>: <span style='color:black;'>Média das métricas ponderada pelo número de amostras de cada classe</span></li>
        </ul>
        <div style='margin-top:6px;font-size:0.97em;color:black;text-align:justify;padding-left:6px;padding-right:2px;'>
        <b>Resumo:</b> O pipeline extrai descritores HOG das imagens do CIFAR-10, treina um SVM para distinguir cachorros de não-cachorros e avalia o desempenho nos dados de teste, apresentando métricas detalhadas e exemplos visuais dos resultados.
        </div>
        </div>
        """, unsafe_allow_html=True)
        # Remover botões de exportação
        # if st.button("Exportar exemplos para relatório (pasta resultados/)"):
        #     exportar_exemplos(X_test_imgs, y_test_bin, y_pred, acertos_cachorro, falsos_positivos, n=n_examples, pasta='resultados', tipo='acertos_cachorro')
        #     exportar_exemplos(X_test_imgs, y_test_bin, y_pred, acertos_nao_cachorro, falsos_negativos, n=n_examples, pasta='resultados', tipo='acertos_nao_cachorro')
        #     st.success(f"Exemplos exportados para a pasta 'resultados/'!")
        # if st.button("Exportar relatório detalhado em PDF (pasta resultados/)"):
        #     exportar_relatorio_pdf(parametros, metrics, exemplos_imgs, exemplos_legendas, pasta='resultados', nome_pdf='relatorio.pdf')
        #     st.success("Relatório PDF exportado para a pasta 'resultados/'!")
    else:
        st.error("Erro ao rodar pipeline.")
else:
    st.info("Configure os parâmetros e clique em 'Rodar Pipeline'.")

# Rodapé fixo com direitos reservados
def rodape_grupo():
    st.markdown(
        """
        <style>
        .footer-grupo {
            position: fixed;
            left: 0; right: 0; bottom: 0;
            background: #f7f7f7;
            color: #7a5c2e;
            text-align: center;
            font-size: 1.05em;
            padding: 10px 0 6px 0;
            border-top: 1.5px solid #e0d6c3;
            z-index: 100;
        }
        </style>
        <div class='footer-grupo'>
            <b>Todos os direitos reservados ao Grupo: Alexandre Valente | Arialan Gomes | Jardel Terci Flores</b>
        </div>
        """,
        unsafe_allow_html=True
    )

rodape_grupo() 