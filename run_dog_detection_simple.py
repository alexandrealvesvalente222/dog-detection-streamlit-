# Trabalho 01 - Visão Computacional e Aprendizado Profundo (VCAP)
# Detecção de Cachorros usando HOG + SVM (versão simplificada)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
import urllib.request
import pickle
import os

# Função para baixar e carregar o dataset CIFAR-10
def download_cifar10():
    """Baixa o dataset CIFAR-10 se não existir"""
    if not os.path.exists('cifar-10-batches-py'):
        print("Baixando dataset CIFAR-10...")
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        urllib.request.urlretrieve(url, "cifar-10-python.tar.gz")
        
        import tarfile
        with tarfile.open("cifar-10-python.tar.gz", "r:gz") as tar:
            tar.extractall()
        
        # Remove o arquivo tar.gz
        os.remove("cifar-10-python.tar.gz")
        print("Dataset baixado com sucesso!")

def load_cifar10():
    """Carrega o dataset CIFAR-10"""
    download_cifar10()
    
    # Carregar dados de treino
    x_train = []
    y_train = []
    
    for i in range(1, 6):
        with open(f'cifar-10-batches-py/data_batch_{i}', 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            x_train.extend(batch[b'data'])
            y_train.extend(batch[b'labels'])
    
    # Carregar dados de teste
    with open('cifar-10-batches-py/test_batch', 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        x_test = batch[b'data']
        y_test = batch[b'labels']
    
    # Converter para numpy arrays e reshape
    x_train = np.array(x_train).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_test = np.array(x_test).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    return (x_train, y_train), (x_test, y_test)

# 1. Carregar o dataset CIFAR-10
print("Carregando dataset CIFAR-10...")
(x_train, y_train), (x_test, y_test) = load_cifar10()

# 2. Filtrar apenas "cachorro" (label 5) vs. outras classes
# 1 = cachorro, 0 = não-cachorro

def preparar_dados(x, y):
    x_cachorro = x[y == 5]
    x_outros = x[y != 5]
    y_cachorro = np.ones(len(x_cachorro))
    y_outros = np.zeros(len(x_outros))

    # Balancear: usar mesma quantidade de negativos e positivos
    x_outros = x_outros[:len(x_cachorro)]
    y_outros = y_outros[:len(x_cachorro)]

    x_total = np.concatenate([x_cachorro, x_outros])
    y_total = np.concatenate([y_cachorro, y_outros])

    return x_total, y_total

print("Preparando dados...")
x_total, y_total = preparar_dados(np.concatenate([x_train, x_test]),
                                   np.concatenate([y_train, y_test]))

# 3. Extrair HOG features (aplicando upscale para 96x96)
def extrair_hog_features(images):
    features = []
    for i, img in enumerate(images):
        if i % 1000 == 0:
            print(f"Processando imagem {i}/{len(images)}")
        gray = rgb2gray(img)
        gray_up = resize(gray, (96, 96), anti_aliasing=True)
        hog_feat = hog(gray_up, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), feature_vector=True)
        features.append(hog_feat)
    return np.array(features)

print("Extraindo HOG features com upscale para 96x96...")
x_features = extrair_hog_features(x_total)

# 4. Treinar e testar classificador SVM
print("Treinando classificador SVM...")
X_train, X_test, y_train_bin, y_test_bin = train_test_split(
    x_features, y_total, test_size=0.2, random_state=42)

clf = LinearSVC()
clf.fit(X_train, y_train_bin)
y_pred = clf.predict(X_test)

# 5. Avaliação do desempenho
print("\nRelatório de classificação:")
print(classification_report(y_test_bin, y_pred, target_names=["Não-cachorro", "Cachorro"]))

# 6. Visualizar exemplos de acertos e erros com imagens ampliadas
X_test_imgs, _, y_test_imgs, _ = train_test_split(x_total, y_total, test_size=0.2, random_state=42)

def redimensionar_imagem(img, tamanho=(128, 128)):
    return resize(img, tamanho, anti_aliasing=True)

def mostrar_resultados(imgs, y_true, y_pred, n=3):
    acertos = np.where(y_true == y_pred)[0]
    erros = np.where(y_true != y_pred)[0]

    print(f"\nExemplos de acertos ({len(acertos)} total):")
    for i in acertos[:n]:
        plt.figure(figsize=(6, 6))
        plt.imshow(redimensionar_imagem(imgs[i]))
        plt.title(f"Verdadeiro: {int(y_true[i])} | Previsto: {int(y_pred[i])}")
        plt.axis('off')
        plt.show()

    print(f"\nExemplos de erros ({len(erros)} total):")
    for i in erros[:n]:
        plt.figure(figsize=(6, 6))
        plt.imshow(redimensionar_imagem(imgs[i]))
        plt.title(f"Verdadeiro: {int(y_true[i])} | Previsto: {int(y_pred[i])}")
        plt.axis('off')
        plt.show()

print("Mostrando resultados...")
mostrar_resultados(X_test_imgs, y_test_bin, y_pred)

print("\nCódigo executado com sucesso!") 