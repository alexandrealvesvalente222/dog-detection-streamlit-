# Trabalho 01 - Visão Computacional e Aprendizado Profundo (VCAP)
# Detecção de Cachorros usando HOG + SVM no CIFAR-10 (com melhoria de resolução)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
from tensorflow.keras.datasets import cifar10

# 1. Carregar o dataset CIFAR-10
print("Carregando dataset CIFAR-10...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()

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