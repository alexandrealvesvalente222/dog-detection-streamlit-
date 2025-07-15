# Trabalho 01 - VCAP: Detecção de Cachorros com HOG + SVM

Este projeto implementa um pipeline clássico de visão computacional para detectar cachorros em imagens do dataset CIFAR-10, utilizando descritores HOG e um classificador SVM.

## Estrutura do Projeto

```
Trabalhp/
├── apresentacao/
│   └── apresentacao_trabalho01.html
├── src/
│   ├── run_dog_detection_simple.py
├── resultados/
│   ├── exemplo_acerto1.png
│   ├── exemplo_acerto2.png
│   ├── exemplo_acerto3.png
│   ├── exemplo_erro1.png
│   ├── exemplo_erro2.png
│   └── exemplo_erro3.png
├── README.md
└── requirements.txt
```

## Como Executar

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute o pipeline:
   ```bash
   python src/run_dog_detection_simple.py
   ```
3. Veja os resultados e exemplos de acertos/erros na pasta `resultados/`.
4. Apresente o roteiro em `apresentacao/apresentacao_trabalho01.html`.

## Descrição do Pipeline
- **Pré-processamento:** Aumento da resolução das imagens para 96x96 pixels.
- **Extração de descritores:** HOG (Histogram of Oriented Gradients).
- **Classificação:** SVM Linear (LinearSVC).
- **Avaliação:** Métricas de acurácia, precisão, revocação e f1-score.

## Créditos
- Integrantes do grupo:
