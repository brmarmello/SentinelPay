# SentinelPay

SentinelPay e uma aplicacao completa de Machine Learning para detectar risco de fraude em transacoes de cartao de credito, com pipeline modular, dashboard Streamlit, ajuste de limiar de decisao e explicabilidade com SHAP.

Este projeto foi reconstruido do zero a partir da ideia do desafio original, com foco em organizacao, didatica e deploy funcional.

## O Problema

O dataset possui 284.807 transacoes e apenas 492 fraudes, ou seja, cerca de 0,17% da base. Em cenarios assim, acuracia isolada quase sempre engana: um modelo que classifica tudo como transacao legitima parece bom, mas falha exatamente onde importa.

Por isso o projeto acompanha metricas mais adequadas para fraude:

- `recall` da classe fraude: quantas fraudes reais foram encontradas.
- `precision` da classe fraude: quantos alertas de fraude estavam corretos.
- `average precision`: qualidade geral do ranqueamento em base desbalanceada.
- matriz de confusao: leitura direta de falsos positivos e falsos negativos.

## Dataset

Fonte: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Na aplicacao e no script de treino, o CSV e carregado pela URL publica mantida pelo TensorFlow:

`https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`

As colunas `V1` a `V28` ja sao componentes PCA anonimizados. As colunas abertas sao:

- `Time`: segundos desde a primeira transacao da base.
- `Amount`: valor da transacao.
- `Class`: alvo, onde `1` indica fraude e `0` indica transacao legitima.

## Arquitetura

```text
.
├── app.py                         # Dashboard Streamlit
├── scripts/train.py               # Treino via terminal e salvamento de artefato
├── src/sentinelpay/
│   ├── config.py                  # Constantes e configuracao do modelo
│   ├── data.py                    # Carregamento e resumo dos dados
│   ├── features.py                # Feature engineering, split e escala
│   ├── model.py                   # Treino e predicao
│   └── evaluation.py              # Metricas e curvas
├── tests/                         # Testes pequenos e rapidos
├── notebooks/                     # Notebook Colab
├── requirements.txt
├── pyproject.toml
└── runtime.txt                    # Python para Streamlit Cloud
```

## Como Rodar Localmente

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

No Linux, macOS ou Google Colab, a ativacao do ambiente muda, mas os demais comandos seguem iguais.

## Treinar Pelo Terminal

```bash
python scripts/train.py
```

O script baixa os dados, treina o XGBoost, imprime metricas e salva um artefato em `artifacts/sentinelpay_model.joblib`.

Parametros uteis:

```bash
python scripts/train.py --threshold 0.20
python scripts/train.py --no-smote
```

## Rodar Testes

```bash
pytest
```

## Google Colab

Abra o notebook em `notebooks/sentinelpay_colab.ipynb`. Ele segue a mesma ordem pedagogica do projeto:

1. instalar dependencias;
2. carregar dados;
3. treinar o modelo;
4. avaliar resultados;
5. interpretar metricas.

Colab e util para exploracao e treino sem depender da maquina local. Para deploy, o caminho mais simples continua sendo Streamlit Cloud.

## Deploy no Streamlit Cloud

1. Suba este projeto para um repositorio GitHub.
2. Acesse [share.streamlit.io](https://share.streamlit.io/).
3. Escolha o repositorio.
4. Configure:
   - Branch: `main`
   - Main file path: `app.py`
   - Python: definido por `runtime.txt`
5. Clique em deploy.

O `requirements.txt` instala o pacote local com `-e .`, entao os modulos dentro de `src/` funcionam no ambiente de deploy.

## Decisoes Tecnicas

- `Amount` foi transformado em `Amount_log` para reduzir assimetria.
- O split e estratificado para preservar a proporcao rara de fraudes.
- `SMOTE` e aplicado somente no treino, evitando vazamento para teste.
- `XGBoost` foi usado por ser forte em dados tabulares.
- O limiar de decisao fica ajustavel porque fraude e um problema de trade-off: reduzir o limiar aumenta recall, mas tende a aumentar falsos positivos.
