# 📊 Análise dos resultados — Detecção de alimentos com YOLO11s

**Entrega 01 — Inteligência Artificial e Aprendizado de Máquina**

---

## 1. Configuração do experimento

O modelo utilizado foi o **YOLO11s** (small), treinado por **70 épocas** com batch size de 16 e resolução de entrada de 640x640. O dataset foi dividido em **90% treino / 10% validação**, totalizando aproximadamente 370 instâncias anotadas distribuídas de forma balanceada entre as três classes:

| Classe | Instâncias de treino |
|--------|----------------------|
| arroz  | 116                  |
| cafe   | 128                  |
| feijao | 126                  |

O dataset, no momento, é relativamente **pequeno** para um problema de detecção de objetos, o que impõe limitações na capacidade de generalização do modelo, especialmente em condições de iluminação e ângulo variados. Porém, atualizações futuras já estão sendo implementadas para aumentar o dataset.

<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/labels.jpg" width="300"><br>
  <i>Distribuição de instâncias e posições das bounding boxes no dataset</i>
</p>

---

## 2. Amostras do dataset de treino

As imagens abaixo mostram amostras dos primeiros batches de treinamento, com as bounding boxes anotadas sobrepostas. É possível observar a diversidade de ângulos, distâncias e condições de iluminação das fotos coletadas.

<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/train_batch0.jpg" width="300">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/train_batch1.jpg" width="300">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/train_batch2.jpg" width="300"><br>
  <i>Batch 0 — Batch 1 — Batch 2</i>
</p>

---

## 3. Curvas de treinamento

As curvas abaixo mostram a evolução das losses e métricas ao longo das 70 épocas de treinamento.

<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/results.png" width="600"><br>
  <i>Curvas de loss (treino e validação) e métricas por época</i>
</p>

### Observações

- **Épocas 1–15**: comportamento instável, com picos expressivos nas losses de validação (`val/cls_loss` chegando a ~44 e `val/dfl_loss` a ~10). Isso é esperado nas fases iniciais com dataset pequeno, quando o modelo ainda não convergiu.
- **Épocas 15–40**: estabilização progressiva. As losses passam a cair de forma consistente e o mAP começa a subir de maneira mais contínua.
- **Épocas 40–70**: convergência clara. As losses de treino chegam próximas de 0.73 (box), 0.54 (cls) e 1.21 (dfl). O mAP@0.5 e mAP@0.5:0.95 se estabilizam nas últimas épocas, indicando que o modelo atingiu seu limite com os dados disponíveis.

---

## 4. Métricas Finais

Valores registrados na **época 70** (melhor checkpoint salvo em `best.pt`):

| Métrica          | Valor  | Interpretação                                      |
|------------------|--------|----------------------------------------------------|
| Precisão         | 0.954  | Quando detecta, raramente erra a classe            |
| Recall           | 0.675  | Ainda perde ~32% dos objetos presentes nas imagens |
| **mAP@0.5**      | **0.739** | Resultado satisfatório para o tamanho do dataset |
| mAP@0.5:0.95     | 0.460  | Localização geométrica das boxes ainda limitada    |

A alta precisão com recall moderado indica que o modelo é **conservador**: prefere não detectar a detectar errado. Isso é desejável em aplicações onde falsos positivos são problemáticos, mas pode ser inadequado quando é importante não perder nenhum produto.

---

## 5. Desempenho por classe

### 5.1 Curva precisão × recall

<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/BoxPR_curve.png" width="300"><br>
  <i>Curva Precisão-Recall por classe</i>
</p>

| Classe | AP@0.5 | Interpretação                                              |
|--------|--------|------------------------------------------------------------|
| feijao | 0.861  | Melhor desempenho — embalagem visualmente mais distinta    |
| cafe   | 0.693  | Desempenho intermediário                                   |
| arroz  | 0.682  | Maior dificuldade — embalagem com menor contraste visual   |
| **média** | **0.745** | mAP geral da detecção                               |

### 5.2 Curva F1 × confiança

<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/BoxF1_curve.png" width="300"><br>
  <i>Curva F1 por classe em função do limiar de confiança</i>
</p>

O F1 máximo geral é **0.80**, atingido com limiar de confiança de **0.439**. Esse é o limiar recomendado para uso na inferência em tempo real, pois equilibra precisão e recall de forma ótima.

O feijão se destaca com F1 máximo de ~0.92, enquanto arroz (~0.77) e café (~0.74) ficam mais próximos.

### 5.3 Curva precisão × confiança

<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/BoxP_curve.png" width="300"><br>
  <i>Curva de precisão por classe em função do limiar de confiança</i>
</p>

Acima de confiança **0.63**, todas as classes atingem precisão próxima de 1.0 — ou seja, com limiar mais restritivo, as detecções são altamente confiáveis, porém à custa de recall menor.

### 5.4 Curva recall × confiança

<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/BoxR_curve.png" width="300"><br>
  <i>Curva de recall por classe em função do limiar de confiança</i>
</p>

Com limiar próximo de zero, o recall máximo geral é **0.81**. O feijão mantém recall acima de 0.85 em uma faixa ampla de confiança (até ~0.85), enquanto arroz e café caem mais rapidamente.

---

## 6. Matriz de confusão

### 6.1 Valores absolutos

<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/confusion_matrix.png" width="450"><br>
  <i>Matriz de confusão — valores absolutos no conjunto de validação</i>
</p>

### 6.2 Normalizada

<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/confusion_matrix_normalized.png" width="450"><br>
  <i>Matriz de confusão normalizada</i>
</p>

### Análise

Dois padrões chamam atenção na matriz normalizada:

**Falsos negativos (classificados como background):**
- 33% das instâncias de arroz foram perdidas pelo modelo
- 35% das instâncias de café não foram detectadas
- 14% das instâncias de feijão não foram detectadas

**Falsos positivos originados do background:**
- 42% das predições de arroz vieram de regiões de fundo
- 50% das predições de café vieram de regiões de fundo
- Apenas 8% das predições de feijão vieram de fundo

Esses números confirmam que o **feijão é a classe mais bem aprendida**, enquanto arroz e café ainda sofrem tanto com detecções perdidas quanto com alarmes falsos. As embalagens do arroz (Namorado, azul/branco) e do café (Melitta, verde/preto) compartilham paletas de cores que podem se confundir com o fundo em certas condições.

---

## 7. Análise visual das predições

### 7.1 Ground truth vs. Predição no conjunto de validação

<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/val_batch0_labels.jpg" width="300"><br>
  <i>Ground truth — anotações reais do conjunto de validação</i>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/val_batch0_pred.jpg" width="300"><br>
  <i>Predições do modelo no conjunto de validação (com scores de confiança)</i>
</p>

### 7.2 Últimos batches de treino

As imagens abaixo mostram as detecções nos últimos batches de treino, evidenciando que o modelo aprendeu a identificar múltiplos produtos simultaneamente em cenas complexas.

<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/train_batch660.jpg" width="300"><br>
  <i>Batch 660 — detecções nas épocas finais de treino</i>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/train_batch661.jpg" width="300"><br>
  <i>Batch 661</i>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/train_batch662.jpg" width="300"><br>
  <i>Batch 662</i>
</p>

### 7.3 Casos problemáticos identificados

Ao analisar visualmente as predições, foram identificados os seguintes padrões de falha:

- **Embalagens fotografadas de costas**: sem logomarca visível, o modelo perde a detecção ou detecta com baixa confiança (~0.3)
- **Iluminação muito baixa**: imagens com fundo escuro comprometem principalmente arroz e café
- **Múltiplas embalagens sobrepostas**: bounding boxes se sobrepõem e confiança cai
- **Produtos parcialmente recortados**: bordas da imagem cortando a embalagem reduzem a confiança da detecção

---

## 8. Conclusão

O modelo YOLO11s treinado atingiu um **mAP@0.5 de 0.739**, resultado satisfatório considerando o dataset de tamanho reduzido (~200 imagens de treino). A classe com melhor desempenho foi o **feijão** (AP 0.861), seguida por **café** (0.693) e **arroz** (0.682).

A alta precisão final (0.954) indica que o modelo é confiável quando realiza uma detecção. O recall mais moderado (0.675) aponta que ainda há espaço para melhora na capacidade de encontrar todos os objetos presentes em uma cena.

### Viões para próxima entrega (11/05/2026)

| Recomendação | Justificativa |
|---|---|
| Ampliar dataset para 300–500 imagens por classe | Reduzir instabilidade e melhorar generalização |
| Aumentar conjunto de validação (atual: ~20 imagens) | Métricas de validação mais estáveis e representativas |
| Usar limiar de confiança entre 0.40 e 0.45 na inferência | Ponto de F1 máximo identificado nas curvas |
| Incluir imagens com iluminação adversa e ângulos extremos | Reduzir falsos negativos em condições reais |
| Explorar data augmentation com variação de brilho e rotação | Melhorar robustez sem necessidade de mais coleta de dados |
| Testar modelo `yolo11m` com mais dados | Avaliar ganho de precisão com maior capacidade de representação |

---
