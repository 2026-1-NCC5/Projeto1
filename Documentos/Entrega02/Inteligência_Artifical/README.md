# 📊 Análise dos resultados — Detecção de alimentos com YOLO11s

**Entrega 02 — Inteligência Artificial e Aprendizado de Máquina**

---

## 1. Configuração do experimento

O modelo utilizado foi o **YOLO11s** (small), treinado por **60 épocas** com batch size de 16 e resolução de entrada de 640x640. O dataset foi dividido em **80% treino / 20% validação**, totalizando **454 instâncias anotadas** distribuídas entre as cinco classes:

| Classe   | Instâncias totais |
|----------|-------------------|
| arroz    | 106               |
| acucar   | 53                |
| cafe     | 134               |
| feijao   | 123               |
| macarrao | 38                |

Em relação à entrega anterior (3 classes, ~370 instâncias), este experimento expande o escopo para **5 classes** e **454 instâncias**, incorporando açúcar e macarrão ao pipeline.

<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/labels2.jpg" width="300"><br>
  <i>Distribuição de instâncias e posições das bounding boxes no dataset</i>
</p>

---

## 2. Amostras do dataset de treino

As imagens abaixo mostram amostras dos primeiros batches de treinamento, com as bounding boxes anotadas sobrepostas. É possível observar a diversidade de ângulos, distâncias, condições de iluminação e a presença simultânea de múltiplas classes por imagem.

<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/train_batch02.jpg" width="300">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/train_batch12.jpg" width="300">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/train_batch22.jpg" width="300"><br>
  <i>Batch 0 — Batch 1 — Batch 2</i>
</p>

---

## 3. Curvas de treinamento

As curvas abaixo mostram a evolução das losses e métricas ao longo das 60 épocas de treinamento.

<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/results2.png" width="600"><br>
  <i>Curvas de loss (treino e validação) e métricas por época</i>
</p>

### Observações

- **Épocas 1–15**: convergência mais rápida que no experimento anterior. A `val/cls_loss` parte de ~3.67 na época 1 e cai para ~1.89 já na época 10, indicando que o modelo absorveu os padrões visuais mais distintos rapidamente.
- **Épocas 15–40**: estabilização progressiva. O mAP@0.5 sobe de ~0.63 na época 20 para ~0.85 na época 40, sugerindo que a maior parte do aprendizado útil ocorreu nessa faixa.
- **Épocas 40–60**: plateau de convergência. O mAP@0.5 oscila entre 0.848 e 0.854, e as losses de treino convergem para 0.678 (box), 0.488 (cls) e 1.204 (dfl). O modelo atingiu seu limite com os dados disponíveis antes das 60 épocas, indicando que épocas adicionais provavelmente não trariam ganhos significativos.

---

## 4. Métricas Finais

Valores do **melhor checkpoint** (`best.pt`), registrado na época 40:

| Métrica          | Valor  | Interpretação                                                   |
|------------------|--------|-----------------------------------------------------------------|
| Precisão         | 0.889  | Quando detecta, erra a classe em apenas ~11% dos casos          |
| Recall           | 0.818  | Encontra ~82% dos objetos presentes nas imagens                 |
| **mAP@0.5**      | **0.854** | Melhora expressiva em relação à entrega anterior (0.739)     |
| mAP@0.5:0.95     | 0.601  | Localização geométrica das bounding boxes substancialmente melhorada |

A comparação direta com a entrega anterior evidencia o avanço obtido com a expansão do dataset e a adição de novas classes:

| Métrica      | Entrega 01 | Entrega 02 | Variação  |
|--------------|------------|------------|-----------|
| Precisão     | 0.954      | 0.889      | −0.065    |
| Recall       | 0.675      | 0.818      | **+0.143**|
| mAP@0.5      | 0.739      | 0.854      | **+0.115**|
| mAP@0.5:0.95 | 0.460      | 0.601      | **+0.141**|

A leve queda de precisão é esperada e saudável: o modelo anterior era excessivamente conservador (detectava pouco, mas raramente errava). Agora, com recall significativamente maior, o modelo encontra mais objetos nas cenas, um trade-off favorável para o caso de uso de contagem de produtos.

---

## 5. Desempenho por classe

### 5.1 Curva precisão × recall

<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/BoxPR_curve2.png" width="300"><br>
  <i>Curva Precisão-Recall por classe</i>
</p>

| Classe   | mAP@0.5 | Interpretação                                                          |
|----------|--------|------------------------------------------------------------------------|
| acucar   | 0.979  | Melhor desempenho: embalagem da União com identidade visual muito distinta (vermelho/branco) |
| macarrao | 0.931  | Segundo melhor, apesar de ter apenas 38 instâncias: embalagem visualmente característica |
| arroz    | 0.863  | Bom desempenho: melhora expressiva em relação à entrega anterior (0.682) |
| cafe     | 0.759  | Desempenho intermediário: caixa Melitta verde ainda gera confusão em alguns ângulos |
| feijao   | 0.680  | Maior dificuldade: inversão de papel em relação à entrega anterior, onde era o melhor |
| **média**| **0.842** | mAP geral da detecção                                              |

O resultado do feijão merece atenção: foi a classe com melhor mAP na entrega anterior (0.861) e agora apresenta o pior desempenho (0.680). Isso provavelmente reflete a maior dificuldade de separação quando o modelo precisa distinguir feijão de outras embalagens semelhantes em cenas com múltiplas classes simultaneamente.

### 5.2 Curva F1 × confiança

<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/BoxF1_curve2.png" width="300"><br>
  <i>Curva F1 por classe em função do limiar de confiança</i>
</p>

O F1 máximo geral é **0.84**, atingido com limiar de confiança de **0.346**. Esse é o limiar recomendado para uso na inferência em tempo real, pois equilibra precisão e recall de forma ótima — ligeiramente abaixo do 0.439 da entrega anterior, refletindo que o modelo agora compensa levemente mais para o lado do recall.

Açúcar e macarrão se destacam com F1 máximos de ~0.95 e ~0.92 respectivamente, enquanto café (~0.80) e feijão (~0.75) ficam mais próximos da média.

### 5.3 Curva precisão × confiança

<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/BoxP_curve2.png" width="300"><br>
  <i>Curva de precisão por classe em função do limiar de confiança</i>
</p>

Acima de confiança **0.90**, todas as classes atingem precisão próxima de 1.0. Para uso em produção com tolerância zero a falsos positivos, esse limiar garante detecções altamente confiáveis,

### 5.4 Curva recall × confiança

<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/BoxR_curve2.png" width="300"><br>
  <i>Curva de recall por classe em função do limiar de confiança</i>
</p>

Com limiar próximo de zero, o recall máximo geral é **0.90**. Açúcar e macarrão mantêm recall acima de 0.90 em uma faixa ampla de confiança (até ~0.80), enquanto café e feijão caem mais rapidamente, indicando maior incerteza do modelo nessas classes.

---

## 6. Matriz de confusão

### 6.1 Valores absolutos

<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/confusion_matrix2.png" width="450"><br>
  <i>Matriz de confusão: valores absolutos no conjunto de validação</i>
</p>

### Análise

A diagonal principal é consistentemente dominante, confirmando que o modelo aprende as cinco classes de forma distinta. Os padrões de erro mais relevantes são:

**Falsos negativos (classificados como background):**
- 4 instâncias de arroz não detectadas (~16% do total de arroz na validação)
- 1 instância de açúcar não detectada
- 9 instâncias de café perdidas (~29% do café na validação), maior taxa de falsos negativos
- 12 instâncias de feijão perdidas (~33% do feijão na validação)
- 2 instâncias de macarrão perdidas

**Confusões entre classes:**
- 1 instância de café classificada como feijão
- 7 instâncias do background preditas como feijão — maior fonte de falsos positivos
- 3 instâncias do background preditas como café

Dois padrões chamam atenção: feijão concentra tanto os maiores falsos negativos (12 perdas) quanto os maiores falsos positivos vindos do background (7 detecções indevidas). Café também apresenta alta taxa de não detecção (9 perdas), reforçando que essas duas classes são as mais desafiadoras do conjunto.

Açúcar e macarrão, por outro lado, apresentam matrizes praticamente limpas, com erros mínimos, o que é notável especialmente para o macarrão dado seu menor volume de exemplos no dataset.

---

## 7. Análise visual das predições

### 7.1 Ground truth vs. Predição no conjunto de validação

<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/val_batch0_labels2.jpg" width="300">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/val_batch0_pred2.jpg" width="300"><br>
  <i>Ground truth (esquerda) e predições do modelo (direita): batch 0</i>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/val_batch1_labels2.jpg" width="300">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/val_batch1_pred2.jpg" width="300"><br>
  <i>Ground truth (esquerda) e predições do modelo (direita): batch 1</i>
</p>

### 7.2 Inferência em tempo real com estimativa de peso e valor
 
Além da validação estática, o modelo foi testado em tempo real utilizando a câmera do notebook. A cada frame, o sistema identifica os produtos presentes na cena e consulta uma tabela De-Para para estimar o peso total (kg) e o valor de mercado (R$) dos itens detectados — sem necessidade de balança ou sensor adicional.
 
<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/Teste_Local.png" width="600"><br>
  <i>Demonstração em tempo real: detecção de macarrão (75%), feijão (87%) e café (73%), com peso e valor acumulados exibidos no canto superior esquerdo</i>
</p>

Na cena acima, o modelo detectou três produtos simultaneamente: Dona Benta Penne (macarrão), Camil Feijão Carioca e Melitta Extraforte (café), com o sistema retornando automaticamente **2,0 kg** e **R$ 31,00** como estimativas agregadas. O pipeline roda a **5,26 FPS** na câmera integrada do notebook, viável para uso em demonstrações e protótipos.
 
A tabela De-Para utilizada associa cada classe a um peso e preço fixos de referência:
 
| Classe   | Peso (kg) | Valor (R$) |
|----------|-----------|------------|
| arroz    | 5,0       | 25,00      |
| feijao   | 1,0       | 8,50       |
| cafe     | 0,5       | 18,00      |
| macarrao | 0,5       | 4,50       |
| acucar   | 2,0       | 6,90       |

### 7.3 Últimos batches de treino

<p align="center">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/train_batch7502.jpg" width="300">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/train_batch7512.jpg" width="300">
  <img src="https://raw.githubusercontent.com/2026-1-NCC5/Projeto1/refs/heads/main/Imagens/train_batch7522.jpg" width="300"><br>
  <i>Batch 750 — Batch 751 — Batch 752</i>
</p>

Nas épocas finais, o modelo demonstra capacidade de detectar múltiplos produtos simultaneamente em cenas complexas, com bounding boxes precisas mesmo em imagens com sobreposição de embalagens e iluminação variada. Detecções com confiança entre 0.8 e 1.0 são frequentes para açúcar, macarrão e arroz, enquanto café e feijão ocasionalmente aparecem com escores mais baixos (~0.3–0.5) em ângulos desfavoráveis.

### 7.4 Casos problemáticos identificados

Ao analisar visualmente as predições, os padrões de falha identificados na entrega anterior persistem e novos foram observados:

- **Embalagens fotografadas de costas**: sem logomarca visível, o modelo mantém dificuldade de detecção, especialmente para café e feijão
- **Múltiplas embalagens sobrepostas**: bounding boxes se sobrepõem e confiança cai, especialmente em cenas com 3 ou mais produtos
- **Macarrão com confiança baixa em poses extremas**: apesar do bom mAP, embalagens muito anguladas ou parcialmente fora do frame geram detecções com ~0.4 de confiança
- **Açúcar em escala reduzida**: embalagens de açúcar em segundo plano ou parcialmente visíveis são por vezes ignoradas

---

## 8. Conclusão

O modelo YOLO11s treinado nesta entrega atingiu um **mAP@0.5 de 0.854**, representando uma melhora de **+11,5 pontos percentuais** em relação à entrega anterior (0.739), com o dobro de classes detectadas. O recall subiu de 0.675 para **0.818**, indicando que o modelo agora encontra a grande maioria dos produtos presentes nas cenas, característica essencial para o caso de uso de contagem automática.

As classes com melhor desempenho foram **açúcar** (AP 0.979) e **macarrão** (AP 0.931), ambas com identidade visual clara e embalagens altamente características. As classes mais desafiadoras permanecem **café** (AP 0.759) e **feijão** (AP 0.680), que concentram a maioria dos falsos negativos e positivos do modelo.

A convergência rápida demonstra que o modelo absorveu o máximo de informação disponível no dataset atual. Ganhos adicionais dependerão essencialmente de mais dados, especialmente para as classes com mais erros

---
