# Detecção e prevenção de colisões espaciais

**Nico Ramos**

GRR20210574
Novembro 2025

[GitHub](https://github.com/nico-ig/Collision-Detection)

---

# Dataset escolhido

## Problema

* Prever o risco final de colisão entre um satélite e outro objeto espacial (lixo ou outro satélite)
* Cada satélite envia periodicamente mensagens à base com alertas de aproximação, formando uma série temporal

* Cada mensagem contém informações como:
    * Data de colisão estimada
    * Risco estimado
    * Incertezas
    
* A base recebe uma grande quantidade de avisos, mas apenas uma parte muito pequena são de alto risco
* As manobras de evasão são planejadas pelo menos 2 dias antes da data de colisão estimada
* A decisão final é tomada 1 dia antes

---

## Desafio

* Proposto pela Agência Espacial Europeia (ESA)

* Treinar um modelo capaz de prever o risco final estimado pelo satélite
* O principal objetivo é minimizar os **Falsos Negativos**
    * Eventos de alto risco identificados como de baixo risco pelo modelo

[Desafio e Dataset](https://kelvins.esa.int/collision-avoidance-challenge)

---

## Séries Temporais

* **O que são**: 
    * Sequência de observações de uma ou mais variáveis
    * Coletadas em intervalos de tempo regulares
    * Ordenadas cronologicamente.

* **No dataset**:
    * Cada linha no dataset representa uma observação de um evento
    * Todas as linhas de um mesmo evento formas a série temporal dele
    * Essa série permite observar a mudança dos atributos ao longo do tempo

| time_to_tca           |   risk   |
|------------------------|----------:|
| 7 03:00:00    | -27.9496 |
| 7 08:00:00    | -7.4126  |
| 7 17:00:00    | -6.8041  |
| 6 02:00:00    | -3.8716  |
| 6 03:00:00    | -3.8716  |
| 6 08:00:00    | -4.3216  |
| 6 19:00:00    | -5.4366  |
| 5 01:00:00    | -9.5442  |
| 5 09:00:00    | -11.8536 |
| 5 18:00:00    | -11.2554 |
| 4 01:00:00    | -9.4447  |
| 4 07:00:00    | -25.0812 |
| 4 15:00:00    | -30.0000 |
| 3 01:00:00    | -30.0000 |
| 3 10:00:00    | -30.0000 |
| 3 16:00:00    | -30.0000 |
| 2 01:00:00    | -30.0000 |
| 2 12:00:00    | -30.0000 |
| 2 18:00:00    | -30.0000 |
| 1 24:00:00    | -30.0000 |
| 1 12:00:00    | -30.0000 |
| 1 04:00:00    | -30.0000 |
| 0 00:00:00    | -30.0000 |

---

# Objetivo (inicial)

* Projetar e treinar um modelo capaz de prever o risco estimado final
* Identificar os eventos de alto risco
* Comparar o desempenho de dois otimizadores *descida de encosta* e *Adam* aplicados ao modelos *Arima*

---

# EDA

---

# EDA

* O EDA foi o principal **desafio** enfrentado
    * Entender, ao mesmo tempo, um problema complexo novo e o/como fazer o EDA
    * Grande quantidade de colunas, por onde começar?

---
# Por onde começar?

* Selecionar algumas colunas mais relevantes para diminuir a dimensionalidade do problema
    * Buscar alguns artigos? Vídeos?
    * Primeiro se familizarizar com o dataset?

* Foram selecionadas algumas colunas mais relevantes com base em dois artigos:
    * [PACEcraft Collision Avoidance Challenge: Design and Results of a Machine Learning Competition](https://arxiv.org/pdf/2008.03069)
    * [Implementation and Comparison of Data-Based Methods for Collision Avoidance in Satellite Operations](https://conference.sdo.esoc.esa.int/proceedings/sdc8/paper/33/SDC8-paper33.pdf)

---

## Dimensões

* 103 colunas

* **Treino**:
    * 162634 linhas
    * 13154 evebntos únicos
    * Média de 12 observações por evento

* **Teste**:
    * 24484 linhas 
    * 2167 eventos únicos
    * Média de 11 observações por evento

---

## Quantidade de Observações por Evento


| Porcentagem | Qtd. Observações |
|:-----------:|:----------------:|
| min         | 1                |
| 25%         | 5                |
| 50%         | 13               |
| 75%         | 20               |
| max         | 23               |

---

## Dataset

* **Tipo dos dados**:
    * Contínuos

* **Frequência das séries**:
    * Aproximadamente a cada 8 horas

* **Valores nulos**:
    * Apenas 4 colunas com valores nulos significativos

* **Detecção de Outliers no tempo**: Método Hampel e IQR
    * window=1 não detecta nenhum valor nulo, window > 1 detecta muitos
    * O IQR também não mostrou resultados significativos

* **Valores Constantes no tempo**: Janela de tamanho 3, threshold de 1% do IQR
    * Todas as colunas possuem uma grande quantidade de valores constantes

---

### Valores Nulos

TODO<TABELA
| Coluna                      | Null Count |
|:---------------------------:|:----------:|
| **SSN**                         | **6,822**      |
| **c_sigma_ndot**                | **9,241**      |
| **c_sigma_rdot**                | **9,241**      |
| **c_crdot_t**                   | **9,241**      |
>

## Outliers

TODO<TABELA
| Coluna                    | Qtd. Outliers |
|:-------------------------:|:-------------:|
| **SSN**                   | **29**        |
| c_sedr                    | 304           |
| c_cr_area_over_mass       | 491           |
| c_cd_area_over_mass       | 500           |
| relative_speed            | 572           |
| c_sigma_rdot              | 632           |
| c_sigma_t                 | 654           |
| c_obs_used                | 742           |
| mahalanobis_distance      | 761           |
| relative_position_n       | 783           |
| miss_distance             | 823           |
| c_position_covariance_det | 856           |
| c_sigma_ndot              | 856           |
| relative_position_r       | 859           |
| c_sigma_r                 | 869           |
| c_sigma_n                 | 877           |
| risk                      | 1,075         |
| max_risk_estimate         | 1,108         |
| c_recommended_od_span     | 1,325         |
| max_risk_scaling          | 1,457         |
| c_crdot_t                 | 1,572         |
| c_time_lastob_end         | 1,783         |
| c_time_lastob_start       | 2,202         |
>

---

## Valores Constantes

| Feature                      | Qtd. Constantes |
|------------------------------|----------------:|
| **miss_distance**                |          **6,964** |
| **relative_position_n**          |          **7,379** |
| **mahalanobis_distance**         |          **8,134** |
| **relative_position_r**          |          **8,778** |
| max_risk_estimate            |         13,532 |
| relative_speed               |         17,718 |
| c_sigma_rdot                 |         31,706 |
| c_sigma_t                    |         32,086 |
| max_risk_scaling             |         38,655 |
| c_crdot_t                    |         40,961 |
| c_position_covariance_det    |         51,916 |
| c_sigma_r                    |         51,874 |
| risk                         |         57,037 |
| c_time_lastob_end            |         67,493 |
| c_sigma_ndot                 |         70,013 |
| c_time_lastob_start          |         70,350 |
| c_cr_area_over_mass          |         72,551 |
| c_obs_used                   |         75,628 |
| c_sigma_n                    |         76,022 |
| c_recommended_od_span        |         77,530 |
| c_sedr                       |         77,669 |
| c_cd_area_over_mass          |         78,316 |
| SSN                          |         85,524 |

---

## Correlações

* **Correlação com o Risco* nas obs finais**: As principais colunas são as que medem a incerteza/confiabilidade da observação
    * As colunas de incerteza na determinação da órbita do chaser possuem a maior correlação, o dobro da segunda maior
    * A incerteza na medida de distância vem em seguida, sendo o dobro das anteriores a ela

* **Correlação com o Risco nos dias**: 
    * A incerteza na determinação da órbita e na posição do chaser se tornam cada vez mais importantes
    * A importância do tamanho do objeto e do arrasto aumenta até a metade dias antes, quando começa a diminuir
    * A estimativa do risco máximo e a quantidade de observações começam como a mais importantes, mas perdem relevância

---

## Evolução da correlação com o risco

TODO<IMAGEM
| Feature                  | -7 days  | -6 days  | -5 days  | -4 days  | -3 days  | -2 days  | -1 day   | TCA     |
|:-------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:-------:|
| **c_time_lastob_start**     | **0.1764**   | **0.2548**   | **0.3311**   | **0.3732**   | **0.3904**   | **0.3968**   | **0.4174**   | **0.4395**  |
| **c_time_lastob_end**       | **0.2102**   | **0.2959**   | **0.3696**   | **0.3997**   | **0.4056**   | **0.3958**   | **0.4018**   | **0.4154**  |
| **c_position_covariance_det**| **-0.0414**  | **-0.0269**  | **0.0054**   | **0.0413**   | **0.0896**   | **0.1404**   | **0.1880**   | **0.2331**  |
| **c_cd_area_over_mass**     | **0.0756**   | **0.1022**   | **0.1106**   | **0.0979**   | **0.1013**   | **0.1062**   | **0.0890**   | **0.0543**  |
| **c_cr_area_over_mass**     | **0.0877**   | **0.1071**   | **0.1288**   | **0.1355**   | **0.1221**   | **0.1128**   | **0.0968**   | **0.0677**  |
| **max_risk_estimate**      | **0.3684**   | **0.3112**   | **0.2518**   | **0.2139**   | **0.2082**   | **0.1962**   | **0.1872**   | **0.1741**  |
>

---

## Correlação com o risco nos dois dias anteriores

TODO<IMAGEM
| Feature                       | -1 day  | -2 days |
|:-----------------------------:|:-------:|:-------:|
| **c_time_lastob_start**         |  **0.4164** |  **0.4036** |
| **c_time_lastob_end**           |  **0.4078** |  **0.3923** |
| **mahalanobis_distance**        | **-0.2593** | **-0.2825** |
| c_obs_used                  | -0.1670 | -0.1653 |
| c_cr_area_over_mass         |  0.1557 |  0.1563 |
| c_sedr                      |  0.1425 |  0.1527 |
| c_sigma_t                   |  0.1410 |  0.1220 |
| c_sigma_r                   |  0.1331 |  0.1137 |
| c_sigma_n                   |  0.1330 |  0.1136 |
| c_position_covariance_det   |  0.1330 |  0.1136 |
| c_cd_area_over_mass         |  0.1239 |  0.1247 |
| c_sigma_rdot                |  0.1180 |  0.1168 |
| c_recommended_od_span       |  0.1171 |  0.0926 |
| c_sigma_ndot                |  0.1095 |  0.1089 |
| max_risk_estimate           |  0.0999 |  0.0891 |
| max_risk_scaling           | -0.0805 | -0.0724 |
| SSN                         |  0.0498 |  0.0422 |
| c_crdot_t                   |  0.0188 |  0.0568 |
| miss_distance               |  0.0037 | -0.0324 |
| relative_position_r         |  0.0008 |  0.0188 |
| relative_speed             | -0.0315 | -0.0213 |
| relative_position_n        | -0.0150 | -0.0084 |
>

---

## Relação entre colunas

* **Multicolinearidade**: Variance Inflation Factor (VIF)
    * As colunas do volume do erro do chaser foram as únicas com alta multicolinearidade

* **Correlação com a observação anterior**: Auto Correlation Function (ACF)
    * As correlações são significativas até lag 3, quando começam a perder importância

---

## Multicolinearidade

TODO<TABLE 
| Feature                       | VIF          |
|-------------------------------|--------------|
| c_crdot_t                    | 0.0161       |
| max_risk_estimate            | 0.0254       |
| relative_speed               | 0.1402       |
| c_recommended_od_span        | 0.3911       |
| miss_distance                | 0.4256       |
| SSN                          | 0.6051       |
| c_time_lastob_end            | 0.6592       |
| c_obs_used                   | 0.7163       |
| c_time_lastob_start          | 0.7828       |
| mahalanobis_distance         | 0.8142       |
| c_cr_area_over_mass          | 0.8983       |
| c_cd_area_over_mass          | 0.9302       |
| c_sedr                       | 0.9587       |
| max_risk_scaling             | 0.9864       |
| relative_position_r          | 0.9961       |
| relative_position_n          | 1.0020       |
| **c_sigma_t**                   | **73.6501**      |
| **c_sigma_rdot**                 | **101.1216**     |
| **c_sigma_r**                    | **217687.1774**  |
| **c_sigma_n**                    | **73747676.0191**|
| **c_sigma_ndot**                 | **87927389.8727**|
| **c_position_covariance_det**    | **311512889.3774**|
>

---

## Auto Correlação com observações anteriores (lag)

TODO<IMAGEM
| Feature                      |  **lag_1**   | **lag_3**  |  lag_6  |
|------------------------------|--------|----------|-----------|
| risk                        | **0.4254** | **0.1849**  | -0.0578 |
| max_risk_scaling            | **0.3090** | **0.0711**  | -0.0689 |
| mahalanobis_distance        | **0.4245** | **0.1867**  | -0.0387 |
| c_sigma_t                   | **0.4428** | **0.2205**  | -0.0231 |
| max_risk_estimate           | **0.3690** | **0.1358**  | -0.0713 |
| c_sigma_rdot                | **0.4424** | **0.2205**  | -0.0228 |
| miss_distance               | **0.3186** | **0.1204**  | -0.0724 |
| c_position_covariance_det   | **0.3748** | **0.1346**  | -0.0628 |
| c_sigma_n                   | **0.3909** | **0.1164**  | -0.0909 |
| c_sigma_r                   | **0.4204** | **0.1652**  | -0.0616 |
| c_obs_used                  | **0.5237** | **0.1668**  | -0.0919 |
| c_sigma_ndot                | **0.4002** | **0.1286**  | -0.0802 |
| relative_position_n         | **0.3285** | **0.1268**  | -0.0707 |
| c_recommended_od_span       | **0.4697** | **0.1200**  | -0.1029 |
| relative_position_r         | **0.3463** | **0.0932**  | -0.1046 |
| c_sedr                      | **0.5479** | **0.1983**  | -0.0765 |
| SSN                         | **0.5142** | **0.1240**  | -0.1108 |
| c_crdot_t                   | **0.2949** | **0.0697**  | -0.0715 |
| relative_speed              | **0.3201** | **0.0556**  | -0.0961 |
| c_time_lastob_end           | **0.3918** |-**0.0334**  | -0.1161 |
| c_time_lastob_start         | **0.3843** |-**0.0204**  | -0.1150 |
| c_cr_area_over_mass         | **0.4894** | **0.1124**  | -0.1052 |
| c_cd_area_over_mass         | **0.5201** | **0.1843**  | -0.0656 |
> Evolução do Risco

* A maior parte dos eventos de alto risco são atualizados para baixo risco antes da última observação
* Poucos eventos oscilam de estados
* As transições diminuem conforme o último evento se aproxima
* A quantidade de eventos que passam de alto para baixo nível é muito maior do que o inverso

TODO<IMAGE
| Categoria                   | 8 dias | 7 dias | 6 dias | 5 dias | 4 dias | 3 dias | 2 dias | 1 dia |
|-----------------------------|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:-----:|
| Low → High → Low            |   8    |   2    |   4    |   0    |   2    |   0    |   0    |   0   |
| High → Low → High           |  21    |  17    |  11    |   6    |   1    |   0    |   0    |   0   |
| Low → High                  | 249    | 254    | 268    | 248    | 199    | 143    |  70    |   0   |
| High → Low                  | 122    | 112    | 105    |  70    |  52    |  14    |  10    |   0   |
| Total High                  | 859    | 884    | 758    | 616    | 441    | 303    | 171    |  94   |
| Total Low                   | 6234   | 7885   | 7948   | 8347   | 8433   | 8836   | 9081   | 8876  |
>

---

## Divisão

* Conjunto de treino foi dividido em 80% treino e 20% validação
* Foi mantida a mesma proporção de eventos de alto risco em cada um

* Os eventos de alto risco são raros, então foram selecionados para super representar a quantidade real

* Todos os eventos no conjunto de *testes* tem:
    * A última observação (avaliação) a menos de um dia do TCA (recente)
    * Todas as outras a pelo menos 2 dias do TCA (tempo para planejar e executar a manobra)

* Os eventos no conjunto de testes não são filtrados

| TCA       | Última observação | Todas antes da última | 
|:---------:|:-----------------:|:---------------------:|
|  Teste    | < 1 dia           | > 2 dias              |
|  Treino   | Qualquer data     | Qualquer data         |


Conjunto    | Alto Risco | Baixo/Médio Risco  | Total de Eventos |
|:---------:|:----------------:|:------------:|:----------------:|
| Treino (Original)  | 2.77%     | 97.23%        | 13154         |
| Validação | 2.89%     | 97.11%        | 2630          |
| Treino    | 2.75%     | 97.25%        | 10524         |
| Teste     | 8.21%     | 91.79%        | 2167          |


---

## Dataset Final

| Coluna                     | Mean        | Std         | Min         | 25%        | 50%        | 75%         | Max         |
|----------------------------|-------------|-------------|-------------|------------|------------|-------------|-------------|
| c_cd_area_over_mass        | 0.7843      | 2.3417      | -128.1786   | 0.1774     | 0.4387     | 0.6928      | 147.9127    |
| c_cr_area_over_mass        | 0.3465      | 0.9655      | -0.7121     | 0.0518     | 0.1813     | 0.3052      | 59.1550     |
| c_obs_used                 | 59.1585     | 84.9738     | 3.0000      | 21.0000    | 30.0000    | 57.0000     | 2227.0000   |
| c_position_covariance_det  | 1.10e+45    | 8.53e+45    | -8.18e+18   | 2.36e+11   | 4.09e+13   | 6.06e+15    | 6.73e+58    |
| c_recommended_od_span      | 12.6636     | 9.9371      | 0.0000      | 6.5900     | 11.5200    | 16.4700     | 234.4100    |
| c_sedr                     | 0.0030      | 0.0150      | -0.1073     | 0.0003     | 0.0007     | 0.0015      | 0.8762      |
| c_time_lastob_end          | 0.5897      | 0.8232      | 0.0000      | 0.0000     | 0.0000     | 1.0000      | 2.0000      |
| c_time_lastob_start        | 40.1508     | 73.8095     | 1.0000      | 1.0000     | 1.0000     | 2.0000      | 180.0000    |
| mahalanobis_distance       | 192.6028    | 433.6808    | 0.0000      | 22.4056    | 71.1696    | 198.4767    | 15427.1608  |
| max_risk_scaling           | 5.37e+4     | 9.09e+5     | 0.0000      | 8.3239     | 31.7449    | 304.7437    | 4.98e+7     |
| risk                       | -19.3406    | 10.0116     | -30.0000    | -30.0000   | -17.8706   | -9.1733     | -1.4429     |

---

## Conclusão do EDA

* Redução de 50% das colunas exógenas originais
    * 11 Removidas de 21

* Dataset final com 12 colunas:
    * 10 Colunas exógenas,
    * 1 Identificador da série,
    * 1 Identificador do tempo

* **O que as colunas quantificam?**
    * A incerteza/confiança da órbita do chaser e a resistência dele ao movimento em alta e baixa órbita

* Os eventos tendem a ter uma parte estável 
* São fortemente correlacionados com eventos próximos, mas as observações anteriores perdem importância rapidamente
* Uma vez que um evento mudou de estado, ele tende a permanecer nele, sem oscilar

---

# Pré-processamento

* Filtragem das colunas
* Ordenar e transformar o tempo em data
* Ressample para as séries terem o mesmo tamanho
* Normalização
* Ressample para as observações de um evento terem a mesma frequência

---

# Metodologia

---

## Treinar um modelo linear de séries temporais


* Arima, Arimax e Varmax


### Problemas

* **Arima**:
    * Só aceita uma variável (univariado)

* **Arimax**:
    * As séries são muito curtas e o modelo não converge
    * Para prever o futuro, preciso saber o valor das outras variáveis no futuro

* **Varmax**:
    * Também não convergiu

* Não generalizam para outras séries

---


## Treinar um Modelo de Séries Multivariadas e Multiséries


* Não foi encontrada uma forma de combinar as séries multivariadas em um modelo global multi séries

* Os algoritmos encontrados ou são para séries multivariadas, ou para múltiplas séries independentes, e não para as duas 

---

# Objetivo ~~(inicial)~~ (novo)

* ~~Prever o valor final do risco estimado na última observação antes da colisão~~
* Classificar os eventos em alto/baixo nível no último evento

---

# Treino

* **Ressample das séries**:
    * Padronizar o tamanho para o tamanho da maior série
    * Padronizar a frequência entre eventos de uma mesma série

* **Normalização com variância média**:
    * Ignorar a diferênça de amplitude e escala entre as colunas
    * Comparar a forma ao invés do valor

* **Clusterização com KMeans**:
    * **Métrica**: Dtw, permite comparar séries desalinhadas
    * **Máximo de iterações**: 100
    * **Número de clusters**: 3

* **Classificação**:
    * **Alto Risco**: 5% dos riscos na última observação são maiores que o threshold
    * **Baixo Risco**: Todos os outros
    * *Eventos de alto risco são raros*

---

# Validação e Teste

* Previsão e classificação feitas com a penúltima observação
* O resultado foi comparado com a última observação daquele evento

## Validação

* Utilizado para ajustar o threshold de classificação

## Teste

* Avaliado com o threshold escolhido na *validação*

---

# Resultados

TODO<TABLE
| Cluster | Mean Risk | 75% Risk |
|---------|-----------|-----------|
| **0**   | -27.3440  | -28.5429  |
| **1**   | -9.7131   | -6.8949   |
| **2**   | -20.7393  | -11.4110  |
>

## Validação

* O Cluster 0 foi o único classificado como baixo risco

TODO<TABLE
| Real \ Previsto | High | Low/Medium |
|-----------------|------|------------|
| **High**        |  71  |     4      |
| **Low/Medium**  | 2117 |    401     |
>

## Teste

TODO<TABLE
| Real \ Previsto | High | Low/Medium |
|-----------------|------|------------|
| **High**        |  172  |     1      |
| **Low/Medium**  | 1663 |    256     |
>





----------------
---

# Conclusões

* O modelo tenta prever o risco com base em quão **incerta** as medidas são
* Modelos de séries temporais lineares (Arima, Arimax, Varmax):
    * **Não são adequados** para identificar e **generalizar padrões entre séries distintas**
    * Os que **aceitam mais de uma feature**, dependem de conseguir **estimar as outras variáveis no futuro** para fazer previsões
    * Não são adequados para simular eventos de outras séries curtas
    * Não funcionam com séries tão curtas como as do dataset
    * Mas parecem ser eficientes e confiáveis para séries com observações suficientes
* O resample das séries pode perder informações e características importantes, afetando a qualidade das previsões
* Os clusters não classificaram bem os eventos, ou o modelo classifica todo mundo como de algo risco, ou todo mundo como de baixo risco

---

# Trabalhos Futuros

* Treinar para outros tamanhos de clusters

* Combinar a classificação do risco com a predição do valor final
    * Identificação de padrões gerais similares (classificação)
    * Predição do valor final, auxiliado por eventos similares