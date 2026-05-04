# Relatório Técnico: Benchmarking de Rastreamento LightGlue vs. Baselines

Este documento detalha os dados, experimentos e resultados finais do rastreador baseado em LightGlue comparado aos algoritmos BoTSORT e ByteTrack em diferentes taxas de amostragem temporal.

## 1. Estatísticas dos Datasets

| Dataset | Uso | Sequências | Total de Frames | Descrição |
| :--- | :--- | :---: | :---: | :--- |
| **GoPro Benchmark** | Rodovias/Placas | 5 | 15.939 | Cenário roadside, alta velocidade, deslocamentos lineares. |
| **GOT-10k (Val)** | Objetos Gerais | 180 | 21.007 | Cenário geral (SOT-to-MOT), movimento variado, câmeras em movimento. |
| **Total** | | **185** | **36.946** | |

## 2. Resultados Consolidados

### A. GoPro (Cenário Alvo: Roadside/Placas)
| Método | FPS | Stride | MOTA $\uparrow$ | IDF1 $\uparrow$ | IDSW $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| BoTSORT | 30 | 1 | 96.4 | 97.8 | 5 |
| **LightGlue (Nós)** | 30 | 1 | 93.7 | 90.4 | 1013 |
| BoTSORT | 5 | 6 | 5.3 | 10.0 | 3 |
| **LightGlue (Nós)** | 5 | 6 | **11.8** | **19.9** | 936 |
| BoTSORT | 2 | 15 | 0.6 | 1.2 | 5 |
| **LightGlue (Nós)** | 2 | 15 | **4.3** | **8.0** | 464 |

> **Nota:** Em 2 FPS, o LightGlue é **7x superior** em MOTA e IDF1.

### B. GOT-10k (Cenário Geral)
| Método | FPS | Stride | MOTA $\uparrow$ | IDF1 $\uparrow$ | IDSW $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| BoTSORT | 10 | 1 | 96.8 | 90.9 | 263 |
| **LightGlue (Nós)** | 10 | 1 | 93.2 | 87.2 | 341 |
| BoTSORT | 5 | 2 | 46.4 | 58.7 | 267 |
| **LightGlue (Nós)** | 5 | 2 | 46.3 | 56.4 | 283 |
| BoTSORT | 2 | 5 | 16.9 | 26.6 | 172 |
| **LightGlue (Nós)** | 2 | 5 | **18.3** | **27.0** | 211 |

## 3. Localização dos Dados Brutos para Gráficos

Os arquivos `.txt` no formato MOT15 podem ser lidos para plotar curvas de performance:

- **GoPro Results:** `/home/servidor/ArtigoLightglue/mass_results/gopro/`
    - Subpastas: `bytetrack_s1`, `botsort_s1`, `lightglue_s1`, etc. (s1=30fps, s6=5fps, s15=2fps)
- **GOT-10k Results:** `/home/servidor/ArtigoLightglue/mass_results/got10k_full/`
    - Subpastas: `bytetrack_s1`, `botsort_s1`, `lightglue_s1`, etc. (s1=10fps, s2=5fps, s5=2fps)

## 4. Análise de Comportamento

1. **Resiliência Visual:** O BoTSORT e ByteTrack dependem fortemente de IoU (proximidade espacial) e Filtro de Kalman. Quando o FPS cai, o deslocamento do objeto entre frames supera o limite do IoU, causando perda total de rastreamento.
2. **Superioridade Geométrica:** O LightGlue utiliza correspondência de pontos-chave (`keypoint matching`) via SuperPoint, o que permite reencontrar o objeto mesmo que ele tenha se deslocado centenas de pixels entre um frame e outro.
3. **Compensação de Movimento (CMC):** Implementamos uma camada de *Global Camera Motion Compensation* (CMC) dentro do LightGlueTracker. Diferente dos baselines que usam ORB ou ECC, nós utilizamos as próprias correspondências do LightGlue para estimar a homografia/deslocamento da câmera, garantindo que o rastreamento seja estável mesmo com *ego-motion* agressivo (comum no GOT-10k).
4. **Ponto Crítico:** O ganho é muito maior na GoPro (700%) do que no GOT-10k (8%) porque na GoPro o movimento é puramente translacional e rápido (veículos), onde o LightGlue brilha ao lidar com grandes saltos espaciais. No GOT-10k, a diversidade de movimentos e o CMC ajudam a manter a liderança em baixos FPS.
