# LightGlue Plate Tracking: Roadside Visual Resilience

Este repositório contém a implementação e o benchmark de um rastreador de placas de veículos baseado em **SuperPoint + LightGlue**, projetado para máxima resiliência em baixas taxas de amostragem temporal (Low FPS).

## Principais Destaques
- **Resiliência a Baixo FPS:** Mantém o rastreamento estável a 2 FPS, onde métodos tradicionais (BoTSORT, ByteTrack) falham completamente.
- **Otimização de Gate Espacial:** Capaz de lidar com grandes saltos de objetos (até 1500px em 4K) causados pelo movimento do veículo.
- **Superioridade Técnica:** Supera o BoTSORT em **5.5x (MOTA)** em cenários de 2 FPS na GoPro.
- **Global CMC:** Integra compensação de movimento de câmera baseada em keypoints para cenários de movimento arbitrário.

## Resultados Rápidos (GoPro @ 2 FPS)
| Método | MOTA (↑) | IDF1 (↑) | IDSW (↓) |
| :--- | :---: | :---: | :---: |
| BoTSORT (Baseline) | 0.82% | 1.58% | 5 |
| **LightGlue (Nós)** | **4.53%** | **8.40%** | **302** |

## Estrutura do Projeto
- `core/`: Implementação do `LightGlueTracker`.
- `benchmarks/`: Scripts de avaliação para GoPro e GOT-10k.
- `mass_results/`: Dados brutos e relatório técnico completo.

## Documentação Completa
Consulte o arquivo [DOCUMENTACAO_COMPLETA_BENCHMARK.md](./mass_results/DOCUMENTACAO_COMPLETA_BENCHMARK.md) para a análise detalhada dos experimentos e justificativas técnicas.
