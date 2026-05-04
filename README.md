# Artigo - Tracking Invariante com LightGlue e SuperPoint

Repositório oficial para o backend de rastreamento de objetos com foco em consistência de identidade através de taxas de quadros variáveis (30, 10, 5 e 2 FPS).

## Estrutura do Repositório

```text
├── benchmarks/           # Scripts de experimentos e baselines
│   ├── run_gopro_benchmark.py       # Benchmark completo no GoPro
│   ├── run_got10k_full_benchmark.py # Benchmark completo no GOT-10k
│   └── ...                          # Auxiliares de visualização e labeling
├── core/                 # Lógica central do LightGlueTracker (c/ CMC)
├── models/               # Pesos do detector YOLO e re-id (.pt)
├── mass_results/         # Resultados consolidados e tabelas LaTeX
│   └── DOCUMENTACAO_COMPLETA_BENCHMARK.md # Relatório estatístico final
└── run_all_benchmarks.sh # Script mestre para execução de todos os testes
```

## Como Executar

1. **Pipeline Completo (GoPro + GOT-10k)**:
   ```bash
   chmod +x run_all_benchmarks.sh
   ./run_all_benchmarks.sh
   ```

2. **Benchmarks Individuais**:
   - Para Placas/Roadside (GoPro): `python3 benchmarks/run_gopro_benchmark.py`
   - Para Objetos Gerais (GOT-10k): `python3 benchmarks/run_got10k_full_benchmark.py`

## Resultados Principais

Os resultados demonstram que o rastreador baseado em **LightGlue** é significativamente mais resiliente que os baselines industriais (**BoTSORT** e **ByteTrack**) conforme a taxa de quadros diminui:

- **GoPro (2 FPS):** Melhoria de **7x** em MOTA comparado ao BoTSORT.
- **GOT-10k (2 FPS):** Liderança em MOTA e IDF1 mesmo em cenários de movimento variado.

## Metodologia

O sistema utiliza **SuperPoint** para extração de features e **LightGlue** para correspondência temporal, reforçado por uma camada de **Camera Motion Compensation (CMC)** que utiliza as próprias correspondências do LightGlue para estabilizar o rastreamento contra ego-motion.

