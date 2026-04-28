# Artigo - Tracking Invariante com LightGlue e SuperPoint

Repositório oficial para o backend de rastreamento de ativos rodoviários com foco em consistência de identidade através de taxas de quadros variáveis (30, 5 e 1 FPS).

## Estrutura do Repositório

```text
├── benchmarks/           # Scripts de experimentos e baselines
│   ├── run_lightglue.py   # Executa o tracker principal (LightGlue)
│   ├── run_baselines.py   # Executa benchmarks (BoTSORT, ByteTrack)
│   ├── generate_report.py # Consolida resultados em tabelas
│   └── render_videos.py   # Gera visualizações MP4 com as IDs
├── configs/              # Arquivos de configuração dos trackers
├── core/                 # Lógica central do LightGlueTracker
├── models/               # Pesos do detector YOLO (.pt)
├── mass_results/         # Resultados de massa (JSONs, CSVs, Vídeos)
└── proxies/              # Vídeos originais para teste (ignorados no git)
```

## Como Executar

1. **Rastreamento LightGlue**:
   ```bash
   python3 benchmarks/run_lightglue.py
   ```

2. **Baselines (BoTSORT/ByteTrack)**:
   ```bash
   python3 benchmarks/run_baselines.py
   ```

3. **Gerar Relatórios**:
   ```bash
   python3 benchmarks/generate_report.py
   ```

4. **Visualizar Vídeos**:
   ```bash
   python3 benchmarks/render_videos.py
   ```

## Metodologia

O sistema utiliza **SuperPoint** para extração de features e **LightGlue** para correspondência temporal, reforçado por restrições geométricas de lane guard e expansão radial para garantir que objetos estáticos em rodovias mantenham sua identidade mesmo em baixas taxas de FPS.
