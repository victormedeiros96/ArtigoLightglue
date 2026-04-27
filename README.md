# ArtigoLightglue

Repositório para os códigos, experimentos de massa e resultados do artigo analisando o rastreamento quase-denso por similaridade estrutural focado em ativos de rodovias usando LightGlue.

## Arquivos Principais
- `trackers/lightglue_tracker.py`: O algoritmo principal de rastreamento robusto a variação de FPS, com proteção espacial (Lane Guard).
- `run_mass_experiment.py`: Executa o tracking para múltiplos vídeos simultaneamente e emula quebras de FPS.
- `run_mass_baselines.py`: Executa algoritmos de baseline (BoTSORT, ByteTrack) para gerar as métricas de comparação.
- `mass_results/`: Logs brutos, comparativos formatados (.csv e .md) das execuções em ambiente de massa.
