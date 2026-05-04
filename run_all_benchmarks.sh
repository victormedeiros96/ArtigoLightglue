#!/bin/bash
# Script mestre para gerar todos os resultados do artigo
echo "Iniciando benchmarks completos para o artigo (GoPro e GOT-10k)..."
date

# Criar pasta de resultados se não existir
mkdir -p mass_results

# 1. GoPro (Roadside Mode) - Já rodamos, mas deixo aqui se precisar refazer
# echo "1. Rodando GoPro Benchmark..."
# python3 benchmarks/run_gopro_benchmark.py > mass_results/gopro_full.log 2>&1

# 2. GOT-10k (SOT-to-MOT Mode) - Foco agora
echo "2. Rodando GOT-10k Benchmark (180 sequências)..."
python3 benchmarks/run_got10k_full_benchmark.py > mass_results/got10k_full.log 2>&1

echo "Benchmarks finalizados!"
date
