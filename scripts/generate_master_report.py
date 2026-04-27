import pandas as pd
import os

root = "/home/servidor/ArtigoLightglue/mass_results"
lg_csv = os.path.join(root, "mass_experiment_report.csv")
base_csv = os.path.join(root, "mass_baselines_report.csv")

if not os.path.exists(lg_csv) or not os.path.exists(base_csv):
    print("Erro: Arquivos de relatório não encontrados.")
    exit()

# Lendo os dados e padronizando colunas
df_lg = pd.read_csv(lg_csv)
df_lg['Metodologia'] = 'LightGlue'
# Renomear para bater com o padrão desejado
df_lg = df_lg.rename(columns={"video": "Video", "fps": "FPS", "ids": "IDs", "sightings": "Sightings"})

df_base = pd.read_csv(base_csv)
df_base = df_base.rename(columns={"Video": "Video", "Tracker": "Metodologia", "FPS": "FPS", "Unique_IDs": "IDs", "Sightings": "Sightings"})
# Limpar FPS se vier como string '30fps'
if df_base['FPS'].dtype == object:
    df_base['FPS'] = df_base['FPS'].str.replace('fps', '').astype(int)

# Unir tudo
df_all = pd.concat([df_lg, df_base], ignore_index=True)

# Pivotar a tabela para comparação direta: Video e FPS nas linhas, Metodologias nas colunas
pivot_df = df_all.pivot_table(index=['Video', 'FPS'], columns='Metodologia', values='IDs').reset_index()

# Ordenar
pivot_df = pivot_df.sort_values(by=['Video', 'FPS'], ascending=[True, False])

# Gerar Markdown
markdown = "# RELATÓRIO COMPARATIVO PARA ARTIGO: INVARIANÇA DE IDENTIDADE\n\n"
markdown += "Esta tabela compara a consistência da contagem de identidades (IDs únicos) entre o nosso método (LightGlue) e os baselines industriais (BoTSORT e ByteTrack).\n\n"
markdown += "| Vídeo | FPS | **LightGlue (Ours)** | BoTSORT | ByteTrack | Diferença LG vs BoTSORT |\n"
markdown += "| :--- | :---: | :---: | :---: | :---: | :---: |\n"

for _, row in pivot_df.iterrows():
    lg = row.get('LightGlue', 0)
    bot = row.get('botsort', 0)
    byte = row.get('bytetrack', 0)
    
    # Formatação amigável (sem .0)
    lg_s = f"{int(lg)}" if not pd.isna(lg) else "N/A"
    bot_s = f"{int(bot)}" if not pd.isna(bot) else "N/A"
    byte_s = f"{int(byte)}" if not pd.isna(byte) else "N/A"
    
    diff = int(lg - bot) if (not pd.isna(lg) and not pd.isna(bot)) else "N/A"
    
    # Destacar a nossa coluna
    markdown += f"| {row['Video']} | {row['FPS']} | **{lg_s}** | {bot_s} | {byte_s} | {diff} |\n"

# Salvar
report_path = os.path.join(root, "TABELA_COMPARATIVA_ARTIGO.md")
with open(report_path, 'w') as f:
    f.write(markdown)

print("\n" + "="*50)
print("TABELA COMPARATIVA GERADA COM SUCESSO!")
print(f"Local: {report_path}")
print("="*50 + "\n")
print(markdown)
