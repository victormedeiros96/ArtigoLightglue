import os
import json
import time

PATH = "/home/servidor/ArtigoLightglue/mass_results/jsons"
VIDEOS_TARGET = 9 * 3 # 9 vídeos x 3 taxas de FPS

def get_progress():
    os.system('clear')
    print("="*50)
    print("      DASHBOARD DE PROGRESSO - ARTIGO LIGHTGLUE")
    print("="*50)
    
    files = [f for f in os.listdir(PATH) if f.endswith('.json')]
    finished = len(files)
    percent = (finished / VIDEOS_TARGET) * 100
    
    print(f"Progresso Total: {finished}/{VIDEOS_TARGET} arquivos ({percent:.1f}%)")
    
    # Tentar ver qual processo está rodando
    try:
        import subprocess
        ps = subprocess.check_output("ps aux | grep run_mass_experiment.py | grep -v grep", shell=True).decode()
        if "30fps" in ps: current = "30fps"
        elif "5fps" in ps: current = "5fps"
        elif "1fps" in ps: current = "1fps"
        else: current = "Detectando..."
        print(f"Status Atual: Processando vídeo (taxa {current})")
    except:
        print("Status Atual: Baselines ou Renderização final...")

    print("\nÚltimos Resultados (Contagem de IDs):")
    print("-" * 40)
    for f in sorted(files)[-10:]: # Mostrar os últimos 10
        try:
            with open(os.path.join(PATH, f), 'r') as j:
                data = json.load(j)
                uids = len(set(d['id'] for d in data))
                print(f" {f:<30} | {uids:>3} IDs")
        except:
            pass
    print("-" * 40)
    print("\nComando para atualizar: python3 scripts/check_progress.py")

if __name__ == "__main__":
    get_progress()
