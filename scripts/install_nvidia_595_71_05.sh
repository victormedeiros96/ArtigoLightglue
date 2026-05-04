#!/usr/bin/env bash
set -euo pipefail

INSTALLER="${INSTALLER:-/home/servidor/Downloads/NVIDIA-Linux-x86_64-595.71.05.run}"
STOP_DISPLAY_MANAGER="${STOP_DISPLAY_MANAGER:-0}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Rode com sudo:"
  echo "  sudo bash $0"
  exit 1
fi

if [[ ! -f "${INSTALLER}" ]]; then
  echo "Instalador nao encontrado: ${INSTALLER}"
  echo "Opcao: sudo INSTALLER=/caminho/NVIDIA-Linux-x86_64-595.71.05.run bash $0"
  exit 1
fi

echo "Instalador: ${INSTALLER}"
echo

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "Driver atual:"
  nvidia-smi --query-gpu=driver_version,name --format=csv,noheader || true
else
  echo "nvidia-smi nao encontrado no PATH."
fi

echo
echo "Modulos NVIDIA carregados:"
lsmod | grep -E '^(nvidia|nouveau)' || true

echo
echo "Processos usando /dev/nvidia*:"
fuser -v /dev/nvidia* 2>&1 || true

echo
read -r -p "Continuar e tentar liberar o driver NVIDIA? [y/N] " answer
case "${answer}" in
  y|Y|yes|YES) ;;
  *) echo "Cancelado."; exit 0 ;;
esac

if [[ "${STOP_DISPLAY_MANAGER}" == "1" ]]; then
  echo "Parando display manager..."
  for service in display-manager gdm3 sddm lightdm; do
    if systemctl list-unit-files "${service}.service" >/dev/null 2>&1; then
      systemctl stop "${service}.service" || true
    fi
  done
else
  echo "Nao vou parar display manager automaticamente."
  echo "Se o unload falhar por nvidia_drm/nvidia_modeset em uso, rode:"
  echo "  sudo STOP_DISPLAY_MANAGER=1 bash $0"
fi

echo
echo "Matando processos compute do usuario que seguram /dev/nvidia*..."
for pid in $(fuser /dev/nvidia-uvm /dev/nvidiactl /dev/nvidia[0-9]* 2>/dev/null | tr ' ' '\n' | sort -u); do
  [[ -z "${pid}" ]] && continue
  comm="$(ps -p "${pid}" -o comm= 2>/dev/null || true)"
  case "${comm}" in
    Xorg|Xwayland|gnome-shell|kwin_x11|kwin_wayland|mutter*|sddm*|gdm*|lightdm*)
      echo "Preservando processo grafico PID ${pid} (${comm})"
      ;;
    *)
      echo "kill -TERM ${pid} (${comm})"
      kill -TERM "${pid}" 2>/dev/null || true
      ;;
  esac
done

sleep 3

for pid in $(fuser /dev/nvidia-uvm /dev/nvidiactl /dev/nvidia[0-9]* 2>/dev/null | tr ' ' '\n' | sort -u); do
  [[ -z "${pid}" ]] && continue
  comm="$(ps -p "${pid}" -o comm= 2>/dev/null || true)"
  case "${comm}" in
    Xorg|Xwayland|gnome-shell|kwin_x11|kwin_wayland|mutter*|sddm*|gdm*|lightdm*)
      ;;
    *)
      echo "kill -KILL ${pid} (${comm})"
      kill -KILL "${pid}" 2>/dev/null || true
      ;;
  esac
done

sleep 2

echo
echo "Removendo modulos NVIDIA..."
modprobe -r nvidia_uvm 2>/dev/null || true
modprobe -r nvidia_drm 2>/dev/null || true
modprobe -r nvidia_modeset 2>/dev/null || true
modprobe -r nvidia 2>/dev/null || true

if lsmod | grep -qE '^nvidia'; then
  echo
  echo "Ainda ha modulos NVIDIA carregados:"
  lsmod | grep -E '^nvidia' || true
  echo
  echo "Processos restantes usando /dev/nvidia*:"
  fuser -v /dev/nvidia* 2>&1 || true
  echo
  echo "Nao consegui descarregar totalmente o driver."
  echo "Tente novamente parando a interface grafica:"
  echo "  sudo STOP_DISPLAY_MANAGER=1 bash $0"
  exit 1
fi

echo "Driver descarregado."
echo
echo "Tornando instalador executavel..."
chmod +x "${INSTALLER}"

echo
echo "Iniciando instalador NVIDIA..."
bash "${INSTALLER}"

echo
echo "Instalacao finalizada. Recarregando modulos basicos..."
modprobe nvidia || true
modprobe nvidia_uvm || true
modprobe nvidia_drm modeset=1 || true

echo
echo "Estado final:"
lsmod | grep -E '^nvidia' || true
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

echo
echo "Recomendo reiniciar a maquina se o instalador atualizou o modulo do kernel."
