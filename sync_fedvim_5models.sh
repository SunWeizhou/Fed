#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
REMOTES=(
  "dell@10.24.1.131"
  "dell7960@10.4.47.203"
)
WINDOWS_HOST="dell@10.24.1.131"

SYNC_ITEMS=(
  "AGENTS.md"
  "README.md"
  "train_federated.py"
  "advanced_fedvim.py"
  "evaluate_fedvim.py"
  "evaluate_act_fedvim.py"
  "evaluate_baselines.py"
  "evaluation_common.py"
  "client.py"
  "server.py"
  "models.py"
  "config.py"
  "data_utils.py"
  "early_stopping.py"
  "eval_utils.py"
  "requirements.txt"
  "utils"
  "paper_tools"
  "docs"
  "Reference"
  "run_fedvim_model_pipeline.sh"
  "sync_fedvim_5models.sh"
  "launch_fedvim_5models.sh"
  "fetch_fedvim_results.sh"
)

resolve_remote_root() {
  local host="$1"
  if [[ "${host}" == "${WINDOWS_HOST}" ]]; then
    ssh "${host}" "cmd /c mkdir %USERPROFILE%\\Desktop\\FedViM >nul 2>nul || exit 0" >/dev/null
    printf "%s" "/c/Users/DELL/Desktop/FedViM"
  else
    ssh "${host}" "bash -lc 'if [ -d \"\$HOME/桌面\" ]; then root=\"\$HOME/桌面/FedViM\"; else root=\"\$HOME/Desktop/FedViM\"; fi; mkdir -p \"\$root\"; printf \"%s\" \"\$root\"'"
  fi
}

remote_has_rsync() {
  local host="$1"
  if [[ "${host}" == "${WINDOWS_HOST}" ]]; then
    return 1
  fi
  ssh "${host}" "bash -lc 'command -v rsync >/dev/null 2>&1'"
}

copy_to_remote() {
  local host="$1"
  local remote_root="$2"
  local src_path="$3"

  if [[ "${host}" == "${WINDOWS_HOST}" ]]; then
    scp -r "${src_path}" "${host}:Desktop/FedViM/"
    return
  fi

  if remote_has_rsync "${host}"; then
    rsync -az "${src_path}" "${host}:${remote_root}/"
  else
    local parent_dir
    local base_name
    parent_dir="$(dirname "${src_path}")"
    base_name="$(basename "${src_path}")"
    tar -cf - -C "${parent_dir}" "${base_name}" | ssh "${host}" "bash -lc \"cd '${remote_root}' && tar -xf -\""
  fi
}

for host in "${REMOTES[@]}"; do
  echo "[sync] resolving target on ${host}"
  remote_root="$(resolve_remote_root "${host}")"
  echo "[sync] ${host} -> ${remote_root}"

  for item in "${SYNC_ITEMS[@]}"; do
    copy_to_remote "${host}" "${remote_root}" "${PROJECT_ROOT}/${item}"
  done

  copy_to_remote "${host}" "${remote_root}" "${PROJECT_ROOT}/Plankton_OOD_Dataset"
done

echo "[sync] completed for all remotes"
