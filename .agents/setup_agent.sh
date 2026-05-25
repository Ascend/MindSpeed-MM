#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AGENTS_DIR="${REPO_ROOT}/.agents"

usage() {
  cat <<'EOF'
Usage:
  bash .agents/setup_agent.sh <agent-name>

Examples:
  bash .agents/setup_agent.sh codex
  bash .agents/setup_agent.sh claude
  bash .agents/setup_agent.sh cursor
  bash .agents/setup_agent.sh trae

The command creates .<agent-name>/ as a local adapter directory, links shared
agent content into it, and excludes that generated directory from Git.
EOF
}

die() {
  echo "$*" >&2
  exit 1
}

agent_name="${1:-}"

if [[ -z "${agent_name}" || "${agent_name}" == "-h" || "${agent_name}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! "${agent_name}" =~ ^[A-Za-z0-9_-]+$ ]]; then
  die "Invalid agent name: ${agent_name}"
fi

adapter_dir="${REPO_ROOT}/.${agent_name}"
exclude_entry="/.${agent_name}/"
exclude_file="${REPO_ROOT}/.git/info/exclude"

mkdir -p "${adapter_dir}"

link_or_replace() {
  local source="$1"
  local target="$2"

  if [[ -L "${target}" ]]; then
    rm "${target}"
  elif [[ -e "${target}" ]]; then
    die "Refusing to overwrite existing path: ${target}"
  fi

  ln -s "${source}" "${target}"
}

link_or_replace "../.agents/skills" "${adapter_dir}/skills"
link_or_replace "../.agents/knowledge" "${adapter_dir}/knowledge"

if [[ -d "${REPO_ROOT}/.git" ]]; then
  mkdir -p "$(dirname "${exclude_file}")"
  touch "${exclude_file}"
  if ! grep -Fxq "${exclude_entry}" "${exclude_file}"; then
    printf '%s\n' "${exclude_entry}" >> "${exclude_file}"
  fi
fi
