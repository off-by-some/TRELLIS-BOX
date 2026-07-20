#!/usr/bin/env bash
set -euo pipefail

APP_USER="${APP_USER:-trellis}"
APP_UID="${APP_UID:-1000}"
APP_GID="${APP_GID:-1000}"

read_token_file() {
    local token_file="$1"
    local token

    [ -f "${token_file}" ] || return 1
    IFS= read -r token < "${token_file}" || return 1
    token="$(first_hf_token "${token}")" || return 1
    printf '%s\n' "${token}"
}

first_hf_token() {
    grep -Eo 'hf_[A-Za-z0-9]{20,}' <<< "${1:-}" | head -n 1
}

load_hf_token() {
    local token

    if [ -n "${HF_TOKEN:-}" ]; then
        token="$(first_hf_token "${HF_TOKEN}" || true)"
        if [ -n "${token}" ]; then
            export HF_TOKEN="${token}"
            export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${token}}"
            echo "[startup] Hugging Face token: found in environment." >&2
            return
        fi
    fi
    if [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
        token="$(first_hf_token "${HUGGING_FACE_HUB_TOKEN}" || true)"
        if [ -n "${token}" ]; then
            export HF_TOKEN="${token}"
            export HUGGING_FACE_HUB_TOKEN="${token}"
            echo "[startup] Hugging Face token: found in environment." >&2
            return
        fi
    fi

    for token_file in \
        "${HF_TOKEN_FILE:-}" \
        "${HF_TOKEN_PATH:-}" \
        "${HF_HOME:-/home/${APP_USER}/.cache/huggingface}/token" \
        "${XDG_CACHE_HOME:-/home/${APP_USER}/.cache}/huggingface/token" \
        "/home/${APP_USER}/.huggingface/token"
    do
        [ -n "${token_file}" ] || continue
        token="$(read_token_file "${token_file}" || true)"
        if [ -n "${token}" ]; then
            export HF_TOKEN="${token}"
            export HUGGING_FACE_HUB_TOKEN="${token}"
            echo "[startup] Hugging Face token: found in ${token_file}." >&2
            return
        fi
    done

    for token_file in \
        "${HF_HOME:-/home/${APP_USER}/.cache/huggingface}/stored_tokens" \
        "${XDG_CACHE_HOME:-/home/${APP_USER}/.cache}/huggingface/stored_tokens"
    do
        [ -f "${token_file}" ] || continue
        token="$(grep -Eo 'hf_[A-Za-z0-9]{20,}' "${token_file}" | head -n 1 || true)"
        if [ -n "${token}" ]; then
            export HF_TOKEN="${token}"
            export HUGGING_FACE_HUB_TOKEN="${token}"
            echo "[startup] Hugging Face token: found in ${token_file}." >&2
            return
        fi
    done

    echo "[startup] Hugging Face token: not found. Gated model downloads may fail." >&2
}

own_dir() {
    local dir="$1"
    mkdir -p "${dir}"
    chown "${APP_UID}:${APP_GID}" "${dir}"
}

own_tree() {
    local dir="$1"
    mkdir -p "${dir}"
    chown -R "${APP_UID}:${APP_GID}" "${dir}"
}

if [ "$(id -u)" = "0" ]; then
    echo "[startup] Preparing mounted cache and output directories..." >&2
    own_dir "/home/${APP_USER}"
    own_dir "${XDG_CACHE_HOME:-/home/${APP_USER}/.cache}"
    own_dir "${HF_HOME:-/home/${APP_USER}/.cache/huggingface}"
    own_dir "${HUGGINGFACE_HUB_CACHE:-/home/${APP_USER}/.cache/huggingface/hub}"
    own_dir "${TRANSFORMERS_CACHE:-/home/${APP_USER}/.cache/huggingface/transformers}"
    own_tree "${TRITON_CACHE_DIR:-/home/${APP_USER}/.cache/triton}"
    own_tree "/app/tmp"
    own_tree "/app/outputs"
    load_hf_token
    echo "[startup] Running as ${APP_UID}:${APP_GID}." >&2

    exec gosu "${APP_UID}:${APP_GID}" "$@"
fi

load_hf_token
exec "$@"
