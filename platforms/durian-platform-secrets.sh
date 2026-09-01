#!/usr/bin/env bash
set -euo pipefail

platform_root="/home/admin01/桌面/Desktop/durian-training/platforms"
ragflow_env="${platform_root}/ragflow/docker/.env"
dify_env="${platform_root}/dify/docker/.env"
credential_file="${platform_root}/.admin-credentials"

replace_token() {
  target_file="$1"
  token="$2"
  value="$3"
  sed -i "s|${token}|${value}|g" "${target_file}"
}

if grep -q '__GENERATE_RAGFLOW_' "${ragflow_env}"; then
  replace_token "${ragflow_env}" "__GENERATE_RAGFLOW_ELASTIC__" "$(openssl rand -hex 32)"
  replace_token "${ragflow_env}" "__GENERATE_RAGFLOW_MYSQL__" "$(openssl rand -hex 32)"
  replace_token "${ragflow_env}" "__GENERATE_RAGFLOW_MINIO__" "$(openssl rand -hex 32)"
  replace_token "${ragflow_env}" "__GENERATE_RAGFLOW_REDIS__" "$(openssl rand -hex 32)"
fi

if grep -q '__GENERATE_DIFY_' "${dify_env}"; then
  dify_init_password="$(openssl rand -base64 24 | tr -d '/+=' | cut -c1-24)"
  replace_token "${dify_env}" "__GENERATE_DIFY_SECRET__" "$(openssl rand -hex 32)"
  replace_token "${dify_env}" "__GENERATE_DIFY_INIT_PASSWORD__" "${dify_init_password}"
  replace_token "${dify_env}" "__GENERATE_DIFY_DB__" "$(openssl rand -hex 32)"
  dify_redis_password="$(openssl rand -hex 32)"
  replace_token "${dify_env}" "__GENERATE_DIFY_REDIS__" "${dify_redis_password}"
  replace_token "${dify_env}" "__GENERATE_DIFY_WEAVIATE__" "$(openssl rand -hex 32)"
  replace_token "${dify_env}" "__GENERATE_DIFY_SANDBOX__" "$(openssl rand -hex 32)"
  replace_token "${dify_env}" "__GENERATE_DIFY_PLUGIN_DAEMON__" "$(openssl rand -hex 32)"
  replace_token "${dify_env}" "__GENERATE_DIFY_PLUGIN_INNER__" "$(openssl rand -hex 32)"
  umask 077
  {
    echo "DIFY_INIT_PASSWORD=${dify_init_password}"
    echo "DIFY_LOCAL_URL=http://127.0.0.1:15001"
    echo "RAGFLOW_LOCAL_URL=http://127.0.0.1:19380"
  } > "${credential_file}"
fi

chmod 600 "${ragflow_env}" "${dify_env}" "${credential_file}"

if grep -Eq '__GENERATE_' "${ragflow_env}" "${dify_env}"; then
  echo "Credential placeholders remain" >&2
  exit 1
fi

echo "Platform credentials generated and stored with mode 600."
