#!/usr/bin/env bash
# Deploy da stack CloudFormation do AbraceAI Analytics em uma conta AWS.
#
# Pre-requisitos:
#   1. AWS CLI instalado e autenticado (aws sts get-caller-identity deve funcionar)
#   2. KeyPair EC2 criada no console (ex.: abrace-ai-analytics) e .pem baixada
#   3. Voce sabe seu IP publico atual (curl -s https://checkip.amazonaws.com)
#
# Uso (interativo - perguntar parametros):
#   bash infra/deploy_aws.sh
#
# Uso (todos os parametros via env):
#   STACK_NAME=abrace-ai REGION=us-east-1 KEY_PAIR=abrace-ai-analytics \
#     ALLOWED_CIDR=1.2.3.4/32 GIT_REPO=https://github.com/USER/repo.git \
#     bash infra/deploy_aws.sh

set -euo pipefail

STACK_NAME="${STACK_NAME:-abrace-ai-analytics}"
REGION="${REGION:-us-east-1}"
TEMPLATE_PATH="$(cd "$(dirname "$0")" && pwd)/cloudformation.yml"

if ! command -v aws >/dev/null 2>&1; then
  echo "[deploy] AWS CLI nao encontrado. Instale: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
  exit 1
fi

echo "[deploy] Conta AWS:"
aws sts get-caller-identity || { echo "[deploy] Faltou 'aws configure'."; exit 1; }

prompt() {
  local var="$1" msg="$2" default="${3:-}"
  local current="${!var:-}"
  if [ -z "${current}" ]; then
    if [ -n "${default}" ]; then
      read -r -p "${msg} [${default}]: " value
      value="${value:-$default}"
    else
      read -r -p "${msg}: " value
    fi
    eval "${var}=\${value}"
  fi
}

prompt KEY_PAIR "Nome da KeyPair EC2 ja criada"
if [ -z "${ALLOWED_CIDR:-}" ]; then
  MY_IP=$(curl -s https://checkip.amazonaws.com || true)
  if [ -n "${MY_IP}" ]; then
    DEFAULT_CIDR="${MY_IP}/32"
  else
    DEFAULT_CIDR=""
  fi
  prompt ALLOWED_CIDR "CIDR autorizado (SSH+dashboard)" "${DEFAULT_CIDR}"
fi
prompt GIT_REPO "URL do repositorio git (vazio = bootstrap manual via scp)" ""
prompt GIT_REF "Branch/tag/commit" "main"

PARAMS=(
  "ParameterKey=KeyPairName,ParameterValue=${KEY_PAIR}"
  "ParameterKey=AllowedSshCidr,ParameterValue=${ALLOWED_CIDR}"
  "ParameterKey=AllowedAppCidr,ParameterValue=${ALLOWED_CIDR}"
  "ParameterKey=GitRepoUrl,ParameterValue=${GIT_REPO}"
  "ParameterKey=GitRef,ParameterValue=${GIT_REF}"
)

echo "[deploy] criando/atualizando stack ${STACK_NAME} em ${REGION}"

if aws cloudformation describe-stacks --stack-name "${STACK_NAME}" --region "${REGION}" >/dev/null 2>&1; then
  ACTION="update-stack"
else
  ACTION="create-stack"
fi

aws cloudformation "${ACTION}" \
  --stack-name "${STACK_NAME}" \
  --region "${REGION}" \
  --template-body "file://${TEMPLATE_PATH}" \
  --parameters "${PARAMS[@]}" \
  --capabilities CAPABILITY_IAM \
  --tags Key=Project,Value=abrace-ai-analytics Key=Owner,Value=academic-poc

echo "[deploy] aguardando stack ficar pronta (pode levar ~5 min)..."
WAIT_VERB=$([ "${ACTION}" = "create-stack" ] && echo "stack-create-complete" || echo "stack-update-complete")
aws cloudformation wait "${WAIT_VERB}" --stack-name "${STACK_NAME}" --region "${REGION}"

echo "[deploy] Outputs:"
aws cloudformation describe-stacks --stack-name "${STACK_NAME}" --region "${REGION}" \
  --query 'Stacks[0].Outputs' --output table
