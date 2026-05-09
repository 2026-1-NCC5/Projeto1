#!/usr/bin/env bash
# TEARDOWN do AbraceAI Analytics na AWS.
# Apaga a stack inteira (EC2, SecurityGroup, EBS) e valida que nao sobrou nada cobravel.

set -euo pipefail

STACK_NAME="${STACK_NAME:-abrace-ai-analytics}"
REGION="${REGION:-us-east-1}"

if ! command -v aws >/dev/null 2>&1; then
  echo "[teardown] AWS CLI nao encontrado."; exit 1
fi

read -r -p "[teardown] CONFIRMAR delecao da stack '${STACK_NAME}' em ${REGION}? (yes/NO) " ans
[[ "${ans:-NO}" == "yes" ]] || { echo "[teardown] cancelado."; exit 0; }

echo "[teardown] deletando stack..."
aws cloudformation delete-stack --stack-name "${STACK_NAME}" --region "${REGION}"
aws cloudformation wait stack-delete-complete --stack-name "${STACK_NAME}" --region "${REGION}"
echo "[teardown] stack deletada."

echo
echo "[teardown] Sanity check de recursos remanescentes:"
echo "[teardown] - EBS volumes na regiao ${REGION} (deve estar vazio se voce so usou esta stack):"
aws ec2 describe-volumes --region "${REGION}" \
  --query 'Volumes[?State!=`deleted`].[VolumeId,State,Size,Tags[?Key==`Project`]|[0].Value]' --output table

echo
echo "[teardown] - Elastic IPs (NAO deve haver alocacoes; cobranca por IP idle):"
aws ec2 describe-addresses --region "${REGION}" \
  --query 'Addresses[].[AllocationId,PublicIp,InstanceId]' --output table

echo
echo "[teardown] - Snapshots manuais (esta stack nao cria snapshots, mas vale checar):"
aws ec2 describe-snapshots --owner-ids self --region "${REGION}" \
  --query 'Snapshots[].[SnapshotId,VolumeSize,StartTime,Description]' --output table

echo
echo "[teardown] OK. Lembre-se de checar Billing > Cost Explorer no proximo dia."
