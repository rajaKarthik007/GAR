#!/usr/bin/env bash
set -euo pipefail

# TAMUS LLM CLI bootstrap for OpenAI-compatible chat endpoint.
# Usage:
#   bash scripts/setup_tamus_llm.sh \
#     --endpoint https://chat-api.tamu.ai \
#     --key-id chat.tamu.ai \
#     --default-model protected.o4-mini

ENDPOINT="https://chat-api.tamu.ai"
KEY_ID="chat.tamu.ai"
DEFAULT_MODEL="protected.o4-mini"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --endpoint)
      ENDPOINT="$2"
      shift 2
      ;;
    --key-id)
      KEY_ID="$2"
      shift 2
      ;;
    --default-model)
      DEFAULT_MODEL="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if ! command -v llm >/dev/null 2>&1; then
  echo "llm CLI not found. Install with: brew install llm  (or pip install llm)" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq not found. Install with: brew install jq" >&2
  exit 1
fi

echo "Set TAMUS API key for identifier: ${KEY_ID}"
llm keys set "${KEY_ID}"

TAMUS_AI_CHAT_API_KEY="$(llm keys get "${KEY_ID}")"

echo "Fetching models from ${ENDPOINT}/api/models"
models=$(curl -s -X GET "${ENDPOINT}/api/models" \
  -H "Authorization: Bearer ${TAMUS_AI_CHAT_API_KEY}" \
  | jq -r '.data[].id')

if [[ -z "${models}" ]]; then
  echo "No models returned from endpoint. Check endpoint/key." >&2
  exit 1
fi

llmpath=$(dirname "$(llm logs path)")
mkdir -p "${llmpath}"
cd "${llmpath}"

cat /dev/null > extra-openai-models.yaml

while IFS= read -r model; do
  [[ -z "${model}" ]] && continue
  cat >> extra-openai-models.yaml <<YAML
- model_id: "TAMUS AI Chat (${KEY_ID}): ${model}"
  model_name: "${model}"
  api_base: "${ENDPOINT}/api"
  api_key_name: "${KEY_ID}"
YAML
done <<< "${models}"

echo "Registered models in $(pwd)/extra-openai-models.yaml"
llm models | grep "TAMUS AI Chat (${KEY_ID})" || true

default_name="TAMUS AI Chat (${KEY_ID}): ${DEFAULT_MODEL}"
if llm models | grep -F "${default_name}" >/dev/null; then
  llm models default "${default_name}"
  echo "Default model set to: ${default_name}"
else
  echo "Default model not set: ${DEFAULT_MODEL} not found in endpoint model list." >&2
fi

echo
echo "Test command:"
echo "llm \"How do you fly a kite?\""
