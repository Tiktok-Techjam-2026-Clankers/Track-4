#!/usr/bin/env bash
set -euo pipefail

readonly CATALOG_URL="https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/catalog.jsonl.gz"
readonly CHECKSUM_URL="https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/SHA256SUMS"
readonly EXPECTED_ROWS=50000

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly DATA_DIR="${REPO_ROOT}/data"
readonly ARCHIVE_PATH="${DATA_DIR}/catalog.jsonl.gz"
readonly CATALOG_PATH="${DATA_DIR}/catalog.jsonl"

for command_name in curl gzip sha256sum awk wc; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing required command: ${command_name}" >&2
    exit 1
  fi
done

if [[ -f "${CATALOG_PATH}" ]]; then
  row_count="$(wc -l < "${CATALOG_PATH}")"
  if [[ "${row_count}" -eq "${EXPECTED_ROWS}" ]]; then
    echo "Catalog is already ready: ${CATALOG_PATH} (${row_count} rows)"
    exit 0
  fi

  echo "Existing catalog has ${row_count} rows; expected ${EXPECTED_ROWS}." >&2
  echo "Remove it and rerun this script to download a fresh copy." >&2
  exit 1
fi

mkdir -p "${DATA_DIR}"

readonly CHECKSUM_PATH="$(mktemp)"
readonly TEMP_CATALOG_PATH="${CATALOG_PATH}.tmp"
cleanup() {
  rm -f "${CHECKSUM_PATH}" "${TEMP_CATALOG_PATH}"
}
trap cleanup EXIT

echo "Downloading catalog archive..."
curl --fail --location --retry 3 --output "${ARCHIVE_PATH}" "${CATALOG_URL}"

echo "Downloading published checksums..."
curl --fail --location --retry 3 --output "${CHECKSUM_PATH}" "${CHECKSUM_URL}"

expected_hash="$(awk '$2 ~ /catalog\.jsonl\.gz$/ {print $1; exit}' "${CHECKSUM_PATH}")"
if [[ -z "${expected_hash}" ]]; then
  echo "The published checksum file does not contain catalog.jsonl.gz." >&2
  exit 1
fi

actual_hash="$(sha256sum "${ARCHIVE_PATH}" | awk '{print $1}')"
if [[ "${actual_hash}" != "${expected_hash}" ]]; then
  echo "Catalog checksum verification failed." >&2
  exit 1
fi
echo "Checksum verified."

echo "Extracting catalog..."
gzip -dc "${ARCHIVE_PATH}" > "${TEMP_CATALOG_PATH}"

row_count="$(wc -l < "${TEMP_CATALOG_PATH}")"
if [[ "${row_count}" -ne "${EXPECTED_ROWS}" ]]; then
  echo "Extracted catalog has ${row_count} rows; expected ${EXPECTED_ROWS}." >&2
  exit 1
fi

mv "${TEMP_CATALOG_PATH}" "${CATALOG_PATH}"
echo "Catalog is ready: ${CATALOG_PATH} (${row_count} rows)"
