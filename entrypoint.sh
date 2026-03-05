#!/bin/bash
# Entrypoint script for OCI Anthropic Gateway Docker container

set -e

echo "=========================================="
echo "  OCI Anthropic Gateway - Starting..."
echo "=========================================="

# Check if config.json exists
if [ ! -f /app/config.json ]; then
    echo "ERROR: config.json not found!"
    echo "Please mount your config.json to /app/config.json"
    echo ""
    echo "Example: docker run -v \$(pwd)/config.json:/app/config.json:ro ..."
    exit 1
fi

# Check authentication method
if [ -n "$OCI_RESOURCE_PRINCIPAL_VERSION" ]; then
    echo "Authentication: OCI Resource Principal (Workload Identity)"
    AUTH_MODE="resource_principal"
elif [ -f /root/.oci/config ]; then
    echo "Authentication: OCI API Key"
    AUTH_MODE="api_key"
else
    echo "ERROR: No authentication method available!"
    echo ""
    echo "Option 1 - API Key (local development):"
    echo "  Mount ~/.oci to /root/.oci"
    echo "  docker run -v ~/.oci:/root/.oci:ro ..."
    echo ""
    echo "Option 2 - Resource Principal (OKE/Functions):"
    echo "  Set OCI_RESOURCE_PRINCIPAL_VERSION=2.2"
    echo "  docker run -e OCI_RESOURCE_PRINCIPAL_VERSION=2.2 ..."
    exit 1
fi

# Load environment variables from .env if present
if [ -f /app/.env ]; then
    echo "Loading environment variables from .env"
    export $(grep -v '^#' /app/.env | xargs)
fi

# Set log level (default to WARNING in production)
export LOG_LEVEL=${LOG_LEVEL:-WARNING}

echo ""
echo "Configuration:"
echo "  - Config file: /app/config.json"
echo "  - Auth mode: ${AUTH_MODE}"
echo "  - Log level: ${LOG_LEVEL}"
echo "  - Port: 8000"
echo ""

# Create debug_dumps directory if it doesn't exist
mkdir -p /app/debug_dumps

# Start the gateway
echo "Starting OCI Anthropic Gateway..."
exec python main.py
