
## Setup oci-anthropic-gateway
export OCI_BASE_URL=https://inference.generativeai.us-chicago-1.oci.oraclecloud.com   # change to your zone
export OCI_AUTH_TOKEN=your OCI Bearer token or API key
export OCI_COMPARTMENT_ID=you COMPARTMENT ID
python oci-anthropic-gateway.py

export ANTHROPIC_BASE_URL=http://localhost:8000/v1
export ANTHROPIC_API_KEY=dummy_key
export ANTHROPIC_MODEL=meta.llama-3.1-70b-instruct   # change to your model, eg. fast code
claude
