
python3.12 -m venv venv312
source venv312/bin/activate
pip install fastapi uvicorn httpx oci python-dotenv

## Setup oci-anthropic-gateway
export OCI_BASE_URL=https://inference.generativeai.us-chicago-1.oci.oraclecloud.com   # change to your zone
export OCI_AUTH_TOKEN=your OCI Bearer token or API key
export OCI_COMPARTMENT_ID=ocid1.compartment.oc1.........
edit config.json

python oci-anthropic-gateway.py

export OCI_COMPARTMENT_ID=ocid1.compartment.oc1.........
export ANTHROPIC_BASE_URL=http://localhost:8000/
export ANTHROPIC_API_KEY=dummy_key
export ANTHROPIC_MODEL=xai.grok-4-fast-non-reasoning
claude
