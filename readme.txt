
export ANTHROPIC_BASE_URL=http://localhost:8000/v1
export ANTHROPIC_API_KEY=dummy_key
export ANTHROPIC_MODEL=meta.llama-3.1-70b-instruct   # 或你想用的 OCI 模型名
claude

OCI_BASE_URL=https://inference.generativeai.us-chicago-1.oci.oraclecloud.com   # 改成你区域的
OCI_AUTH_TOKEN=your OCI Bearer token or API key
export OCI_COMPARTMENT_ID=you COMPARTMENT ID
python oci-anthropic-gateway.py
