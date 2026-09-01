# Durian GPT: LangGraph + RAGFlow + Dify

This directory contains the private retrieval/control-plane services used only
by Durian GPT.

## Runtime ownership

- LangGraph owns conversation state, topic routing, user memory, and exact
  answer gating.
- RAGFlow owns document parsing and hybrid vector + keyword retrieval.
- Dify provides an optional visual retrieval workflow/control plane. It does
  not own or duplicate chat memory.
- The existing Durian vLLM remains the only GPU language model.

## Private endpoints

- RAGFlow console: `127.0.0.1:19380`
- Dify console: `127.0.0.1:15001`
- Docker-only embedding bridge: `172.17.0.1:18001`
- Docker-only RAGFlow bridge: `172.17.0.1:18002`
- Docker-only existing-vLLM bridge: `172.17.0.1:18003`

The management consoles are intentionally not exposed to the public network.
Use SSH local forwarding when administration is required.

## Configuration and validation

Credentials are stored in `.admin-credentials` and
`/etc/durian-gpt-platform.env`, both mode `0600`. Do not commit or print them.

```bash
/home/admin01/miniconda3/envs/durian/bin/python platforms/durian-configure-ragflow.py
/home/admin01/miniconda3/envs/durian/bin/python platforms/durian-validate-ragflow.py
/home/admin01/miniconda3/envs/durian/bin/python platforms/durian-configure-dify.py
/home/admin01/miniconda3/envs/durian/bin/python platforms/durian-validate-dify.py
/home/admin01/miniconda3/envs/durian/bin/python platforms/durian-e2e.py
```

Only run `durian-configure-ragflow.py --enable` after the RAGFlow validator
passes. Exact questions never fall back to free generation: if no matching
source passage survives the gate, the answer explicitly says that the source
was not found and returns no misleading reference.
