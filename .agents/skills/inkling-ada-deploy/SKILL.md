---
name: inkling-ada-deploy
description: Deploy Inkling Ada on a local or user-selected Linux NVIDIA GPU host and verify a real Inkling 2 Mini response over its loopback OpenAI-compatible API. Use when asked to install, start, or bring up the project in one agent task.
---

# Deploy Inkling Ada

Bring up the patched Inkling Ada server on the requested host and finish with a verified chat completion. The default target is one RTX 4090 with SM89 support. A supported Linux installation or existing GPU-enabled WSL2 installation is suitable. Use a stock NInfer build only when the user chooses that variant and label it as upstream NInfer.

1. **Select the host and check capacity.** Work on the user-selected host, or the current eligible Linux host when the request says local. Confirm the GPU with `nvidia-smi --query-gpu=index,name,uuid,compute_cap,memory.free --format=csv`, then check disk space and the requested or default port 8080. The reference FP16 configuration peaked near 18 GiB on a 24 GiB RTX 4090. Use the available VRAM and current processes to decide whether the chosen GPU can run it. Keep existing inference services running. If a production pause, network exposure, or host configuration change is needed, get authorization for that change.
2. **Prepare the engine, model, and project checkout.** Reuse verified artifacts when available. Otherwise follow [Prepare the pinned build and model](references/prepare.md). Resolve every path on the deployment host. Use a separate checkout when work could disturb an existing working tree.
3. **Start and verify in one call.** Copy `scripts/start_server.py` to the target host when running over SSH. Set `gpu_uuid` from the selected GPU in the inventory above.

   ```bash
   CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 /path/to/start_server.py \
     --repo "$repo" --engine "$engine" --model "$model" --port 8080
   ```

   The launcher calls the project's `serve.sh`, which binds to `127.0.0.1` and serves `inkling-2-mini` with a 16,384-token context, one active sequence, FP16 KV, and speculation disabled. Set `--spec dspark` only for a requested draft run. The launcher verifies the model listing and a completed `READY` reply. An identical healthy deployment is reused. A conflicting listener stays intact. Use another port if the user did not require the occupied port.
4. **Hand back a working endpoint.** Report the host, loopback URL ending in `/v1`, model ID, engine variant, PID, and log path from the launcher. When deploying over SSH, rerun the same launcher command from a fresh SSH session and verify that it reuses the PID. A client on another machine can use an SSH tunnel to the loopback listener when needed. If startup or the completion fails, inspect the launcher log and fix the specific issue. Report the blocker if the target cannot be made ready within scope.

The server remains running for the user after a successful start. State and logs default to `${XDG_STATE_HOME:-~/.local/state}/inkling-ada`. To stop a deployment later, verify the PID still identifies that Inkling process before sending SIGTERM.
