# RunPod training endpoint for *You.*

This deploys a serverless GPU endpoint that fine-tunes a tiny LoRA adapter on a user's wiki and (optionally) pushes it to Hugging Face.

## One-time setup

### 1. Build and push the image

You'll need a Docker registry your RunPod can pull from (Docker Hub is easiest).

```bash
cd runpod
docker build -t YOUR-DOCKERHUB-USER/you-trainer:latest .
docker push YOUR-DOCKERHUB-USER/you-trainer:latest
```

> If you don't have Docker locally, you can also use RunPod's [GitHub integration](https://docs.runpod.io/serverless/workers/deploy/github) — point it at this repo's `runpod/` folder and RunPod builds the image for you.

### 2. Create a Serverless endpoint on RunPod

1. Go to **https://www.runpod.io/console/serverless**
2. Click **+ New Endpoint**
3. Configure:
   - **Container image**: `YOUR-DOCKERHUB-USER/you-trainer:latest`
   - **GPU**: A40 / RTX 4090 / L40S (24 GB+) — A40 is a good default
   - **Active workers**: 0 (cold-start; saves money)
   - **Max workers**: 1 (single-tenant per user)
   - **Idle timeout**: 5 s
   - **Execution timeout**: 600 s (10 min — long enough for 200 iters on a 1B model)
   - **Container disk**: 30 GB (base model is ~5 GB plus dependencies)
4. **Environment variables**: leave empty for now — they'll come per-request
5. **Create**

After creation you'll have an **Endpoint ID** (like `abc123def456`). Note it down.

### 3. Get a RunPod API key

**https://www.runpod.io/console/user/settings → API Keys → Create API Key**

### 4. Add the secrets to Vercel

The *You.* Vercel project needs these env vars (Production):

| Name | Value |
|---|---|
| `RUNPOD_API_KEY` | from step 3 |
| `RUNPOD_ENDPOINT_ID` | from step 2 |
| `HF_TOKEN` | *(optional)* a Hugging Face write token for pushing adapters |
| `HF_REPO_PREFIX` | *(optional)* e.g. `vizuara/you-` — visitor's user-id is appended |

Add via the Vercel dashboard → **Settings → Environment Variables → Production**.

Then redeploy: **Deployments → ⋯ on latest → Redeploy**.

## How a training run works

1. Frontend (Brain tab) builds a Q&A dataset by calling `/api/claude` for each wiki page.
2. POSTs the dataset to `/api/train/start`, which forwards to `https://api.runpod.ai/v2/<ENDPOINT_ID>/run` with `RUNPOD_API_KEY`.
3. RunPod cold-starts a worker, downloads the base model on first run (~2 min), then fine-tunes for `iters` steps (~3 min on A40).
4. On success, pushes the adapter to `HF_REPO_PREFIX + user_id`.
5. Frontend polls `/api/train/status?id=<job_id>` and animates the brain accordingly.

## Estimated cost per training run

- **A40, 200 iters, 1B model:** ~5 minutes of compute → **~$0.04 per run**
- **L40S, 500 iters, 3B model:** ~12 minutes → **~$0.18 per run**

Cold-start adds 60–90s the first time after a deploy or long idle. After that, the worker stays warm for ~5s, so back-to-back runs share a worker.

## Local-only test (no GPU)

If you don't have a RunPod yet but want to verify the handler signature, the harness can be simulated:

```bash
cd runpod
pip install runpod
python -c "
from handler import handler
print(handler({'input': {
  'dataset': [{'messages': [{'role':'user','content':'hi'}, {'role':'assistant','content':'hello'}]}],
  'iters': 1, 'base_model': 'sshleifer/tiny-gpt2'
}}))
"
```

(That uses a 1.5 MB toy model — won't actually train but exercises the code path.)
