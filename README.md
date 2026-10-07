# ACE-Step-Intel-XPU-Docker

**Intel Arc (XPU) Docker + mobile UI for [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)**

Run full-song local music generation on **Intel Arc GPUs** under **Linux Docker** (including OpenMediaVault / headless NAS), with a **Spotify-style web UI** that works on phones — not only Windows Gradio.

| | |
|--|--|
| **Upstream model** | [ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5) |
| **This fork’s focus** | Linux Docker, Level Zero / XPU, ace-step-ui, Arc A/B/Pro settings |
| **Verified** | Intel Arc **A770 16GB** on OMV — generate, AI Format, **Save dataset**, **preprocess → `.pt`** |
| **Detailed guide** | **[README-DOCKER-XPU.md](./README-DOCKER-XPU.md)** |

Upstream supports XPU on Windows and ships **NVIDIA CUDA Docker only**. This repository adds the missing path.
EDIT: Also working now on a B60

---

## Quick start

```bash
git clone https://github.com/Manya3084/ACE-Step-Intel-XPU-Docker.git
cd ACE-Step-Intel-XPU-Docker

cp .env.xpu.example .env
# edit .env for your GPU (see GPU tables in README-DOCKER-XPU.md)

docker compose -f docker-compose.xpu.yml up -d --build
```

Open on your LAN / phone:

```text
http://YOUR_SERVER_IP:3003
```

| Service | Role | Port |
|---------|------|------|
| `acestep-xpu` | ACE-Step Gradio + API on Intel XPU | **8001** |
| `acestep-ui` | Mobile-friendly frontend | **3003** |
| `acestep-ui` API (proxied) | Express backend | **3004** |

Host needs Intel Level Zero drivers and `/dev/dri` passed into the container.

---

## What you get

- **Dockerfile.xpu** — Ubuntu + Intel GPU packages + PyTorch XPU nightly + Gradio `--enable-api` + **soundfile/ffmpeg** (no CUDA TorchCodec)
- **Dockerfile.ui** — [ace-step-ui](https://github.com/fspecii/ace-step-ui) with CORS, local **Save dataset**, Gradio arg fixes, ffmpeg/ffprobe
- **docker-compose.xpu.yml** — full stack + shared `datasets` / `lora_output` / `checkpoints`
- **CPU offload** — dual DiT offload + LM RAM path for all **1.5** and **1.5XL** models on 16GB Arc (see below)
- **Live model switch** — Create dropdown switches DiT/LM via `/v1/init` without container restart
- **AI Format** via `/format_lyrics` (threaded; works with CPU offload on 16GB)
- **LoRA path** — Save JSON → preprocess helper → `.pt` tensors → train (see [README-DOCKER-XPU.md](./README-DOCKER-XPU.md))
- **Settings tables** for Arc **A-series**, **B-series**, **Pro**, and **Flex** by VRAM

Default profile (A770 16GB): `acestep-v15-turbo` + `acestep-5Hz-lm-1.7B` + CPU offload + `pt` backend.

### CPU offload at a glance

Edit `.env` then `docker compose -f docker-compose.xpu.yml up -d --force-recreate acestep-xpu`:

```bash
# DiT (music) — keep true on ≤16GB / when using XL
ACESTEP_OFFLOAD_TO_CPU=true
ACESTEP_OFFLOAD_DIT_TO_CPU=true

# LM (lyrics / Format / CoT) — use system RAM on large hosts
ACESTEP_LM_OFFLOAD_TO_CPU=true
ACESTEP_ALLOW_4B_LM=true
```

Set any of those to `false` only if you have VRAM headroom. Full table and guidance: **[CPU offload in README-DOCKER-XPU.md](./README-DOCKER-XPU.md#cpu-offload-enable--disable)**.

---

## Docs in this repo

| Doc | Contents |
|-----|----------|
| **[README-DOCKER-XPU.md](./README-DOCKER-XPU.md)** | Full setup, **CPU offload**, GPU recommendations, LoRA preprocess, troubleshooting |
| **[FORK.md](./FORK.md)** | Short “why this fork exists” |
| Upstream ACE-Step guides | Still under `docs/` (API, Gradio, install, etc.) |
| Windows XPU (upstream-style) | [README-XPU.md](./README-XPU.md) |

---

## Smoke tests

```bash
curl -sS http://127.0.0.1:8001/health

curl -sS -m 300 -X POST http://127.0.0.1:8001/format_lyrics \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"pop rock","lyrics":"walking down the street"}'
```

Format can take **1–3 minutes** on first call with 1.7B + offload — that is normal.

---

## Keeping up with upstream

```bash
git remote add upstream https://github.com/ace-step/ACE-Step-1.5.git   # once
git fetch upstream
git merge upstream/main
# resolve conflicts in Docker/UI overlays → rebuild → smoke test
```

Keep packaging changes small (`Dockerfile.*`, compose, `.env.xpu.example`, documented patches). Pin ace-step-ui to a known-good commit in `Dockerfile.ui` when the stack is stable.

---

## License & upstream

ACE-Step model code remains under the upstream [MIT License](./LICENSE).  
This fork does not rebrand the model; it packages it for **Intel Arc + Docker + mobile UI**.

Upstream project page, models, and research: [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5).
