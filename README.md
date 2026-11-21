# AI Customer Support and Refund Assistance Copilot

Final project for Applied Generative AI. A FastAPI assistant for a simulated online store: staff enter a customer message and select an order, and the assistant identifies the request (cancellation, refund, refund status), checks the order against store policies with Python rules, and drafts a reply that cites the policy. Staff approve every reply through `/approve`. No refunds or messages are ever sent.

This repo extends the course API built since Assignment 1 (sps_genai). The earlier endpoints are kept below.

## Status

Project setup. Project endpoints are being built; see the team plan and decisions log.

## Folder layout

- `app/`: API code and model loading (Member B, with Member A's loading module). Also holds the earlier assignment models.
- `store/`: policies, orders, rule engine, drafting prompt, commitment check (Member C)
- `training/`: Bitext data scripts and the LoRA training notebook (Member A)
- `eval/`: test sets, scenario tests, unit tests, results (Member C, with Member A)
- `data/`: downloaded datasets, git-ignored, created by scripts
- `main.py`: FastAPI entry point

## Team workflow

- Work on your own branch (`member-a`, `member-b`, `member-c`, or a short feature name) and merge into `main` by pull request.
- Never commit datasets, model weights, the SQLite database or `.env`.
- Any change to the kickoff agreements (labels, response fields, order fields, interfaces) goes in the decisions log first.
- Pull the latest `main` before each work session.

Note for Member B: the Dockerfile currently copies only `app/`, `helper_lib/` and `main.py`. Add `store/` when the project endpoints use it.

## Run with Docker

    docker build -t cs-copilot .
    docker run --rm -p 8000:80 cs-copilot

Docs: http://127.0.0.1:8000/docs

## Earlier course endpoints (kept)

- `POST /generate`: bigram text generation, `{"start_word": "bigram", "length": 5}`
- `POST /embedding`: spaCy word vector, `{"word": "apple"}`
- `POST /similarity`: `{"word1": "apple", "word2": "orange"}`
- `POST /classify`: CNN image classifier (CIFAR10), image upload
- `POST /generate_digits`: GAN digits, `{"num_samples": 10, "seed": 42}`
- `POST /generate_diffusion`: diffusion model images, `{"num_samples": 10, "diffusion_steps": 50, "seed": 42}`
- `POST /generate_ebm`: energy-based model images, `{"num_samples": 10, "langevin_steps": 256, "seed": 42}`

## Data and licenses

- Bitext Customer Support dataset (CDLA-Sharing-1.0), downloaded by script, not redistributed
- Qwen2.5-Instruct (Apache 2.0)
- Store policies, orders and the hand-made test set were written by the team