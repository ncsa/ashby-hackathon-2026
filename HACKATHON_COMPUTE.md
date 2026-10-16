# Hackathon computing: Delta, models, data, and Lumen

Student guide to the computing for the 2026 Ashby Prize Hackathon
(Sat–Sun, Oct 17–18, 8:30am–5:00pm, with Delta access through Mon Oct 19).
It follows the opening-session slides, with the commands written out in full.

Help: **#compute-issues** on Slack for Delta, OnDemand, Slurm and Lumen.

**GPUs are the scarcest thing this weekend.** There are 16 reserved A100 GPUs
for about 60 teams. Use the hosted model servers and Lumen first, run CPU jobs
where you can, and use one GPU per team at a time for short jobs.

Contents:

1. [Log in to Delta](#1-log-in-to-delta)
2. [Storage](#2-storage)
3. [Shared datasets](#3-shared-datasets)
4. [Pathology models: CONCH, MUSK, QuiltNet, PLIP](#4-pathology-models)
5. [GPU jobs](#5-gpu-jobs)
6. [Embed tiles, then train a classifier](#6-embed-tiles-then-train-a-classifier)
7. [LLMs on Lumen](#7-llms-on-lumen)
8. [Agentic coding with opencode](#8-agentic-coding-with-opencode)
9. [Illinois Chat and your own resources](#9-illinois-chat-and-your-own-resources)

## 1. Log in to Delta

**Find your username.** Log in at <https://allocations.access-ci.org>, go to
**My Projects** → the hackathon's Delta project → **My Username**. (Some NCSA
docs mention a "Resource Provider Site Usernames" table on your profile page.
The current portal doesn't have it; use My Projects.)

**SSH** from a terminal (macOS/Linux Terminal, Windows PowerShell):

```bash
ssh <username>@login.delta.ncsa.illinois.edu
```

Enter your NCSA password, then answer Duo (type `1` for a push). You land on a
**login node**: edit files, install packages, submit jobs. Don't run anything
heavy there.

**Or use Open OnDemand** in a browser: <https://openondemand.delta.ncsa.illinois.edu>.
This is the easiest start if you haven't used a cluster before. It has a file
browser, a shell, and apps like Jupyter and Code Server. For a CPU session,
use account `bfqi-delta-cpu`, partition `cpu-interactive`, duration
`00-01:00:00`.

**Check you're on the hackathon project:**

```bash
accounts   # should list bfqi-delta-cpu and bfqi-delta-gpu
quota      # your storage use and limits
```

If either account is missing, post in #compute-issues.

## 2. Storage

| Where | Use it for |
|---|---|
| `/u/<username>` (home, 100 GB) | Code, environments, small files |
| `/projects/bfqi/<username>` | Your data and outputs |
| `/projects/bfqi/teams/<team>` | Files your team shares, e.g. embeddings computed once and reused. Create it with `mkdir -p /projects/bfqi/teams/<team>` |
| `/projects/bfqi/shared/` | Read-only datasets |

Avoid `/work` this weekend: `/work/hdd` is having filesystem problems. If a
write fails with `Disk quota exceeded`, use your home directory and tell us in
#compute-issues.

Everyone in the hackathon can read `/projects/bfqi`, so don't put anything
private there.

## 3. Shared datasets

Read these in place; don't copy them.

```
/projects/bfqi/shared/datasets/
    NCT-CRC-HE-100K/data-samples/   # CRC: 100k .tif tiles, one folder per class (9)
    quilt/                          # Quilt-1M: .jpg images + quilt_1M_lookup CSV of captions
    pathgen_dataset/                # PathGen-1.6M: JSON captions + a 2k-slide sample
    pathmmu/                        # PathMMU: ARCH, BACH, MHIST, NCT-CRC, Osteo folders
```

These are already tiles and captions, so no slide preprocessing is needed. If
the path changes, it'll be posted in #compute-issues.

## 4. Pathology models

| Model | Name to pass to our scripts | Access |
|---|---|---|
| CONCH | `conch` | Gated: get your own access first |
| MUSK | `musk` | Gated: get your own access first |
| QuiltNet-B-32 | `openclip:wisdomik/QuiltNet-B-32` | Open |
| PLIP | `hfclip:vinid/plip` | Open |

**CONCH and MUSK: everyone needs their own access.** Their licenses forbid
sharing copies, and CONCH's says each person must register individually.
That also applies to using them through the hosted servers.

1. Request access on <https://huggingface.co/MahmoodLab/CONCH> and
   <https://huggingface.co/xiangjx/musk>. Use your .edu email; approval is
   usually fast.
2. Create a Hugging Face token (Settings → Access Tokens), then on Delta run
   `huggingface-cli login` and paste it.
3. The weights download to your home directory the first time you use them.

**Two ways to use the models:**

- **Hosted servers (no GPU job needed).** The organizers run all four models,
  one endpoint each, on a reserved A40 node. Host, port and API key are posted
  in #compute-issues. Call them from **inside Delta** (login node, OnDemand,
  or a job), not from your laptop. See [`STUDENT_GUIDE.md`](STUDENT_GUIDE.md).
  The servers run until **Sun 6:00pm**.
- **Your own GPU job.** Load the model yourself, as in section 6.

## 5. GPU jobs

The reserved A100 nodes are on `bfqi-delta-gpu`, partition `gpuA100x4`. You
don't need a `--reservation` flag: event jobs go to the reserved nodes
automatically.

Interactive, for debugging (release it when you're done):

```bash
srun --account=bfqi-delta-gpu --partition=gpuA100x4 --gpus-per-node=1 \
     --cpus-per-task=16 --mem=32G --time=01:00:00 --pty bash
```

Rules:

- **One GPU per team at a time.**
- **Keep jobs to an hour or two.** The reserved nodes run Sat 7:30am to
  Sun 6:00pm, in two blocks: **Sat 7:30am – Sun 7:59am** and
  **Sun 8:00am – 6:00pm**. A job whose time limit crosses Sun 8:00am or
  6:00pm can't use the reserved nodes and waits in the regular queue, which
  this weekend means days. If you leave a job running Saturday night, make sure
  it finishes by 7:59am Sunday.
- **Nothing heavy on the login node.**
- `squeue -u $USER` lists your jobs, and `scancel <jobid>` ends one. End idle
  Jupyter or interactive sessions; they keep holding a GPU.
- After Sun 6:00pm there are no reserved nodes, and GPU jobs go through the
  normal (slow) queue.

## 6. Embed tiles, then train a classifier

One-time setup on the login node:

```bash
git clone https://github.com/ncsa/ashby-hackathon-2026.git
cd ashby-hackathon-2026
module load pytorch-conda/2.8
pip install --user "open_clip_torch>=2.24"
# only if you'll run CONCH or MUSK yourself:
pip install --user git+https://github.com/mahmoodlab/CONCH.git
pip install --user git+https://github.com/lilab-stanford/MUSK.git
```

**Embed a folder of tiles** (GPU batch job, 1 hour). Run it from inside the
`ashby-hackathon-2026` folder, where its logs go:

```bash
# sbatch ... submit.sbatch <tiles dir> <output csv> <model>
sbatch --account=bfqi-delta-gpu submit.sbatch \
    /projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples/TUM \
    tum_embeddings.csv openclip:wisdomik/QuiltNet-B-32
```

This writes one row per tile (`image_path`, `embedding`). The script reads
folders recursively, and rerunning the same command resumes where it stopped.
Logs go to `logs/<jobid>.out`. Pass a JSON file of `{label: prompt}` as a 4th
argument to get zero-shot predictions instead of embeddings; `prompts.json` is
an example.

**Train a classifier on the embeddings.** This runs in seconds on CPU, so you
don't need a GPU:

```bash
python train_probe.py embeddings.csv                      # label = tile's folder name
python train_probe.py embeddings.csv --labels labels.csv  # your own labels
```

`labels.csv` has columns `image_path,label`. Add a `group` column (a patient
or slide ID) to split by patient, so tiles from one patient never end up in
both train and test. CRC-100K has no patient IDs, so its random split
overstates accuracy. To run it as a CPU batch job, use
`--account=bfqi-delta-cpu --partition=cpu`.

## 7. LLMs on Lumen

[Lumen](https://lumen.ncsa.illinois.edu) is NCSA's gateway to open-weight
LLMs: coding models and general chat models, in a browser chat or through an
OpenAI-compatible API. Help pages are at <https://lumen.ncsa.illinois.edu/help/>.

**Log in** with your NCSA ID. Hackathon participants are on Lumen's access
list; if you can't get in, post in #compute-issues.

**Coins.** Each request costs coins, and bigger models cost more. You start
with 10, and your balance slowly refills every hour. Your **Profile** page
shows your balance. If you run out, post in #compute-issues and an organizer
will top you up.

**Model access.** The *Model Access* table on your Profile lists every model.
A model marked **Need Consent** requires a one-time click to accept its terms
before you can use it.

**Get an API key** (one key works for every model):

1. Open your **Profile**.
2. Click **+ New API Key** and **copy the key right away**. It's shown only
   once and starts with `sk_`.
3. Give it a name and click **Save Key**.

Keep it in an environment variable, not in your code:

```bash
export LUMEN_API_KEY=sk_...       # add to ~/.bashrc to keep it across logins
```

**Check it works and list the models.** This works from your laptop or Delta:

```bash
curl https://lumen.ncsa.illinois.edu/v1/models \
     -H "Authorization: Bearer $LUMEN_API_KEY"
```

Model IDs available as of Oct 16 include `qwen3-coder-next`, `glm-5.3`,
`gemma-4-31b-it` and `nemotron-3-super-120b-a12b`; the call above has the
current list.

**From Python** (`pip install openai`):

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["LUMEN_API_KEY"],
                base_url="https://lumen.ncsa.illinois.edu/v1")
reply = client.chat.completions.create(
    model="qwen3-coder-next",   # any ID from /v1/models
    messages=[{"role": "user", "content": "Explain H&E staining in two sentences."}],
)
print(reply.choices[0].message.content)
```

Any tool that speaks the OpenAI API works the same way: set the base URL to
`https://lumen.ncsa.illinois.edu/v1` and use your key.

**If you leak a key** (committed it, pasted it in Slack), delete it on your
Profile page and make a new one. Deleting it takes effect immediately.

## 8. Agentic coding with opencode

[opencode](https://opencode.ai) is an open-source coding agent for the
terminal. Pointed at Lumen, it runs on university-hosted models at no cost to
you. Install it in your home directory on Delta, so the agent works next to
your data and jobs (or on your laptop):

```bash
curl -fsSL https://opencode.ai/install | bash
```

Then create `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "lumen/qwen3-coder-next",
  "provider": {
    "lumen": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Lumen (NCSA)",
      "options": {
        "baseURL": "https://lumen.ncsa.illinois.edu/v1",
        "apiKey": "{env:LUMEN_API_KEY}"
      },
      "models": {
        "qwen3-coder-next": { "name": "Qwen3 Coder Next" },
        "glm-5.3": { "name": "GLM 5.3" }
      }
    }
  }
}
```

With `LUMEN_API_KEY` exported, run `opencode` in your project folder.
`opencode models` should list the Lumen models. Add more from `/v1/models` in
the same format.

Read what the agent writes before you run it, especially anything that
submits GPU jobs.

## 9. Illinois Chat and your own resources

- **Illinois Chat** (<https://chat.illinois.edu>), free with your NetID. Ask
  about papers, models and code; the hackathon's pathology space is
  <https://chat.illinois.edu/multimodal-pathology/chat>.
- **Your laptop** works for Lumen, Illinois Chat, and anything that uses
  embeddings you've copied off Delta. The hosted pathology servers need Delta.
- **Google Colab, or your own lab or cloud GPUs**, are fine for extra GPU
  work.
