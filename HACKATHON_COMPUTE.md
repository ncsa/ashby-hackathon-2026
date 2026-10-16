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
| `/u/<username>` (home, **100 GB each**) | Your code, environments, embeddings and results: **start here** |
| `/projects/bfqi/shared/datasets/` | Read-only datasets |
| `/projects/bfqi/teams/<team>` | Small files your team shares. Create it with `mkdir -p /projects/bfqi/teams/<team>` |

The hackathon's project space (`/projects/bfqi`) is nearly full until NCSA
expands it, so keep big outputs in your home directory. Avoid `/work` this
weekend: `/work/hdd` is having filesystem problems. If a write fails with
`Disk quota exceeded`, tell us in #compute-issues.

Everyone in the hackathon can read `/projects/bfqi`, so don't put anything
private there.

## 3. Shared datasets

Read these in place; don't copy them.

```
/projects/bfqi/shared/datasets/
    NCT-CRC-HE-100K/data-samples/   # CRC: 100k .tif tiles, one folder per class (9)
    pathgen_dataset/                # PathGen-1.6M: JSON captions + a 2k-slide sample
    pathmmu/                        # PathMMU: ARCH, BACH, MHIST, NCT-CRC, Osteo folders
```

These are already tiles and captions, so no slide preprocessing is needed.
**Quilt-1M** will be added once NCSA expands the project's storage; watch
#compute-issues. If
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
   `module load pytorch-conda/2.8` and `hf auth login`, and paste it. Without
   the module, the command isn't found.
3. The weights download to your home directory the first time you use them.

**Two ways to use the models:**

- **Hosted servers (no GPU job needed).** The organizers run all four models,
  one endpoint each, on a reserved A40 node. Host, port and API key are posted
  in #compute-issues. Call them from **inside Delta** (login node, OnDemand,
  or a job), not from your laptop. See [`STUDENT_GUIDE.md`](STUDENT_GUIDE.md).
  The servers run until **Sun 6:00pm**. Each one handles about 60 images per
  second, shared by everyone: use it for up to a few thousand tiles at a time.
  To embed a whole dataset (tens of thousands of tiles), run your own GPU job
  (section 6) so you don't block other teams.
- **Your own GPU job.** Load the model yourself, as in section 6.

### Bringing your own models, data or tools

You're free to use any model, dataset or tool, as long as you follow its
terms:

- **Licences:** many pathology models (CONCH, MUSK, UNI, TITAN, …) are gated
  and non-commercial. Each person requests their own access, and copies of the
  weights can't be shared.
- **Patient data:** use public, de-identified data (TCGA open access, the
  shared datasets, published sets). Controlled-access data needs your own
  approval, and patient data must never be pasted into LLM chats or APIs.
- **Size:** downloads go in your home directory (100 GB). Ask in
  #compute-issues before pulling anything over ~50 GB.
- **Credit:** cite every model, dataset and tool in your final presentation.

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
- **Each person has a budget of 10 GPU-hours** on `gpuA100x4` for the event.
  Slurm won't start a job if it would go over what's left; the job waits with
  reason `QOSGrpBillingMinutes`. You're charged only for time actually used,
  so ask for a realistic `--time`. Two things use the budget faster: asking
  for more than 16 CPUs or about 62 GB of memory per GPU (you're billed for
  whichever is largest), and the `*-interactive` partitions, which cost double.
  Need more? Ask in #compute-issues.
- **Keep jobs to an hour or two.** The reserved nodes run Sat 7:30am to
  Sun 6:00pm, in two blocks: **Sat 7:30am – Sun 7:59am** and
  **Sun 8:00am – 6:00pm**. A job whose time limit crosses Sun 8:00am or
  6:00pm can't use the reserved nodes and waits in the regular queue, which
  this weekend means days. If you leave a job running Saturday night, make sure
  it finishes by 7:59am Sunday.
- **Nothing heavy on the login node.**
- The `*-interactive` partitions (e.g. `cpu-interactive` for OnDemand)
  allow **one job per person at a time** and up to 1 hour each. A second
  request is rejected with `QOSMaxSubmitJobPerUserLimit`; end the first one.
- `squeue -u $USER` lists your jobs, and `scancel <jobid>` ends one. End idle
  Jupyter or interactive sessions; they keep holding a GPU.
- **Don't use DeltaAI** (`bfqi-dtai-gh` in `accounts`). The project has only
  a few dozen hours there, kept as an organizer backup.
- After Sun 6:00pm there are no reserved nodes, and GPU jobs go through the
  normal (slow) queue.

## 6. Embed tiles, then train a classifier

One-time setup on the login node:

```bash
git clone https://github.com/ncsa/ashby-hackathon-2026.git
cd ashby-hackathon-2026
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE   # needed after every module load, or --user packages are invisible
pip install --user "open_clip_torch>=2.24"
# only if you'll run CONCH or MUSK yourself:
pip install --user git+https://github.com/mahmoodlab/CONCH.git
pip install --user git+https://github.com/lilab-stanford/MUSK.git fairscale sentencepiece
```

Put `unset PYTHONNOUSERSITE` right after `module load pytorch-conda/2.8` in
your own scripts and notebooks too: the module sets that variable, and without
it Python ignores everything you installed with `pip install --user`. Our
`submit.sbatch` already does it.

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
`gemma-4-31b-it`, `nemotron-3-super-120b-a12b` and `ornith-1.0-35b`; the call
above has the current list. **Costs differ about 40× between models**: as of
Oct 16, `ornith-1.0-35b` was cheapest, `qwen3-coder-next` and `gemma-4-31b-it`
cheap, and `glm-5.3` by far the most expensive. Your 10 coins go much further
on the cheap ones, especially for coding agents, which resend the whole
conversation at every step.

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
terminal. **On Delta it's already installed** as NCSA's HPC-GPT module, set
up to use Lumen models with no API key of your own:

```bash
module load hpc-gpt
hpc-gpt          # run in your project folder
```

It starts in a Delta-support mode using `lumen/qwen3.8-27b`. Switch models
with `/models`; `opencode models` lists them.

**Choosing a model in hpc-gpt:**

| Use | Model | Why |
|---|---|---|
| Everyday questions, Delta/Slurm help | `lumen/qwen3.8-27b` (default) | Good all-rounder |
| Writing and editing code | `lumen/qwen3-coder-next` | Tuned for code, 131k context, cheap |
| Long or simple tasks on a budget | `lumen/ornith-1.0-35b` | Cheapest by far |
| A hard problem the others can't crack | `lumen/glm-5.3` | Strongest, but about 5–20× the cost; switch back afterwards |

Skip `lumen/gemma-4-31b`, `lumen/nemotron-3-super-120b` and `lumen/qwen3.6-35b`
in the hpc-gpt list. Their names don't match Lumen's current model IDs, so
they're likely to fail. If `ncsahosted/...` doesn't respond, use a `lumen/`
model.

**Tips:**

- **hpc-gpt runs on one Lumen budget shared by every Delta user.** Long agent
  sessions use a lot of tokens, because each step resends the conversation.
  Start a fresh session (`/new`) when you switch tasks, and point the agent at
  specific files rather than your whole dataset.
- **Pick the right mode.** The default `support` mode can edit files and run
  commands. The `debug` and `learning` modes ask before editing or running
  anything, which is a better fit if you're new to Delta.
- **Don't let the agent submit GPU jobs unattended**, and never let it set up
  scripts that resubmit jobs in a loop. Every GPU job counts against your
  10 GPU-hour budget. Read any `sbatch`/`srun` command before approving it.
- **Never paste passwords, API keys or tokens** into a chat. Anything you
  paste is sent to the model.
- Agents make confident mistakes. Run their code on a few tiles first, and
  check numbers against something you trust.

**On your laptop**, install opencode (`curl -fsSL https://opencode.ai/install | bash`)
and point it at Lumen with your own key (section 7). Create
`~/.config/opencode/opencode.json`:

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
