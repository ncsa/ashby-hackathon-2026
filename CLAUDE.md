# TA runbook: CONCH/MUSK/CLIP pathology tooling

Context for whoever is running the pathology track at the hackathon (TA-facing
— not for students, see `STUDENT_GUIDE.md` for that) and for Claude when
asked to help operate this during the event. Lives in
`ncsa/ashby-hackathon-2026` (split out of LLMFlux's `pathology-hackathon-tooling`
branch on 2026-10-16, history preserved). Run the tests with
`python -m pytest`.

## The plan this exists for

About 180 students in about 60 teams (up from the ~15 originally planned).
One persistent micro-batching server per model (`serve.py`), not one Slurm
job per team, so GPU use doesn't grow with the number of teams.

Event plan (2026-10-16): four servers on the A40 reservation node, one per
GPU: `conch`, `musk`, `openclip:wisdomik/QuiltNet-B-32`, `hfclip:vinid/plip`.
Students get endpoints via #compute-issues and call them from inside Delta
only. Compute nodes aren't reachable from outside; gpub080's ports timed out
from off-cluster.

**No shared copy of CONCH or MUSK weights.** Both licenses forbid
redistribution, and CONCH requires each user to register individually. That
is why students are told to get their own HF access before using those
endpoints. Shared copies of open models (QuiltNet, PLIP) are fine; check their
model-card licenses first.

Student-facing compute details: `HACKATHON_COMPUTE.md` (dataset paths in
§ 3; see checklist item 8 below).

## Lessons from 2026: do these next time

The 2026 event (Oct 17–18) went well overall. Most of what's below was
improvised over the weekend and should be set up in advance next year.

**Ask NCSA for everything in one request, about 2 weeks ahead:**

1. **A CPU reservation alongside the GPU ones.** In 2026 there wasn't one at
   first, and CPU jobs (downloads, tiling, probe training, OnDemand
   sessions) sat in Delta's general `cpu` queue for hours until Brett added
   `sup-30434-cpu` (1 node, cn004) at 12:37 Saturday. That node ran out of
   **memory** before cores (250 of 258 GB allocated with only 57 of 128 cores
   in use), so ask for 2 nodes, or a high-memory node.
2. **Limits set before the event and announced in the first post**, not
   worked out midway:
   - The per-user GPU cap (2026: a QOS per student, `bfqi-delta-gpu-<user>`,
     `GrpTRESMins=billing=600000` = 10 GPU-hours on `gpuA100x4`;
     `*-interactive` bills 2×). Our first estimate (~5h) was wrong because
     we calibrated it on an interactive job.
   - Enforce "one GPU per team" with `MaxTRESPerUser=gres/gpu=1` on the QOS
     rather than asking nicely. Several students ran 2–3 GPU jobs at once.
   - A per-user cap on concurrent CPU jobs or memory. One student queued 8
     jobs of 32–48 GB and blocked the CPU node for everyone else.
   - Decide the cap from the budget. 2026 started with 5,083 GPU-hours, and
     3,914 were still left on Sunday morning. The reserved A100s were mostly
     idle (2–5 of 16 GPUs in use), while a few heavy users hit the cap. A
     20–30 hour cap would have been safe. Check with the PI whether the
     leftover allocation is earmarked for after the event before raising it.
3. **Storage set up and checked a week ahead**: quota room on `/projects`
   and `/work/nvme` (2026: `/projects/bfqi` was over quota two days out,
   `/work/hdd` had an outage, `/work/nvme` was expanded to ~4.9 TB on the
   morning of the event, and `/work/hdd` opened with 2 TB Saturday night).

**Accounts.** Get the final roster at least a week ahead, then check every
username with `id <user>` on Delta (it must exist and list `delta_bfqi`).
In 2026, 12 of 13 late additions had no Delta account on Saturday, so
password resets silently sent nothing. A user being on the ACCESS allocation
doesn't mean their Delta account exists.

**Data.** Stage every shared dataset before the event, read-only for the
group (`chmod g-w` plus a default ACL `d:g::rx`), and post the final paths
in the first message. In 2026, Quilt-1M arrived Saturday ~11:00 via Globus,
and it landed in `datasets/` rather than `datasets/quilt/` (check where a
transfer actually wrote). Make the event page match what's actually staged:
it implied TCGA slides were preloaded, and they weren't. Create
`/work/nvme/bfqi/teams/` yourself, with a read-only README, before students
arrive. In 2026 the first student to run `mkdir -p` owned it.

**Model servers.**
- Give each server `--mem=64G`. With 32G, CONCH was OOM-killed after 7h
  (see "Restarting one server" below). Peaks under normal traffic were
  1.5–9.7 GB, so it was large requests, not a leak.
- Before next year, add a limit on images per request in `serve.py` (return 413
  over ~64), and access logging, so a crash can be traced to its cause.
  The student guide now says ≤64 per request.
- Set server walltimes to end exactly when the reservation does: in 2026
  three servers stopped at 17:42, 18 minutes early.
- Usage, for sizing: by Sunday 14:00, MUSK had ~44k requests, CONCH ~9.4k,
  PLIP ~7k and QuiltNet ~6.2k. MUSK and CONCH were the ones that mattered.

**Login nodes.** On Saturday afternoon all four Delta login nodes degraded
(hundreds of processes stuck in D state, ~2-minute logins), and NCSA
rebooted them one by one. Have fallback guidance ready: point students at a
specific healthy `dt-login0N.delta.ncsa.illinois.edu`. OnDemand's "Shell
Access" goes through the login nodes too. Only Interactive Apps (Code
Server, Jupyter) run on compute nodes.

**Supporting students.** Useful queries during the event:
`scontrol show assoc_mgr flags=qos` (each student's GPU-cap usage, including
running jobs), `squeue -R <reservation>`, and `sacct -a -A <account> -S
<time> -X` grouped by user and state (fast repeated failures mean someone is
stuck). Student logs live in home directories, which TAs can't read, so ask
students for the error text. Common problems: MUSK's extra dependencies,
`command not found` (exit 127) from a missing `module load`, and long jobs
that fail at the end. Suggest testing the end of a pipeline on a small
subset first.

## Before the event: checklist

1. **Get `HF_TOKEN`** and accept the license on each gated model's HF page
   (CONCH, MUSK — see README.md § Setup). Not needed for the generic
   OpenCLIP/CLIP adapters unless the specific repo is gated.
2. **Install deps** on the login node (network access, fine there).
   `module load pytorch-conda/2.8` covers everything except `open_clip_torch`
   and the CONCH/MUSK packages, which go in with `pip install --user`. As of
   2026-10-16 the module sets `PYTHONNOUSERSITE=1`, which hides `--user`
   installs, so `unset PYTHONNOUSERSITE` right after loading it.
   `submit.sbatch`/`submit_serve.sbatch` load the module and unset it.
3. **Dry-run the serving infra without a GPU or gated access**, using the
   dependency-free `test` adapter, from the login node (HTTP only, no model
   load):
   ```bash
   sbatch --account=bfqi-delta-gpu submit_serve.sbatch test
   # once connection.json exists:
   python loadtest.py --endpoint http://<node>:<port> --api-key <key> \
       --teams 20 --images-per-team 3
   ```
   This validates request batching and per-team result isolation without
   touching a real model. **Verified 2026-09-17**: 20 teams, 0 errors, no
   cross-request mix-ups.
4. **Smoke-test each real model you intend to stage** on a real allocation
   with a handful of tiles before the event — `run_embeddings.py --model
   conch --input-dir <a few test tiles> --output smoke.csv`. Never on the
   login node (see below). Not yet done — CONCH/MUSK need `HF_TOKEN` and
   license acceptance, which the 2026-09-17 test run didn't have set up.
5. **Re-run `loadtest.py` against the real model** at your expected team
   count to check latency/OOM before committing to `PATHOLOGY_MAX_BATCH_SIZE`
   / `PATHOLOGY_MAX_WAIT_MS` defaults (see README.md's serve-mode section for
   what those trade off).
6. **Auto-resubmit chain: verified working 2026-09-17**, with one real gap
   found — see "Verified on a real Delta allocation" in README.md's
   serve-mode section. Short version: the resubmit itself, the API key
   carry-over, and `STOP_CHAIN-<job-name>` all work; the `--nodelist`-pinned
   successor queued for 7+ minutes on the shared partition without a
   reservation active. Re-test once the reservation exists — if it still
   queues for minutes even with dedicated nodes, revisit dropping
   `--nodelist` (see the design question below).
7. **Reservations (NCSA ticket sup-30434, created 2026-10-16)**:

   | Name | Use | Partition | Nodes | Window | Account | Magnetic |
   |---|---|---|---|---|---|---|
   | `sup-30434` | model servers | `gpuA40x4` | 1 (gpub055 as of Sat 07:30, 4× A40) | Sat 07:30 → Sun 18:00 | `bfqi-delta-gpu` | no — pass `--reservation` |
   | `sup-30434-17` | students | `gpuA100x4` | 4 | Sat 07:30 → Sun 07:59 | see below | yes |
   | `sup-30434-18` | students | `gpuA100x4` | 4 | Sun 08:00 → Sun 18:00 | see below | yes |
   | `sup-30434-cpu` | students (CPU) | `cpu` | 1 (cn004, 128 cores) | Sat 12:37 → Sun 18:00 | `bfqi-delta-cpu` | yes |

   `sup-30434-cpu` was added by Brett on Saturday after CPU jobs sat behind
   Delta's general `cpu` queue. OnDemand `cpu-interactive` sessions land on
   cn004 too (21 of the first 28 did).

   The three GPU ones are on `bfqi-delta-gpu` (Brett fixed the student ones on
   2026-10-16; they were first created on `bhws-delta-gpu`). Checked by
   submitting 1-minute jobs: all three pend with reason `(Reservation)`,
   including a job with no `--reservation` flag (magnetic). `sbatch
   --test-only` is NOT a reliable check here. It gave start estimates days
   out, on nodes outside the reservation. Since jobs pend until the
   reservation opens, the serve jobs can be submitted the night before and
   start at 07:30 on their own.

   The serve window (34.5h) is under the 48h cap, so the auto-resubmit chain
   isn't needed: `touch ~/.pathology/serve/STOP_CHAIN-<job-name>` *before*
   submitting so the job doesn't try to resubmit at the end of the event.
8. **Shared read-only datasets**: the event runs entirely under `bfqi`
   (students included; the earlier `bhws` group is not being used). As of
   2026-10-17 the data is in two places, both group `delta_bfqi`, read-only
   for students (group `r-x`, default ACL too, so new files arrive read-only):
   - `/projects/bfqi/shared/datasets/`: `NCT-CRC-HE-100K/`, `pathgen_dataset/`,
     `pathmmu/` (copied 2026-10-16).
   - `/work/nvme/bfqi/shared/datasets/quilt/`: full Quilt-1M
     (`images_part_1` … `images_part_10` + `quilt_1M_lookup-002.csv`). The Globus
     transfer finished ~11:00 on 2026-10-17 (653,213 images, 115 GB; it
     landed in `datasets/` and was moved into `quilt/`). Permissions checked:
     every file group-readable, none group-writable.
   Both paths are in `HACKATHON_COMPUTE.md` § 3.

**Open design question from the 2026-09-17 test**: is pinning the
resubmitted job to the same node (`--nodelist` in `resubmit_chain()`,
`submit_serve.sbatch`) worth the queue-delay risk it showed without a
reservation? The stable `current-<job-name>/connection.json` file already
handles a host that changes between links, so dropping `--nodelist` (and
the `--dependency=afterany` that exists to serialize onto that same node)
would very likely let Slurm schedule the successor on any free node much
faster — probably eliminating the downtime gap — at the cost of the host
no longer being guaranteed stable across a resubmission. Worth deciding
before the event, not during it.

## Load test, 2026-10-16 (CONCH, one A40, called from the login node)

60 teams x 3 images: 0 errors, p50 3.2s, p95 3.9s. 60 teams x 20 images
(1,200 images): 0 errors, p50 11.9s, p95 18.3s. That's about 65 images/s.
GPU memory was 4.1 of 46 GB. The limit is per-image preprocessing in the
single uvicorn event loop (serve.py `adapter.preprocess` in the handler), not
the GPU or batch size, so default PATHOLOGY_MAX_BATCH_SIZE/MAX_WAIT_MS are
kept. If it's ever needed, moving preprocessing to a thread pool is the fix.

## Never run this on the login node

Same rule as README.md, worth repeating here since it's the mistake most
likely to eat your allocation time mid-event: `serve.py` and
`run_embeddings.py` load the model and do real inference — **interactive
allocation or batch job only**. `client.py`, `loadtest.py`, and `pip install`
are lightweight/network-only and are fine on the login node.

## Day-of: staging a model

For the event itself, `./stage_event_models.sh submit` submits all four
servers to `sup-30434` (and turns off their resubmit chains). It's safe to
run the night before: jobs pend until 07:30. `./stage_event_models.sh status`
prints one block of endpoints and keys to post in #compute-issues. The manual
steps below are what it does for one model.

```bash
touch ~/.pathology/serve/STOP_CHAIN-conch   # single window, no resubmit
sbatch --account=bfqi-delta-gpu --partition=gpuA40x4 --reservation=sup-30434 \
       --time=34:25:00 --job-name=conch submit_serve.sbatch conch
tail -f logs/<job_id>.out   # watch for "Server ready on port ..."
cat ~/.pathology/serve/current-<job-name>/connection.json   # host, port, api_key -- stable across chain restarts, job name defaults to pathology-serve
```

`<model>` is `conch`, `musk`, or a generic spec — see next section. The API
key is auto-generated by the sbatch script into `connection.json`; you don't
set it yourself. Hand `host`/`port`/`api_key` to teams. Students can't read
your home directory, so post the values in the event channel.

Storage as of 2026-10-17 (`quota`): `/projects/bfqi` 124 GB of 500 GB soft
(308k of 850k files); `/work/nvme/bfqi` 500 GB of 4.88 TB soft (2.3M of 10.2M
files), so neither is near its limit. `/work/hdd/bfqi` was opened to
students on the night of 2026-10-17 with 2 TB (Brett), after NCSA's earlier
"avoid /work/hdd" request; 864 GB used by Sun 08:00.

The node has 4 A40s (48 GB each) and a server needs 1, so up to four models
can run side by side on `sup-30434`, one job each with a distinct
`--job-name` (and a matching `STOP_CHAIN-<job-name>`).

The auto-resubmit chain (README.md's "Multi-day events") is for windows
longer than 48h. This event's serve window isn't, hence the `STOP_CHAIN`
touch above. If the chain is ever used, don't just `scancel` at the end —
that only stops the *current* link; touch `STOP_CHAIN-<job-name>` too.

**If a team says the endpoint stopped responding**, check first whether the
SLURM job actually ended (`squeue -j <job_id>`) before assuming a bug —
`connection.json` is deleted automatically on job exit (including `scancel`
and walltime expiry), so "endpoint unreachable" and "job ended" are usually
the same event.

**Restarting one server with the same endpoint and key** (done 2026-10-17
after the CONCH job hit `OUT_OF_MEMORY` at 14:35, ~7h in, under `--mem=32G`.
Likely cause: very large requests. `serve.py` decodes and preprocesses every
image in a request in host RAM before batching, with no cap on images per
request, and handlers run concurrently; the student guide had told teams to
"send your whole batch at once". Guide and `example_student_usage.py` now say
≤64 per request. After the event: reject oversized requests in `serve.py`):

```bash
API_KEY=<that server's old key> SBATCH_MEM_PER_NODE=64G \
WALLTIME=<time left in the reservation, minus a margin> \
MODELS="conch=conch" ./stage_event_models.sh submit
```

`API_KEY` carries the old key into the new job, and the server takes the
lowest free port from its `SERVE_PORT`, so on the single reserved node it gets
its old port back. `WALLTIME` must end before the reservation does, or the job
never starts.

## A team wants a model you didn't stage

As of `5510fa3` (generic OpenCLIP/transformers-CLIP adapters), most CLIP-style
pathology or general-purpose models need **no new code** — just a spec
string. Decision order:

1. **Is it CONCH or MUSK?** → `--model conch` / `--model musk` (bespoke
   adapters already written).
2. **Is it published on HF in OpenCLIP's own hub format** (has an
   `open_clip_config.json`)? → `--model openclip:<hf-repo>`, e.g.
   `openclip:wisdomik/QuiltNet-B-32`. Confirm on the model's HF page; if
   unsure, try it — it fails loudly with a clear error, not silently.
3. **Is it OpenCLIP-trained but only published as a bare checkpoint file**
   (no hub config, e.g. PathGen-CLIP)? → `--model
   openclip:<arch>@<hf-repo>/<filename>`, e.g.
   `openclip:ViT-B-16@jamessyx/PathGen-CLIP/pathgenclip.pt`. You need to know
   the architecture name from the model's own card/README.
4. **Is it a `transformers`-native CLIP checkpoint** (plain CLIP variants,
   e.g. `openai/clip-vit-base-patch32`)? → `--model
   hfclip:<hf-repo>`.
5. **None of the above** (genuinely custom research code, another CONCH/MUSK
   situation) → needs a bespoke adapter added to `ADAPTERS` in `models.py`
   (see `load_conch`/`load_musk` as the pattern) — this is a scoped,
   contained change, but budget real time for it and don't attempt it live
   against a team's deadline. If there's no time, tell the team to get their
   own allocation and run their own script — same as any team going fully
   their own way.

**Restaging** to change models means resubmitting `submit_serve.sbatch` with
a new spec (a new job, new `connection.json`, new node/port — hand those
out again).

## Scaling beyond one instance

`serve.py` batches concurrent requests but is single-model, single-GPU. If
one instance is saturated (see `loadtest.py` p95 latency / OOM signals),
submit `submit_serve.sbatch` again with a different `--job-name` and split
teams across the two endpoints — README.md's serve-mode section covers the
`PATHOLOGY_MAX_BATCH_SIZE`/`PATHOLOGY_MAX_WAIT_MS` tuning knobs for the
memory/latency trade-off on each instance.

## Env var reference (set via `submit_serve.sbatch` or by hand for testing)

| Var | Default | Meaning |
| --- | --- | --- |
| `PATHOLOGY_MODEL` | `conch` | Model spec — see decision list above |
| `PATHOLOGY_DEVICE` | `cuda` if available | `cuda` or `cpu` |
| `PATHOLOGY_MAX_BATCH_SIZE` | `32` | Max images per GPU forward pass |
| `PATHOLOGY_MAX_WAIT_MS` | `50` | Max wait to fill a batch before running it anyway |
| `API_KEY` | (auto-generated by sbatch) | Bearer token teams must send |
| `HF_TOKEN` | — | Needed for gated models (CONCH, MUSK, or any gated `openclip:`/`hfclip:` repo) |

## Where things live

- Connection info: `~/.pathology/serve/<job_id>/connection.json` (mode 600,
  auto-deleted on job exit)
- Logs: `logs/<job_id>.out` / `.err` in this directory
- Student-facing docs to hand out: `STUDENT_GUIDE.md` + `example_student_usage.py`
- Setup/architecture docs (for you, longer-form): `README.md`
