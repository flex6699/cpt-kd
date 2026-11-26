"""
Run a list of training jobs on several GPUs, a few at a time per GPU.
Whenever a job finishes, the next one starts on that GPU. Run it inside tmux.

Jobs file: one command per line, WITHOUT --gpu (the queue adds it). Lines starting with # are ignored.
  python train_student.py --data data/tea_kd/cub --method kd --seed 1
  python train_student.py --data data/tea_kd/cub --method cpt --seed 1

Usage
  tmux new -s queue
  conda activate teakd && cd ~/ml_research
  python run_queue.py jobs_cub_batch2.txt --gpus 1 2 --per-gpu 2
  (Ctrl-b then d to leave it running; tmux attach -t queue to come back)

Each job's output goes to logs/<job name>.log. Finished runs are skipped by train_student.py itself,
so re-running the same jobs file after an interruption only does what is missing.
Before starting a job, the queue waits until the GPU has at least --min-free-gb free (shared GPUs).
"""
import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
import time


def job_name(cmd):
    """Readable log name from the command, e.g. cub_cpt_trust-temp4_seed1."""
    toks = shlex.split(cmd)
    parts = []
    if "--data" in toks:
        parts.append(os.path.basename(toks[toks.index("--data") + 1].rstrip("/")))
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--") and t not in ("--data", "--gpu", "--workers"):
            val = toks[i + 1] if i + 1 < len(toks) and not toks[i + 1].startswith("--") else ""
            parts.append(t[2:] + val if t != "--method" else val)
            i += 2 if val else 1
        else:
            i += 1
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", "_".join(parts))[:150] or "job"


def free_gb(gpu):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
                              "-i", str(gpu)], capture_output=True, text=True, timeout=20)
        return float(out.stdout.strip().splitlines()[0]) / 1024
    except Exception:
        return None                    # nvidia-smi unavailable: do not block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs_file")
    ap.add_argument("--gpus", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--per-gpu", type=int, default=2, help="jobs running at the same time on each GPU")
    ap.add_argument("--min-free-gb", type=float, default=9.0, help="wait until a GPU has this much free memory")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--poll", type=float, default=5.0)
    args = ap.parse_args()

    with open(args.jobs_file) as f:
        jobs = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
    os.makedirs(args.logs, exist_ok=True)
    print(f"{len(jobs)} jobs, GPUs {args.gpus}, {args.per_gpu} per GPU -> up to "
          f"{len(args.gpus) * args.per_gpu} at a time. Logs in {args.logs}/\n", flush=True)

    pending = list(enumerate(jobs, 1))
    running = {}                       # pid -> (popen, gpu, name, start, logfile, number)
    done, failed = [], []

    def stop_all(*_):
        print("\nStopping all running jobs ...", flush=True)
        for p, *_ in running.values():
            p.terminate()
        sys.exit(1)
    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)

    t_all = time.time()
    while pending or running:
        # collect finished jobs
        for pid in list(running):
            p, gpu, name, start, logf, num = running[pid]
            if p.poll() is not None:
                logf.close()
                mins = (time.time() - start) / 60
                status = "done" if p.returncode == 0 else f"FAILED (exit {p.returncode})"
                (done if p.returncode == 0 else failed).append(name)
                print(f"[{time.strftime('%H:%M')}] #{num} {name} on GPU {gpu}: {status} after {mins:.1f} min",
                      flush=True)
                del running[pid]
        # start new jobs where there is room
        for gpu in args.gpus:
            while pending and sum(1 for v in running.values() if v[1] == gpu) < args.per_gpu:
                fg = free_gb(gpu)
                if fg is not None and fg < args.min_free_gb:
                    break              # not enough memory right now; try again next poll
                num, cmd = pending.pop(0)
                name = job_name(cmd)
                logf = open(os.path.join(args.logs, f"{name}.log"), "w")
                p = subprocess.Popen(f"{cmd} --gpu {gpu}", shell=True, stdout=logf, stderr=subprocess.STDOUT)
                running[p.pid] = (p, gpu, name, time.time(), logf, num)
                print(f"[{time.strftime('%H:%M')}] #{num} start {name} on GPU {gpu}"
                      + (f" ({fg:.1f} GB free)" if fg is not None else ""), flush=True)
                time.sleep(3)          # let it allocate memory before checking again
        time.sleep(args.poll)

    print(f"\nAll jobs finished in {(time.time() - t_all) / 60:.1f} min: {len(done)} ok, {len(failed)} failed.")
    for name in failed:
        print(f"  FAILED: {name}   (see {args.logs}/{name}.log)")


if __name__ == "__main__":
    main()
