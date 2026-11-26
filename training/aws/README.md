# Training on AWS

Doom RL is bottlenecked by CPU, because the game engine and sensors run on CPU. A many-core
instance gives a near-linear speed-up and needs no GPU, since the network is small.

## Cost (us-east-1, approximate; check the EC2 pricing page)

| Instance | vCPU | On-demand $/h | Spot $/h (approx) | Speed (approx) | Hours for $50 |
|---|---|---|---|---|---|
| **c7i.8xlarge** (default) | 32 | 1.43 | 0.5–0.6 | ~2.5k steps/s | 35 on-demand / ~90 spot |
| c7i.16xlarge | 64 | 2.86 | 1.0–1.2 | ~5k steps/s | 17 / ~45 |
| g4dn.8xlarge | 32 + T4 | 2.18 | ~0.7 | ~2.5k steps/s | 23 / ~70 |

Disk is 40 GB gp3, about $0.10 a day. Data transfer is negligible.

Typical run: 8M distillation steps plus 50M RL steps on c7i.8xlarge spot comes to about 6–7 h,
roughly $4.

## Before the first launch

1. Install the AWS CLI and run `aws configure` with your access key and region (e.g. `us-east-1`).
2. In the console, open **Service Quotas → EC2** and ask for at least 32 vCPU in both of these:
   - *Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances*
   - *All Standard (A, C, D, H, I, M, R, T, Z) Spot Instance Requests*

   New accounts often start at 5 vCPU.
3. Check that your credits cover EC2. Promotional credits usually do.

## Use (from Git Bash in the repo root)

```bash
bash training/aws/launch.sh                       # c7i.8xlarge spot, distill -> RL, uploads local checkpoints
TYPE=c7i.16xlarge bash training/aws/launch.sh     # bigger box
MARKET=on-demand bash training/aws/launch.sh      # no spot interruptions
PHASE=rl bash training/aws/launch.sh              # skip distillation

bash training/aws/status.sh                       # latest log lines
bash training/aws/fetch.sh                        # copy weights/logs to training/checkpoints/aws/
bash training/aws/terminate.sh                    # fetch, then TERMINATE (billing stops)
```

**Always run `terminate.sh` when you are done.** A forgotten c7i.8xlarge costs about $34 a day
on-demand.

Spot instances can be reclaimed by AWS. When that happens, `fetch.sh` whatever is left, then
`launch.sh` again. The new box uploads the local checkpoints, and `run.sh` resumes distillation
or RL from them.
