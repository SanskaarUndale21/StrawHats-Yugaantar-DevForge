#!/usr/bin/env bash
# Run LOCALLY (Git Bash / Linux / macOS) with a configured AWS CLI (`aws configure`).
# Launches a CPU-heavy EC2 instance (spot by default), installs everything, uploads your local
# checkpoints and starts the training pipeline in tmux.
#
#   bash training/aws/launch.sh                         # c7i.8xlarge spot, full pipeline
#   TYPE=c7i.16xlarge bash training/aws/launch.sh
#   MARKET=on-demand PHASE=rl bash training/aws/launch.sh
set -euo pipefail
export MSYS_NO_PATHCONV=1   # Git Bash on Windows: keep /aws/... and /dev/sda1 as-is
cd "$(dirname "$0")/../.."
export AWS_PROFILE="${AWS_PROFILE:-doom}"   # dedicated profile only; never the default credentials
if ! aws configure list-profiles 2>/dev/null | grep -qx "$AWS_PROFILE"; then
  echo "AWS profile '$AWS_PROFILE' not found. Create it with:  aws configure --profile doom" >&2
  exit 1
fi

TYPE="${TYPE:-c7i.8xlarge}"
MARKET="${MARKET:-spot}"
MAXPRICE="${MAXPRICE:-1.00}"      # spot: never pay more than this per hour
MAX_HOURS="${MAX_HOURS:-16}"      # instance terminates itself after this, no matter what
UPLOAD="${UPLOAD:-latest.pt student.pt distill.pt}"
PHASE="${PHASE:-full}"
REGION="${REGION:-$(aws configure get region || echo us-east-1)}"
KEY="${KEY:-doom-train-$AWS_PROFILE-$REGION}"
SG_NAME="doom-train-ssh"
KEYFILE="$HOME/.ssh/$KEY.pem"
export AWS_DEFAULT_REGION="$REGION"
STATE="training/aws/.instance-${NAME:-doom-train}"   # gitignored: instance id / ip

echo "profile=$AWS_PROFILE name=${NAME:-doom-train} region=$REGION type=$TYPE market=$MARKET maxprice=$MAXPRICE max_hours=$MAX_HOURS phase=$PHASE seed=${SEED:-1}"
aws sts get-caller-identity --query Account --output text >/dev/null

# key pair
if [ ! -f "$KEYFILE" ]; then
  mkdir -p "$HOME/.ssh"
  aws ec2 delete-key-pair --key-name "$KEY" >/dev/null 2>&1 || true
  aws ec2 create-key-pair --key-name "$KEY" --query KeyMaterial --output text > "$KEYFILE"
  chmod 600 "$KEYFILE" 2>/dev/null || true
fi

# security group: SSH from this machine's IP only
MYIP=$(curl -s https://checkip.amazonaws.com | tr -d '\r\n')
SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values=$SG_NAME --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
if [ "$SG" = "None" ] || [ -z "$SG" ]; then
  SG=$(aws ec2 create-security-group --group-name $SG_NAME --description "ssh for doom training" --query GroupId --output text)
fi
aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP/32" >/dev/null 2>&1 || true

# AMI_SSM: set to a GPU image (NVIDIA driver preinstalled) for a box whose learner runs on a GPU
AMI=$(aws ssm get-parameters --names "${AMI_SSM:-/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id}" \
      --query 'Parameters[0].Value' --output text)
MKT=()
if [ "$MARKET" = "spot" ]; then
  MKT=(--instance-market-options "MarketType=spot,SpotOptions={MaxPrice=$MAXPRICE,SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}")
fi
PLACE=()
if [ -n "${AZ:-}" ]; then PLACE=(--placement "AvailabilityZone=$AZ"); fi   # pick a zone that has spot capacity
ID=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY" --security-group-ids "$SG" --instance-initiated-shutdown-behavior terminate "${PLACE[@]}" \
     --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=${DISK:-40},VolumeType=gp3,DeleteOnTermination=true}" \
     --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${NAME:-doom-train}}]" "${MKT[@]}" \
     --query 'Instances[0].InstanceId' --output text)
echo "instance $ID starting..."
aws ec2 wait instance-running --instance-ids "$ID"
IP=$(aws ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
printf 'ID=%s\nIP=%s\nREGION=%s\nKEYFILE=%s\n' "$ID" "$IP" "$REGION" "$KEYFILE" > "$STATE"
SSH="ssh -i $KEYFILE -o StrictHostKeyChecking=no -o ConnectTimeout=10 ubuntu@$IP"
echo "waiting for ssh on $IP ..."
until $SSH true 2>/dev/null; do sleep 5; done
# hard stop: OS shutdown -> instance terminates (shutdown behaviour = terminate)
$SSH "sudo shutdown -h +$((MAX_HOURS * 60)) >/dev/null 2>&1 || true"
echo "self-terminate timer: $MAX_HOURS h"

scp -i "$KEYFILE" -o StrictHostKeyChecking=no training/aws/setup_instance.sh ubuntu@$IP:~/
$SSH "sed -i 's/\r\$//' ~/setup_instance.sh; bash ~/setup_instance.sh https://github.com/SanskaarUndale21/StrawHats-Yugaantar-DevForge.git ${BRANCH:-main}"
# carry local progress over (whatever exists)
for f in $UPLOAD; do
  if [ -f "training/checkpoints/$f" ]; then
    scp -i "$KEYFILE" -o StrictHostKeyChecking=no "training/checkpoints/$f" ubuntu@$IP:~/doom/training/checkpoints/
  fi
done
$SSH "tmux new-session -d -s train 'bash ~/doom/training/aws/run.sh $PHASE ${DSTEPS:-8000000} 300000000 ${SEED:-1}; bash'"
echo
echo "training started on $TYPE ($MARKET)  id=$ID  ip=$IP"
echo "  watch:   $SSH -t tmux attach -t train      (detach: Ctrl-b d)"
echo "  status:  bash training/aws/status.sh"
echo "  fetch:   bash training/aws/fetch.sh        (copies weights + logs back)"
echo "  STOP:    bash training/aws/terminate.sh    (billing continues until you do this!)"
