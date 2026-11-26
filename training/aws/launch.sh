#!/usr/bin/env bash
# Run LOCALLY (Git Bash / Linux / macOS) with a configured AWS CLI (`aws configure`).
# Launches a CPU-heavy EC2 instance (spot by default), installs everything, uploads your local
# checkpoints and starts the training pipeline in tmux.
#
#   bash training/aws/launch.sh                         # c7i.8xlarge spot, full pipeline
#   TYPE=c7i.16xlarge bash training/aws/launch.sh
#   MARKET=on-demand PHASE=rl bash training/aws/launch.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

TYPE="${TYPE:-c7i.8xlarge}"
MARKET="${MARKET:-spot}"
PHASE="${PHASE:-full}"
REGION="${REGION:-$(aws configure get region || echo us-east-1)}"
KEY="${KEY:-doom-train}"
SG_NAME="doom-train-ssh"
KEYFILE="$HOME/.ssh/$KEY.pem"
export AWS_DEFAULT_REGION="$REGION"
STATE=training/aws/.instance   # gitignored: holds the instance id / ip

echo "region=$REGION type=$TYPE market=$MARKET phase=$PHASE"
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

AMI=$(aws ssm get-parameters --names /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
      --query 'Parameters[0].Value' --output text)
MKT=()
if [ "$MARKET" = "spot" ]; then
  MKT=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
fi
ID=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY" --security-group-ids "$SG" \
     --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=40,VolumeType=gp3,DeleteOnTermination=true}' \
     --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=doom-train}]' "${MKT[@]}" \
     --query 'Instances[0].InstanceId' --output text)
echo "instance $ID starting..."
aws ec2 wait instance-running --instance-ids "$ID"
IP=$(aws ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
printf 'ID=%s\nIP=%s\nREGION=%s\nKEYFILE=%s\n' "$ID" "$IP" "$REGION" "$KEYFILE" > "$STATE"
SSH="ssh -i $KEYFILE -o StrictHostKeyChecking=no -o ConnectTimeout=10 ubuntu@$IP"
echo "waiting for ssh on $IP ..."
until $SSH true 2>/dev/null; do sleep 5; done

scp -i "$KEYFILE" -o StrictHostKeyChecking=no training/aws/setup_instance.sh ubuntu@$IP:~/
$SSH "bash ~/setup_instance.sh"
# carry local progress over (whatever exists)
for f in latest.pt student.pt distill.pt; do
  if [ -f "training/checkpoints/$f" ]; then
    scp -i "$KEYFILE" -o StrictHostKeyChecking=no "training/checkpoints/$f" ubuntu@$IP:~/doom/training/checkpoints/
  fi
done
$SSH "tmux new-session -d -s train 'bash ~/doom/training/aws/run.sh $PHASE; bash'"
echo
echo "training started on $TYPE ($MARKET)  id=$ID  ip=$IP"
echo "  watch:   $SSH -t tmux attach -t train      (detach: Ctrl-b d)"
echo "  status:  bash training/aws/status.sh"
echo "  fetch:   bash training/aws/fetch.sh        (copies weights + logs back)"
echo "  STOP:    bash training/aws/terminate.sh    (billing continues until you do this!)"
