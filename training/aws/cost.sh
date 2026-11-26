#!/usr/bin/env bash
# Running cost of every doom-training instance in a profile (spot price x hours since launch).
#   AWS_PROFILE=nilesh bash training/aws/cost.sh
#   AWS_PROFILE=doom REGION=eu-north-1 bash training/aws/cost.sh
set -euo pipefail
export MSYS_NO_PATHCONV=1
export AWS_PROFILE="${AWS_PROFILE:-nilesh}"
REGION="${REGION:-$(aws configure get region || echo us-east-1)}"
export AWS_DEFAULT_REGION="$REGION"
NOW=$(date -u +%s)
TOTAL=0
while read -r ID TYPE AZ LT LIFE NAME; do
  [ -z "$ID" ] && continue
  START=$(date -u -d "$LT" +%s)
  HOURS=$(awk -v a=$NOW -v b=$START 'BEGIN{printf "%.2f",(a-b)/3600}')
  if [ "$LIFE" = "spot" ]; then
    PRICE=$(aws ec2 describe-spot-price-history --instance-types "$TYPE" --product-descriptions Linux/UNIX \
            --availability-zone "$AZ" --start-time "$LT" --query 'SpotPriceHistory[].SpotPrice' --output text \
            | tr '\t' '\n' | awk '{s+=$1;n++} END{printf "%.4f", n?s/n:0}')
  else
    PRICE="${ONDEMAND_PRICE:-0.10}"   # free-plan m7i-flex.large: paid from credits
  fi
  COST=$(awk -v h=$HOURS -v p=$PRICE 'BEGIN{printf "%.2f", h*p + h*40*0.08/730}')
  TOTAL=$(awk -v t=$TOTAL -v c=$COST 'BEGIN{printf "%.2f", t+c}')
  printf "%-10s %-14s %-8s %6s h  x  \$%s/h  =  \$%s\n" "$NAME" "$TYPE" "$LIFE" "$HOURS" "$PRICE" "$COST"
done < <(aws ec2 describe-instances --filters Name=instance-state-name,Values=running \
          --query 'Reservations[].Instances[].[InstanceId,InstanceType,Placement.AvailabilityZone,LaunchTime,InstanceLifecycle||`on-demand`,Tags[?Key==`Name`]|[0].Value]' \
          --output text)
echo "running total so far: \$$TOTAL  (profile $AWS_PROFILE, $REGION; disk included)"
