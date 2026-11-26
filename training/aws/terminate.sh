#!/usr/bin/env bash
# Fetch results, then terminate the instance (stops billing). Pass --no-fetch to skip the copy.
set -euo pipefail
cd "$(dirname "$0")/../.."
export AWS_PROFILE="${AWS_PROFILE:-doom}"   # dedicated profile only; never the default credentials
if ! aws configure list-profiles 2>/dev/null | grep -qx "$AWS_PROFILE"; then
  echo "AWS profile '$AWS_PROFILE' not found. Create it with:  aws configure --profile doom" >&2
  exit 1
fi
source "training/aws/.instance-${NAME:-doom-train}"
if [ "${1:-}" != "--no-fetch" ]; then bash training/aws/fetch.sh || true; fi
AWS_DEFAULT_REGION=$REGION aws ec2 terminate-instances --instance-ids "$ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text
rm -f "training/aws/.instance-${NAME:-doom-train}"
echo "terminated $ID"
