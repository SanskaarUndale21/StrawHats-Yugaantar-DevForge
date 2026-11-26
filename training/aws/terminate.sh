#!/usr/bin/env bash
# Fetch results, then terminate the instance (stops billing). Pass --no-fetch to skip the copy.
set -euo pipefail
cd "$(dirname "$0")/../.."
source training/aws/.instance
if [ "${1:-}" != "--no-fetch" ]; then bash training/aws/fetch.sh || true; fi
AWS_DEFAULT_REGION=$REGION aws ec2 terminate-instances --instance-ids "$ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text
rm -f training/aws/.instance
echo "terminated $ID"
