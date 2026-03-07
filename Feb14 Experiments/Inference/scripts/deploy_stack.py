import argparse, subprocess, json

def sh(cmd):
    print(" ".join(cmd))
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--stack", default="geoai-byoc")
    ap.add_argument("--image-uri", required=True)
    ap.add_argument("--bucket-name", default="")
    ap.add_argument("--schedule", default="cron(0 2 * * ? *)")
    ap.add_argument("--instance-type", default="ml.m5.large")
    ap.add_argument("--instance-count", type=int, default=1)
    args = ap.parse_args()

    params = [
        f"ProjectName={args.stack}",
        f"EcrImageUri={args.image_uri}",
        f"BucketName={args.bucket_name}",
        f"ScheduleExpression={args.schedule}",
        f"InstanceType={args.instance_type}",
        f"InstanceCount={args.instance_count}",
    ]

    sh([
        "aws","cloudformation","deploy",
        "--region", args.region,
        "--template-file","infra/cloudformation.yaml",
        "--stack-name", args.stack,
        "--capabilities","CAPABILITY_NAMED_IAM",
        "--parameter-overrides", *params
    ])

    out = subprocess.check_output([
        "aws","cloudformation","describe-stacks",
        "--region", args.region,
        "--stack-name", args.stack,
        "--query","Stacks[0].Outputs",
        "--output","json"
    ])
    print(json.dumps(json.loads(out), indent=2))

if __name__ == "__main__":
    main()
