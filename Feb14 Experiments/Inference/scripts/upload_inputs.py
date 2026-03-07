import argparse, os, glob, json, subprocess
import boto3
import datetime

def get_outputs(region, stack):
    out = subprocess.check_output([
        "aws","cloudformation","describe-stacks",
        "--region", region,
        "--stack-name", stack,
        "--query","Stacks[0].Outputs",
        "--output","json"
    ])
    return {o["OutputKey"]: o["OutputValue"] for o in json.loads(out)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--stack", required=True)
    ap.add_argument("--local-dir", required=True)
    ap.add_argument("--run-date", default=None)
    args = ap.parse_args()

    run_date = args.run_date or datetime.datetime.utcnow().strftime("%Y-%m-%d")
    outs = get_outputs(args.region, args.stack)
    bucket = outs["BucketOut"]

    # Upload dummy model tar for SageMaker ModelDataUrl
    subprocess.check_call([
        "python","scripts/upload_dummy_model_tar.py",
        "--region", args.region,
        "--bucket", bucket,
        "--key", "model/model.tar.gz"
    ])

    s3 = boto3.client("s3", region_name=args.region)
    files = glob.glob(os.path.join(args.local_dir, "features_*.csv"))
    if not files:
        raise SystemExit("No features_*.csv found")

    for f in files:
        key = f"features/run_date={run_date}/{os.path.basename(f)}"
        s3.upload_file(f, bucket, key)
        print("Uploaded", f"s3://{bucket}/{key}")

    print("Done. Invoke Lambda or wait for schedule.")

if __name__ == "__main__":
    main()
