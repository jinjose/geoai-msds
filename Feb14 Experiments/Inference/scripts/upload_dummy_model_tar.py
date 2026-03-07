import argparse, io, tarfile
import boto3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--key", required=True)
    args = ap.parse_args()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        content = b"Dummy artifact. Real model baked into container image."
        info = tarfile.TarInfo(name="README.txt")
        info.size = len(content)
        tf.addfile(info, io.BytesIO(content))
    buf.seek(0)

    s3 = boto3.client("s3", region_name=args.region)
    s3.put_object(Bucket=args.bucket, Key=args.key, Body=buf.read())
    print(f"Uploaded s3://{args.bucket}/{args.key}")

if __name__ == "__main__":
    main()
