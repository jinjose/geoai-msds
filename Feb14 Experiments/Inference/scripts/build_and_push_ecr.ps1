param(
  [Parameter(Mandatory=$true)][string]$Region,
  [string]$Repo="geoai-lgbm-group6-byoc",
  [string]$Tag="v1"
)
$AccountId = (aws sts get-caller-identity --query Account --output text)
aws ecr describe-repositories --repository-names $Repo --region $Region *> $null
if ($LASTEXITCODE -ne 0) { aws ecr create-repository --repository-name $Repo --region $Region *> $null }
aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin "$AccountId.dkr.ecr.$Region.amazonaws.com"
docker build -t "$Repo`:$Tag" -f docker/Dockerfile .
docker tag "$Repo`:$Tag" "$AccountId.dkr.ecr.$Region.amazonaws.com/$Repo`:$Tag"
docker push "$AccountId.dkr.ecr.$Region.amazonaws.com/$Repo`:$Tag"
Write-Host "ECR Image URI: $AccountId.dkr.ecr.$Region.amazonaws.com/$Repo`:$Tag"
