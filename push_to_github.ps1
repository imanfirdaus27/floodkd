# push_to_github.ps1
# Run this from the project folder in PowerShell:
#     cd "C:\Users\firda\Desktop\Master\Sem 3 20252026\MAXU 5214\9 floodsegmentation"
#     .\push_to_github.ps1 -RepoUrl "https://github.com/<username>/floodkd.git"

param(
    [Parameter(Mandatory = $true)]
    [string]$RepoUrl
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# Clear the stale lock left behind by the sandbox
if (Test-Path ".git\index.lock") { Remove-Item ".git\index.lock" -Force }
if (Test-Path "__t.tmp")        { Remove-Item "__t.tmp" -Force }

git config user.name  "Muhammad Iman Firdaus Bin Md Rostan"
git config user.email "firdausrostan@gmail.com"

git add -A

Write-Host "`n--- Files to be committed ---" -ForegroundColor Cyan
git status --short

git commit -m @"
Initial commit: cross-modal knowledge distillation for flood segmentation

- run.py: single CLI entry point (explore/download/stats/train/distill/eval)
- src/: config, data, models, losses, metrics, engine
- configs/default.yaml: all hyperparameters in one place
- baseline Sentinel-1 run log (30 epochs) and qualitative figures

Data chips (~14 GB) and model checkpoints excluded via .gitignore.
"@

git branch -M main
git remote remove origin 2>$null
git remote add origin $RepoUrl
git push -u origin main

Write-Host "`nDone. Repo pushed to $RepoUrl" -ForegroundColor Green
