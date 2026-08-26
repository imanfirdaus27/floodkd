# push_to_github.ps1
# Run from the project folder in PowerShell:
#     cd "C:\Users\firda\Desktop\Master\Sem 3 20252026\MAXU 5214\9 floodsegmentation"
#     .\push_to_github.ps1
#
# The remote is already set to github.com/imanfirdaus27/floodkd, so no argument
# is needed. Pass -RepoUrl only if you want to point it somewhere else.

param(
    [string]$RepoUrl = ""
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# A lock left behind by another process stops git add before it starts
if (Test-Path ".git\index.lock") { Remove-Item ".git\index.lock" -Force }
if (Test-Path "__t.tmp")        { Remove-Item "__t.tmp" -Force }

git config user.name  "Muhammad Iman Firdaus Bin Md Rostan"
git config user.email "firdausrostan@gmail.com"

git add -A

Write-Host "`n--- Files to be committed ---" -ForegroundColor Cyan
git status --short

# Nothing to do if the tree is clean; committing would fail and stop the script
$pending = git status --porcelain
if ([string]::IsNullOrWhiteSpace($pending)) {
    Write-Host "`nNothing to commit. Pushing whatever is already committed." -ForegroundColor Yellow
} else {
    git commit -m @"
Teacher, distillation runs, and tools for the two additional datasets

Experiments
- teacher_s2: Sentinel-2 optical teacher, best validation IoU 0.8261 at epoch 23
- student_kd: distillation with alpha 1.0, beta 1.0, no gate, 0.6266 at epoch 29
- student_kd_v2: beta 0.1 with the confidence gate at 0.7, 0.6206 at epoch 29
- against the Sentinel-1 baseline of 0.6244, a modality gap of 0.2017 IoU

Fixes
- engine.py: mixed precision now wraps the distillation forward pass, which it
  never did, so the two networks ran in fp32 and exhausted a 4 GB card
- engine.py: the three loss terms are logged separately instead of only their sum

Tools
- scripts/fetch_s1s2.py: reads S1S2-Water scenes out of the Zenodo zips over HTTP
  range requests, so one scene costs about 2 GB rather than the full 165 GB
- scripts/fetch_sturm.py: the same for STURM-Flood, matching tiles to flood maps
  by name, a few hundred kilobytes instead of a 3.71 GB archive
- scripts/explore_s1s2.py, scripts/explore_sturm.py: overview figures for both

Data and checkpoints stay out of the repository via .gitignore.
"@
}

git branch -M main

if ($RepoUrl -ne "") {
    $ErrorActionPreference = "Continue"
    git remote remove origin 2>&1 | Out-Null
    $ErrorActionPreference = "Stop"
    git remote add origin $RepoUrl
}

$target = git remote get-url origin
git push -u origin main

Write-Host "`nDone. Pushed to $target" -ForegroundColor Green
