# Git History - IsoScopeXX Project

1. This file tracks all significant git operations and commits for the IsoScopeXX project. 

2. Do not commit versions of this file.

3. Only add operation of commit, checkout, merge, and branch creation.


[2025-10-30 10:15:23] git checkout -b feature-branch
  └─> Branch: feature-branch
[2025-10-30 10:16:45] git add .
  └─> Branch: feature-branch
[2025-10-30 10:17:02] git commit -m "Add new feature"
  └─> Branch: feature-branch
[2025-10-30 10:18:30] git checkout main
  └─> Branch: main
[2025-10-30 10:19:15] git merge feature-branch
  └─> Branch: main

---

## Actual Git Operations

[2025-10-30 18:29:25] git init
  └─> Branch: main
  └─> Action: Initialized empty Git repository

[2025-10-30 18:29:34] git commit -m "Initial commit: IsoScopeXX project setup"
  └─> Branch: main (root-commit)
  └─> Commit: 5b9ca45
  └─> Stats: 82 files changed, 20884 insertions(+)
  └─> Summary: Complete VQGAN-based 3D medical image enhancement framework

[2025-10-31 19:12:30] git commit -m "start fixing OmegaConf"
  └─> Branch: main
  └─> Commit: d58228c
  └─> Stats: 8 files changed, 69 insertions(+), 59 deletions(-)
  └─> Summary: Migrate from JSON to YAML configuration
  └─> Changes:
      • Created env/aisr.yaml (YAML config)
      • Deleted env/jsn/aisr.json
      • Removed unused --hbranch argument from ae0iso0tccutvqq.py
      • Updated train.py to use yaml.safe_load() instead of json.load()
      • Changed --jsn argument to --yaml in utils/get_args.py
      • Updated run.sh to use --yaml flag

[2025-10-31 19:52:00] git reset --hard HEAD
  └─> Branch: main
  └─> Action: Reset to commit d58228c (start fixing OmegaConf)
  └─> Summary: Reverted uncommitted OmegaConf experimental changes
  └─> Note: Decided to keep simpler YAML approach without OmegaConf for now

