# Publish to Choongmh

Destination account: https://github.com/Choongmh

Suggested independent repository name: `dwc2026-portfolio`. Local preparation is not a successful upload. The source URL currently returns 404 to the signed-out browser; authenticate before comparing upstream branches and remote-only files.

## Finish locally and publish

Run in a normal authenticated terminal from this project folder. Review `git status` first. These commands include the large datasets using the supplied `.gitattributes` and Git LFS; do not remove the LFS rules.

```powershell
git lfs install --local
git add .
git commit -m "Include complete challenge data and study materials"
```

The four task branches already exist. After the data commit, fast-forward each to the complete integrated snapshot before first publication:

```powershell
git switch t1-data
git merge --ff-only main
git switch t2-proxy
git merge --ff-only main
git switch t3-forecast
git merge --ff-only main
git switch t4-storytelling
git merge --ff-only main
git switch main
```

If a fast-forward fails, inspect the branch history rather than force-updating it. The initial preparation branches are starting points for subsequent study, not invented historical work.

Create an empty public repository named `dwc2026-portfolio` under Choongmh in GitHub, without adding an extra README or license. Once it exists, use:

```powershell
git remote add origin https://github.com/Choongmh/dwc2026-portfolio.git
git push -u origin main
git push origin t1-data t2-proxy t3-forecast t4-storytelling
git lfs push --all origin
```

If origin already exists, inspect `git remote -v` rather than adding it again. Git LFS availability and account allowance must support approximately 1.7 GB of binary material. Do not approve an unexpected paid upgrade automatically.

Verify the public repository in a signed-out browser. Verify four task branches, the learning document, the code, and all LFS files. A fresh clone followed by `git lfs pull` should recover real Excel files, not pointer text.

Compare the authenticated upstream's branches and files with `docs/SOURCE_MAP.md` before describing this edition as a verified full remote copy.
