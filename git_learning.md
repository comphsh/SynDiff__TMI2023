# Git 命令学习记录 — SynDiff 实验适配

本文档记录在 SynDiff 代码适配过程中使用的所有 Git 命令及其解释。

---

## 1. 仓库状态检查

```bash
# 查看工作区状态（已修改/已暂存/未跟踪文件）
git status

# 查看当前分支
git branch

# 查看所有分支（含远程）
git branch -a

# 查看提交历史
git log --oneline
```

**解释**: `git status` 显示工作区和暂存区状态，是最常用的诊断命令。`git branch -a` 列出所有本地和远程分支。

---

## 2. 创建实验分支

```bash
# 创建并切换到新分支 experiment/SynDiff_adapt
git checkout -b experiment/SynDiff_adapt
```

**解释**: `git checkout -b <branch_name>` 等价于 `git branch <name> && git checkout <name>`。我们使用 `experiment/<method>_adapt` 命名约定，清晰标识这是对比方法适配实验。

**注意**: 如果分支已存在，`git checkout -b` 会失败。此时可用 `git checkout experiment/SynDiff_adapt` 直接切换。

---

## 3. 文件暂存与提交

```bash
# 将所有修改和新增文件加入暂存区
git add dataset.py train.py utils/EMA.py datalist/

# 提交暂存区内容，附带描述性信息
git commit -m "base: existing modifications to dataset, train, EMA, and datalist files"

# 只暂存特定文件
git add dataset.py .gitignore

# 查看即将提交的内容
git diff --staged
```

**解释**:
- `git add <files>`: 将文件更改添加到暂存区（staging area），准备提交
- `git commit -m "message"`: 将暂存区内容永久记录到 Git 历史
- 提交信息遵循约定式提交 (Conventional Commits) 格式: `<type>: <description>`
- 使用 `Co-Authored-By` 尾部标记辅助工具

---

## 4. 提交信息规范

```bash
# 数据加载变更
git commit -m "feat: replace .mat data loading with MONAI-based nifti DataLoader"

# 训练适配变更
git commit -m "feat: adapt training for single-GPU with TensorBoard logging"

# 功能新增
git commit -m "feat: add eval.py for 14 missing modality pattern inference"

# 脚本新增
git commit -m "feat: add run_train.sh and run_eval.sh launch scripts"
```

**解释**: 采用 Conventional Commits 规范:
- `feat:` — 新功能 (feature)
- `fix:` — 错误修复
- `refactor:` — 代码重构
- `docs:` — 文档更新
- `chore:` — 杂项任务

---

## 5. 忽略文件配置

```bash
# 编辑 .gitignore 文件，添加不需要跟踪的文件模式
# 例如: results/, .idea/, __pycache__/, .DS_Store
```

**解释**: `.gitignore` 告诉 Git 忽略某些文件和目录。实验结果 `results/` 加入 `.gitignore` 避免将大文件提交到仓库。IDE 配置文件 `.idea/` 和 Python 缓存 `__pycache__/` 也应忽略。

---

## 6. 文件权限管理

```bash
# 为 shell 脚本添加可执行权限
chmod +x run_train.sh run_eval.sh
```

**解释**: `chmod +x` 使脚本可执行。虽然 Git 也跟踪文件权限（通过 `git update-index --chmod=+x`），但 `chmod +x` 是直接修改文件系统权限。

---

## 7. 查看提交历史

```bash
# 紧凑的单行历史
git log --oneline

# 查看最近 N 次提交的详细差异
git log -p -2

# 查看文件修改历史
git log --follow -p dataset.py
```

**解释**:
- `--oneline`: 每个提交一行
- `-p`: 显示每次提交的完整 diff
- `-N`: 限制显示最近 N 次提交
- `--follow`: 跟踪文件重命名历史

---

## 8. 差异比较

```bash
# 工作区 vs 暂存区
git diff

# 暂存区 vs 最新提交
git diff --staged

# 工作区 vs 最新提交
git diff HEAD

# 两个提交之间
git diff commit1 commit2
```

**解释**: `git diff` 比较文件差异。理解三个状态（工作区、暂存区、提交历史）之间的关系很重要。

---

## 9. 撤销操作

```bash
# 撤销工作区修改（恢复到最新提交状态）
git checkout -- <file>

# 取消暂存（保留工作区修改）
git reset HEAD <file>

# 修改最后一次提交信息
git commit --amend -m "new message"
```

**解释**: 这些命令在出错时非常有用。注意 `git checkout -- <file>` 会**永久丢弃**工作区修改，使用前需确认。

---

## 10. 协同工作（备用）

```bash
# 推送分支到远程
git push origin experiment/SynDiff_adapt

# 拉取远程更新
git pull origin main

# 合并分支
git checkout main && git merge experiment/SynDiff_adapt
```

**解释**: 当需要将实验分支推送到远程仓库或合并回主分支时使用。

---

## 完整提交记录

本次实验分支的完整提交序列:
```
d123eba feat: add run_train.sh and run_eval.sh launch scripts
46eb229 feat: add eval.py for 14 missing modality pattern inference
d9ff0e4 feat: adapt training for single-GPU with TensorBoard logging
16bcbe2 feat: replace .mat data loading with MONAI-based nifti DataLoader
83078d9 base: existing modifications to dataset, train, EMA, and datalist files
fff3d84 (main) Update README.md
```

---

## 关键概念总结

| 概念 | 说明 |
|------|------|
| **Working Directory** | 工作区 — 你正在编辑的文件 |
| **Staging Area** | 暂存区 — `git add` 后的状态，准备提交 |
| **Repository** | 仓库 — `.git/` 目录，存储所有版本历史 |
| **HEAD** | 指向当前分支最新提交的指针 |
| **Branch** | 分支 — 独立的开发线，指向特定提交 |
| **Commit** | 提交 — 项目某一时刻的快照，由 SHA-1 哈希唯一标识 |
| **Remote** | 远程仓库 — 通常托管在 GitHub/GitLab 上 |

---
*记录时间: 2026-06-04*
*分支: experiment/SynDiff_adapt*
