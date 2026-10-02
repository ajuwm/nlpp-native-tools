# 推送到 GitHub

本地仓库已经准备好（已 init + 首次提交）。**只差认证这一步——我没有你的 GitHub 凭据，
也不能替你索取，所以需要你自己执行下面的命令。**

---

## 方案 A：用 GitHub CLI（推荐，一条命令建仓并推送）

```powershell
winget install --id GitHub.cli          # 安装 gh
gh auth login                           # 浏览器登录
cd D:\dsh\nlpp\github_ready
gh repo create nlpp-native-tools --public --source=. --push
```

`gh repo create` 会自动建仓、设置 remote、并推送当前分支。

---

## 方案 B：先到网页建仓，再推

1. 打开 https://github.com/new
2. 仓库名填 `nlpp-native-tools`，**不要**勾选 "Add README / .gitignore / license"（本地已有）
3. 建好后执行：

```powershell
cd D:\dsh\nlpp\github_ready
git branch -M main
git remote add origin https://github.com/<你的用户名>/nlpp-native-tools.git
git push -u origin main
```

推送时 Git 会弹出浏览器/凭据窗口，用你的 GitHub 账号授权即可。

---

## 推送前的两处修改建议

1. **提交者署名**：现在的提交用的是占位身份
   ```powershell
   cd D:\dsh\nlpp\github_ready
   git config user.name  "你的名字"
   git config user.email "你的邮箱"
   git commit --amend --reset-author --no-edit
   ```

2. **LICENSE 里的版权人**：`LICENSE` 第 3 行是 `<YOUR NAME>`，改成你的名字或 ID。

---

## 已包含 / 未包含

**已提交（22 个文件）**

```
README.md          汇总文档（已确证格式 + 未解问题 + 给社区的话）
LICENSE            MIT
.gitignore         排除所有游戏资源与提取文本
native/            19 个 Python 工具
```

**`.gitignore` 已排除了所有游戏资源与提取产物**（`*.bin` `*.mot` `*.trb` `*.jpg` `*.tsv` 等），
所以**不会误传游戏数据**。上传前可以用下面这条确认：

```powershell
git ls-files | Select-String -Pattern '\.(bin|mot|trb|jpg|png|tsv)$'
# 应该没有任何输出
```
