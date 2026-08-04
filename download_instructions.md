# 模型权重下载与目录结构说明

本文档详细说明了如何使用 `curl` 或 `gh` (GitHub CLI) 下载指定随机种子（42, 3407, 9999, 2025, 1234）的模型权重文件，并解释了项目所需的预期目录结构。

## 1. 预期的目录结构

所有下载的模型权重应放置在 `d:\NEOschool\2026BMI-\results` 目录下。针对不同的实验设置（被试内 `intra-subjects` 和被试间 `inter-subjects`）、不同的被试（`sub-01` 至 `sub-10`）以及不同的随机种子，目录结构应如下所示：

```text
d:\NEOschool\2026BMI-\results\
├── intra-subjects-seed42\
│   ├── sub-01-seed42\
│   │   └── checkpoint_test_best.pth
│   ├── ...
│   └── sub-10-seed42\
│       └── checkpoint_test_best.pth
├── inter-subjects-seed3407\
│   ├── sub-01-seed3407\
│   │   └── checkpoint_test_best.pth
│   ├── ...
```
这总计将有 2(modes) * 10(subjects) * 5(seeds) = 100 个权重文件。

## 2. 下载模型权重

请根据您的网络环境和偏好，选择使用 `curl` 或 `gh` 进行下载。
*注：由于您提到权重在 GitHub 上，请将 `<owner>/<repo>` 和真实链接替换为实际值。*

### 方法一：使用 `curl` 下载 (假设有直链)

如果您有直接的 Release 资产下载链接，可以使用 `curl`。建议在终端（如 PowerShell）中运行：

```bash
# 示例：下载 intra-subjects seed 42 的 sub-01 权重
curl -L -o "d:\NEOschool\2026BMI-\results\intra-subjects-seed42\sub-01-seed42\checkpoint_test_best.pth" "https://github.com/your-org/your-repo/releases/download/v1.0/intra_sub-01_seed42.pth"
```

### 方法二：使用 GitHub CLI (`gh`) 下载 (推荐)

如果模型权重作为 Release 资产托管在 GitHub 上，使用 `gh` 可以按模式匹配下载：

```bash
# 首先确保您已经登录
# gh auth login

# 下载整个 Release 中的所有相关文件到本地当前目录
gh release download v1.0 -R your-org/your-repo

# 然后可以使用脚本将它们分发到指定的目录结构中
```

## 3. 批量下载与解压分发脚本 (PowerShell)

通常如果文件很多，最好的方式是将 100 个文件打包成一个 `weights.zip` 上传到 GitHub，然后使用命令行下载并解压：

```powershell
# 1. 下载 zip 文件
gh release download v1.0 -R your-org/your-repo -p "weights.zip" --dir "d:\NEOschool\2026BMI-\results"

# 2. 解压缩 (假设压缩包内已经包含了完整的 intra-subjects-seed* 目录结构)
Expand-Archive -Path "d:\NEOschool\2026BMI-\results\weights.zip" -DestinationPath "d:\NEOschool\2026BMI-\results" -Force
```
这种方式是下载 100 个权重文件最稳妥高效的方式。
