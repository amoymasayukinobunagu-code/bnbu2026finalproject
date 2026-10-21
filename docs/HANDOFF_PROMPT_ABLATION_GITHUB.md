# Prompt 消融：搭档执行 + GitHub 上传说明

目标仓库：https://github.com/amoymasayukinobunagu-code/bnbu2026finalproject  
本机工程：`E:\Project\FYP`（当前还不是 git 仓库；GitHub 上目前只有 `PROJECT_BRIEF.md` 与 `README.md`）

## 1. 消融跑哪些模型

必跑五模型（box 偏移 0 / 10 / 30 / 50；±30 可抄主表）：

| 模型 | 角色 | ±30 锚点（Kvasir / CVC Dice） |
|---|---|---|
| SAM ViT-L | 零样本 | 0.8140 / 0.6588 |
| MedSAM | 医学预训练 | 0.7034 / 0.5172 |
| SAMed LoRA full-880 | PEFT | 0.8955 / 0.8608 |
| MedSAM+LoRA r4 | MedSAM PEFT | 0.9049 / 0.8476 |
| MSA polyp ViT-B | Adapter | 0.8524 / 0.7685 |

选跑：MSA ViT-L、PP-SAM full-880（表注写 full-FT 对照）。  
跳过：Chen SAM-Adapter、MedSAM+MSA。

协议：v3，Kvasir val 120 + CVC 612，GT box ±N px。不训新权重。

建议顺序：MedSAM+LoRA r4 的 0→50 → MedSAM → SAMed → MSA-B → SAM ViT-L；再补各模型的 10。

详细清单：https://app.notion.com/p/3fe3092ac6c481769949e3a5298a8415

## 2. 本机结果目录（跑完再上传）

```
E:\Project\FYP\results\prompt_ablation_v3\
  SUMMARY.md
  medsam_lora_r4\box_0\summary.json
  medsam_lora_r4\box_50\summary.json
  ...（其余模型同理）
```

`SUMMARY.md` 可从模板拷贝后回填：聊天附件 / bot 侧 `prompt_ablation_v3_SUMMARY_TEMPLATE.md`。

## 3. 哪些进 GitHub，哪些绝对不要

要上传（小、可复现叙事）：
- `SUMMARY.md`
- 各模型各偏移下的 `summary.json`（只要均值指标，不要整图 mask）
- 可选：很短的复现说明（协议、偏移、ckpt 本地路径）

不要上传：
- `checkpoints/` 权重
- 数据集 `材料/` / 原始图
- 逐图像素预测、大 CSV、训练 log、整包 zip
- 现有 `.gitignore` 已排除 `checkpoints/`、`results/`、`材料/` —— 保持这个原则

推荐做法：不要强行 `git add results/`，而是把要入库的汇总拷到可跟踪路径：

```
docs/experiments/prompt_ablation_v3/SUMMARY.md
docs/experiments/prompt_ablation_v3/summaries/*.json
```

## 4. 第一次把本机工程接到 GitHub（君涵或搭档二选一做一次）

在 `E:\Project\FYP`：

```powershell
cd E:\Project\FYP
git init
git remote add origin https://github.com/amoymasayukinobunagu-code/bnbu2026finalproject.git
git fetch origin
git checkout -b main
git pull origin main --allow-unrelated-histories
```

确认 `.gitignore` 仍排除 `checkpoints/`、`results/`、`材料/`。  
注意：当前 `.gitignore` 后半段有一批无路径前缀的名字（含 `code`、`results` 等），若整库提交前发现 `code/` 被忽略，把那几行改成只忽略 `results/` 下对应子目录，或删掉误伤行。

## 5. 消融结果上传步骤（搭档跑完后）

```powershell
cd E:\Project\FYP
mkdir docs\experiments\prompt_ablation_v3\summaries -Force
copy results\prompt_ablation_v3\SUMMARY.md docs\experiments\prompt_ablation_v3\SUMMARY.md
# 把各 summary.json 拷到 summaries\，文件名如 medsam_lora_r4_box0.json

git checkout -b feat/prompt-ablation-v3
git add docs/experiments/prompt_ablation_v3/
git status   # 确认没有 .pth / 大数据集
git commit -m "Add prompt ablation box 0/10/30/50 summary (v3 protocol)"
git push -u origin feat/prompt-ablation-v3
```

然后在 GitHub 开 PR 合进 `main`，标题建议：`prompt ablation box 0/10/30/50 under v3`。

若坚持直接放在 `results/prompt_ablation_v3/` 下入库，用强制添加汇总文件即可（仍不要 add 整个 results）：

```powershell
git add -f results/prompt_ablation_v3/SUMMARY.md
git add -f results/prompt_ablation_v3/**/summary.json
```

## 6. 完成标准

- 五模型 ×（至少 0/50；最好含 10；30 可抄表）Kvasir + CVC mean Dice 写进 `SUMMARY.md`
- PR 里只有文档/json，无权重无数据
- Notion Goals #3 勾选；可选补一页阶段说明

## 7. 搭档最小 checklist

1. 读 Notion 消融清单  
2. 本机建 `results\prompt_ablation_v3\`  
3. 按五模型跑 box 0→50→10  
4. 填 `SUMMARY.md`  
5. 拷到 `docs/experiments/prompt_ablation_v3/`  
6. 开分支 → commit → push → PR  
