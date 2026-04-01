# Thesis Bundle

这个文件夹用于单独存放当前论文稿与相关 LaTeX 文件，不影响原工作区结构。

## 目录说明

- `markdown/FedViM_论文草稿.md`
  - 当前论文 Markdown 草稿
- `latex/`
  - 已整理好的北师大 `bnuthesis` 模板版本
  - 包含章节内容、参考文献、图片和当前编译出的 `main.pdf`

## 编译方式

如果本机已安装 TinyTeX，可用下面命令重新编译：

```bash
export PATH=/home/dell7960/.TinyTeX/bin/x86_64-linux:$PATH
cd /home/dell7960/桌面/FedOOD/thesis_bundle/latex
latexmk -xelatex main.tex
```

## 当前输出

- 最终 PDF：`latex/main.pdf`
