# Streamlit Community Cloud 部署

## 部署参数

- Repository：`jiuxin77/ShopMind-AI`
- Branch：`main`
- Main file path：`src/shopmind/dashboard.py`
- Python：3.12（由 `runtime.txt` 指定）
- Secrets：无；当前演示不依赖 API Key

## 本地部署检查

```bash
source .venv/bin/activate
python scripts/release_check.py
python scripts/deployment_check.py
```

只有两项检查都通过才进行公网部署。`requirements.txt` 使用 `-e .` 安装当前项目及 `pyproject.toml` 中声明的运行依赖。

## 公网边界

- 演示数据为仓库内的确定性样本；UCI 原始数据不随应用分发。
- M5–M9 的实验指标来自仓库保存的离线产物，不会在网页启动时重新跑完整实验。
- 视觉模块使用合成演示缩略图和轻量特征，不是 CLIP 或真实商品图模型。
- 应用不收集账号、支付、地址等个人敏感信息。
