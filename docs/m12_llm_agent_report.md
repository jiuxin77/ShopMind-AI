# M12：LLM 商品导购 Agent 与独立后端

## 目标

证明项目具备真实 LLM 应用工程能力，而不仅是聊天界面：模型负责判断是否追问或检索，应用负责执行工具、约束商品证据并处理失败。

## 执行链路

1. 用户提交自然语言需求。
2. Responses API 必须调用 `ask_clarification` 或 `search_catalog`。
3. 应用执行本地混合召回，将有限商品字段回传模型。
4. 模型输出 `GroundedAnswer` Pydantic 结构。
5. Grounding Guard 删除不在检索结果中的商品 ID。
6. 外部服务异常时回退至规则解析与混合召回。

## 评测

`data/llm_agent_eval_cases.json` 覆盖明确检索、预算、品类、排除条件和信息不足追问。运行：

```bash
OPENAI_API_KEY=... python scripts/evaluate_llm_agent.py --model gpt-4.1-mini
```

输出 `artifacts/m12_llm_agent_eval.json`，包括工具选择准确率、有依据商品比例、约束满足率、降级率和平均延迟。真实模型结果依赖模型版本和运行日期，因此只接受脚本实际输出，不手工填写。

## 独立服务

FastAPI 暴露 `/recommend`、`/shopping-assistant`、`/llm-shopping-agent`、`/health/live` 和 `/health/ready`。配置由环境变量管理；日志为 JSON 格式并包含请求 ID、状态码与耗时；Docker 使用非 root 用户和健康检查；CI 运行 lint、测试、编译与部署检查。
