# AnimeTrace LLM 识图插件

这是一个面向 AstrBot 的 AnimeTrace 图片识别插件 fork。它保留手动命令识图能力，并额外把识图能力注册为 LLM 工具，让主 LLM 在用户发图、引用图片并询问“这是谁/出处/识图”时可以主动调用工具，再根据候选结果回答。

## 上游来源

- 上游插件：[`Aurora-xk/astrbot_plugin_shitu`](https://github.com/Aurora-xk/astrbot_plugin_shitu)
- 上游插件名：`astrbot_plugin_shitu`
- 本 fork 基于上游 `main` 分支，参考提交：`1a30fdd803806f4fb644476ddc0b65312b607d1c`
- 本 fork 插件名：`astrbot_plugin_animetrace_llm`

为避免和上游插件同时安装时冲突，本 fork 已更换：

- 插件注册名：`astrbot_plugin_animetrace_llm`
- Python 插件类名：`AnimeTraceLLMPlugin`
- 配置节：`animetrace_llm_settings`
- 手动识图命令：`/at识图`
- 头像识别命令：`/at头像识图`
- 模型切换命令：`/at模型`
- LLM 工具名：`animetrace_identify_image`
- 临时裁剪目录前缀：`astrbot_animetrace_llm_crops_`

## 功能

- 基于 [AnimeTrace](https://ai.animedb.cn/) 识别动漫、GalGame、二次元游戏角色。
- 支持图片和命令一起发送、引用图片识别、先发命令再补图。
- 支持 QQ 头像识别。
- 支持启动时拉取 AnimeTrace 可用模型，并通过命令查看/切换。
- 支持手动命令返回裁剪角色图。
- 注册 LLM 工具 `animetrace_identify_image`，可由主 LLM 在需要时调用。
- 在消息包含图片/引用图片，或文本命中识图关键词时，向 LLM 注入工具提示，减少误触发。

## 命令

| 命令 | 说明 |
| --- | --- |
| `/at识图` | 识别图片中的二次元角色 |
| `/at头像识图` | 识别 QQ 头像 |
| `/at模型` | 查看当前可用模型列表 |
| `/at模型 1` | 切换到列表中的第 1 个模型 |

## LLM 工具

工具名：`animetrace_identify_image`

参数：

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `image_url` | 否 | 图片 URL、本地路径或 `file://` URI。留空时自动读取当前消息或引用消息中的图片 |
| `model` | 否 | AnimeTrace 模型 ID。不确定时留空 |
| `qq` | 否 | 识别 QQ 头像时填写 QQ 号 |
| `max_results` | 否 | 每个检测区域最多返回多少个候选 |

典型效果：

1. 用户发送图片并问“这是谁？”。
2. LLM 看到本轮请求里有图片，并收到工具提示。
3. LLM 调用 `animetrace_identify_image`。
4. 工具返回 AnimeTrace 候选角色和作品。
5. LLM 根据候选结果自然回答，并说明结果可能不完全确定。

## 配置

配置文件：`_conf_schema.json`

主要配置项：

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `timeout_seconds` | `30` | 手动等待模式的补图超时时间 |
| `return_crops` | `true` | 手动命令是否发送裁剪角色图 |
| `max_crops` | `5` | 手动命令最多发送多少张裁剪图 |
| `max_characters_per_role` | `5` | 手动命令每个检测区域显示多少个候选 |
| `forward_threshold` | `0` | aiocqhttp 下多少个角色起使用合并转发，`0` 为关闭 |
| `llm_tool_enabled` | `true` | 是否注册 LLM 识图工具 |
| `inject_llm_tool_hint` | `true` | 是否在相关 LLM 请求中注入工具提示 |
| `llm_tool_max_results` | `5` | LLM 工具每个检测区域返回多少个候选 |
| `tool_request_keywords` | 见配置文件 | 命中后会提醒 LLM 可调用识图工具 |
| `tool_description` | 见配置文件 | LLM 工具描述，可影响模型调用倾向 |

## 安装依赖

```bash
pip install -r requirements.txt
```

## 说明

- 数据来源为 AnimeTrace，识别结果仅供参考。
- LLM 工具调用只返回文本候选，不会直接向聊天发送额外消息。
- 手动命令仍会按配置发送普通文本、裁剪图或 QQ 合并转发。
