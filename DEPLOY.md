# 部署说明（Deployment notes）

本项目部署与修改说明，支持本地运行、局域网访问与 Agent 工具调用。

## 当前状态

| 项目 | 值 |
| --- | --- |
| 监听 | `0.0.0.0:8000`（所有网卡，在 `.env` 里配置） |
| 访问地址 | 本机 http://127.0.0.1:8000 ・ 局域网 http://<局域网IP>:8000 |
| 可用模型 | `deepseek-chat`（唯一模型，对应 快速模式） |
| 虚拟环境 | `venv\`（Python 3.9+） |
| 登录会话 | `session\session.json`（已捕获，约 6 小时后自动无头刷新） |
| 浏览器 | 复用系统 Chrome（`playwright install chromium`） |

## 启动 / 停止

```powershell
cd Deepseek-API

.\start.ps1                 # 启动（推荐，见下方“为什么不用 python app.py”）
```

停止：在运行它的终端按 `Ctrl+C`；若是后台任务，结束对应的 python 进程即可。

当前监听地址写在 `.env` 里（`HOST=0.0.0.0`）。临时改端口 / 只在本机监听：

```powershell
$env:PORT = "8080"; .\start.ps1          # 换端口
$env:HOST = "127.0.0.1"; .\start.ps1     # 仅本机可访问
```

> 进程环境变量优先于 `.env`，所以上面这种临时覆盖不用改文件。

### ⚠️ 关于对外开放

服务**默认无鉴权**：若配置监听 `0.0.0.0`，同一局域网内任何能访问该端口的设备都能直接调用该服务消耗 DeepSeek 账号额度。建议仅在受信任的局域网内开放，或在 `.env` 中设置 `HOST=127.0.0.1` 仅限本机访问。

现有的唯一约束是上游自带的限流：每个客户端 IP 每分钟 30 次（`RATE_LIMIT_PER_MINUTE`），这只防打爆，不防未授权使用。

想收回去：把 `.env` 里的 `HOST` 改回 `127.0.0.1` 并重启。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | 对话，支持 `stream`、`conversation_id`、`thinking`、`search` |
| `GET` | `/v1/models` | 模型列表 |
| `GET` | `/healthz` | 健康检查（不受限流） |

## 调用示例

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")

r = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello!"}],
    # DeepThink / 联网搜索是独立开关，不属于 OpenAI schema，走 extra_body：
    # extra_body={"thinking": True, "search": True, "conversation_id": "..."},
)
print(r.choices[0].message.content)
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"Hello!"}]}'
```

多轮对话：把响应的 `conversation_id` 回传即可续接（注意续接时**不能**再传 `model`，线程的模型在创建时就固定了）。

从**其它设备**调用时，把上面的 `127.0.0.1` 换成本机在局域网中的实际 IP（在终端执行 `ipconfig` 查看）即可。注意若客户端自己配了系统代理，可能需要把该地址加入 `NO_PROXY` 才会直连。

## 本项目相对上游的改动

除了 `venv/`、`.env`、`session/` 这些运行产物，源码改动包括兼容修复、启动入口、模型列表更新与工具调用支持：

### 1. `deepseek/_envfix.py`（新增）+ `deepseek/__init__.py`

**问题**：httpx 会为 `NO_PROXY` 的每个条目构造 URL 匹配模式，条目 `[::1]` 会变成非法模式 `all://*[::1]`，导致 `httpx.Client` **在构造阶段**就抛 `InvalidURL: Invalid port: ':1]'` —— 请求还没发出去就 500 了。

**排查**：在某些 Windows 终端环境中，环境变量可能会存在带有方括号的 IPv6 回环条目（如 `[::1]`）。Windows 下 `os.environ` 会把键统一成大写，覆盖正确的配置。

**处理**：在 Python 进程内改写 `os.environ`（httpx 读的就是它），去掉带方括号的 IPv6 条目。保留裸写的 `::1`，所以 IPv6 回环绕过能力不变。

> 从**新开的终端**启动的程序会直接读到注册表里正确的值，本来就不受影响。

### 2. `deepseek/client.py` + `server/openai_format.py` — 思考过程分离与 `reasoning_content`

**问题**：
1. 开启 DeepThink（`thinking=True`）时，响应包含多个 fragment：先 `THINK`（深度思考过程），再追加 `RESPONSE`（真正回答）。两者在内部流中都通过 `-1` 索引寻址，原版解析器把思考过程当正文混入，破坏了答案结构。
2. 移动端应用（如 OPERIT、ChatBox 等）接入 OpenAI 兼容接口时，期待的是 DeepSeek 官方标准的 `choices[0].delta.reasoning_content` 字段。原版未向外暴露该通道。

**处理**：
1. 底层将 SSE 流解析为带标签的 `(kind, text)` 事件流（`think` vs `response`），分别汇入 `reply.reasoning` 与 `reply.text`。
2. 流式响应向客户端发送 `delta.reasoning_content`（思考中）与 `delta.content`（正文），非流式在 `message.reasoning_content` 中附带完整思考过程，手机客户端能折叠/展开原生渲染思考过程。
3. 修复了末尾补丁帧省略 `o: APPEND` 字段时尾部字符（如标点）被意外静默丢弃的问题。

### 3. `run_server.py` + `start.ps1`（新增）

启动入口。`start.ps1` 调用 `run_server.py`，后者在导入任何 httpx 之前先应用上面的 NO_PROXY 修复，再启动 uvicorn。

> 之所以不直接 `python app.py`：`app.py` 只能沿用继承来的环境变量，无法修复上述问题。

### 4. `server/config.py` — 只保留一个模型

`MODEL_MAP` 现在只有 `deepseek-chat -> default`，`deepseek-expert` 不再对外暴露（传它返回 404）。README / examples 里的相关说明和示例一并更新。原因见下一节。

顺带修了 `examples/06_server_openai_sdk.py` 里三个会让它跑不起来的问题：补上 NO_PROXY 修复的引入（它只 import `openai`，拿不到包级的修复）、删掉硬编码的 `conversation_id`（那是原作者账号的会话，在别人账号上必然失败）、加一行 stdout UTF-8 保护（该示例要求用印地语回答，CP936 控制台输出会抛 `UnicodeEncodeError`）。

### 5. `server/schemas.py` — `thinking` 默认开启

手机端客户端（如 OPERIT）使用标准 OpenAI 接口接入，无法在请求体中附带自定义的 `extra_body: {"thinking": true}`。为了开箱即用看到完整思考过程，服务端已将 `thinking` 默认值设为 `True`（传 `{"thinking": false}` 可主动关闭以加快响应速度）。

### 6. 上游限流透传（HTTP 429）

当触发 DeepSeek 官方账号每分钟请求频率限制时，官方接口会返回包含 `rate_limit_reached` 的 hint 错误帧且不包含任何 fragment。原代码会静默返回一个空内容的 HTTP 200。现已增加 `DeepSeekError` 捕获：
- 非流式：转换为带 `Retry-After: 5` 的标准 HTTP 429 错误，驱动 OpenAI SDK 与客户端自动退避重试。
- 流式：在 SSE 流内输出标准 error 帧告知客户端，不再静默返回空回复。

### 7. Agent 工具调用转译器（Function Calling Adapter）

**问题**：
DeepSeek 网页端（`chat.deepseek.com`）是面向人类聊天的纯文本接口，底层**没有**原生 Function Calling API。原开源项目收到底层传来的 `tools` 列表会直接忽略丢弃，且响应只产生纯文本，导致像 DSH（DeepSeek Harness）、Cursor、Aider 这样的自主智能体（Agent）框架无法捕获工具调用，模型只能在回复里打出字面 `Tool: ...` 而无法驱动宿主执行工具。

**处理**：
在服务端实现了完整的标准 OpenAI Function Calling 垫片（Polyfill）：
1. **Schema 支持**：`server/schemas.py` 扩展支持 `tools`、`tool_choice` 参数，以及 `ChatMessage` 的 `tool_calls` 与 `role: "tool"` 结果消息。
2. **Prompt 注入与重构**：`server/openai_format.py` 将传入的工具集合（名称、描述、JSON Schema 参数）自动编译为模型易读的标准工具声明，并在对话历史中完整对齐上一轮生成的 `tool_calls` 与工具执行返回值（`role: "tool"`）。
3. **输出拦截与转译**：
   - **流式（SSE）**：引入轻量状态机实时拦截流中的 `<tool_call>` 标签。普通正文正常以 `delta.content` 流出，检测到工具调用时暂停并提取 JSON，打包为 OpenAI 标准的 `delta.tool_calls: [...]` 结构分片，并在尾部发送 `finish_reason: "tool_calls"`，精准触发 DSH 拦截并执行宿主工具。
   - **非流式**：解析完整正文中的 `<tool_call>`，提取为 `choices[0].message.tool_calls` 并标记 `finish_reason: "tool_calls"`。
   - **普通对话兼顾**：未传入 `tools` 或模型判定无需调工具时，无损保持普通纯文本与思考过程流式输出。

> 示例 04 / 05 用的是 `requests`，它不在 `requirements.txt` 里。需要时另装：
> `.\venv\Scripts\python.exe -m pip install requests`

## 模型说明：为什么现在只有一个模型

DeepSeek 于 **2026-09-10** 把网页版的快速 / 专家 / 识图三种模式**合并**，不再单独设专家模式（专家模式 2026-04-08 上线，识图模式 2026-06-18 上线）。

实测网页版服务端下发的模型配置（存在浏览器的 feature store 里）：

| model_type | 名称 | is_default | enabled | switchable |
| --- | --- | --- | --- | --- |
| `default` | 快速模式 | ✅ | ✅ | ✅ |
| `expert` | 专家模式 | ❌ | ❌ | ❌ |
| `vision` | 识图模式 | ❌ | ❌ | ❌ |

三个条目服务端都还留着，但**只有 `default` 是启用的**，另外两个被标为禁用且不可切换，所以界面上不再显示。直接拦截网页版发出的请求，它传的正是 `model_type: "default"`。

也就是说：`deepseek-expert` 之前"还能通"，是因为后端没有删除这个值，而**不是**它还在产品功能面上。这属于随时可能失效的遗留值，所以已从对外接口移除。

**现在要用深度思考 / 联网搜索，应该传 `thinking` / `search` 开关**——这正是专家模式过去独有的能力，合并后变成了通用开关：

```python
client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "..."}],
    extra_body={"thinking": True, "search": True},
)
```

> 顺带一提：网页版现在发 `x-client-version: 2.4.0`，本项目仍发 `2.0.0`。目前不影响使用，但若日后接口开始校验版本号，这里是第一个要改的地方。

## 注意事项（来自上游 README）

- **单账号串行**：后端共用一个已登录账号，PoW 求解器不可重入，上游调用是**串行**的，并发请求会排队。请勿高并发压测。
- **限流**：默认每 IP 每分钟 30 次，超限返回 429 + `Retry-After`；用 `RATE_LIMIT_PER_MINUTE` 调整。客户端建议指数退避。
- **`usage` 是估算值**（约 4 字符/token），不是真实计数。
- **多数 OpenAI 参数被忽略**：只有 `model`、`messages`、`stream`、`conversation_id`、`thinking`、`search` 生效。
- **会话私有**：`session/` 内含 cookie 和 token，已 git-ignore，请勿外传。
- 该项目为非官方逆向实现，使用需自行遵守 DeepSeek 的服务条款。

## 常见问题

**Q：返回 503 `login_required` / 浏览器又弹出来了？**
会话过期了。重新登录：

```powershell
.\venv\Scripts\python.exe -m deepseek.auth
```

**Q：所有请求都 500，报 `Invalid port: ':1]'`？**
`NO_PROXY` 又被写回了带方括号的 IPv6 条目。用 `.\start.ps1` 启动即可（它会自动修）；外部客户端可参考 `deepseek/_envfix.py` 自查。

**Q：想升级到上游最新代码？**
本地改的是 `deepseek/__init__.py`、`deepseek/client.py`、`server/config.py`，另有 3 个新文件（`deepseek/_envfix.py`、`run_server.py`、`start.ps1`）。`git stash` 或 `git diff` 备份后再 `git pull`，然后重新套用上述修改。注意 `MODEL_MAP`：如果上游之后新增了模型条目，需要手动合并，别被 `git pull` 覆盖回旧的两模型版本。
