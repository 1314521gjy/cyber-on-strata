# 10 · 把它做成服务（OpenAI / Anthropic 兼容端点）

> 证据等级：`A` = 本机自实测 ｜ `B` = 同机对照实现 ｜ `C` = 外部实测 ｜ `D` = 推导或出处自报。
> 本文记录**把这台引擎接到一个聊天客户端上**这件事：上游的 serve 层怎么用、为这个模型改了什么、以及哪些是实测过的。

---

## §1 形态

引擎有一个 `--serve` 模式：它不再一次算完就退出，而是在 **stdin/stdout 上收 token id、吐 token id**，权重、专家竞技场与显存缓存只加载一次。

上面套一层 Python（`serve/server.py`，随仓携带），把它变成标准端点：

| 端点 | 协议 |
|---|---|
| `POST /v1/chat/completions` | OpenAI（流式与非流式） |
| `POST /v1/messages` | Anthropic（流式与非流式） |
| `GET /v1/models`、`/models`、`/props`、`/health`、`/slots`、`/mcp` | 元信息 |
| `GET /` | 自带聊天页 |

一次只跑一条序列（FIFO）：并发请求排队，不并行。**提示词 + max_tokens 超过上下文会被 400 拒绝，不截断。**

启动：

```
python serve/server.py --engine strata --config <runconfig>.json --port 8091
```

`<runconfig>.json` 的键（`A` 级，实测可用）：`exe`、`args[]`、`cwd`、`tokenizer`、`model_name`、`log`、`lib_dirs`、`port`，以及两个与"思考"有关的：`reasoning_budget_tokens`（**0 = 不设上限**）、`anthropic_thinking`（`model` = 模型怎么做就怎么做）。模板见 `examples/cyber/strata-cyber.example.json`。

**分词器**：`tokenizer` 指向 pack 里的 `tokenizer/`（`vocab.json` + `merges.txt` + `token_type.json` + `chat_template.jinja`）。引擎自己不带分词器，这一层替它做文本 ↔ token id 的转换 —— 与 `03-run.md` 里命令行启动器做的事同一件。

---

## §2 与上游 serve 层的两处硬冲突（本模型的）

上游那套 serve 是围绕**参考模型**（`qwen4exp` 系）写的，接到 `qwen35moe` 上时有两处对不上。两处都不是配置问题，是要改代码的：

### 2.1 serve 强制要求一个 MTP 草稿器

`src/program/generate.cpp` 的 serve 入口原本是：

```c
if (o.spec < 2 || o.mtp.empty() || o.prefill_chunk <= 0 || ...) { ... return 2; }
```

也就是**必须**给 `--spec T`（T ≥ 2）与 `--mtp DIR`（MTP 运行时目录）。

而 MTP 运行时目录的几何**写死在参考模型上**：

- `tools/mtp_rt.py` 顶部常量 `H, FF, NE = 2560, 640, 512`；
- `generate.cpp` 里绑定草稿层用的是默认 `ModelGeometry`（同一套参考几何），注释也写着"草稿层就是规范模型的 MTP 头（512 专家）"。

这个模型（40 层 / 256 专家 / hidden 2048 / FF 512）的 MTP 块与参考几何不同，实测在草稿层的**融合 GR 读**上直接失败（`fused_gr_read_multi: invalid arguments for token 0`），并且是在**第一次请求的 prompt 阶段**就 `exit(1)`。我们试过三个不同的运行时目录变体，**同一个崩溃点**。

⇒ 结论：**这个模型在 serve 模式下不能用 MTP 草稿器**。而 MTP 在命令行下早有实测结论：接受率能做到 0.55–0.58，但草稿头会占掉 124–125 个专家槽，净效果是负的（见 `08-known-issues.md`）。也就是说：**它本来就不该开**。

### 2.2 改法（本 fork 的 serve 补丁）

让 serve **不带草稿器**照跑，等同命令行下 `--mtp` 缺省时的行为（有 verify 窗口、无草稿；草稿会被目标模型验证，所以"没有草稿"只损失速度，不影响输出）。共 6 处，全在 `src/program/generate.cpp`：

| 位置 | 改动 |
|---|---|
| serve 入口守卫 | 去掉 `o.spec < 2` 与 `o.mtp.empty()` 两条前置条件，只留 `--prefill` 与图化命中路径的检查 |
| `mtp.bind(...)` | 仅在 `!o.mtp.empty()` 时调用 |
| 对话检查点的草稿 K/V | 无草稿器时传 `nullptr`（两处：`draft0` 与 `draft_of`） |
| prompt 阶段的草稿前填 | `batched` 与 `mtp.prefill(...)` 都加 `!o.mtp.empty()` 前置；诊断打印里的 `kv_mode` 无草稿器时打 -1 |
| 解码循环取草稿 | `o.mtp.empty()` 时直接视为"已出草稿"（即不取草稿），窗口照跑 |

**没有改动任何数值路径**：这些守卫只在"没有草稿器"时为真，有草稿器时的行为与上游一致。

### 2.3 另一处开关（性质不同）

启动器里设了 `STRATA_HC_SPLIT=0`（hyper-connection 读走 plain 版本）。它是在定位 2.1 的过程中加的；**崩溃最终归因于草稿器，这条是否必需我们没有单独验证**（`D`）。引擎自己打印的自证行是：

```
strata hc: CUDA0: the hyper-connection read runs as the plain one (STRATA_HC_SPLIT=0)
```

---

## §3 实测（`A` 级）

本机（RTX 4080 SUPER）：

| 项 | 读数 |
|---|---|
| 服务就绪 | `GET /health` 通；`/v1/models` 返回配置里的 `model_name` |
| 一次真实对话 | prompt **192** tokens → completion **19** tokens，`finish_reason=stop`，**143.1 tok/s**（短生成，只作方向） |
| 加载显存 | `strata serve: 9711 MiB of VRAM free with everything loaded` |
| 专家缓存 | 本例是**全常驻**档（`reserve 1500` ⇒ 10240 槽，18.32 GiB 显存） |
| KV | `int8` + `--kv-resident 20480` ⇒ `KV streaming: 20480 of 32768 cells per QSA layer in VRAM, the K/V in 0.32 GiB of pinned RAM` |

**没测**：并发、工具调用（MCP）、图像、长上下文下的服务吞吐、多条请求后的 K/V 复用。这些都要看引擎与 serve 层的组合，不在本文的判据内。

---

## §4 接到客户端上

任何 OpenAI 兼容客户端都行：`baseURL` 指 `http://127.0.0.1:<port>/v1`，模型名用配置文件里的 `model_name`。

**要遵守的两条**：

1. **客户端声明的上下文窗口必须等于 runconfig 里的 `--max-context`**。声明得比引擎大，客户端会按更大的窗口去算 `max_tokens`，把上下文撑爆。
2. **思考档位**：服务端接受的 wire 值只有 `none | low | medium | high`（模型模板里是"开/关"二值，low/medium/high 在服务端是同一个"开"）。所以"默认思考开、不设上限"= 模板 `enable_thinking` 默认 true（不需要额外传参）+ `reasoning_budget_tokens: 0`；想要更高档位只能靠请求里带的标记（模板支持 `<|think_high|>` 之类的开关）。

---

## §5 复现清单

```text
1) 跑起引擎（见 01-build.md 构建，02-pack-and-profile.md 生成 pack 与 profile）
2) 写 runconfig：exe / args / cwd / tokenizer / model_name / log / port / reasoning_budget_tokens
3) 起服务： python serve/server.py --engine strata --config <runconfig>.json --port <port>
4) 探活：   GET /health  →  GET /v1/models  →  POST /v1/chat/completions
5) 日志：   引擎自身的输出落在 runconfig 的 log；服务自身的输出建议重定向到一个文件
           （否则服务报错信息只出现在控制台窗口里，排查时看不到）
```

第 5 条是踩过的：服务层自己的报错（例如"引擎在就绪前退出"）**不会**写进 runconfig 的 `log`，只打到它自己的 stdout/stderr。
