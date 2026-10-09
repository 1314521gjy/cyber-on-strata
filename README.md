# cyber-on-strata

> **项目目标：让显存不足的人也能体验 Ornith-1.5 MoE（Cyber-Tiel-Coder-35B-A3B）。**
> **Goal: let people without enough VRAM still run Ornith-1.5 MoE.**

**在 8 GB 显存的显卡上跑起 35B MoE：专家权重流式上卡、未命中专家由 CPU 池就地计算。**

> An expert-streaming inference setup for **Cyber-Tiel-Coder-35B-A3B** (`qwen35moe`, 256 routed
> experts, top-8), built as a fork of the [Strata](https://github.com/Niko1221/Strata) engine.
> Weights stay resident in system RAM and are paged onto the GPU on demand; experts that miss the
> GPU cache are computed in place by a CPU thread pool. Measured on an RTX 4080 SUPER:
> **57.7–63.5 tok/s decode** with a 4.2 GiB expert cache, **134–136 tok/s** when the whole expert
> set fits in VRAM. See [`docs/cyber/`](docs/cyber/) for the full engineering record.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CUDA arch](https://img.shields.io/badge/CUDA%20arch-sm__75%20%7C%20sm__86%20%7C%20sm__89%20%7C%20sm__120-blue.svg)](#显卡要求)
[![Model](https://img.shields.io/badge/model-Cyber--Tiel--Coder--35B--A3B-orange.svg)](https://huggingface.co/peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP)

---

## 它解决什么问题

35B 的 MoE 模型（22.75 GB）里 **89% 的字节是路由专家**，而每个 token 只用到其中 8/256。
把这 18.85 GiB 的专家全部塞进显存，小显存卡做不到；全部走 CPU 又慢得没法用。
本项目的做法是**两级专家内存**：

```
        ┌──────────────── 显存（按卡自适应）────────────────┐
        │  常驻权重（dense / attention / GDN）≈ 2.3 GiB    │
        │  专家缓存（命中即在 GPU 上算）                    │
        │  KV（可压位宽、可挪进内存）                       │
        └──────────────────────────────────────────────────┘
                              ↕ 按需搬运
        ┌──────────────── 系统内存 ────────────────────────┐
        │  专家银行（约 19 GiB，钉住）                      │
        │  CPU 线程池：未命中的专家在这里直接算             │
        └──────────────────────────────────────────────────┘
                ↕ 冷数据
                           磁盘上的 GGUF（22.75 GB）
```

三个关键点：**专家缓存按剩余显存自适应**、**KV 可以整体挪进钉住内存换显存**、
**未命中的专家不搬回 GPU，而是交给 CPU 池就地算**。

---

## 实测效果

同一台机器（RTX 4080 SUPER；模型 `…MTP-UD-Q4_K_XL.gguf`，22,749,880,640 B）：

| | **8 GB 卡档**（专家缓存 3.0 GiB） | **全常驻档**（10240 槽，全部专家驻显存） |
|---|---|---|
| decode | **57.7–63.5 tok/s** | **134–136 tok/s** |
| prefill | 1736–2203 tok/s | 3026–3086 tok/s |
| 专家缓存命中率 | 0.78 | —（不需要缓存） |
| CPU 专家池 | 1.54 专家/层，≈ 5.4 ms/token | **0 dispatches** |
| 显存峰值（含系统） | **7,260 MiB** | 22,862 MiB |
| 内存净占 | **≈ 21 GiB** | ≈ 21 GiB |

**每 token 的账（8 GB 档）**：`15.75 ms = 7.35（GPU 地板）+ 5.42（CPU 池）+ ≈3（搬运与验证）`。
GPU 地板 7.35 ms/token 是这张卡的上界；CPU 池那 5.4 ms 是带宽在还债
（每 token 要从内存读约 117 MB 专家字节 ⇒ 60 tok/s 就是约 7 GB/s）。

---

## 快速开始

```text
1) 下载模型（本仓库不包含权重）
   https://huggingface.co/peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP
   文件：Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf
   大小：22,749,880,640 B
   sha256：0BBCF3CC9BE4C976BAD20E641BAF629DAD9C178D39EBDC9CD72129179943C06A
   放进 <package>/models/

2) 跑

   run.cmd -Prompt "用一句话解释什么是快速排序"

3) 看结果
   启动器会打印回答，并给出三行读数：expert cache / hits / decode tok/s
```

启动器（`examples/cyber/run.cmd` + `run.ps1`）负责**文本进、文本出**：它用
`llama-tokenize --ids` 分词、用一张 `id → 字节` 表反分词，因为**引擎本身不带 tokenizer**
（它吃预分词好的 token id）。

---

## 显卡要求

发布版 `strata.exe` 是**多架构构建**，`cuobjdump` 实测 SASS 与 PTX **同时包含四代**：

| 显卡 | SM | 能否运行 |
|---|---|---|
| RTX 50 系 | 12.0 | ✅ |
| RTX 40 系 | 8.9 | ✅ **本机实机验证过** |
| RTX 30 系 | 8.6 | ✅ 有对应机器码，**未实机验证** |
| RTX 20 系 | 7.5 | ✅ 同上 |
| 更老 / 非 NVIDIA | ≤ 7.5 | ❌ |

```bash
# 自己构建时务必把架构写全，否则产物只有一代能用
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75;86;89;120 -DSTRATA_MMQ_KQUANTS=ON
```

## 内存要求

| 内存 | 结论 |
|---|---|
| 16 GB | ❌ 跑不了 —— 光钉住的专家银行就要 **18.32 GiB** |
| **32 GB** | ⚠️ **能跑**：引擎 21 GiB + 系统 6 GiB；但没余量留给文件缓存（模型 21.2 GiB），会偏慢 |
| **48 GB** | ✅ **推荐**：21（引擎）+ 21（文件缓存）+ 6（系统）正好 |
| 64 GB+ | ✅ 最稳 |

## KV 放在哪

- **默认全在显存**，占 `--vram-reserve-mib` 留出的那块。
- `--max-context` 决定它多大：**实测 fp16 约 21.6 KiB/token** ⇒ 4k≈86 MiB、32k≈690 MiB、128k≈2.76 GiB。
- 它**可以挪进钉住内存**：`--kv-resident N`（每层留 N 格在显存）。实测 128k 上下文下
  **花 2.5 GiB 内存换回 2.11 GiB 显存**。
- 也可以**压精度**：`int8`（= k8v8）/ `k8v4` / `q4_0`。**注意 `k8v4` 与 `--kv-resident` 互斥**。
  本项目的默认是 **`int8` + `--kv-resident 20480`** —— 选它是因为 `int8` 的数值**与 fp16 逐位一致**，
  不破坏已经建立的正确性锚。

---

## 参数（发布默认）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--vram-reserve-mib` | **1500** | **唯一必须按显卡改的参数**：从专家缓存里扣掉、留给 KV/激活/桌面的显存 |
| `--expert-cache auto` | — | 按 `空闲显存 − reserve` 自动定槽数 |
| `--max-context` | 4096 | KV 容量 |
| `--kv` | **int8** | KV 精度（引擎自带默认是 fp16） |
| `--kv-resident` | **20480** | KV 挪进钉住内存 |
| `--prefill` | 2048 | 预填分块 |
| `--expert-profile` | — | 静态预填，**命中率的主要来源**（不带它，默认策略实测只有 2.97% 命中） |
| `--pool-workers` / `--pool-affinity` | 0 / auto | CPU 专家池线程数与绑核 |

**按显存分档**（`空闲显存 ≈ 显存总量 − 3.8 GiB`，本机实测的 3.8 GiB 含常驻权重、CUDA 上下文与激活）：

| 显存 | `-Reserve` 建议 | 预计专家缓存 | 落进哪档 |
|---|---|---|---|
| 8 GB | 800–1200 | ≈ 2.8 GiB | 流式 |
| 12 GB | 1500–2500 | ≈ 6.7 GiB | 流式 |
| 16 GB | 2500–4000 | ≈ 10.7 GiB | 流式 |
| 24 GB | 4000–8000 | ≈ 18.7 GiB | **全常驻** |
| 32 GB | 1500（常驻）/ 24568（流式基准） | 26.7 / 4.2 GiB | 两档都可以 |

---

## 仓库结构

```
.
├── README.md               本文件
├── LICENSE                 MIT（保留上游版权行 + 本项目一行）
├── NOTICE                  第三方归属：Strata / ggml / llama.cpp / 模型
├── docs/
│   ├── cyber/              本项目的工程记录（10 篇，从总览到已知问题）
│   └── upstream/           上游 Strata 自带的 README（含多语言版本，原样保留）
├── examples/cyber/         一键启动器与辅助脚本（文本进、文本出）
├── tools/                  打包与采样工具（iq_pack.py / make_profile.py …）
├── src/  include/  cmake/  引擎源码
├── third_party/ggml/       随仓携带的 ggml（MIT）
└── tests/  bench/  serve/  上游自带的测试与基准
```

工程记录（`docs/cyber/`）：

| 文档 | 内容 |
|---|---|
| [`00-overview.md`](docs/cyber/00-overview.md) | 它是什么、达成什么、边界在哪 |
| [`01-build.md`](docs/cyber/01-build.md) | 完整构建复现（含多架构参数） |
| [`02-pack-and-profile.md`](docs/cyber/02-pack-and-profile.md) | pack 与专家 profile 的生成与校验 |
| [`03-run.md`](docs/cyber/03-run.md) | 跑起来：参数、启动器、分词与反分词 |
| [`04-correctness.md`](docs/cyber/04-correctness.md) | 与外部参考的逐位对照与"首个分岔"判定 |
| [`05-performance.md`](docs/cyber/05-performance.md) | 速度账逐项拆分与测量纪律 |
| [`06-memory-budget.md`](docs/cyber/06-memory-budget.md) | 显存/内存总账与分档表 |
| [`07-shipping-params.md`](docs/cyber/07-shipping-params.md) | 发布参数与调优指引 |
| [`08-known-issues.md`](docs/cyber/08-known-issues.md) | 已知限制与未验清单 |
| [`09-fork-changelog.md`](docs/cyber/09-fork-changelog.md) | 本 fork 相对上游的全部改动 |

---

## 从源码构建

```bash
# 依赖：CUDA 13.x、Visual Studio Build Tools（C++ 与 CUDA 支持）、CMake ≥ 3.24、Ninja
cmake -S . -B build -G Ninja \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75;86;89;120
ninja -C build strata native_expert_parity
```

## 正确性

对照外部 `llama.cpp` 在**同一 prompt、同一贪心**下的逐位输出：

- 修复 **GDN z 闸门**（sigmoid → SiLU；原实现把门算错，导致输出从第一个 token 起就不可用）
  与 **MTP 草稿放置的 off-by-one** 之后，本引擎与参考**逐位相同到生成位 6**；
- 第一个分岔点在第 7 位，而参考在该位的 top-2 是 `198`(0.5046) 对 `271`(0.4724)，
  Δlogit ≈ 0.066 —— 属于**近并列的数值翻转**，不是结构性分歧；
- 专家缓存放开后（全常驻档）首个分岔推到第 15 位。

判定规则与负控见 [`docs/cyber/04-correctness.md`](docs/cyber/04-correctness.md)。

---

## 已知限制（详见 `docs/cyber/08-known-issues.md`）

- **没有服务端**：该构建没有 `serve` 子命令；`--serve` 旗标存在，但本项目未使用过。
- **引擎不自带 tokenizer**：必须由调用方提供 token id，仓库里的启动器负责这一步。
- **只在 Ada（sm_89）实机验证过**；其余架构是"二进制里确实装了对应机器码"的推断。
- **长上下文下 KV 搬运的速度代价未定论**。
- **MTP 投机解码当前不划算**：接受率能做到 0.55–0.58，但草稿头会吃掉 124–125 个专家槽，
  净效果仍是负的。

## 许可与致谢

本项目以 **MIT** 许可发布，是 [Strata](https://github.com/Niko1221/Strata)（MIT，
© 2026 Niko1221 and the Strata contributors）的 fork；随仓携带 [ggml](third_party/ggml)（MIT）；
构建期依赖 [llama.cpp](https://github.com/ggml-org/llama.cpp)（MIT，© 2023-2026 The ggml authors），
运行时配套分发的分词器二进制同样来自 llama.cpp。模型
[Cyber-Tiel-Coder-35B-A3B](https://huggingface.co/peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP)
由第三方发布、声明为 **MIT**，**本仓库不分发其权重**。完整归属见 [`NOTICE`](NOTICE)。
