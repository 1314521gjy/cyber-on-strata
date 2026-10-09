# 01 · 构建复现

> 目标：从本 fork 源码编出 `strata.exe`（多架构 CUDA 构建）。
> 证据等级：`A` 本机自实测 ｜ `B` 同机对照实现 ｜ `C` 外部实测 ｜ `D` 推导或出处自报。

---

## §1 依赖

| 依赖 | 版本 | 备注 |
|---|---|---|
| **CUDA Toolkit** | **13.x**（实测 13.3） | < 13.0 不要建 `sm_120`：`CMakeLists.txt` 对该组合发警告（上游实测：CUDA 12.8 建的 RTX 50 引擎长 prompt 崩溃） |
| CUDA 运行时 DLL | `cudart64_13.dll` / `cublas64_13.dll` | ⚠️ **必须把 CUDA 13 的 `bin` 目录放进 `PATH`**。否则进程以 `STATUS_DLL_NOT_FOUND (0xC0000135)` 立刻死亡、**日志 0 字节、批处理仍报成功** |
| **VS Build Tools** | MSVC **14.51.36231** | ⚠️ `cl.exe` / `link.exe` **不在 PATH**、`INCLUDE` / `LIB` 为空 ⇒ 必须先 `call` `vcvars64.bat` |
| CMake | 走 CUDA 语言支持（`enable_language(CUDA)`） | `find_package(CUDAToolkit REQUIRED)` |
| **Ninja** | 生成器 | 只改 `.cpp` 增量重编约 **1–3 分钟**；全量约 **78 s** |
| Python（仅工具链） | 带 `regex` / `numpy` | `tools/iq_pack.py`、`tools/make_profile.py`、`tools/strata_tokenizer.py` 需要 |

**`vcvars64` 前置（Windows）**：`call "<VS BuildTools>\VC\Auxiliary\Build\vcvars64.bat"`

> **已实测的坑**：`cl.exe` 是中文 MUI，CMake 认不出 `/showIncludes` 前缀 ⇒ **头文件依赖表是空的** ⇒
> **改 `.hpp` 后 ninja 不会重编依赖它的 `.cpp`**。改头文件必须显式碰一下消费者，或全量重编。

---

## §2 配置与构建

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_CUDA_ARCHITECTURES=75;86;89;120 ^
      -DSTRATA_MMQ_KQUANTS=ON ^
      -DSTRATA_BUILD_TESTS=ON
ninja -C build strata
```

### 2.1 `CMAKE_CUDA_ARCHITECTURES=75;86;89;120`

| 值 | 显卡代 | 说明 |
|---|---|---|
| `75` | RTX 20 系（Turing） | 引擎的最低门槛 |
| `86` | RTX 30 系（Ampere 消费级） | |
| `89` | RTX 40 系（Ada） | **本机验证的这一代** |
| `120` | RTX 50 系（Blackwell） | CUDA 13.0+ 才建 |

**为什么必须四个都进**：产物里 **SASS 与 PTX 两者都含这四代**（`cuobjdump` 实测），同一份 `strata.exe`
在 20/30/40/50 系上都能原生执行。CMake 侧有硬闸：`_base LESS 75` ⇒ `FATAL_ERROR`
（"Strata needs an NVIDIA GPU of compute capability 7.5 or newer"，`CMakeLists.txt`）。已验证的一条：
**多架构版与单架构版逐 token 等价**（同 prompt 同参数，输出 id 完全相同：`514 21 19 516 6000 11 514 9167`）。

> ⚠️ **40 系以外没有实机验证过**。判据是"文件里确实装了那一代的机器码"，不是"跑过"。

### 2.2 `STRATA_MMQ_KQUANTS=ON`

- `CMakeLists.txt` 的 `option(STRATA_MMQ_KQUANTS "Build the CUDA MMQ prompt kernels for Q4_K / Q5_K / Q5_1 experts" OFF)` —— **默认 OFF**。
- 打开后 `target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_KQUANTS=1)`，`src/prefill/moe_mmq.cu` 里由 `#ifdef` 守着的 Q4_K / Q5_K / Q5_1 路径才进编译。
- **为什么必须开**：Cyber 的专家正是 **Q4_K / Q5_K / Q6_K** 家族（不是 Q2_0 canvas）。

**生效自证（怎么证明它不是"只落了码"）**：日志必须出现

```
mmq plan: any=1 fallback=1 layers=40 gu_type=12 d_type=13 supported(gu)=1 supported(d)=1
```

`gu_type=12` = Q4_K、`d_type=13` = Q5_K，两个 `supported(...)=1` 才是"核在场且被选上"。

### 2.3 `STRATA_BUILD_TESTS=ON`

`CMakeLists.txt` 的 `option(STRATA_BUILD_TESTS ...)` 分支会额外带上：CUDA prompt-attention parity、
native experts 测试、`bench/micro/native_mmvq_multi`、图注册 / pinned capture 微基准、MMQ 相关测试
（`STRATA_MMQ_KQUANTS` 同时为真时）、命中顺序 parity（`bench/micro/hit_cpu_order_parity.cu`）。

⚠️ 一条**不同步**：`src/kernels/gdn_parity.cpp` 的期望值仍按 sigmoid 硬写 ⇒ **该测试预期会红**，
本次**未跑该测试、未同步改它**。

---

## §3 产物与校验

| 产物 | 读数 |
|---|---|
| `build/strata.exe`（多架构发货版） | **173,992,960 B** |
| sha256 | `144FB261CC8A9B23EC879714DBE0DDF72892F197758146415A56A2D4808F39F2` |
| 架构自证 | `cuobjdump` 实测 SASS 与 PTX **两者都含 `sm_75 / sm_86 / sm_89 / sm_120`** |
| 另一可选产物 | `strata-overlap.exe`（`ninja -C build strata strata-overlap` 顺带编出） |

| # | 判据 | 怎么变红 |
|---|---|---|
| 1 | `BUILD_EXIT = 0` | 非零 ⇒ 停 |
| 2 | 链接行在场：`[N/N] Linking CXX executable strata.exe` | 无此行 ⇒ 产物可能是陈旧件 |
| 3 | `error C` / `error LNK` / `error MSB` **= 0** | 任一命中 ⇒ 停（只有 C4324 / C4996 警告正常） |
| 4 | **exe 换代**：mtime 与 sha256 都与编译前不同 | sha 不变 ⇒ **改了没编进去**（依赖表为空时最易发生） |
| 5 | `cuobjdump` 四个架构都在 | 缺一代 ⇒ 那一代卡上无法原生执行 |
| 6 | `strata.exe --help` 退出码 = 0 | `-1073741515` ⇒ CUDA 13 DLL 不在 PATH |

---

## §4 出处

| 结论 | 证据 |
|---|---|
| 依赖清单、`vcvars64` 前置、MSVC 版本、头文件依赖表为空 | 本机工具链实测 + `CMakeLists.txt` |
| CUDA 13 DLL 不在 PATH 时的静默退出形态 | 本机实测：退出码 `-1073741515`、日志 0 字节 |
| `STRATA_MMQ_KQUANTS` / `STRATA_BUILD_TESTS` 的语义与默认值 | `CMakeLists.txt` 的 `option(...)` 与各分支逐行核对 |
| `sm_75` 门槛、`CUDA < 13.0 + sm_120` 警告 | `CMakeLists.txt` 的 `FATAL_ERROR` / `message(WARNING)` 分支 |
| 四架构产物、逐 token 等价 | 本机 `cuobjdump` 实测 + 同 prompt 同参数输出 id 逐位相同 |
| `mmq plan` 自证行 | 本机引擎日志自报行 |
| 产物大小与 sha256 | 本机 `Get-FileHash` |
| 增量 / 全量编译耗时 | 本机 `.ninja_log` + 构建日志时间戳 |
| `gdn_parity` 预期红 | `src/kernels/gdn_parity.cpp` 的硬写期望值 |
| `bench/micro/` 下的微基准目标 | `CMakeLists.txt` 的 `STRATA_BUILD_TESTS` 分支 |
