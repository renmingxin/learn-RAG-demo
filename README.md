# RAG 学习示例（LangChain.js + pgvector）

通过 6 个循序渐进的可独立运行示例，演示 RAG（检索增强生成）的完整链路：

> 文档加载 → 文本切片 → 向量化 → 存储（内存 / PostgreSQL）→ 相似度检索 → LLM 生成答案

## 技术栈

- [LangChain.js](https://www.langchain.com/)：`@langchain/openai`、`@langchain/community`、`@langchain/textsplitters`
- 大模型 / Embedding：火山方舟（Ark）OpenAI 兼容接口，`deepseek-v4.1-flash` + `doubao-embedding-vision`
- 向量数据库：PostgreSQL 18 + [pgvector](https://github.com/pgvector/pgvector) 扩展
- 构建：[tsdown](https://github.com/rolldown/tsdown)（Node 24 也可直接运行 TS 源码）

## 分阶段示例（src/）

| 文件 | 内容 |
|---|---|
| `01-load-documents.ts` | 文档加载：支持 word（docx）、csv、srt 三种格式 |
| `02-text-split.ts` | 文档切片：`RecursiveCharacterTextSplitter`（chunkSize=50，overlap=20） |
| `03-document-embeddings.ts` | 文档向量化：`embedDocuments` / `embedQuery`，手写余弦相似度验证排序 |
| `04-memory-vector-store.ts` | 存储到内存：`MemoryVectorStore`，score 为相似度（越大越相关） |
| `05-pg-vector-store.ts` | 存储到 PG：`PGVectorStore` + pgvector，score 为余弦距离（越小越相关） |
| `06-rag-pipeline.ts` | 集大成：加载 → 切片 → 入库 → 检索 → 拼上下文 → LLM 回答 |

## 快速开始

### 1. 安装依赖

```powershell
pnpm install
```

### 2. 启动 PostgreSQL（带 pgvector）

```powershell
pnpm docker:up
```

容器把主机 `6543` 映射到容器内 `5432`（`ports: 6543:5432`），数据库为 `embedding`。

### 3. 配置环境变量

在项目根目录创建 `.env`（参考以下键名，填入你自己的值）：

```ini
LLM_BASE_URL=https://ark.cn-beijing.volces.com/api/plan/v3
LLM_MODEL=deepseek-v4.1-flash
LLM_VECTOR_MODEL=doubao-embedding-vision
LLM_API_KEY=你的方舟 API Key
```

### 4. 运行

直接跑源码（Node 24+）：

```powershell
node src/01-load-documents.ts
node src/06-rag-pipeline.ts
```

或先构建再跑产物：

```powershell
pnpm build          # tsdown 会把 src/*.ts 每个文件都打成独立入口
node dist/06-rag-pipeline.mjs
```

## 项目结构

```
rag/
├── docker/
│   └── docker-compose.yml   # pgvector/pgvector:pg18，端口 6543->5432
├── rag-docs/                # 示例文档：demo.docx / student.csv / demo.srt
├── src/                     # 01~06 分阶段示例 + index.ts（原始整合版）
├── .env                     # 模型与数据库配置（不提交）
├── package.json
└── tsdown.config.ts         # entry: src/*.ts，每个文件独立入口
```

## 踩坑记录

1. **`extension "vector" is not available`**：官方 `postgres:18` 镜像不带 pgvector，必须换 `pgvector/pgvector:pg18` 镜像。
2. **连接报 `early eof`**：端口映射写成了 `6543:6543`，但 PG 在容器内监听 `5432`，应写 `6543:5432`。
3. **检索分数理解反了**：`PGVectorStore`（cosine 策略）返回的是**距离** = `1 - 余弦相似度`，越小越相关；而 `MemoryVectorStore` 返回的是**相似度**，越大越相关。
4. **脚本跑完不退出**：`PGVectorStore` 持有连接池，结束时需调用 `await vectorStore.end()`。

## License

ISC
