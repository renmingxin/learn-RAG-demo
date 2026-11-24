# 3.检索增强生成 (RAG) 精讲

随堂笔记与源码课后更新

## 课程目标

**初中级**

1. **建立大模型应用架构认知：** 理解大模型两个固有约束（上下文窗口有限、静态知识截止），明确 RAG 作为"外部记忆系统"的核心定位，理解为什么事实性知识应当与模型参数解耦。
2. **掌握 RAG 模块化流水线实现：** 能够拆解离线索引与在线服务两条独立流水线，独立完成文档加载、递归字符切分、嵌入模型、向量库、检索器、提示词模板的组件化开发，理解每个组件面向接口、可替换的设计原则。
3. **掌握知识库构建与文档切分：** 能够使用标准 Document 对象组织数据，理解切分粒度对检索效果的影响，掌握 RecursiveCharacterTextSplitter 的递归切分原理和 chunk size、overlap、separators 的配置方法，了解索引更新策略。
4. **掌握三种主流 RAG 架构：** 能够对比并实现 2-Step RAG（生产基线）、Agentic RAG（工具化检索）、Hybrid RAG（验证增强），理解三者在控制度、灵活性、延迟上的权衡与适用场景。
5. **掌握核心参数调优与基线验证：** 理解 chunk size、chunk overlap、top-k、相似度阈值、检索策略的影响机制，能够通过控制变量实验调优召回率、精确率，建立系统效果基线。

**高级**

1. **精通 RAG 架构选型：** 能够基于业务 SLA、延迟预算与成本目标，在三种 RAG 架构之间做选型，并判断何时需要引入查询改写、混合检索、重排序等增强手段。
2. **理解 GraphRAG 与知识图谱增强检索：** 理解向量 RAG 在全局问答和多跳推理上的盲区，掌握 GraphRAG 实体关系抽取、社区聚合、社区摘要的核心思想，了解 TypeScript 生态的落地现状。
3. **构建量化评估与持续调优体系：** 能够设计分层评估与消融实验，建立覆盖 MRR、NDCG、召回率的检索评估和基于 LLM-as-judge 的生成事实一致性评估，形成闭环优化。
4. **实现生产级可观测性与工程化交付：** 能够设计端到端执行追踪、结构化日志与黄金指标监控，制定增量索引策略与向量库版本规范，具备故障定位与根因分析能力。

## 课程大纲

> 原文档此节为内嵌组件（课程大纲），Markdown 无法呈现，请回原文档查看。

## 补充资料

- LangChain JS Retrieval：https://docs.langchain.com/oss/javascript/langchain/retrieval
- LangChain JS RAG Tutorial：https://docs.langchain.com/oss/javascript/langchain/rag
- LangChain.js RAG 教程：https://js.langchain.com/docs/tutorials/rag/
- 微软 GraphRAG 博客：https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/
- 微软 GraphRAG GitHub（Python）：https://github.com/microsoft/graphrag
- Neo4j × LangChain.js：https://www.neo4j.com/developer/genai-ecosystem/langchain-js/
- Neo4j GraphAcademy（GraphRAG）：https://graphacademy.neo4j.com/courses/genai-integration-langchain/
- LLM-as-Judge 评估方法：https://arxiv.org/abs/2306.05685
- 12-Factor 应用原则：https://12factor.net/zh_cn/

## 相关面试真题

### 向量 RAG 为什么答不好"全局性问题"和"多跳推理问题"？GraphRAG 是如何解决的？

**参考答案要点**：要回答这个问题，先要理解向量检索的本质。向量 RAG 把文档切成扁平的文本块，每个块独立嵌入，查询时按语义相似度返回 top-k。它的检索范围被限制在"与问题最相似的少数几个块"内，这带来两个结构性盲区。

**全局性问题**，如"这批文档主要讲了哪些主题""整个数据集中有哪些共性趋势"，答案并不存在于任何一个文本块中，而是分散在所有文档里，需要对整个语料做归纳。任何单个块都不包含完整答案，向量检索无论怎么调 top-k 都无法把分散的信息聚合成全局结论。**多跳推理问题**，如"A 公司收购的企业的创始人曾就读于哪所学校"，需要沿着"A 公司 → 被收购企业 → 创始人 → 学校"这条关系链跨多个文档跳转。向量检索只做单跳的相似度匹配，块与块之间没有显式的关系连接，无法沿关系链逐步追溯。

GraphRAG 的解决思路是把检索对象从"文本块"换成"关系结构"。离线阶段，它先用 LLM 从文档中抽取实体（人、组织、概念）和实体间关系，构建知识图谱；再用 Leiden 等社区发现算法把连接紧密的实体聚合成层级化社区，并为每个社区预先生成摘要。查询时，全局性问题直接用社区摘要做 global search——社区摘要本身就是对一组关联实体的归纳，天然适合回答全局问题；实体相关问题则从问题中的种子实体出发，沿图遍历收集相关子图（local search），多跳关系在图上就是一条可遍历的路径，从而支持多跳推理。

需要补充的是，GraphRAG 并非替代向量 RAG：单点事实查询、细节检索上向量 RAG 更快更准，而 GraphRAG 的图谱构建成本高、离线计算量大，应在确实存在全局问答和多跳推理需求时再引入。

### 请对比 2-Step RAG、Agentic RAG、Hybrid RAG 三种架构的优缺点，并说明各自的适用场景。

**参考答案要点**：三种 RAG 架构代表了从简单到复杂的演进路径，在可控性、灵活性和延迟之间做出了不同权衡，没有绝对的优劣，只有是否适合业务需求。

**2-Step RAG** 是最基础也最可靠的生产基线，遵循固定的"先检索后生成"流程。它的优势是行为完全可预测、延迟稳定、成本最低、最容易建立自动化测试体系；缺点是灵活性有限，无法处理多步推理问题。最适合 FAQ、内部文档助手、政策查询等需要强可控性的标准化场景。

**Agentic RAG** 将检索封装为标准工具，由 Agent 自主决定调用时机。它的优势是灵活性极高，能处理开放式研究、多工具协作等复杂任务；缺点是行为不可预测、调试难度大、成本波动明显。更适合研究助手、数据分析等对灵活性要求高于可控性的场景。

**Hybrid RAG** 介于两者之间，结合了多种检索策略和中间验证步骤。通过向量检索与关键词检索的互补提升召回鲁棒性，同时引入检索验证和回答校验环节保障质量。虽然系统复杂度更高，但能满足对召回率和准确性有较高要求的场景，特别是包含专有名词、编号、代码片段等精确匹配需求的查询。

架构选型核心原则：永远从最简单的 2-Step RAG 开始建立基线；只有当 2-Step 明确无法满足需求时再逐步增加复杂度；避免过早引入 Agentic RAG，大多数业务场景 2-Step 或 Hybrid 已足够；复杂度的增加必须带来可量化的业务价值。

### 生产环境中 RAG 系统最常见的问题是幻觉，请问你会从哪些方面系统性地解决这个问题？

参考答案要点：幻觉不是单一环节的缺陷，而是 RAG 全链路多个因素共同作用的结果，必须采用分层治理的系统性方案，不能只靠优化提示词解决。

**数据层（基础）：**确保数据源本身的准确性和完整性，建立定期清洗和更新机制；优化文档切分策略，保证每个 chunk 的语义独立性，避免关键信息被拆分；完善元数据管理，为每个文档块添加来源、版本、可信度等信息。

**检索层（核心）：**采用混合检索策略，结合向量检索的语义匹配和关键词检索的精确匹配；合理调整 top-k 和相似度阈值，在召回率和噪声之间取得平衡；高要求场景引入重排序（Reranker）模型，对初筛结果二次排序，把真正相关的内容排到前面。

**生成层（关键）：**用严格的提示词约束，明确要求"只基于上下文回答，不知道就说不知道"；强制模型引用来源，要求回答中的事实标注对应文档；使用低 temperature（0-0.3）减少生成随机性。

**验证层（兜底）：**实现回答自一致性检查，让模型验证回答是否与上下文一致；引入 LLM-as-judge 做自动化事实一致性评估；高风险场景建立人工审核兜底。

整体原则：预防优先，从数据和检索环节减少幻觉；分层治理，每个环节都有质量控制；闭环迭代，建立用户反馈收集错误案例持续优化；同时保持合理预期，幻觉无法 100% 消除，重点是控制在可接受范围内。

## 大模型应用的核心架构约束

所有大模型应用架构设计都必须围绕以下两个不可突破的约束展开：

1. **有限上下文窗口**：模型只能处理固定长度的输入序列，无法一次性摄入完整知识库
2. **静态知识截止**：模型知识固化于训练完成时刻，无法获取实时信息与私有领域知识

这两个约束决定了：**试图通过增大模型规模或延长上下文窗口来解决所有问题的方案，在工程上是不可行的**。即使上下文窗口持续扩大，把全部私有文档、实时数据每次都塞进提示词，也会带来成本失控、关键信息被淹没（"大海捞针"）和无法增量更新等问题。

![RAG 原理示意图](./rag-原理图.png)

*图：向量 RAG 与直接调用大模型两条流程的对比（课程讲义配图）*

### RAG：外部知识注入的主流方案

针对知识截止和私有知识缺失，业界的主流解法是 RAG（Retrieval-Augmented Generation，检索增强生成）：不在训练阶段把知识写进模型参数，而是在运行时先从外部知识库检索出相关内容，再把这些内容连同问题一起交给模型，让模型基于检索到的资料作答。

| **维度** | **把知识写进模型（训练/微调）** | **RAG 运行时检索注入** |
| --- | --- | --- |
| 知识更新 | 需重新训练，周期长、成本高 | 更新知识库即可，秒级生效 |
| 可追溯性 | 知识固化在参数中，无法溯源 | 每个回答可追溯到原始文档 |
| 推理成本 | 训练成本高，推理无额外检索开销 | 增加一次检索，但模型本身不变 |
| 适用知识 | 稳定的行为模式、领域风格 | 频繁变化、需审计的事实性信息 |

需要说明的是，微调（如 LoRA 等 PEFT 技术）解决的是另一类正交问题——模型的行为、格式与风格适配，而不是知识注入。本课程聚焦 RAG，微调不在本节范围内。一个重要的架构原则是：**不要试图用微调去记忆事实性知识**，频繁变化、需要可追溯和审计的事实信息应当走 RAG。

### 验证案例：学生信息查询系统

本课程将以一个学生信息查询系统贯穿始终。学生的姓名、学号、成绩等数据频繁变化、需要准确可查，显然不适合也不可能通过训练模型来记忆。RAG 的做法是：把学生数据整理成文档并构建向量索引，用户提问时先检索出相关学生记录，再让模型严格基于检索到的记录回答，数据更新时只需更新知识库，无需改动模型。后续的最小 RAG 实现、三种架构对比都将围绕这个案例展开。

## RAG 模块化检索流水线

Retrieval-Augmented Generation（检索增强生成）

### RAG的核心设计哲学

RAG的本质是**将知识从模型参数中解耦出来，构建独立的外部记忆系统**。这种解耦带来了三个关键的架构优势：

1. **知识可独立更新**：无需重新训练模型即可更新知识库
2. **来源可追溯**：所有回答都可以追溯到原始文档
3. **系统可水平扩展**：检索系统与生成系统可以独立扩容

### 两条独立流水线

一个生产级RAG系统由两条完全解耦的流水线组成：

#### 离线索引流水线（Indexing Pipeline）

```text
数据源 → 文档加载器 → 标准化Document → 文档切分器 → 嵌入模型 → 向量数据库
```

**架构特性**：

- 批量执行，对延迟不敏感
- 计算密集型，适合离线调度
- 幂等操作，支持增量更新

#### 在线服务流水线（Serving Pipeline）

```text
用户查询 → 查询嵌入 → 向量检索 → 上下文聚合 → 提示词模板 → 大模型 → 回答
```

**架构特性**：

- 实时执行，对延迟敏感
- IO密集型，需要高并发支持
- 无状态设计，支持水平扩展

### 七大核心组件的职责边界

LangChain官方将RAG系统抽象为七个标准组件，每个组件都有明确的职责与替换接口：

| 组件 | 核心职责 | 替换原则 | 官方实现示例 |
| --- | --- | --- | --- |
| **Document Loader** | 将异构数据源转换为统一Document对象 | 支持任意数据源接入 | CSVLoader、PDFLoader、WebLoader |
| **Text Splitter** | 将长文档切分为适合嵌入的语义块 | 根据文档类型调整切分策略 | RecursiveCharacterTextSplitter |
| **Embedding Model** | 将文本映射到高维向量空间 | 语义相似度匹配效果优先 | OpenAIEmbeddings、  <br>OllamaEmbeddings |
| **Vector Store** | 向量存储与近似最近邻搜索 | 根据规模与性能需求选择 | MemoryVectorStore、Chroma、Pinecone |
| **Retriever** | 面向应用的查询接口 | 实现不同的检索策略 | SimilarityRetriever、MMRRetriever |
| **Prompt Template** | 上下文与查询的标准化格式化 | 可插拔的提示词策略 | ChatPromptTemplate |
| **LLM** | 基于上下文生成最终回答 | 效果与成本的权衡 | ChatOpenAI、ChatAnthropic |

**架构原则**：所有组件都应该面向接口编程，替换任何一个组件都不应该影响其他组件的逻辑。

在进入实现之前，有必要理解嵌入模型（Embedding Model）为什么能完成"语义匹配"，这是整个 RAG 检索的数学基础。

嵌入模型是一个经过专门训练的神经网络，它把一段文本映射为一个高维数值向量（如 768 维、1536 维），这个向量可以理解为文本在"语义空间"中的坐标。训练的目标是让**语义相近的文本在空间中距离也相近**——"如何重置密码"和"忘记密码怎么办"虽然用词不同，但向量距离很近；而"今天天气真好"与它们语义无关，向量距离就很远。

因此，检索时不需要关键词字面匹配，只需把用户问题也转成向量，计算它与各文本块向量的距离（常用余弦相似度），距离最近的几个块就是语义最相关的内容。这也解释了两个常见现象：一是纯向量检索对专有名词、编号、错别字不敏感（语义相近即可命中），这是它相比传统关键词检索的优势；二是嵌入模型的选择、领域适配程度会直接决定检索质量——用通用嵌入模型去检索高度专业的医学、法律文本，语义空间可能区分不开，这正是需要挑选或微调嵌入模型的原因。

### 最小RAG系统实现

```typescript
// src/shared/knowledge-base.ts
import { CSVLoader } from "@langchain/community/document_loaders/fs/csv";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

/**
 * 构建学生知识库向量存储
 * 实现了离线索引流水线的完整逻辑
 */
export async function createStudentVectorStore() {
  // 1. 数据加载：异构数据源标准化
  const loader = new CSVLoader("data/students.csv");
  const documents = await loader.load();

  // 2. 文档切分：语义块划分
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 50,
    separators: ["\n\n", "\n", " ", ""],
  });
  const splits = await splitter.splitDocuments(documents);

  // 3. 向量索引：语义空间映射
  const embeddings = new OpenAIEmbeddings({
    model: process.env.RAG_EMBEDDING_MODEL || "doubao-embedding-vision",
  });

  // 返回标准Retriever接口
  return MemoryVectorStore.fromDocuments(splits, embeddings);
}
```

**架构解读**：

这段代码不是一个简单的示例，而是生产级RAG系统的核心骨架。它完美体现了RAG的模块化设计思想：

- 替换`CSVLoader`可以接入任何数据源
- 调整`TextSplitter`参数可以适配不同文档类型
- 替换OpenAIEmbeddings可以使用任何嵌入模型
- 替换`MemoryVectorStore`可以使用任何向量数据库

## 知识库构建——离线索引流水线设计

### Document对象标准化

LangChain定义了标准的`Document`接口，这是整个RAG系统的数据契约：

```typescript
interface Document {
  pageContent: string;  // 文档的纯文本内容
  metadata: Record<string, any>;  // 文档元数据
}
```

**架构重点**：`metadata`是RAG系统可观测性与可追溯性的基础。生产环境中必须至少包含以下字段：

- `source`：文档来源标识
- `last_updated`：最后更新时间
- `document_id`：唯一文档ID
- `chunk_id`：块在文档中的序号

### 文档切分策略设计

文档切分是RAG系统中最影响检索效果的环节，也是最容易被忽视的环节。

#### 切分粒度的权衡

- **过粗**：单个块包含过多无关信息，增加模型噪声
- **过细**：语义完整性被破坏，单个块无法独立表达完整含义

理解切分为何如此关键，要看到它同时影响检索和生成两端，而 chunk overlap（重叠）正是为缓解"语义被切断"设计的。一个文本块最终要被单独嵌入、单独检索、并作为相对独立的上下文交给模型。如果切分点恰好落在一个完整论述的中间——比如前半句"该功能的配置方法如下"在块 A，具体步骤在块 B——那么块 A 嵌入后语义残缺，可能无法被正确检索；即使检索到块 B，模型也可能缺少前提。这就是"过细破坏语义完整性"的含义。

chunk overlap 的作用是让相邻块之间保留一部分重叠文本（如块 A 的结尾若干字符也出现在块 B 开头），为跨块的语义提供缓冲，降低关键信息恰好被切在边界上的概率。但 overlap 也不是越大越好：它会产生冗余内容、增加索引数量和成本，需要在语义完整性与冗余之间权衡。RecursiveCharacterTextSplitter 之所以成为通用首选，是因为它不按固定长度硬切，而是优先按段落、再按句子、最后才按字符逐级回退，尽量在自然语义边界处断开——下一小节会专门解析它的工作原理。

#### 通用切分参数指南

| 文档类型 | 推荐chunkSize | 推荐chunkOverlap |
| --- | --- | --- |
| 结构化数据（CSV/JSON） | 100-300 | 20-50 |
| 纯文本（Markdown/TXT） | 300-1000 | 50-200 |
| 代码文件 | 100-500 | 20-100 |
| PDF文档 | 500-1500 | 100-300 |

#### RecursiveCharacterTextSplitter工作原理

这是LangChain官方推荐的通用文本切分器，它按照以下优先级递归切分文档：

1. 按段落分隔符（`\n\n`）切分
2. 按行分隔符（`\n`）切分
3. 按空格切分
4. 按字符切分

这种切分策略能够最大程度地保持语义单元的完整性。

### 生产级切分实现

```typescript
// src/shared/text-splitter.ts
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import type { Document } from "@langchain/core/documents";

/**
 * 生产级文档切分器
 * 保留完整元数据，支持不同文档类型的参数配置
 */
export async function splitDocuments(
  documents: Document[],
  options: {
    chunkSize?: number;
    chunkOverlap?: number;
    separators?: string[];
  } = {}
): Promise<Document[]> {
  const {
    chunkSize = 200,
    chunkOverlap = 50,
    separators = ["\n\n", "\n", " ", ""],
  } = options;

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize,
    chunkOverlap,
    separators,
    keepSeparator: false,
    stripWhitespace: true,
  });

  const splits = await splitter.splitDocuments(documents);

  // 为每个块添加唯一chunk_id
  return splits.map((doc, index) => ({
    ...doc,
    metadata: {
      ...doc.metadata,
      chunk_id: `${doc.metadata.source || "unknown"}_chunk_${index}`,
    },
  }));
}
```

### 索引更新策略

生产环境中，知识库不可能是静态的。常见的索引更新策略有三种：

1. **全量重建**：简单可靠，但更新延迟高，适合小规模知识库
2. **增量更新**：只更新变化的文档，需要维护文档版本号
3. **实时更新**：文档变化时立即触发索引更新，适合高频变化场景

**架构建议**：从全量重建开始，随着数据规模增长逐步演进到增量更新。

## RAG范式对比：三种核心架构的权衡

LangChain官方明确区分了三种主流的RAG架构，每种架构都有明确的适用场景与权衡。

### 官方架构对比矩阵

| 架构 | 核心描述 | 控制度 | 灵活性 | 延迟特性 | 典型用例 |
| --- | --- | --- | --- | --- | --- |
| **2-Step RAG** | 检索总是在生成之前执行 | ✅ 高 | ❌ 低 | ⚡ 可预测 | FAQ、文档助手、政策查询 |
| **Agentic RAG** | Agent自主决定何时及如何检索 | ❌ 低 | ✅ 高 | ⏳ 可变 | 研究助手、多工具协作、复杂推理 |
| **Hybrid RAG** | 结合固定流程与Agentic验证 | ⚖️ 中 | ⚖️ 中 | ⏳ 可变 | 领域QA、质量敏感型应用 |

**架构原则**：永远从最简单的2-Step RAG开始，只有当它无法满足需求时，再考虑更复杂的架构。

### 2-Step RAG：生产级基线架构

#### 架构定义

2-Step RAG是最简单也最可靠的RAG架构，它遵循严格的执行顺序：

```text
用户查询 → 检索 → 上下文格式化 → 生成 → 回答
```

#### 核心特性

- **确定性执行**：每次查询的执行路径完全相同
- **单次LLM调用**：延迟稳定，成本可控
- **易于评估**：可以建立完整的自动化测试体系
- **故障点清晰**：问题可以明确归因到检索或生成环节

#### 生产级实现

```typescript
// src/2.rag-patterns/1.two-step-rag.ts
import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { createStudentVectorStore } from "../shared/knowledge-base";

async function main(question: string) {
  // 初始化依赖（生产环境应使用依赖注入）
  const model = new ChatOpenAI({ temperature: 0 });
  const vectorStore = await createStudentVectorStore();
  const retriever = vectorStore.asRetriever({ k: 3 });

  // 标准化提示词模板
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `你是一个严谨的学生信息助手。
      请严格基于以下<context>中的内容回答问题。
      如果上下文中没有相关信息，请明确回答"上下文中没有足够信息"。
      禁止编造任何信息。

      <context>
      {context}
      </context>`,
    ],
    ["human", "{question}"],
  ]);

  // 构建RAG链
  const ragChain = prompt.pipe(model).pipe(new StringOutputParser());

  // 执行流程
  const docs = await retriever.invoke(question);
  const context = docs.map((doc) => doc.pageContent).join("\n\n");
  const answer = await ragChain.invoke({ context, question });

  // 生产环境应记录完整的执行日志
  console.log(JSON.stringify({
    question,
    retrieved_docs: docs.map((d) => d.metadata),
    answer,
  }, null, 2));

  return answer;
}

main(process.argv[2] || "heyi的成绩是多少？");
```

#### 适用场景

- 所有需要强可控性的业务场景
- 问题类型固定且可枚举的场景
- 对延迟和成本敏感的场景
- 需要严格合规与审计的场景

### Agentic RAG：工具化检索架构

#### 架构定义

Agentic RAG将检索能力封装为一个工具，由LLM驱动的Agent自主决定是否调用工具：

```text
用户查询 → Agent推理 → 工具调用决策 → 检索 → 上下文 → 生成 → 回答
```

#### 核心特性

- **自主决策**：Agent可以根据问题复杂度决定是否需要检索
- **多步推理**：支持多次工具调用和复杂推理链
- **可组合性**：可以轻松集成其他工具（数据库、API等）
- **不确定性**：执行路径不可预测，调试难度大

#### 官方标准实现

```typescript
// src/2.rag-patterns/2.agentic-rag.ts
import "dotenv/config";
import { tool } from "@langchain/core/tools";
import { createAgent } from "langchain";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import { createStudentVectorStore } from "../shared/knowledge-base";

async function main(question: string) {
  const model = new ChatOpenAI({ temperature: 0 });
  const vectorStore = await createStudentVectorStore();
  const retriever = vectorStore.asRetriever({ k: 3 });

  // 将检索器封装为标准LangChain工具
  const retrieveStudentKnowledge = tool(
    async ({ query }) => {
      const docs = await retriever.invoke(query);
      return docs.map((doc) =>
        `[来源: ${doc.metadata.source}] ${doc.pageContent}`
      ).join("\n\n");
    },
    {
      name: "retrieve_student_knowledge",
      description: "检索学生知识库，用于回答学生姓名、学号、成绩相关问题",
      schema: z.object({
        query: z.string().describe("用于检索的自然语言查询"),
      }),
    }
  );

  // 创建Agent
  const agent = createAgent({
    model,
    tools: [retrieveStudentKnowledge],
    systemPrompt: `
      你是学生信息助手。
      所有关于学生信息的问题，必须先调用retrieve_student_knowledge工具。
      只能使用工具返回的信息回答问题。
      如果工具返回的信息不足以回答问题，请明确说明。
    `,
  });

  // 执行Agent
  const result = await agent.invoke({
    messages: [{ role: "user", content: question }],
  });

  return result.messages.at(-1)?.content;
}

main(process.argv[2] || "heyi和lihua谁的数学成绩更高？");
```

#### 架构要点

- 工具的`description`和`schema`是Agent正确调用工具的关键
- 系统提示词必须明确约束工具的使用规则
- 生产环境必须记录所有工具调用的输入和输出
- Agentic RAG不是"更智能"的2-Step RAG，而是解决不同问题的架构

### Hybrid RAG：验证增强型架构

#### 架构定义

Hybrid RAG结合了2-Step RAG的可控性和Agentic RAG的灵活性，引入了中间验证步骤：

```text
用户查询 → 查询增强 → 多源检索 → 检索验证 → 上下文聚合 → 生成 → 回答验证 → 回答
```

#### 核心特性

- **多策略检索**：同时使用向量检索、关键词检索等多种策略
- **检索验证**：评估检索结果的相关性与充分性
- **回答验证**：检查生成回答与上下文的一致性
- **迭代优化**：支持多轮检索与生成循环

#### 实现

```typescript
// src/2.rag-patterns/3.hybrid-rag.ts
import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { createStudentVectorStore } from "../shared/knowledge-base";
import { mergeHybridResults } from "../shared/rag-utils";

async function main(question: string) {
  const model = new ChatOpenAI({ temperature: 0 });
  const vectorStore = await createStudentVectorStore();

  // 1. 查询增强：生成多个语义相似的查询
  const rewritePrompt = ChatPromptTemplate.fromTemplate(`
    将以下用户查询改写为3个不同的搜索查询，用于从学生知识库中检索信息。
    只输出查询，每行一个。

    原始查询：{question}
  `);

  const rewriteChain = rewritePrompt.pipe(model).pipe(new StringOutputParser());
  const rewrittenQueries = (await rewriteChain.invoke({ question }))
    .split("\n")
    .filter((q) => q.trim());

  // 2. 多策略并行检索
  const vectorResults = await Promise.all(
    rewrittenQueries.map((q) => vectorStore.similaritySearch(q, 2))
  );
  const keywordResults = await vectorStore.similaritySearch(question, 2);

  // 3. 结果合并与去重
  const mergedResults = mergeHybridResults(
    [...vectorResults.flat(), ...keywordResults],
    4
  );

  // 4. 生成回答
  const prompt = ChatPromptTemplate.fromTemplate(`
    基于以下上下文回答问题。
    上下文：{context}
    问题：{question}
  `);

  const ragChain = prompt.pipe(model).pipe(new StringOutputParser());
  const context = mergedResults.map((doc) => doc.pageContent).join("\n\n");
  const answer = await ragChain.invoke({ context, question });

  return answer;
}

main(process.argv[2] || "heyi 数学 分数");
```

## GraphRAG 与知识图谱增强检索【了解】

前面介绍的三种 RAG 架构，无论 2-Step、Agentic 还是 Hybrid，检索的基本单位都是"文本块"，核心匹配手段都是向量相似度。它们能覆盖绝大多数知识库问答场景，但在两类问题上存在结构性局限。GraphRAG（微软研究院 2024 年提出）通过引入知识图谱来补齐这些局限，是 RAG 进阶的重要方向。

### 向量 RAG 的两类检索盲区

第一类是**全局性问题（Global / Summarization Queries）**。如"这批文档反映了哪些主要主题""整个知识库中有哪些关键趋势"。这类问题需要理解整个语料，答案分散在大量文档中，没有任何一个文本块包含完整答案。向量检索只返回与问题最相似的少数块，无法对全库做归纳，容易以偏概全。

第二类是**多跳推理问题（Multi-hop Reasoning Queries）**。如"张三所在公司的创始人毕业于哪所大学"，需要沿"张三 → 公司 → 创始人 → 学校"的关系链跨多个文档逐步推导。向量检索是"单跳"的相似度匹配，文本块之间没有显式的关系连接，无法沿关系链追溯。

这两类问题的共同根因是：**向量 RAG 只保存了文本块的语义表示，却丢失了实体以及实体之间的关系结构**。GraphRAG 正是把这部分结构显式地建模出来。

### GraphRAG 的核心流程

GraphRAG 同样分为离线构建和在线查询两个阶段，但离线阶段的产物从"向量集合"变成了"知识图谱 + 社区摘要"。

**离线构建阶段：**

1. **文本切分与实体关系抽取**：先用 LLM 遍历文档，抽取其中的实体（人、组织、地点、概念等）以及实体之间的关系，例如"张三 —— 任职于 —— A 公司"。
2. **构建知识图谱**：将抽取的实体作为节点、关系作为边，合并跨文档出现的同一实体，形成一张全局的实体关系图。
3. **社区发现（Community Detection）**：使用 Leiden 等图聚类算法，把连接紧密的实体划分为若干"社区"，并形成层级化结构（小社区可进一步聚合成大社区）。
4. **生成社区摘要**：让 LLM 阅读每个社区内的实体和关系，预先生成该社区的主题摘要和关键结论。

**在线查询阶段**，GraphRAG 提供两种检索模式：

- **Local Search（局部检索）**：从问题中识别种子实体，在知识图谱上沿关系遍历，收集邻近实体、关系、相关文本块组成子图，再交给模型回答，适合实体相关和多跳推理问题。
- **Global Search（全局检索）**：把问题与各社区摘要做匹配，选出相关社区，基于这些社区摘要（而非原始文本块）生成回答，适合需要跨全库归纳的主题性、总结性问题。

### 向量 RAG 与 GraphRAG 的本质对比

| **对比维度** | **向量 RAG** | **GraphRAG** |
| --- | --- | --- |
| 检索对象 | 扁平的文本块 | 实体、关系构成的图结构与社区 |
| 匹配方式 | 语义向量相似度 | 图遍历 + 社区摘要匹配 |
| 擅长问题 | 单点事实查询、细节检索 | 全局总结、多跳推理、关系比较 |
| 跨文档关联 | 弱，块之间无显式连接 | 强，实体关系天然跨文档连接 |
| 离线构建成本 | 低，主要是嵌入计算 | 高，需大量 LLM 调用做抽取和摘要 |
| 结果可解释性 | 可溯源到文本块 | 可展示推理路径与关系网络 |

一句话概括差异：**向量 RAG"检索相似的文本"，GraphRAG"检索关联的结构"**。需要强调的是，GraphRAG 并不是用来替代向量 RAG 的——在单跳事实查询和细节问答上，向量 RAG 更简单、更快、往往也更准；GraphRAG 的价值集中在需要全局视角和关系推理的问题上，实践中两者常结合使用。

### TypeScript 生态现状与落地建议

了解概念之后，需要客观认识工程落地现状，避免在课程中给出无法运行的方案：

- **微软官方 GraphRAG 仅有 Python 实现**（开源在 microsoft/graphrag），完整的"实体抽取 + Leiden 社区发现 + 社区摘要 + Local/Global Search"流水线目前没有官方的 JavaScript/TypeScript 版本。
- 社区虽有 graphrag-ts 等第三方移植包，但成熟度很低、下载量极小，尚不足以用于生产环境，不建议直接采用。
- TypeScript 生态中相对可落地的，是**"知识图谱 + Text-to-Cypher + 图增强向量检索"这一路线**，以图数据库为核心：使用 Neo4j、FalkorDB 等图数据库存储实体关系，通过自然语言生成图查询语句（Cypher）检索，或在向量检索时叠加图关系扩展。

以 Neo4j 配合 LangChain.js 为例，能力包括：向量索引与混合检索、图查询包装器、GraphCypherQAChain（自然语言转 Cypher 查图再回答）、以及用 LLM 从文档抽取实体关系建图的 Graph Transformer。需要注意的是，部分能力依赖的集成包存在维护状态变化，实际选型时应以官方最新文档为准。FalkorDB 则提供了独立维护的 LangChain TypeScript 集成，是 Redis 生态下的新兴选择。

**落地建议：**如果团队是 Python 技术栈且确实有大量全局问答、多跳推理需求，可以直接评估微软官方 GraphRAG；如果是 TypeScript 技术栈，更现实的路径是先引入图数据库做"知识图谱 + Text-to-Cypher"，把实体关系用起来，而不是追求一步到位复刻完整 GraphRAG。无论哪种路线，都应先用向量 RAG 建立基线，确认瓶颈确实来自全局/多跳问题后再引入图谱，避免为复杂而复杂。

## RAG系统优化：调参与评估体系【了解】

### 分层评估模型

生产级RAG系统必须建立分层评估体系，不能只看最终回答的质量：

| 评估层次 | 评估指标 | 评估方法 |
| --- | --- | --- |
| **数据层** | 覆盖率、准确率、新鲜度 | 自动化数据校验 |
| **索引层** | 切分准确率、嵌入质量 | 人工抽样评估 |
| **召回层** | 召回率、精确率、MRR、NDCG | 基于标注数据集的自动化评估 |
| **上下文层** | 相关性、冗余度、噪声水平 | 自动化评分 + 人工审核 |
| **生成层** | 事实一致性、相关性、有用性 | LLM-as-judge + 人工评估 |

### 核心调参策略

#### 检索调参

- **k值**：控制返回的文档数量，通常3-5是比较均衡的选择
- **检索策略**：

  - Similarity：纯相似度搜索，速度最快
  - MMR：最大边际相关性，兼顾相关性与多样性
  - ScoreThreshold：过滤相似度低于阈值的文档

除了调整 k 值和检索策略，生产系统还常引入**重排序（Rerank）**来进一步提升检索质量，需要理解它与向量检索的分工。向量检索的优势是快——可以通过近似最近邻（ANN）索引在百万、千万级向量中毫秒级召回，但它为了速度牺牲了一定精度，并且只用一个"语义相似度"维度排序，容易召回一些"看着相关、其实答非所问"的块。Rerank 的做法是两阶段：先用向量检索快速召回一个较大的候选集（如 top-20），再用一个更精确但更慢的 Cross-Encoder 重排序模型，对"问题 + 每个候选块"逐对精细打分，重新排序后取前若干个交给模型。

为什么 Rerank 更准？向量检索中问题和文档是**分别独立编码**的，两者在编码时看不到对方，丢失了很多细粒度交互；而 Cross-Encoder 把问题和候选块**拼在一起输入模型**，能捕捉"这个块是否真的能回答这个问题"的深度关联。代价是计算量大、不能对全库做，所以只对召回的小候选集使用。一个典型组合是"向量召回 top-20 → Rerank 精排取 top-5"，在延迟可接受的前提下显著提升最终上下文的准确率，是 Hybrid RAG 和高质量场景的常用手段。

#### 提示词调参

- 明确约束模型只能基于上下文回答
- 加入明确的拒答指令
- 要求模型引用来源信息
- 使用结构化输出格式

### RAG系统优化：调参与评估体系的生产级落地

RAG系统的质量不是靠感觉判断的，而是靠**量化指标**和**系统化实验**来提升的。本节将提供一套完整的、可直接落地的调参方法论和评估体系实现。

#### 调参实验设计：科学的参数优化流程

调参每次只改变一个参数，保持其他参数不变，观察指标的变化。

##### 标准调参实验流程

1. **建立基线**：使用默认参数运行，记录所有评估指标
2. **单变量实验**：每次只调整一个参数，运行完整的评估流程
3. **结果对比**：对比调整前后的指标变化，确定最优值
4. **迭代优化**：重复步骤2-3，直到所有关键参数都优化完毕

##### 核心参数调优顺序与实验设计

| 参数 | 默认值 | 实验范围 | 评估指标 | 最优值判断标准 |
| --- | --- | --- | --- | --- |
| **chunkSize** | 200 | 100, 200, 300, 500 | 召回率、精确率 | 召回率最高的最小值 |
| **chunkOverlap** | 50 | 0, 25, 50, 100 | 召回率 | 召回率不再提升的最小值 |
| **k值** | 3 | 1, 2, 3, 5, 10 | 召回率、上下文长度 | 召回率最高且上下文长度不超过模型窗口的20% |
| **检索策略** | Similarity | Similarity, MMR | 精确率、多样性 | 精确率最高的策略 |
| **嵌入模型** | nomic-embed-text | nomic-embed-text, text-embedding-3-small | 召回率 | 召回率最高的模型 |

##### 自动化调参脚本示例

```typescript
// src/experiments/tune-chunk-size.ts
import { createStudentVectorStore } from "../shared/knowledge-base";
import { evaluateRetriever } from "../evaluation/retriever-evaluator";

async function main() {
  const chunkSizes = [100, 200, 300, 500];
  const results = [];

  for (const chunkSize of chunkSizes) {
    console.log(`\n=== 测试 chunkSize: ${chunkSize} ===`);

    // 创建不同chunkSize的向量库
    const vectorStore = await createStudentVectorStore({ chunkSize, chunkOverlap: 50 });
    const retriever = vectorStore.asRetriever({ k: 3 });

    // 评估召回效果
    const metrics = await evaluateRetriever(retriever, "data/test-queries.json");

    results.push({
      chunkSize,
      ...metrics,
    });

    console.log(`召回率: ${metrics.recall.toFixed(2)}`);
    console.log(`精确率: ${metrics.precision.toFixed(2)}`);
  }

  // 输出最优结果
  const bestResult = results.reduce((best, current) =>
    current.recall > best.recall ? current : best
  );

  console.log("\n=== 最优参数 ===");
  console.log(`最佳 chunkSize: ${bestResult.chunkSize}`);
  console.log(`最高召回率: ${bestResult.recall.toFixed(2)}`);
}

main();
```

#### 召回层评估：RAG质量的基础

召回层是RAG系统的基础，如果召回不到正确的文档，再好的模型也无法生成正确的回答。召回层的评估可以完全自动化，不需要人工参与。

##### 标注数据集格式

首先需要创建一个标注数据集，包含查询和对应的正确文档ID：

文件：`data/test-queries.json`

```json
[
  {
    "query": "heyi的成绩是多少？",
    "relevant_doc_ids": ["student.csv_chunk_0"]
  },
  {
    "query": "liuyi的学号是多少？",
    "relevant_doc_ids": ["student.csv_chunk_1"]
  },
  {
    "query": "谁的数学成绩最高？",
    "relevant_doc_ids": ["student.csv_chunk_0", "student.csv_chunk_1", "student.csv_chunk_2"]
  }
]
```

##### 召回层评估实现

```typescript
// src/evaluation/retriever-evaluator.ts
import type { Retriever } from "@langchain/core/retrievers";
import type { Document } from "@langchain/core/documents";

type TestQuery = {
  query: string;
  relevant_doc_ids: string[];
};

type RetrieverMetrics = {
  recall: number;  // 召回率：正确召回的文档数 / 总相关文档数
  precision: number; // 精确率：正确召回的文档数 / 总召回文档数
  f1: number; // F1分数：召回率和精确率的调和平均
  mrr: number; // 平均倒数排名：第一个相关文档的排名倒数的平均值
};

export async function evaluateRetriever(
  retriever: Retriever,
  testQueriesPath: string
): Promise<RetrieverMetrics> {
  const testQueries: TestQuery[] = await import(testQueriesPath);

  let totalRecall = 0;
  let totalPrecision = 0;
  let totalMrr = 0;

  for (const testQuery of testQueries) {
    const docs = await retriever.invoke(testQuery.query);
    const retrievedDocIds = docs.map((doc) => doc.metadata.chunk_id);
    const relevantDocIds = testQuery.relevant_doc_ids;

    // 计算真阳性：既在召回结果中又在相关文档中的数量
    const truePositives = retrievedDocIds.filter((id) =>
      relevantDocIds.includes(id)
    ).length;

    // 计算召回率和精确率
    const recall = truePositives / relevantDocIds.length;
    const precision = truePositives / retrievedDocIds.length;

    // 计算MRR
    let mrr = 0;
    for (let i = 0; i < retrievedDocIds.length; i++) {
      if (relevantDocIds.includes(retrievedDocIds[i])) {
        mrr = 1 / (i + 1);
        break;
      }
    }

    totalRecall += recall;
    totalPrecision += precision;
    totalMrr += mrr;
  }

  return {
    recall: totalRecall / testQueries.length,
    precision: totalPrecision / testQueries.length,
    f1: 2 * (totalPrecision * totalRecall) / (totalPrecision + totalRecall),
    mrr: totalMrr / testQueries.length,
  };
}
```

#### 生成层评估：事实一致性与相关性

生成层的评估比召回层复杂，需要判断回答的事实一致性、相关性和有用性。常用的方法是**LLM-as-judge**，即使用另一个大模型来评估生成的回答。

##### 生成层评估实现

```typescript
// src/evaluation/generation-evaluator.ts
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

type GenerationTest = {
  query: string;
  context: string;
  expected_answer: string;
};

type GenerationMetrics = {
  factuality: number; // 事实一致性：0-1分
  relevance: number; // 相关性：0-1分
  helpfulness: number; // 有用性：0-1分
  overall: number; // 综合得分：0-1分
};

export async function evaluateGeneration(
  testCases: GenerationTest[],
  modelName: string = "doubao-seed-2.0-mini"
): Promise<GenerationMetrics> {
  const model = new ChatOpenAI({ model: modelName, temperature: 0 });

  const prompt = ChatPromptTemplate.fromTemplate(`
    你是一个专业的RAG系统评估员。请根据以下上下文和预期答案，评估模型生成的回答质量。

    上下文：{context}
    问题：{query}
    预期答案：{expected_answer}
    模型回答：{generated_answer}

    请从以下三个维度进行评分，每个维度0-1分：
    1. 事实一致性：回答是否与上下文一致，没有编造信息
    2. 相关性：回答是否与问题相关，没有答非所问
    3. 有用性：回答是否清晰、准确、有用

    只输出三个分数，用逗号分隔，不要其他内容。
    例如：0.8,0.9,0.7
  `);

  const chain = prompt.pipe(model).pipe(new StringOutputParser());

  let totalFactuality = 0;
  let totalRelevance = 0;
  let totalHelpfulness = 0;

  for (const testCase of testCases) {
    // 这里需要调用你的RAG系统生成回答
    const generated_answer = await yourRAGSystem(testCase.query);

    const result = await chain.invoke({
      context: testCase.context,
      query: testCase.query,
      expected_answer: testCase.expected_answer,
      generated_answer,
    });

    const [factuality, relevance, helpfulness] = result.split(",").map(Number);

    totalFactuality += factuality;
    totalRelevance += relevance;
    totalHelpfulness += helpfulness;
  }

  const count = testCases.length;
  const factuality = totalFactuality / count;
  const relevance = totalRelevance / count;
  const helpfulness = totalHelpfulness / count;

  return {
    factuality,
    relevance,
    helpfulness,
    overall: (factuality + relevance + helpfulness) / 3,
  };
}
```

#### 生产级可观测性

生产环境中的RAG系统必须具备完整的可观测性，能够记录每一次请求的完整执行过程，便于排查问题和优化系统。

##### 标准日志格式

```typescript
// src/shared/logger.ts
import { createLogger, format, transports } from "winston";

const logger = createLogger({
  level: "info",
  format: format.combine(
    format.timestamp(),
    format.json()
  ),
  transports: [
    new transports.File({ filename: "logs/rag-error.log", level: "error" }),
    new transports.File({ filename: "logs/rag-combined.log" }),
  ],
});

// RAG请求日志结构
type RagRequestLog = {
  request_id: string;
  timestamp: string;
  user_query: string;
  retrieved_docs: Array<{
    chunk_id: string;
    source: string;
    similarity: number;
  }>;
  context: string;
  generated_answer: string;
  latency_ms: number;
  model: string;
  status: "success" | "error";
  error_message?: string;
};

export function logRagRequest(log: RagRequestLog) {
  logger.info(log);
}
```

##### 关键监控指标

生产环境中需要监控以下核心指标：

1. **请求量**：每秒请求数（QPS）
2. **延迟**：平均延迟、P95延迟、P99延迟
3. **错误率**：请求失败的比例
4. **召回指标**：平均召回率、平均精确率
5. **生成指标**：平均事实一致性、平均相关性
6. **成本**：平均每次请求的token消耗和费用

#### 常见问题排查流程

当RAG系统出现质量问题时，按照以下流程排查：

1. **检查召回层**：运行召回层评估脚本，确认是否召回了正确的文档

   - 如果召回率低：优化chunk参数、嵌入模型或检索策略
   - 如果召回率正常：继续检查生成层
2. **检查上下文**：查看日志中记录的上下文，确认是否包含回答问题所需的所有信息

   - 如果上下文不完整：增加k值或优化chunk切分
   - 如果上下文完整：继续检查生成层
3. **检查提示词**：查看日志中记录的完整prompt，确认是否有足够的约束

   - 如果提示词约束不足：加强提示词，明确要求只基于上下文回答
   - 如果提示词正常：尝试降低temperature或更换模型
4. **检查模型**：使用相同的上下文和问题，直接调用模型API，观察输出

   - 如果模型输出错误：尝试使用更大的模型或PEFT微调
   - 如果模型输出正确：检查RAG系统的其他环节
