/**
 * 第 6 步：集大成示例 —— 完整的 RAG 问答链路
 *
 * 加载 → 切片 → 向量化入库（PG）→ 检索 → 拼上下文 → 交给大模型生成答案
 *
 * 运行：node src/06-rag-pipeline.ts
 */
import 'dotenv/config'
import path from 'node:path'
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai'
import { DocxLoader } from '@langchain/community/document_loaders/fs/docx'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
import { DistanceStrategy, PGVectorStore } from '@langchain/community/vectorstores/pgvector'
import { Document } from '@langchain/core/documents'
import type { PoolConfig } from 'pg'

const BASE_URL = process.env.LLM_BASE_URL || 'https://api.deepseek.com'
const API_KEY = process.env.LLM_API_KEY || ''
const MODEL = process.env.LLM_MODEL || 'deepseek-flash'
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || process.env.LLM_VECTOR_MODEL || 'doubao-embedding-vision'

const DOC_PATH = path.resolve(import.meta.dirname, '../rag-docs/demo.docx')

// 向量模型：负责把文本变成向量
const embeddings = new OpenAIEmbeddings({
  model: EMBEDDING_MODEL,
  apiKey: API_KEY,
  batchSize: 10,
  timeout: 30000,
  configuration: {
    baseURL: BASE_URL,
  },
})

// 对话模型：负责根据检索到的资料生成答案
const llm = new ChatOpenAI({
  model: MODEL,
  apiKey: API_KEY,
  timeout: 30000,
  configuration: {
    baseURL: BASE_URL,
  },
})

const pgConfig = {
  postgresConnectionOptions: {
    type: 'postgres',
    host: '127.0.0.1',
    port: 6543,
    user: 'renmx',
    password: '123456',
    database: 'embedding',
  } as PoolConfig,
  tableName: 'rag_demo_pipeline',
  columns: {
    idColumnName: 'id',
    vectorColumnName: 'vector',
    contentColumnName: 'content',
    metadataColumnName: 'metadata',
  },
  distanceStrategy: 'cosine' as DistanceStrategy,
}

const main = async () => {
  // ---------- 1. 文档加载 ----------
  const documents = await new DocxLoader(DOC_PATH).load()
  console.log('加载文档数：', documents.length)

  // ---------- 2. 文档切片 ----------
  const chunks = await new RecursiveCharacterTextSplitter({
    chunkSize: 50,
    chunkOverlap: 20,
  }).splitText(documents[0]?.pageContent ?? '')
  console.log('切片数：', chunks.length)

  // ---------- 3. 向量化 + 存储到 PG ----------
  const vectorStore = await PGVectorStore.initialize(embeddings, pgConfig)
  await vectorStore.delete({ filter: {} }) // 清空演示表，重复运行也不会积累重复数据
  await vectorStore.addDocuments(chunks.map(text => new Document({ pageContent: text })))
  console.log('已入库片段数：', chunks.length)

  // ---------- 4. 检索 ----------
  const question = 'RAG 的四个步骤分别是什么？文本切分的作用是什么？'
  const queryVector = await embeddings.embedQuery(question)
  const results = await vectorStore.similaritySearchVectorWithScore(queryVector, 4)

  // ---------- 5. 拼上下文 ----------
  const context = results.map(([doc]) => doc.pageContent).join('\n\n')
  console.log('\n检索到的片段：')
  results.forEach(([doc, distance], index) => {
    console.log(`[${index}] distance=${distance.toFixed(4)}：${doc.pageContent.replace(/\n/g, ' ').slice(0, 60)}`)
  })

  // ---------- 6. 生成答案 ----------
  const prompt = `你是一个严谨的文档问答助手。请只根据下面提供的资料回答用户问题；资料中没有的信息就回答"资料中未提及"。

资料：
${context}

问题：${question}`

  const answer = await llm.invoke(prompt)
  console.log('\n========== 模型回答 ==========')
  console.log(typeof answer.content === 'string' ? answer.content : JSON.stringify(answer.content))

  // 关闭连接池
  await vectorStore.end()
}

main()
