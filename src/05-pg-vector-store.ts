/**
 * 第 5 步：把向量存到 PostgreSQL（PGVectorStore + pgvector 扩展）
 *
 * 前提：
 * 1. 容器镜像必须带 pgvector：pgvector/pgvector:pg18（官方 postgres:18 没有该扩展）
 * 2. 目标数据库执行过一次 CREATE EXTENSION vector（PGVectorStore.initialize 会自动执行）
 *
 * 注意：cosine 策略下 similaritySearchVectorWithScore 返回的是“距离”，
 * 距离 = 1 - 余弦相似度，所以数值越小越相关（和内存版正好相反）。
 *
 * 运行：node src/05-pg-vector-store.ts
 */
import 'dotenv/config'
import path from 'node:path'
import { OpenAIEmbeddings } from '@langchain/openai'
import { DocxLoader } from '@langchain/community/document_loaders/fs/docx'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
import { DistanceStrategy, PGVectorStore } from '@langchain/community/vectorstores/pgvector'
import { Document } from '@langchain/core/documents'
import type { PoolConfig } from 'pg'

const BASE_URL = process.env.LLM_BASE_URL || 'https://api.deepseek.com'
const API_KEY = process.env.LLM_API_KEY || ''
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || process.env.LLM_VECTOR_MODEL || 'doubao-embedding-vision'

const DOC_PATH = path.resolve(import.meta.dirname, '../rag-docs/demo.docx')

const embeddings = new OpenAIEmbeddings({
  model: EMBEDDING_MODEL,
  apiKey: API_KEY,
  batchSize: 10,
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
  tableName: 'rag_demo_pgvector',
  columns: {
    idColumnName: 'id',
    vectorColumnName: 'vector',
    contentColumnName: 'content',
    metadataColumnName: 'metadata',
  },
  distanceStrategy: 'cosine' as DistanceStrategy,
}

const main = async () => {
  // 1. 加载 + 切片
  const content = (await new DocxLoader(DOC_PATH).load())[0]?.pageContent ?? ''
  const chunks = await new RecursiveCharacterTextSplitter({ chunkSize: 50, chunkOverlap: 20 }).splitText(content)

  // 2. 初始化向量库：建表 / 建扩展 / 建立索引
  const vectorStore = await PGVectorStore.initialize(embeddings, pgConfig)

  // 3. 先清空演示表，避免重复运行时插入重复数据（空 filter 等于删除全部）
  await vectorStore.delete({ filter: {} })

  // 4. 入库：addDocuments 内部会把文本批量向量化，再写入 PostgreSQL
  await vectorStore.addDocuments(chunks.map(text => new Document({ pageContent: text })))
  console.log('已写入 PostgreSQL 的片段数：', chunks.length)

  // 5. 检索
  const query = '完整的 RAG 问答通常包含四个步骤'
  const queryVector = await embeddings.embedQuery(query)
  const results = await vectorStore.similaritySearchVectorWithScore(queryVector, 3)

  console.log(`\n问题：${query}`)
  results.forEach(([doc, distance], index) => {
    // cosine 距离越小越相关；相似度 = 1 - 距离
    console.log(`\n[${index}] distance=${distance.toFixed(4)}  similarity=${(1 - distance).toFixed(4)}`)
    console.log(doc.pageContent.replace(/\n/g, ' '))
  })

  // 6. 关闭连接池，否则进程不会退出
  await vectorStore.end()
}

main()
