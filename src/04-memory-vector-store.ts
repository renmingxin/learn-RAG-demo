/**
 * 第 4 步：把向量存到内存中（MemoryVectorStore）
 *
 * 特点：进程内数组保存，不落地、重启即丢失，适合学习和快速验证。
 * 注意：MemoryVectorStore 返回的 score 是余弦“相似度”，越大越相似（和 PG 相反）。
 *
 * 运行：node src/04-memory-vector-store.ts
 */
import 'dotenv/config'
import path from 'node:path'
import { OpenAIEmbeddings } from '@langchain/openai'
import { DocxLoader } from '@langchain/community/document_loaders/fs/docx'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
import { MemoryVectorStore } from '@langchain/classic/vectorstores/memory'
import { Document } from '@langchain/core/documents'

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

const main = async () => {
  // 1. 加载 + 切片
  const content = (await new DocxLoader(DOC_PATH).load())[0]?.pageContent ?? ''
  const chunks = await new RecursiveCharacterTextSplitter({ chunkSize: 50, chunkOverlap: 20 }).splitText(content)

  // 2. 建库：fromDocuments 内部先向量化，再把向量存进内存
  const vectorStore = await MemoryVectorStore.fromDocuments(
    chunks.map(text => new Document({ pageContent: text })),
    embeddings,
  )
  console.log('已存入内存的片段数：', chunks.length)

  // 3. 检索：问题先向量化，再和库里每条向量算相似度
  const query = '完整的 RAG 问答通常包含四个步骤'
  const queryVector = await embeddings.embedQuery(query)
  // 向量相似度搜索，根据查询向量，返回最相似的向量,返回3个
  const results = await vectorStore.similaritySearchVectorWithScore(queryVector, 3)

  console.log(`\n问题：${query}`)
  results.forEach(([doc, score], index) => {
    // MemoryVectorStore 的 score 就是余弦相似度，越接近 1 越相关
    console.log(`\n[${index}] similarity=${score.toFixed(4)}`)
    console.log(doc.pageContent.replace(/\n/g, ' '))
  })
}

main()
