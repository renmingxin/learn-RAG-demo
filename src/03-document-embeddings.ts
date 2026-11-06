/**
 * 第 3 步：文档向量化（Embedding）
 *
 * 作用：把文本变成一组浮点数（向量）。语义越接近的文本，向量方向越接近，
 * 之后就能用余弦相似度衡量“两段文本有多像”。
 *
 * - embedDocuments：批量把多个片段转成向量（入库时用）
 * - embedQuery：把用户的问题转成向量（检索时用，字符串形式相同但语义上区分开）
 *
 * 运行：node src/03-document-embeddings.ts
 */
import 'dotenv/config'
import path from 'node:path'
import { OpenAIEmbeddings } from '@langchain/openai'
import { DocxLoader } from '@langchain/community/document_loaders/fs/docx'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'

const BASE_URL = process.env.LLM_BASE_URL || 'https://api.deepseek.com'
const API_KEY = process.env.LLM_API_KEY || ''
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || process.env.LLM_VECTOR_MODEL || 'doubao-embedding-vision'

const DOC_PATH = path.resolve(import.meta.dirname, '../rag-docs/demo.docx')

const embeddings = new OpenAIEmbeddings({
  model: EMBEDDING_MODEL,
  apiKey: API_KEY,
  batchSize: 10, // 方舟 Embedding 单次最多 10 条输入
  timeout: 30000,
  configuration: {
    baseURL: BASE_URL,
  },
})

// 余弦相似度：范围 [-1, 1]，越接近 1 表示语义越接近
const cosineSimilarity = (a: number[], b: number[]) => {
  let dot = 0
  let normA = 0
  let normB = 0
  for (let i = 0; i < a.length; i++) {
    const ai = a[i] ?? 0
    const bi = b[i] ?? 0
    dot += ai * bi
    normA += ai * ai
    normB += bi * bi
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB))
}

const main = async () => {
  // 1. 加载 + 切片
  const content = (await new DocxLoader(DOC_PATH).load())[0]?.pageContent ?? ''
  const chunks = await new RecursiveCharacterTextSplitter({ chunkSize: 50, chunkOverlap: 20 }).splitText(content)

  // 2. 批量向量化：返回 number[][]，每个片段对应一条向量
  const vectors = await embeddings.embedDocuments(chunks)
  console.log('片段数量：', chunks.length)
  console.log('向量维度：', vectors[0]?.length)
  console.log('前 5 个数值：', vectors[0]?.slice(0, 5))

  // 3. 把问题也向量化，再算余弦相似度，验证语义相近的片段得分更高
  const query = 'RAG 的四个步骤分别是什么'
  const queryVector = await embeddings.embedQuery(query)

  const ranked = chunks
    .map((chunk, index) => ({
      chunk,
      // 这里计算的是“相似度”，不是距离，所以越大越相关
      similarity: cosineSimilarity(queryVector, vectors[index] ?? []),
    }))
    .sort((a, b) => b.similarity - a.similarity)

  console.log(`\n问题：${query}`)
  ranked.forEach(({ chunk, similarity }, index) => {
    console.log(`\n[${index}] similarity=${similarity.toFixed(4)}`)
    console.log(chunk.replace(/\n/g, ' '))
  })
}

main()
