//接模型
import 'dotenv/config'
import 'dev-log-monitor/auto';
import path from "node:path"
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai'
import { CSVLoader } from '@langchain/community/document_loaders/fs/csv'
import { SRTLoader } from '@langchain/community/document_loaders/fs/srt'
import { DocxLoader } from '@langchain/community/document_loaders/fs/docx'
// 文档切片，递归字符文本分割器，将文档切分成多个段落
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
// 向量量化，储到内存中，将文档转换为多维向量,
import { MemoryVectorStore } from '@langchain/classic/vectorstores/memory'
// 向量存储到DB数据库中
import { DistanceStrategy, PGVectorStore } from '@langchain/community/vectorstores/pgvector'
// LangChain 的文档对象，向量库需要 Document 实例而不是纯字符串
import { Document } from '@langchain/core/documents'
import pg, { PoolConfig } from 'pg'


const BASE_URL = process.env.LLM_BASE_URL || 'https://api.deepseek.com'
const MODEL: string = process.env.LLM_MODEL || 'deepseek-flash'
const EMBEDDING_MODEL: string = process.env.EMBEDDING_MODEL || 'doubao-embedding-vision'
const API_KEY = process.env.LLM_API_KEY || 'sk-12e809b239ec45e49596c0d08c8350eb'

const pgConfig = {
  postgresConnectionOptions: {
    type: "postgres",
    host: "127.0.0.1",
    port: 6543,
    user: "renmx",
    password: "123456",
    database: "embedding",
  } as PoolConfig,
  tableName: "renmxlangchainjs",
  columns: {
    idColumnName: "id",
    vectorColumnName: "vector",
    contentColumnName: "content",
    metadataColumnName: "metadata",
  },
  // 支持的距离策略：cosine（默认）、innerProduct、euclidean
  distanceStrategy: "cosine" as DistanceStrategy,
};

// 加载文档
// 支持csv、srt、word格式
export const loadDocumentsFn: any = async (fileType: string) => {
  let data: any = null;
  switch (fileType) {
    case 'csv':
      const csvloader = new CSVLoader(path.resolve(import.meta.dirname, '../rag-docs/student.csv'))
      data = await csvloader.load();
      console.log('csv', data)
      return data;
    case 'srt':
      const srtloader = new SRTLoader(path.resolve(import.meta.dirname, '../rag-docs/demo.srt'))
      data = await srtloader.load();
      console.log('srt', data)
      return data;
    case "word":
      const wordloader = new DocxLoader(path.resolve(import.meta.dirname, '../rag-docs/demo.docx'))
      data = await wordloader.load();
      console.log('word', data)
      return data;
    default:
      break;
  }
  return data;
}

// 文档切片，将文档切分成多个段落，每个段落的字符数为4，每个段落之间的重叠字符数为2
const textSplit = async (fileType: string = 'word') => {
  const textSplitter = new RecursiveCharacterTextSplitter({
    // 每个段落的字符数
    chunkSize: 50,
    // 每个段落之间的重叠字符数
    chunkOverlap: 20,
  })
  const docs = await loadDocumentsFn(fileType);
  const documents = await textSplitter.splitText(docs[0]?.pageContent || '')
  console.log('documents', documents)
  return documents
}


const llm = new ChatOpenAI({
  model: MODEL,
  apiKey: API_KEY,
  timeout: 30000, // 30秒超时，防止卡住
  configuration: {
    baseURL: BASE_URL,
  }
})

const invoke = async () => {
  const res = await llm.invoke('你好,你是什么模型')
  console.log('res', res)

}

// 向量量化，将openai的embedding模型将切片后的文档进行量化
const embeddings = new OpenAIEmbeddings({
  model: EMBEDDING_MODEL,
  apiKey: API_KEY,
  batchSize: 10, // 方舟 Embedding 单次最多 10 条输入
  timeout: 30000, // 30秒超时，防止卡住
  configuration: {
    baseURL: BASE_URL,
  }
})

const invokeEmbeddings = async () => {
  const documents = await textSplit();
  const resEmbeddings = await embeddings.embedDocuments(documents)
  console.log('resEmbeddings', resEmbeddings)
}

// 向量存储到内存中
const invokeEmbeddingsStoreToMemory = async () => {
  const documents = await textSplit();
  // MemoryVectorStore存储到内存中，向量量化，将文档转换为多维向量，
  const vectorStore = await MemoryVectorStore.fromDocuments(
    documents.map(text => new Document({ pageContent: text })),
    embeddings,
  )
  const vector = await embeddings.embedQuery('本文档为示例资料，用于验证 RAG 系统能否正确读取 Word')
  // 向量相似度搜索，根据查询向量，返回最相似的向量,返回3个
  const vectorStoreRes = await vectorStore.similaritySearchVectorWithScore(vector, 3)
  console.log('vectorStoreRes', vectorStoreRes)

}
// invokeEmbeddingsStoreToMemory();


// 向量存储到数据库中
const invokeEmbeddingsStoreToDB = async () => {
  const documents = await textSplit('word');
  // MemoryVectorStore存储到数据库中，向量量化，将文档转换为多维向量，
  const vectorStore = await PGVectorStore.initialize(
    embeddings,
    pgConfig,
  )
  // vectorStore.addDocuments([new Document({ pageContent: documents[0] ?? '' })])
  await vectorStore.addDocuments(documents.map((text, index) => new Document({ pageContent: text, id: index.toString() })))
  const vector = await embeddings.embedQuery('完整的 RAG 问答通常包含四个步骤')
  // console.log('vectorStore', vectorStore)
  // console.log('vector', vector)

  const vectorStoreRes = await vectorStore.similaritySearchVectorWithScore(vector, 3)
  console.log('vectorStoreRes', vectorStoreRes)

  const matchValue = vectorStoreRes[0]?.[0]?.pageContent ?? ''
  // 调用llm，根据最相似的向量，返回四个步骤,拼接最相似的向量内容 
   const llmRes = await llm.invoke('四个步骤是？' + matchValue)
  console.log('llmRes', llmRes)

  // return vectorStore
}

invokeEmbeddingsStoreToDB();
