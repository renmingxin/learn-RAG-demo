/**
 * 第 2 步：文档切片（Text Splitter）
 *
 * 作用：把一份长文档拆成多个小片段（chunk），方便向量化和检索。
 * - chunkSize：每个片段的最大字符数
 * - chunkOverlap：相邻片段的重叠字符数，避免把一句话/一个语义单元切断
 *
 * 运行：node src/02-text-split.ts
 */
import 'dotenv/config'
import path from 'node:path'
import { DocxLoader } from '@langchain/community/document_loaders/fs/docx'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'

const DOC_PATH = path.resolve(import.meta.dirname, '../rag-docs/demo.docx')

const textSplit = async () => {
  // 1. 先加载文档，拿到完整文本
  const documents = await new DocxLoader(DOC_PATH).load()
  const content = documents[0]?.pageContent ?? ''

  // 2. 配置切分器：每段最多 50 字，相邻段重叠 20 字
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 50,
    chunkOverlap: 20,
  })

  // splitText 只返回字符串数组；如果要保留 metadata，用 splitDocuments(documents)
  const chunks = await textSplitter.splitText(content)

  console.log(`原文长度：${content.length}`)
  console.log(`切片数量：${chunks.length}`)

  chunks.forEach((chunk, index) => {
    console.log(`\n[${index}] 长度=${chunk.length}`)
    console.log(chunk.replace(/\n/g, ' '))
  })

  return chunks
}

textSplit()
