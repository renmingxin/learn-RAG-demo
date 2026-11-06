/**
 * 第 1 步：文档加载
 *
 * 作用：把不同格式的文件读成 LangChain 的 Document 对象。
 * 每个 Document 至少包含 pageContent（文本）和 metadata（来源等信息）。
 *
 * 运行：node src/01-load-documents.ts
 */
import 'dotenv/config'
import path from 'node:path'
import { CSVLoader } from '@langchain/community/document_loaders/fs/csv'
import { SRTLoader } from '@langchain/community/document_loaders/fs/srt'
import { DocxLoader } from '@langchain/community/document_loaders/fs/docx'

type FileType = 'word' | 'csv' | 'srt'

// rag-docs 与 src 平级，所以从 src 往外退一层
const DOC_DIR = path.resolve(import.meta.dirname, '../rag-docs')

const loadDocuments = async (fileType: FileType) => {
  switch (fileType) {
    case 'word':
      return new DocxLoader(path.join(DOC_DIR, 'demo.docx')).load()
    case 'csv':
      return new CSVLoader(path.join(DOC_DIR, 'student.csv')).load()
    case 'srt':
      return new SRTLoader(path.join(DOC_DIR, 'demo.srt')).load()
    default:
      return []
  }
}

const main = async () => {
  for (const fileType of ['word', 'csv', 'srt'] as const) {
    const documents = await loadDocuments(fileType)
    console.log(`\n========== ${fileType} ==========`)
    console.log('文档数量：', documents.length)

    const first = documents[0]
    if (first) {
      console.log('metadata：', first.metadata)
      console.log('内容预览：', first.pageContent.slice(0, 120).replace(/\n/g, ' '))
    }
  }

  // 注意：csv 是一行一个 Document，srt 是一条字幕一个 Document，docx 是整个文件一个 Document
}

main()
