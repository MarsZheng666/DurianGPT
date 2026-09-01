<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import MarkdownIt from 'markdown-it'
import Cropper from 'cropperjs'

const md = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: true,
  // Markdown 表格对换行比较敏感，先关闭硬换行，再由预处理函数统一修复换行。
  breaks: false,
})

const STREAM_RENDER_INTERVAL_MS = 80
const DEFAULT_CHAT_MAX_TOKENS = 1024
const IMAGE_ANALYSIS_MAX_WAIT_MS = 180000

const API_BASE_URL = '/api'
const API_KEY = import.meta.env.VITE_DURIAN_API_KEY || 'change-me'
const LOGIN_ACCOUNTS = [
  { username: 'admin', password: 'admin' },
  { username: 'admin2', password: 'admin2' },
]
const CURRENT_USER_KEY = 'durian_current_user'

const endpoints = {
  health: '/health',
  chat: '/chat',
  chatStream: '/chat/stream',
  conversations: '/conversations',
  classifyPest: '/classify_pest',
  analyzePest: '/analyze_pest',
  analyzePestJob: '/analyze_pest_job',
  ragStatus: '/rag/status',
  ragUploadPdf: '/rag/upload-pdf',
  ragRebuild: '/rag/rebuild',
}

function getAuthHeaders() {
  const user = localStorage.getItem(CURRENT_USER_KEY) || ''
  const headers = {
    'X-API-Key': API_KEY,
  }
  if (user) {
    headers['X-User-Id'] = user
  }
  return headers
}

function getUserStorageKey(key, user = localStorage.getItem(CURRENT_USER_KEY) || 'anonymous') {
  return `durian_${user}_${key}`
}

const i18n = {
  zh: {
    pageTitle: '榴莲GPT',
    subtitle: '榴莲种植专家助手',
    loginTitle: '登录榴莲GPT',
    loginSubtitle: '请选择 admin 或 admin2 账号进入独立对话空间',
    username: '账号',
    password: '密码',
    login: '登录',
    logout: '退出',
    account: '账号',
    loginFailed: '账号或密码错误',
    newChat: '新对话',
    clearHistory: '清空历史',
    history: '对话历史',
    noHistory: '暂无对话历史',
    examples: '示例问题',
    references: '参考资料',
    noEvidence: '暂无参考资料',
    modelInfo: '模型信息',
    status: '状态',
    connected: '已连接',
    disconnected: '未连接',
    model: '模型',
    device: '设备',
    ragStatus: 'RAG 状态',
    ragChunks: 'RAG 数据块',
    ragFiles: 'PDF 文件',
    ragAutoChunks: 'PDF 数据块',
    ragUploadPdf: '上传 PDF',
    ragUploading: '正在导入 PDF...',
    ragUploadDone: 'PDF 已导入 RAG',
    ragUploadFailed: 'PDF 导入失败',
    ragRebuild: '重建 RAG',
    ragRebuildDone: 'RAG 已重建',
    ragRebuildFailed: 'RAG 重建失败',
    tokensUsed: '已用 Token',
    streamMode: '流式输出',
    send: '发送',
    placeholder: '请输入你的问题，按 Enter 发送，Shift+Enter 换行',
    errorEmpty: '请输入问题',
    errorConnection: '连接失败，请检查服务器',
    loaded: '对话已加载',
    loadFailed: '加载对话失败',
    clearConfirm: '确定要清空所有对话历史吗？',
    uploadImage: '上传病虫害图片',
    analyzePest: '病虫害分析',
    classifyOnly: '仅分类',
    selectedImage: '已选图片',
    noImageSelected: '请先选择图片',
    classification: '分类结果',
    confidence: '置信度',
    quality: {
      high: '高度相关',
      medium: '部分相关',
      low: '相关度较低',
      none: '无相关',
    },
    ragEnabled: '✓ 启用',
    ragDisabled: '✗ 禁用',
    historyClearedSuccess: '✓ 历史已清空',
    clearHistoryFailed: '清空历史失败',
    hallucination: '⚠️ 检测到可能的不可靠信息，请参考右侧证据资料',
    lowConfidence: '⚠️ 该回答可信度较低，请谨慎参考',
    imageAnalyzePrompt: '请分析这张图片中的病虫害情况，并提供防治建议。',
    welcomeFeatures: 'RAG + 对话历史 + 流式输出',
    uploadImageTitle: '上传图片',
    imageMessagePrefix: '[图片]',
    evidenceLabel: '文献',
    expandText: '展开全文',
    collapseText: '收起',
    unknownSource: '未知来源',
    messageCountUnit: '条消息',
    unknownClassification: '未知',
    summaryTitle: '结论摘要',
    stepsTitle: '操作步骤',
    risksTitle: '风险提示',
    dataTitle: '参数参考',
    tempLabel: '温度',
    topPLabel: 'Top P',
    tokensLabel: 'Token 数',
    userAvatar: '我',
    assistantAvatar: 'AI',
    idLabel: '编号',
    deleteConversation: '删除对话',
    deleteConversationConfirm: '确定要删除这条对话吗？',
    deleteConversationSuccess: '对话已删除',
    deleteConversationFailed: '删除对话失败',
    cropImageTitle: '裁剪图片',
    cancel: '取消',
    confirmCrop: '确认裁剪',
    imageUploading: '图片正在上传，请稍候...',
    imageTaskCreating: '正在创建图片分析任务...',
    imageAnalyzing: '图片分析中，请稍候...',
    imageAnalysisDone: '图片分析完成',
    imageAnalysisFailed: '图片分析失败，请检查服务器或稍后重试。',
    imageAnalysisTimeout: '图片分析超时，请稍后重试。',
    imageJobPolling: '正在获取图片分析结果...',
    imageClassifying: '正在识别病虫害类别...',
    imageGeneratingAdvice: '正在生成详细建议...',
    classificationResultTitle: '分类结果',
  },
  en: {
    pageTitle: 'DurianGPT',
    subtitle: 'Durian cultivation expert assistant',
    loginTitle: 'Sign in to DurianGPT',
    loginSubtitle: 'Use admin or admin2 to enter a separate chat space',
    username: 'Username',
    password: 'Password',
    login: 'Sign In',
    logout: 'Log Out',
    account: 'Account',
    loginFailed: 'Invalid username or password',
    newChat: 'New Chat',
    clearHistory: 'Clear History',
    history: 'Chat History',
    noHistory: 'No chat history',
    examples: 'Example Questions',
    references: 'References',
    noEvidence: 'No references',
    modelInfo: 'Model Info',
    status: 'Status',
    connected: 'Connected',
    disconnected: 'Disconnected',
    model: 'Model',
    device: 'Device',
    ragStatus: 'RAG Status',
    ragChunks: 'RAG Chunks',
    ragFiles: 'PDF Files',
    ragAutoChunks: 'PDF Chunks',
    ragUploadPdf: 'Upload PDF',
    ragUploading: 'Importing PDF...',
    ragUploadDone: 'PDF imported into RAG',
    ragUploadFailed: 'PDF import failed',
    ragRebuild: 'Rebuild RAG',
    ragRebuildDone: 'RAG rebuilt',
    ragRebuildFailed: 'RAG rebuild failed',
    tokensUsed: 'Tokens Used',
    streamMode: 'Stream mode',
    send: 'Send',
    placeholder: 'Ask a question. Press Enter to send, Shift+Enter for newline',
    errorEmpty: 'Please enter a question',
    errorConnection: 'Connection failed, please check the server',
    loaded: 'Conversation loaded',
    loadFailed: 'Failed to load conversation',
    clearConfirm: 'Clear all conversation history?',
    uploadImage: 'Upload pest image',
    analyzePest: 'Pest analysis',
    classifyOnly: 'Classify only',
    selectedImage: 'Selected image',
    noImageSelected: 'Please select an image first',
    classification: 'Classification',
    confidence: 'Confidence',
    quality: {
      high: 'Highly relevant',
      medium: 'Partially relevant',
      low: 'Low relevance',
      none: 'No relevance',
    },
    ragEnabled: '✓ Enabled',
    ragDisabled: '✗ Disabled',
    historyClearedSuccess: '✓ History cleared',
    clearHistoryFailed: 'Failed to clear history',
    hallucination: '⚠️ Detected potentially unreliable information, please refer to the evidence on the right',
    lowConfidence: '⚠️ This answer has low credibility, please refer to it carefully',
    summaryTitle: 'Summary',
    stepsTitle: 'Steps',
    risksTitle: 'Risk Warning',
    dataTitle: 'Parameter Reference',
    highTrust: '🟢 Highly Trustworthy',
    mediumTrust: '🟡 Moderately Trustworthy',
    lowTrust: '🔴 Low Trustworthy',
    noRelevance: '⚪ No Relevance',
    expandText: 'Expand',
    collapseText: 'Collapse',
    unknownSource: 'Unknown source',
    imageAnalyzePrompt: 'Please analyze the pests and diseases in this image and provide prevention suggestions.',
    welcomeFeatures: 'RAG + Chat History + Stream Output',
    uploadImageTitle: 'Upload Image',
    imageMessagePrefix: '[Image]',
    evidenceLabel: 'Reference',
    messageCountUnit: 'messages',
    unknownClassification: 'Unknown',
    tempLabel: 'Temperature',
    topPLabel: 'Top P',
    tokensLabel: 'Tokens',
    deleteConversation: 'Delete conversation',
    deleteConversationConfirm: 'Delete this conversation?',
    deleteConversationSuccess: 'Conversation deleted',
    deleteConversationFailed: 'Failed to delete conversation',
    cropImageTitle: 'Crop Image',
    cancel: 'Cancel',
    confirmCrop: 'Confirm Crop',
    imageUploading: 'Uploading image, please wait...',
    imageTaskCreating: 'Creating image analysis task...',
    imageAnalyzing: 'Analyzing image, please wait...',
    imageAnalysisDone: 'Image analysis completed',
    imageAnalysisFailed: 'Image analysis failed. Please check the server or try again later.',
    imageAnalysisTimeout: 'Image analysis timed out. Please try again later.',
    imageJobPolling: 'Fetching image analysis result...',
    imageClassifying: 'Identifying pest or disease category...',
    imageGeneratingAdvice: 'Generating detailed recommendations...',
    classificationResultTitle: 'Classification Result',
  },
  ms: {
    pageTitle: 'DurianGPT',
    subtitle: 'Pembantu pakar penanaman durian',
    loginTitle: 'Log Masuk DurianGPT',
    loginSubtitle: 'Gunakan admin atau admin2 untuk ruang perbualan berasingan',
    username: 'Akaun',
    password: 'Kata laluan',
    login: 'Log Masuk',
    logout: 'Log Keluar',
    account: 'Akaun',
    loginFailed: 'Akaun atau kata laluan salah',
    newChat: 'Perbualan Baharu',
    clearHistory: 'Kosongkan Sejarah',
    history: 'Sejarah Perbualan',
    noHistory: 'Tiada sejarah perbualan',
    examples: 'Soalan Contoh',
    references: 'Rujukan',
    noEvidence: 'Tiada rujukan',
    modelInfo: 'Maklumat Model',
    status: 'Status',
    connected: 'Bersambung',
    disconnected: 'Tidak Bersambung',
    model: 'Model',
    device: 'Peranti',
    ragStatus: 'Status RAG',
    ragChunks: 'Chunk RAG',
    ragFiles: 'Fail PDF',
    ragAutoChunks: 'Chunk PDF',
    ragUploadPdf: 'Muat Naik PDF',
    ragUploading: 'Mengimport PDF...',
    ragUploadDone: 'PDF telah diimport ke RAG',
    ragUploadFailed: 'Import PDF gagal',
    ragRebuild: 'Bina Semula RAG',
    ragRebuildDone: 'RAG telah dibina semula',
    ragRebuildFailed: 'Bina semula RAG gagal',
    tokensUsed: 'Token Digunakan',
    streamMode: 'Mod aliran',
    send: 'Hantar',
    placeholder: 'Masukkan soalan. Tekan Enter untuk hantar, Shift+Enter untuk baris baru',
    errorEmpty: 'Sila masukkan soalan',
    errorConnection: 'Sambungan gagal, sila periksa pelayan',
    loaded: 'Perbualan dimuatkan',
    loadFailed: 'Gagal memuatkan perbualan',
    clearConfirm: 'Kosongkan semua sejarah perbualan?',
    uploadImage: 'Muat naik imej perosak',
    analyzePest: 'Analisis perosak',
    classifyOnly: 'Klasifikasi sahaja',
    selectedImage: 'Imej dipilih',
    noImageSelected: 'Sila pilih imej dahulu',
    classification: 'Klasifikasi',
    confidence: 'Keyakinan',
    quality: {
      high: 'Sangat berkaitan',
      medium: 'Sebahagian berkaitan',
      low: 'Kurang berkaitan',
      none: 'Tiada kaitan',
    },
    ragEnabled: '✓ Diaktifkan',
    ragDisabled: '✗ Dimatikan',
    historyClearedSuccess: '✓ Sejarah telah dikosongkan',
    clearHistoryFailed: 'Gagal mengosongkan sejarah',
    hallucination: '⚠️ Mengesan maklumat yang mungkin tidak boleh dipercayai, sila rujuk bukti di sebelah kanan',
    lowConfidence: '⚠️ Jawapan ini mempunyai kredibiliti yang rendah, sila rujuk dengan berhati-hati',
    summaryTitle: 'Ringkasan',
    stepsTitle: 'Langkah-langkah',
    risksTitle: 'Amaran Risiko',
    dataTitle: 'Rujukan Parameter',
    highTrust: '🟢 Sangat Boleh Dipercayai',
    mediumTrust: '🟡 Sederhana Boleh Dipercayai',
    lowTrust: '🔴 Kurang Boleh Dipercayai',
    noRelevance: '⚪ Tiada Kaitan',
    expandText: 'Kembangkan',
    collapseText: 'Tutup',
    unknownSource: 'Sumber tidak diketahui',
    imageAnalyzePrompt: 'Sila analisis perosak dan penyakit dalam imej ini dan berikan cadangan pencegahan.',
    welcomeFeatures: 'RAG + Sejarah Perbualan + Output Aliran',
    uploadImageTitle: 'Muat Naik Imej',
    imageMessagePrefix: '[Imej]',
    evidenceLabel: 'Rujukan',
    messageCountUnit: 'mesej',
    unknownClassification: 'Tidak Diketahui',
    tempLabel: 'Suhu',
    topPLabel: 'Top P',
    tokensLabel: 'Token',
    deleteConversation: 'Padam perbualan',
    deleteConversationConfirm: 'Padam perbualan ini?',
    deleteConversationSuccess: 'Perbualan telah dipadam',
    deleteConversationFailed: 'Gagal memadam perbualan',
    cropImageTitle: 'Pangkas Imej',
    cancel: 'Batal',
    confirmCrop: 'Sahkan Pangkas',
    imageUploading: 'Imej sedang dimuat naik, sila tunggu...',
    imageTaskCreating: 'Sedang mencipta tugas analisis imej...',
    imageAnalyzing: 'Imej sedang dianalisis, sila tunggu...',
    imageAnalysisDone: 'Analisis imej selesai',
    imageAnalysisFailed: 'Analisis imej gagal. Sila semak pelayan atau cuba lagi kemudian.',
    imageAnalysisTimeout: 'Analisis imej tamat masa. Sila cuba lagi kemudian.',
    imageJobPolling: 'Sedang mendapatkan keputusan analisis imej...',
    imageClassifying: 'Sedang mengenal pasti kategori perosak atau penyakit...',
    imageGeneratingAdvice: 'Sedang menjana cadangan terperinci...',
    classificationResultTitle: 'Keputusan Klasifikasi',
  },
  th: {
    pageTitle: 'DurianGPT',
    subtitle: 'ผู้ช่วยผู้เชี่ยวชาญด้านการปลูกทุเรียน',
    loginTitle: 'เข้าสู่ระบบ DurianGPT',
    loginSubtitle: 'ใช้ admin หรือ admin2 เพื่อแยกพื้นที่สนทนา',
    username: 'บัญชี',
    password: 'รหัสผ่าน',
    login: 'เข้าสู่ระบบ',
    logout: 'ออกจากระบบ',
    account: 'บัญชี',
    loginFailed: 'บัญชีหรือรหัสผ่านไม่ถูกต้อง',
    newChat: 'สนทนาใหม่',
    clearHistory: 'ล้างประวัติ',
    history: 'ประวัติการสนทนา',
    noHistory: 'ไม่มีประวัติการสนทนา',
    examples: 'คำถามตัวอย่าง',
    references: 'อ้างอิง',
    noEvidence: 'ไม่มีอ้างอิง',
    modelInfo: 'ข้อมูลโมเดล',
    status: 'สถานะ',
    connected: 'เชื่อมต่อแล้ว',
    disconnected: 'ไม่ได้เชื่อมต่อ',
    model: 'โมเดล',
    device: 'อุปกรณ์',
    ragStatus: 'สถานะ RAG',
    ragChunks: 'ชิ้น RAG',
    ragFiles: 'ไฟล์ PDF',
    ragAutoChunks: 'ชิ้น PDF',
    ragUploadPdf: 'อัปโหลด PDF',
    ragUploading: 'กำลังนำเข้า PDF...',
    ragUploadDone: 'นำเข้า PDF เข้า RAG แล้ว',
    ragUploadFailed: 'นำเข้า PDF ไม่สำเร็จ',
    ragRebuild: 'สร้าง RAG ใหม่',
    ragRebuildDone: 'สร้าง RAG ใหม่แล้ว',
    ragRebuildFailed: 'สร้าง RAG ใหม่ไม่สำเร็จ',
    tokensUsed: 'โทเค็นที่ใช้',
    streamMode: 'โหมดสตรีม',
    send: 'ส่ง',
    placeholder: 'ป้อนคำถาม กด Enter เพื่อส่ง Shift+Enter สำหรับบรรทัดใหม่',
    errorEmpty: 'โปรดป้อนคำถาม',
    errorConnection: 'การเชื่อมต่อล้มเหลว โปรดตรวจสอบเซิร์ฟเวอร์',
    loaded: 'โหลดการสนทนาแล้ว',
    loadFailed: 'ไม่สามารถโหลดการสนทนา',
    clearConfirm: 'ล้างประวัติการสนทนาทั้งหมด?',
    uploadImage: 'อัปโหลดรูปภาพศัตรูพืช',
    analyzePest: 'วิเคราะห์ศัตรูพืช',
    classifyOnly: 'จำแนกเท่านั้น',
    selectedImage: 'เลือกรูปภาพแล้ว',
    noImageSelected: 'โปรดเลือกรูปภาพก่อน',
    classification: 'การจำแนก',
    confidence: 'ความมั่นใจ',
    quality: {
      high: 'เกี่ยวข้องมาก',
      medium: 'เกี่ยวข้องบางส่วน',
      low: 'เกี่ยวข้องน้อย',
      none: 'ไม่เกี่ยวข้อง',
    },
    ragEnabled: '✓ เปิดใช้งาน',
    ragDisabled: '✗ ปิดใช้งาน',
    historyClearedSuccess: '✓ ล้างประวัติแล้ว',
    clearHistoryFailed: 'ไม่สามารถล้างประวัติ',
    hallucination: '⚠️ ตรวจพบข้อมูลที่อาจไม่น่าเชื่อถือ โปรดดูหลักฐานทางขวา',
    lowConfidence: '⚠️ คำตอบนี้มีความน่าเชื่อถือต่ำ โปรดอ้างอิงอย่างระมัดระวัง',
    summaryTitle: 'สรุป',
    stepsTitle: 'ขั้นตอน',
    risksTitle: 'คำเตือนความเสี่ยง',
    dataTitle: 'อ้างอิงพารามิเตอร์',
    highTrust: '🟢 น่าเชื่อถือมาก',
    mediumTrust: '🟡 น่าเชื่อถือปานกลาง',
    lowTrust: '🔴 น่าเชื่อถือน้อย',
    noRelevance: '⚪ ไม่เกี่ยวข้อง',
    expandText: 'ขยาย',
    collapseText: 'ยุบ',
    unknownSource: 'แหล่งที่มาไม่ทราบ',
    imageAnalyzePrompt: 'โปรดวิเคราะห์ศัตรูพืชและโรคในรูปภาพนี้ และให้คำแนะนำในการป้องกัน',
    welcomeFeatures: 'RAG + ประวัติการสนทนา + เอาต์พุตสตรีม',
    uploadImageTitle: 'อัปโหลดรูปภาพ',
    imageMessagePrefix: '[รูปภาพ]',
    evidenceLabel: 'อ้างอิง',
    messageCountUnit: 'ข้อความ',
    unknownClassification: 'ไม่ทราบ',
    tempLabel: 'อุณหภูมิ',
    topPLabel: 'Top P',
    tokensLabel: 'โทเค็น',
    deleteConversation: 'ลบการสนทนา',
    deleteConversationConfirm: 'ต้องการลบการสนทนานี้หรือไม่?',
    deleteConversationSuccess: 'ลบการสนทนาแล้ว',
    deleteConversationFailed: 'ลบการสนทนาไม่สำเร็จ',
    cropImageTitle: 'ครอบตัดรูปภาพ',
    cancel: 'ยกเลิก',
    confirmCrop: 'ยืนยันการครอบตัด',
    imageUploading: 'กำลังอัปโหลดรูปภาพ โปรดรอสักครู่...',
    imageTaskCreating: 'กำลังสร้างงานวิเคราะห์รูปภาพ...',
    imageAnalyzing: 'กำลังวิเคราะห์รูปภาพ โปรดรอสักครู่...',
    imageAnalysisDone: 'วิเคราะห์รูปภาพเสร็จแล้ว',
    imageAnalysisFailed: 'วิเคราะห์รูปภาพล้มเหลว โปรดตรวจสอบเซิร์ฟเวอร์หรือลองใหม่ภายหลัง',
    imageAnalysisTimeout: 'การวิเคราะห์รูปภาพหมดเวลา โปรดลองใหม่ภายหลัง',
    imageJobPolling: 'กำลังดึงผลการวิเคราะห์รูปภาพ...',
    imageClassifying: 'กำลังระบุประเภทศัตรูพืชหรือโรค...',
    imageGeneratingAdvice: 'กำลังสร้างคำแนะนำโดยละเอียด...',
    classificationResultTitle: 'ผลการจำแนก',
  },
}

function normalizeNewlines(text) {
  if (!text) return ''
  return String(text)
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .replace(/\\n/g, '\n')
    .replace(/\uFFFD/g, '')
}

function normalizeMarkdownTables(text) {
  if (!text) return ''

  let result = normalizeNewlines(text)

  // 标题前后补空行，避免 "上一句## 标题" 被当成普通文本。
  result = result.replace(/([^\n])\n?(#{1,6}\s+)/g, '$1\n\n$2')
  result = result.replace(/(#{1,6}[^\n]+)\n(?!\n|#|[-*+\d]+\.|\|)/g, '$1\n\n')

  // 标准化 Markdown 表格分隔线。
  result = result.replace(/^\s*\|?\s*((?::?-{3,}:?\s*\|\s*)+(?::?-{3,}:?)?)\s*\|?\s*$/gm, (line) => {
    const cells = line
      .trim()
      .replace(/^\|/, '')
      .replace(/\|$/, '')
      .split('|')
      .map(() => '---')
    return `| ${cells.join(' | ')} |`
  })

  // 表格行前后补空行。Markdown-it 需要表格与普通段落分开。
  const lines = result.split('\n')
  const output = []
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i].trimEnd()
    const isTableRow = /^\s*\|.+\|\s*$/.test(line)
    const prev = output[output.length - 1] || ''
    const prevIsTableRow = /^\s*\|.+\|\s*$/.test(prev)

    if (isTableRow && output.length > 0 && prev.trim() !== '' && !prevIsTableRow) {
      output.push('')
    }

    output.push(line)

    const next = lines[i + 1] || ''
    const nextIsTableRow = /^\s*\|.+\|\s*$/.test(next)
    if (isTableRow && !nextIsTableRow && next.trim() !== '') {
      output.push('')
    }
  }

  return output.join('\n').replace(/\n{3,}/g, '\n\n').trim()
}

function normalizeModelOutput(text) {
  return normalizeMarkdownTables(filterResponse(normalizeNewlines(text)))
}

function normalizeStreamChunk(text) {
  return normalizeNewlines(text)
}

function renderMarkdown(text) {
  return md.render(normalizeModelOutput(text))
}

const exampleQuestions = {
  zh: [
    '猫山王榴莲适合什么土壤？',
    '榴莲幼苗为什么叶片发黄？',
    '榴莲开花期如何施肥？',
    '榴莲常见病虫害有哪些？',
  ],
  en: [
    'What soil is suitable for Musang King durian?',
    'Why are durian seedling leaves turning yellow?',
    'How to fertilize during the flowering stage of durian?',
    'What are common durian pests and diseases?',
  ],
  ms: [
    'Tanah apa sesuai untuk durian Musang King?',
    'Mengapa daun anak pokok durian menjadi kuning?',
    'Bagaimana membaja semasa peringkat berbunga durian?',
    'Apakah perosak dan penyakit durian yang biasa?',
  ],
  th: [
    'ดินที่เหมาะสมสำหรับทุเรียนมุสังคิง?',
    'ทำไมใบของต้นทุเรียนอ่อนจึงเหลือง?',
    'วิธีการให้ปุ๋ยในช่วงการออกดอกของทุเรียน?',
    'ศัตรูพืชและโรคทุเรียนทั่วไปมีอะไรบ้าง?',
  ],
}

const language = ref(localStorage.getItem('language') || 'zh')
const currentUser = ref(localStorage.getItem(CURRENT_USER_KEY) || '')
const loginUsername = ref(currentUser.value || 'admin')
const loginPassword = ref('')
const loginError = ref('')
const userInput = ref('')
const streamMode = ref(localStorage.getItem('streamMode') !== 'false')
const temperature = ref(0.7)
const topP = ref(0.9)
const maxTokens = ref(Number(localStorage.getItem('maxTokens') || DEFAULT_CHAT_MAX_TOKENS))

const responseLanguage = computed(() => language.value)

const isConnected = ref(false)
const modelInfo = ref({
  model_path: '-',
  device: '-',
  rag_enabled: false,
  rag_chunks: 0,
  rag_pdf_files: 0,
  rag_auto_chunks: 0,
})

const messages = ref([])
const totalTokens = ref(Number(localStorage.getItem(getUserStorageKey('totalTokens')) || 0))
const currentConversationId = ref(null)
const conversations = ref([])
const evidence = ref([])
const evidenceQuality = ref('none')
const toasts = ref([])
const isSending = ref(false)
const evidenceExpanded = ref({})
const pestImageFile = ref(null)
const pestImagePreview = ref(null)
const pestClassification = ref(null)
const showHistorySidebar = ref(false)
const messageImages = ref({})
const showCropperModal = ref(false)
const cropperInstance = ref(null)
const originalImagePreview = ref(null)
const isAnalyzingImage = ref(false)
const ragPdfInput = ref(null)
const isUploadingRagPdf = ref(false)
const ragUploadStatus = ref('')
const rightSidebarCollapsed = ref(false)
const rightSidebarWidth = ref(420)
const isResizingRightSidebar = ref(false)

// 右侧边栏拉伸处理
function startResizeRightSidebar(e) {
  isResizingRightSidebar.value = true
  const startX = e.clientX
  const startWidth = rightSidebarWidth.value

  const handleMouseMove = (moveEvent) => {
    const diff = startX - moveEvent.clientX
    const newWidth = Math.max(340, Math.min(720, startWidth + diff))
    rightSidebarWidth.value = newWidth
  }

  const handleMouseUp = () => {
    isResizingRightSidebar.value = false
    document.removeEventListener('mousemove', handleMouseMove)
    document.removeEventListener('mouseup', handleMouseUp)
  }

  document.addEventListener('mousemove', handleMouseMove)
  document.addEventListener('mouseup', handleMouseUp)
}

// 对话管理函数
async function loadConversations() {
  if (!isLoggedIn.value) return

  try {
    const response = await fetch(`${API_BASE_URL}/conversations`, {
      headers: getAuthHeaders(),
    })
    if (!response.ok) throw new Error('加载对话列表失败')
    const data = await response.json()
    conversations.value = data.conversations || []
  } catch (error) {
    console.error('加载对话列表失败:', error)
  }
}

async function createConversationIfNeeded(firstUserText) {
  if (!isLoggedIn.value) {
    throw new Error('未登录')
  }

  if (currentConversationId.value) return currentConversationId.value

  try {
    const response = await fetch(`${API_BASE_URL}/conversations`, {
      method: 'POST',
      headers: getAuthHeaders(),
    })
    if (!response.ok) throw new Error('创建对话失败')
    const data = await response.json()
    currentConversationId.value = data.id
    await loadConversations()
    return data.id
  } catch (error) {
    console.error('创建对话失败:', error)
    pushToast('创建对话失败', 'error')
    return null
  }
}

  async function saveMessageToConversation(role, content = '', imageUrl = null, activeContextCard = null) {
    if (!currentConversationId.value) {
      const convId = await createConversationIfNeeded(content || t.value.imageMessagePrefix)
      if (!convId) return
    }

    try {
      const response = await fetch(
        `${API_BASE_URL}/conversations/${currentConversationId.value}/messages`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...getAuthHeaders(),
          },
          body: JSON.stringify({
            role,
            content: normalizeModelOutput(content || ''),
            image_url: imageUrl,
            active_context_card: activeContextCard || null,
          }),
        }
      )

      if (!response.ok) throw new Error('保存消息失败')
      await loadConversations()
    } catch (error) {
      console.error('保存消息失败:', error)
    }
  }

async function deleteConversation(convId) {
  if (!isLoggedIn.value) return
  if (!convId) return

  const ok = window.confirm(t.value.deleteConversationConfirm || '确定要删除这条对话吗？')
  if (!ok) return

  try {
    const response = await fetch(`${API_BASE_URL}${endpoints.conversations}/${convId}`, {
      method: 'DELETE',
      headers: getAuthHeaders(),
    })

    if (!response.ok) {
      throw new Error(`Delete failed: ${response.status}`)
    }

    conversations.value = conversations.value.filter((item) => item.id !== convId)
    removeEvidenceCache(convId)

    if (currentConversationId.value === convId) {
      startNewChat()
    }

    pushToast(t.value.deleteConversationSuccess || '对话已删除', 'success')
  } catch (error) {
    console.error('删除对话失败:', error)
    pushToast(t.value.deleteConversationFailed || '删除对话失败', 'error')
  }
}

async function loadConversation(convId) {
  if (!isLoggedIn.value) return

  try {
    const response = await fetch(`${API_BASE_URL}/conversations/${convId}`, {
      headers: getAuthHeaders(),
    })
    if (!response.ok) throw new Error('加载对话失败')
    const data = await response.json()
    
    currentConversationId.value = convId
    
    const restoredImages = {}
    
    messages.value = data.messages.map((m, index) => {
      const msg = {
        role: m.role,
        content: normalizeModelOutput(m.content || ''),
        active_context_card: m.active_context_card || null,
      }
      
      if (m.image_url) {
        const imageKey = `history_img_${convId}_${index}`
        msg.imageKey = imageKey
        restoredImages[imageKey] = normalizeImageUrl(m.image_url)
      }
      
      return msg
    })
    
    messageImages.value = restoredImages
    
    const cachedEvidence = loadEvidenceCache(convId)
    evidence.value = cachedEvidence.evidence
    evidenceQuality.value = cachedEvidence.evidence_quality
    evidenceExpanded.value = {}
    
    await scrollToBottom(true)
  } catch (error) {
    console.error('加载对话失败:', error)
    pushToast('加载对话失败', 'error')
    
    if (localStorage.getItem(getUserStorageKey('currentConversationId')) === convId) {
      localStorage.removeItem(getUserStorageKey('currentConversationId'))
      currentConversationId.value = null
    }
  }
}

const chatMessagesRef = ref(null)
const composerInputRef = ref(null)
let healthTimer = null
let toastCounter = 0
let sendingLock = false

function normalizeImageUrl(url) {
  if (!url) return ''
  if (url.startsWith('data:')) return url
  if (url.startsWith('blob:')) return url
  if (url.startsWith('http://') || url.startsWith('https://')) return url
  if (url.startsWith(`${API_BASE_URL}/`)) return url
  if (url.startsWith('/uploads/')) return `${API_BASE_URL}${url}`
  return url
}


function buildHiddenImageContext({ analyzeData, finalResponse, imageQuestion, fileName }) {
  const context = {
    type: 'durian_image_context',
    user_question: imageQuestion || '',
    image_url: analyzeData?.image_url || '',
    file_name: fileName || '',
    visual_label: analyzeData?.visual_observation || '',
    assistant_summary: finalResponse || '',
    created_at: new Date().toISOString(),
  }
  // MarkdownIt 开启 html:true 时 HTML 注释不会显示，但会保存在历史消息里，后续追问可带给后端。
  return `<!-- [Image context]\n${JSON.stringify(context, null, 2)}\n-->`
}

function isImageRelatedMessage(msg) {
  const content = msg.content || ''

  if (msg.imageKey) return true
  if (msg.image_url) return true
  if (msg.messageType === 'image') return true
  if (msg.messageType === 'image_analysis') return true

  return (
    content.includes('## 分类结果') ||
    content.includes('## Classification') ||
    content.includes('## Klasifikasi') ||
    content.includes('## การจำแนก') ||
    content.includes('图像分类模型识别结果') ||
    content.includes('image classification model') ||
    content.includes('model klasifikasi imej') ||
    content.includes('โมเดลจำแนกรูปภาพ') ||
    content.includes('识别结论') ||
    content.includes('该图片被分类为')
  )
}

function getEvidenceStorageKey(convId) {
  return getUserStorageKey(`evidence_${convId}`)
}

function saveEvidenceCache(convId, evidenceList, quality = 'none') {
  if (!convId) return

  const payload = {
    evidence: evidenceList || [],
    evidence_quality: quality || 'none',
    saved_at: Date.now(),
  }

  localStorage.setItem(getEvidenceStorageKey(convId), JSON.stringify(payload))
}

function loadEvidenceCache(convId) {
  if (!convId) {
    return {
      evidence: [],
      evidence_quality: 'none',
    }
  }

  try {
    const raw = localStorage.getItem(getEvidenceStorageKey(convId))
    if (!raw) {
      return {
        evidence: [],
        evidence_quality: 'none',
      }
    }

    const parsed = JSON.parse(raw)
    return {
      evidence: Array.isArray(parsed.evidence) ? parsed.evidence : [],
      evidence_quality: parsed.evidence_quality || 'none',
    }
  } catch (error) {
    console.error('读取参考资料缓存失败:', error)
    return {
      evidence: [],
      evidence_quality: 'none',
    }
  }
}

function removeEvidenceCache(convId) {
  if (!convId) return
  localStorage.removeItem(getEvidenceStorageKey(convId))
}

const t = computed(() => ({
  ...i18n.zh,
  ...(i18n[language.value] || {}),
}))
const currentExamples = computed(() => exampleQuestions[language.value] || exampleQuestions.zh)
const showWelcome = computed(() => messages.value.length === 0)
const isLoggedIn = computed(() => LOGIN_ACCOUNTS.some((account) => account.username === currentUser.value))

function getUITextByLang(lang, key) {
  const langPack = i18n[lang] || i18n.zh
  return langPack[key] || i18n.zh[key] || ''
}

function getCurrentUILang() {
  return language.value || localStorage.getItem('language') || 'zh'
}


const pestClassNameById = {
  th: {
    0: 'พื้นหลังหรืออื่น ๆ',
    1: 'ใบที่แข็งแรง',
    2: 'ลำต้น/ผลที่แข็งแรง',
    3: 'โรคใบไหม้ลำต้น',
    4: 'ลำต้นแตกร้าว/ยางไหล',
    5: 'โรคใบไหม้ไรโซคโทเนีย',
    6: 'โรคแอนแทรคโนสที่ใบ',
    7: 'โรคแคงเกอร์',
    8: 'โรคราสีชมพู',
    9: 'โรครากขาว',
    10: 'โรคจุดใบสาหร่าย',
    11: 'โรคจุดใบโฟมอปซิส',
    12: 'โรคผลเน่า',
    13: 'ราดำ',
    14: 'ปลวก/มด',
    15: 'จักจั่น',
    16: 'เพลี้ยไก่แจ้',
    17: 'เพลี้ยแป้ง',
    18: 'เพลี้ยจักจั่น',
    19: 'เพลี้ยไฟ',
    20: 'ไรแดง',
    21: 'เพลี้ยหอย',
    22: 'หนอนเจาะลำต้น',
    23: 'ด้วงเปลือกไม้',
    24: 'หนอนเจาะผล',
    25: 'ใบเหลือง',
  },
}

function getPestDisplayName(classifyData, lang) {
  if (!classifyData) return getUITextByLang(lang, 'unknownClassification')

  const directName = {
    zh: classifyData.class_name,
    en: classifyData.class_name_en,
    ms: classifyData.class_name_ms,
    th: classifyData.class_name_th,
  }[lang]

  if (directName) return directName

  const classId = Number(classifyData.class_id)
  const mappedName = pestClassNameById[lang]?.[classId]
  return mappedName || classifyData.class_name_en || classifyData.class_name || getUITextByLang(lang, 'unknownClassification')
}

function formatConfidence(confidence) {
  const value = Number(confidence)
  if (!Number.isFinite(value)) return ''
  return `${(value * 100).toFixed(1)}%`
}

function getRuntimeText(lang, key) {
  const mapping = {
    zh: {
      chatPreparing: '正在连接模型，请稍候...',
      chatGenerating: '模型正在生成回答，请稍候...',
      imageClassified: '分类完成，正在生成详细建议...',
      imageGeneratingWithClass: '正在生成详细建议，请稍候...',
    },
    en: {
      chatPreparing: 'Connecting to the model, please wait...',
      chatGenerating: 'The model is generating the answer, please wait...',
      imageClassified: 'Classification completed. Generating detailed recommendations...',
      imageGeneratingWithClass: 'Generating detailed recommendations, please wait...',
    },
    ms: {
      chatPreparing: 'Sedang menyambung kepada model, sila tunggu...',
      chatGenerating: 'Model sedang menjana jawapan, sila tunggu...',
      imageClassified: 'Klasifikasi selesai. Sedang menjana cadangan terperinci...',
      imageGeneratingWithClass: 'Sedang menjana cadangan terperinci, sila tunggu...',
    },
    th: {
      chatPreparing: 'กำลังเชื่อมต่อกับโมเดล โปรดรอสักครู่...',
      chatGenerating: 'โมเดลกำลังสร้างคำตอบ โปรดรอสักครู่...',
      imageClassified: 'จำแนกเสร็จแล้ว กำลังสร้างคำแนะนำโดยละเอียด...',
      imageGeneratingWithClass: 'กำลังสร้างคำแนะนำโดยละเอียด โปรดรอสักครู่...',
    },
  }
  return mapping[lang]?.[key] || mapping.zh[key] || ''
}

function buildImageProgressText(lang, classifyData, elapsedMs = 0) {
  const className = getPestDisplayName(classifyData, lang)
  const confidence = formatConfidence(classifyData?.confidence)
  const elapsedSeconds = Math.floor(elapsedMs / 1000)
  const elapsedText = elapsedSeconds > 0 ? ` ${elapsedSeconds}s` : ''

  if (lang === 'en') {
    return `${getRuntimeText(lang, 'imageGeneratingWithClass')}\n\nCategory: ${className}${confidence ? ` (${confidence})` : ''}${elapsedText}`
  }
  if (lang === 'ms') {
    return `${getRuntimeText(lang, 'imageGeneratingWithClass')}\n\nKategori: ${className}${confidence ? ` (${confidence})` : ''}${elapsedText}`
  }
  if (lang === 'th') {
    return `${getRuntimeText(lang, 'imageGeneratingWithClass')}\n\nประเภท: ${className}${confidence ? ` (${confidence})` : ''}${elapsedText}`
  }
  return `${getRuntimeText(lang, 'imageGeneratingWithClass')}\n\n分类结果：${className}${confidence ? `（${confidence}）` : ''}${elapsedText}`
}

watch(language, (value) => {
  localStorage.setItem('language', value)
  document.documentElement.lang = value
})

watch(streamMode, (value) => {
  localStorage.setItem('streamMode', String(Boolean(value)))
})

watch(maxTokens, (value) => {
  const normalized = Number(value) || DEFAULT_CHAT_MAX_TOKENS
  localStorage.setItem('maxTokens', String(normalized))
})

watch(totalTokens, (value) => {
  if (!currentUser.value) return
  localStorage.setItem(getUserStorageKey('totalTokens'), String(value))
})

watch(currentConversationId, (value) => {
  if (!currentUser.value) return

  if (value) {
    localStorage.setItem(getUserStorageKey('currentConversationId'), value)
  } else {
    localStorage.removeItem(getUserStorageKey('currentConversationId'))
  }
})


function renderStructuredAnswer(text) {
  return renderMarkdown(text)
}

function containsHallucination(text) {
  const normalized = text || ''
  return (
    (normalized.includes('据研究表明') && !normalized.includes('来源')) ||
    normalized.includes('一般来说') ||
    normalized.length < 20
  )
}

function qualityBadgeText(quality) {
  const mapping = {
    zh: {
      high: '🟢 高可信',
      medium: '🟡 中等可信',
      low: '🔴 低可信',
      none: '⚪ 无相关',
    },
    en: {
      high: '🟢 Highly Trustworthy',
      medium: '🟡 Moderately Trustworthy',
      low: '🔴 Low Trustworthy',
      none: '⚪ No Relevance',
    },
    ms: {
      high: '🟢 Sangat Boleh Dipercayai',
      medium: '🟡 Sederhana Boleh Dipercayai',
      low: '🔴 Kurang Boleh Dipercayai',
      none: '⚪ Tiada Kaitan',
    },
    th: {
      high: '🟢 น่าเชื่อถือมาก',
      medium: '🟡 น่าเชื่อถือปานกลาง',
      low: '🔴 น่าเชื่อถือน้อย',
      none: '⚪ ไม่เกี่ยวข้อง',
    },
  }
  const langMapping = mapping[language.value] || mapping.zh
  return langMapping[quality] || langMapping.none
}

function getEvidencePreview(text, index) {
  if (!text) return ''
  if (evidenceExpanded.value[index] || text.length <= 700) return text
  return `${text.slice(0, 700)}...`
}

function toggleEvidence(index) {
  evidenceExpanded.value = {
    ...evidenceExpanded.value,
    [index]: !evidenceExpanded.value[index],
  }
}

function getQualityLabel(quality) {
  return qualityBadgeText(quality)
}

function getEvidenceSource(item) {
  return item?.doc || item?.source || t.value.unknownSource
}

function getEvidenceScore(item) {
  const score = Number(item?.score || 0)
  if (!Number.isFinite(score) || score <= 0) return ''
  return `${Math.round(score * 100)}%`
}

function getInlineEvidencePreview(text, limit = 180) {
  const normalized = (text || '').replace(/\s+/g, ' ').trim()
  if (!normalized) return ''
  return normalized.length > limit ? `${normalized.slice(0, limit)}...` : normalized
}

function getMessageEvidence(message) {
  return Array.isArray(message?.evidence) ? message.evidence.filter((item) => item && item.text).slice(0, 3) : []
}

function filterResponse(text) {
  if (!text) return ''
  let result = text
  
  // 移除旧版本可能混入 content 的隐藏上下文卡片。v4 起上下文卡片走 active_context_card 字段，不应显示在正文。
  result = result.replace(/<!--\s*\[(?:Image context|Durian context card|视觉识别上下文)\][\s\S]*?(?:-->|$)/gi, '')
  result = result.replace(/\[Durian context card\][\s\S]*$/gi, '')
  result = result.replace(/[\r\n]*["“]?created_at["”]?\s*[:：]\s*["“][^\n{}]+["”]\s*}\s*(?:-->)?\s*$/gi, '')

  // 移除完整闭合的思考标签
  result = result.replace(/<think>[\s\S]*?<\/think>/gi, '')
  result = result.replace(/<analysis>[\s\S]*?<\/analysis>/gi, '')
  result = result.replace(/<reasoning>[\s\S]*?<\/reasoning>/gi, '')

  // 移除未闭合的标签本身，但不删除后续内容
  result = result.replace(/<\/?think>/gi, '')
  result = result.replace(/<\/?analysis>/gi, '')
  result = result.replace(/<\/?reasoning>/gi, '')

  result = result.replace(/^[\s]*(?:思考|思维|分析|推理)[:：][\s\S]*?(?=\n\n|$)/gm, '')
  result = result.replace(/^[\s]*(?:让我|我来|首先)(?:思考|分析|推理).*?[\n：]/gm, '')
  result = result.replace(/\n?\{"evidence"[\s\S]*$/i, '')
  result = result.replace(/\b\d{14}\.json\b/gi, '')
  result = result.replace(/^[ \t]*[\w-]+(?:\.[\w-]+)*\.json[ \t]*$/gim, '')
  result = result.replace(/[ \t]+\n/g, '\n')
  result = result.replace(/\n\n\n+/g, '\n\n')
  // 移除 Classification 行
  result = result.replace(/^[\s]*Classification[\s]*:[\s]*[^\n]*[\n]*/gm, '')
  result = result.replace(/^[\s]*分类[\s]*:[\s]*[^\n]*[\n]*/gm, '')
  result = result.replace(/^[\s]*Klasifikasi[\s]*:[\s]*[^\n]*[\n]*/gm, '')
  result = result.replace(/^[\s]*การจำแนก[\s]*:[\s]*[^\n]*[\n]*/gm, '')
  
  // 保留 <br> 标签用于换行
  result = result.replace(/<br\s*\/?>/gi, '<br>')
  
  return result.trim()
}

function extractStreamPayload(text) {
  if (!text) {
    return {
      content: '',
      evidence: [],
      quality: 'none',
    }
  }

  const markerIndex = text.lastIndexOf('{"evidence"')
  if (markerIndex === -1) {
    return {
      content: filterResponse(text),
      evidence: [],
      quality: 'none',
    }
  }

  const contentPart = text.slice(0, markerIndex)
  const jsonPart = text.slice(markerIndex).trim()

  try {
    const parsed = JSON.parse(jsonPart)
    return {
      content: filterResponse(contentPart),
      evidence: parsed.evidence || [],
      quality: parsed.evidence_quality || 'none',
    }
  } catch {
    return {
      content: filterResponse(text),
      evidence: [],
      quality: 'none',
    }
  }
}

function splitSSEEvents(buffer) {
  const normalized = buffer.replace(/\r\n/g, '\n')
  const parts = normalized.split('\n\n')
  const rest = parts.pop() || ''
  return { events: parts, rest }
}

function getSSEData(rawEvent) {
  if (!rawEvent) return ''
  const trimmed = rawEvent.trim()
  if (trimmed.startsWith('data:')) {
    return trimmed.replace(/^data:\s*/, '')
  }
  return trimmed
}

let pendingMessageRender = 0
function scheduleMessagesRender(forceScroll = false) {
  if (pendingMessageRender) return
  pendingMessageRender = window.requestAnimationFrame(async () => {
    pendingMessageRender = 0
    messages.value = [...messages.value]
    await scrollToBottom(forceScroll)
  })
}

function isUserNearBottom() {
  if (window.matchMedia('(max-width: 900px)').matches) {
    const page = document.scrollingElement || document.documentElement
    return page.scrollHeight - page.scrollTop - window.innerHeight < 120
  }

  const el = chatMessagesRef.value
  if (!el) return true
  const diff = el.scrollHeight - el.scrollTop - el.clientHeight
  return diff < 100
}

async function scrollToBottom(force = false) {
  await nextTick()
  if (window.matchMedia('(max-width: 900px)').matches) {
    const page = document.scrollingElement || document.documentElement
    if (force || isUserNearBottom()) {
      window.scrollTo({ top: page.scrollHeight, behavior: 'auto' })
    }
    return
  }

  const el = chatMessagesRef.value
  if (!el) return
  if (force || isUserNearBottom()) {
    el.scrollTop = el.scrollHeight
  }
}

function resizeComposerInput(event) {
  const input = event?.target || composerInputRef.value
  if (!input) return
  input.style.height = 'auto'
  input.style.height = `${Math.min(Math.max(input.scrollHeight, 48), 140)}px`
}

function handleComposerFocus() {
  window.setTimeout(() => {
    scrollToBottom(true)
  }, 250)
}

function pushToast(message, type = 'info') {
  const id = ++toastCounter
  toasts.value.push({ id, message, type })
  setTimeout(() => {
    toasts.value = toasts.value.filter((item) => item.id !== id)
  }, 3000)
}

function detectInputLanguage(text) {
  const value = (text || '').trim()
  if (!value) return language.value

  if (/[\u0E00-\u0E7F]/.test(value)) return 'th'
  if (/[\u4E00-\u9FFF]/.test(value)) return 'zh'

  const lower = value.toLowerCase()

  const enKeywords = [
    'what', 'why', 'how', 'should', 'could', 'would',
    'do', 'does', 'is', 'are', 'if', 'tree', 'trunk',
    'leaf', 'leaves', 'soil', 'root', 'fruit', 'disease',
    'pest', 'drainage', 'gum', 'exudation'
  ]

  const enHitCount = enKeywords.filter((word) => {
    return new RegExp(`\\b${word}\\b`, 'i').test(lower)
  }).length

  if (enHitCount >= 2) return 'en'

  const msKeywords = [
    'apakah', 'mengapa', 'bagaimana', 'kenapa', 'punca',
    'tanah', 'daun', 'batang', 'akar', 'buah',
    'saliran', 'penanaman', 'pokok', 'penyakit', 'perosak',
    'mengeluarkan', 'getah', 'semasa', 'musim', 'hujan',
    'sesuai', 'perlu', 'dilakukan', 'menjadi', 'kuning'
  ]

  const msHitCount = msKeywords.filter((word) => {
    return new RegExp(`\\b${word}\\b`, 'i').test(lower)
  }).length

  if (msHitCount >= 2) return 'ms'

  if (/[a-z]/i.test(value)) return 'en'

  return language.value
}

function removeThink(text) {
  return normalizeModelOutput(text)
}

function blobToDataURL(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result)
    reader.onerror = reject
    reader.readAsDataURL(blob)
  })
}

async function pollAnalyzeJob(jobId, maxWaitMs = IMAGE_ANALYSIS_MAX_WAIT_MS, onProgress = null) {
  const start = Date.now()

  while (Date.now() - start < maxWaitMs) {
    const response = await fetch(`${API_BASE_URL}${endpoints.analyzePestJob}/${jobId}`, {
      method: 'GET',
      headers: getAuthHeaders(),
    })

    if (!response.ok) {
      const text = await response.text()
      throw new Error(`查询图片分析任务失败: HTTP ${response.status} ${text}`)
    }

    const job = await response.json()
    if (typeof onProgress === 'function') {
      onProgress(job, Date.now() - start)
    }

    if (job.status === 'done') {
      return job.result
    }

    if (job.status === 'error') {
      throw new Error(job.error || '图片分析任务失败')
    }

    await new Promise((resolve) => setTimeout(resolve, 1200))
  }

  throw new Error(t.value.imageAnalysisTimeout)
}

function postFormDataWithXHR(url, formData, headers = {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()

    xhr.open('POST', url, true)
    xhr.responseType = 'text'
    xhr.timeout = 300000

    Object.entries(headers || {}).forEach(([key, value]) => {
      xhr.setRequestHeader(key, value)
    })

    xhr.onload = () => {
      const rawText = xhr.responseText || ''

      console.log('XHR analyze_pest status:', xhr.status)
      console.log('XHR analyze_pest raw response:', rawText)

      if (xhr.status < 200 || xhr.status >= 300) {
        reject(new Error(`分析请求失败: HTTP ${xhr.status} ${rawText}`))
        return
      }

      try {
        const data = JSON.parse(rawText)
        resolve(data)
      } catch (parseError) {
        console.error('XHR JSON 解析失败:', parseError)
        console.error('XHR 原始响应:', rawText)
        reject(new Error('分析接口返回的不是合法 JSON'))
      }
    }

    xhr.onerror = () => {
      console.error('XHR analyze_pest 网络错误:', {
        status: xhr.status,
        readyState: xhr.readyState,
        responseText: xhr.responseText,
      })
      reject(new Error('图片分析请求网络错误'))
    }

    xhr.ontimeout = () => {
      console.error('XHR analyze_pest 请求超时')
      reject(new Error('图片分析请求超时'))
    }

    xhr.onabort = () => {
      console.error('XHR analyze_pest 请求被取消')
      reject(new Error('图片分析请求被取消'))
    }

    xhr.send(formData)
  })
}


function onImageSelected(event) {
  const file = event.target.files?.[0]
  if (!file) return

  pestImageFile.value = file
  const reader = new FileReader()
  reader.onload = (e) => {
    originalImagePreview.value = e.target?.result
    showCropperModal.value = true
    // 延迟初始化 Cropper，确保 DOM 已挂载
    nextTick(() => {
      initCropper()
    })
  }
  reader.readAsDataURL(file)
  event.target.value = ''
}

function initCropper() {
  const imageElement = document.getElementById('cropper-image')
  if (!imageElement || cropperInstance.value) return

  cropperInstance.value = new Cropper(imageElement, {
    aspectRatio: NaN,
    viewMode: 1,
    autoCropArea: 1,
    responsive: true,
    restore: true,
    guides: true,
    center: true,
    highlight: true,
    cropBoxMovable: true,
    cropBoxResizable: true,
    toggleDragModeOnDblclick: true,
  })
}

function closeCropperModal(options = {}) {
  const keepSelectedImage = Boolean(options.keepSelectedImage)
  showCropperModal.value = false
  if (cropperInstance.value) {
    cropperInstance.value.destroy()
    cropperInstance.value = null
  }
  originalImagePreview.value = null
  if (!keepSelectedImage) {
    pestImageFile.value = null
    pestImagePreview.value = null
  }
}

function clearPendingImage() {
  pestImageFile.value = null
  pestImagePreview.value = null
  originalImagePreview.value = null
  if (cropperInstance.value) {
    cropperInstance.value.destroy()
    cropperInstance.value = null
  }
  showCropperModal.value = false
}

async function confirmCrop() {
  if (!cropperInstance.value) return

  // 关键修复：分类模型必须吃“原始上传文件”，不能吃 canvas 裁剪/压缩后的 JPEG。
  // 裁剪结果只用于前端预览展示，避免公网手机端 canvas 重采样破坏病斑纹理。
  const originalFile = pestImageFile.value
  if (!originalFile) {
    pushToast(t.value.noImageSelected, 'warning')
    return
  }

  try {
    const canvas = cropperInstance.value.getCroppedCanvas({
      // 这里只生成预览图，不作为模型输入。尺寸适中即可。
      maxWidth: 1600,
      maxHeight: 1600,
      fillColor: '#fff',
      imageSmoothingEnabled: true,
      imageSmoothingQuality: 'high',
    })

    canvas.toBlob(
      async (blob) => {
        let previewUrl = originalImagePreview.value

        if (blob) {
          // 只用于聊天窗口显示，不上传给分类接口。
          previewUrl = await blobToDataURL(blob)
          console.log(`预览裁剪图大小: ${(blob.size / 1024).toFixed(1)}KB`)
        } else {
          console.warn('裁剪预览生成失败，改用原图预览')
        }

        console.log(`已选择图片原图: ${originalFile.name}, ${(originalFile.size / 1024).toFixed(1)}KB, ${originalFile.type}`)

        // 不立即发送图片。图片先作为输入框上方的 tag 等待用户确认发送，避免出现“图片 + 文本”混合消息占位。
        pestImagePreview.value = previewUrl
        closeCropperModal({ keepSelectedImage: true })
        await nextTick()
      },
      'image/jpeg',
      0.92,
    )
  } catch (error) {
    console.error('裁剪失败:', error)
    pushToast(t.value.errorConnection, 'error')
  }
}

async function classifyAndAnalyze(file, previewUrl) {
  if (isAnalyzingImage.value) {
    console.warn('图片分析正在进行，忽略重复请求')
    return
  }

  const uiLang = getCurrentUILang()

  console.log('图片分析使用语言:', uiLang)
  console.log('新版图片分析：跳过本地分类模型，直接调用视觉大模型')

  isAnalyzingImage.value = true
  isSending.value = true
  let assistantMessage = null
  let hasAnalysisResult = false

  try {
    console.log('开始视觉大模型图片分析...')
    console.log(`图片大小: ${(file.size / 1024).toFixed(1)}KB`)

    // 图片问题也要进入历史上下文，否则后续“严重吗/要不要喷药/怎么处理”无法追问。
    // 但前端展示不再强制显示“[图片]”这类占位文本；只有用户真的输入了问题，才作为图片说明显示。
    const typedImageQuestion = (userInput.value || '').trim()
    const imageQuestion = typedImageQuestion || getUITextByLang(uiLang, 'imageAnalyzePrompt') || '请分析这张榴莲图片'
    userInput.value = ''

    // 确保有对话 ID
    await createConversationIfNeeded(imageQuestion)

    // 立即添加用户图片消息：图片单独展示；只有用户输入了真实问题，才显示文字说明。
    const userMsg = {
      role: 'user',
      content: typedImageQuestion,
      imageKey: `img_${Date.now()}`,
      image_url: null,
      messageType: 'image',
    }
    messages.value.push(userMsg)
    messageImages.value = {
      ...messageImages.value,
      [userMsg.imageKey]: previewUrl,
    }

    // 立即添加助手消息占位符
    assistantMessage = {
      role: 'assistant',
      content: getRuntimeText(uiLang, 'imageGeneratingWithClass') || getUITextByLang(uiLang, 'imageAnalyzing'),
      isStreaming: true,
      messageType: 'image_analysis',
      evidence: [],
      evidence_quality: 'none',
    }
    messages.value.push(assistantMessage)

    await scrollToBottom(true)
    await scrollToBottom(true)

    // 直接创建视觉大模型分析任务：不再调用 /classify_pest
    console.log('创建视觉大模型分析任务...')
    const analyzeFormData = new FormData()
    analyzeFormData.append('file', file)
    analyzeFormData.append('query', imageQuestion)
    analyzeFormData.append('response_language', uiLang)

    const jobResponse = await fetch(`${API_BASE_URL}${endpoints.analyzePestJob}`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: analyzeFormData,
    })

    if (!jobResponse.ok) {
      const text = await jobResponse.text()
      throw new Error(`创建视觉分析任务失败: HTTP ${jobResponse.status} ${text}`)
    }

    const jobData = await jobResponse.json()
    console.log('视觉分析任务创建成功:', jobData)

    if (!jobData.job_id) {
      throw new Error('任务未返回 job_id')
    }

    // 轮询任务结果
    console.log('轮询视觉分析任务结果...')
    const jobStartTime = Date.now()
    const analyzeData = await pollAnalyzeJob(jobData.job_id, IMAGE_ANALYSIS_MAX_WAIT_MS, (_job, elapsedMs) => {
      if (!hasAnalysisResult && assistantMessage) {
        const elapsedSeconds = Math.floor(elapsedMs / 1000)
        const elapsedText = elapsedSeconds > 0 ? ` ${elapsedSeconds}s` : ''
        assistantMessage.content = `${getRuntimeText(uiLang, 'imageGeneratingWithClass') || getUITextByLang(uiLang, 'imageAnalyzing')}${elapsedText}`
        scheduleMessagesRender(true)
      }
    })
    console.log('任务完成耗时:', `${Date.now() - jobStartTime}ms`)
    console.log('分析结果:', analyzeData)

    const finalResponse = normalizeModelOutput(analyzeData.response || '')

    if (!finalResponse) {
      console.error('分析接口返回内容为空，完整返回:', analyzeData)
      throw new Error('分析接口返回内容为空')
    }

    // 显示完整分析结果；上下文卡片单独走 active_context_card 字段，不再塞进正文。
    assistantMessage.content = finalResponse
    assistantMessage.active_context_card = analyzeData.active_context_card || null
    assistantMessage.isStreaming = false
    assistantMessage.messageType = 'image_analysis'
    hasAnalysisResult = true
    messages.value = [...messages.value]
    await scrollToBottom(true)

    // 更新 Token 计数
    totalTokens.value += Number(analyzeData.tokens_generated || 0)

    // 更新证据
    const imageEvidence = Array.isArray(analyzeData.evidence) ? analyzeData.evidence : []
    const imageEvidenceQuality = analyzeData.evidence_quality || 'none'
    assistantMessage.evidence = imageEvidence
    assistantMessage.evidence_quality = imageEvidenceQuality
    evidence.value = imageEvidence
    evidenceQuality.value = imageEvidenceQuality
    evidenceExpanded.value = {}

    try {
      saveEvidenceCache(currentConversationId.value, evidence.value, evidenceQuality.value)
    } catch (cacheError) {
      console.warn('参考资料缓存失败:', cacheError)
    }

    // 保存用户图片消息
    try {
      await saveMessageToConversation('user', imageQuestion, analyzeData.image_url || null)
    } catch (saveUserError) {
      console.warn('保存用户图片消息失败:', saveUserError)
    }

    // 保存助手分析消息
    try {
      await saveMessageToConversation('assistant', assistantMessage.content, null, assistantMessage.active_context_card || null)
    } catch (saveAssistantError) {
      console.warn('保存助手分析消息失败:', saveAssistantError)
    }

    await scrollToBottom(true)
  } catch (error) {
    console.error('分析失败:', error)
    console.error('图片分析请求异常类型:', error?.name)
    console.error('图片分析请求异常信息:', error?.message)

    if (error?.message && (error.message.includes('NetworkError') || error.message.includes('Failed to fetch'))) {
      console.error('这通常表示请求没有拿到 HTTP 响应，可能是 Vite 代理断开、后端进程中断、CORS 或网络连接问题。')
    }

    // 只有在没有真实分析结果时才显示错误提示
    if (!hasAnalysisResult && assistantMessage) {
      assistantMessage.content = t.value.imageAnalysisFailed
      assistantMessage.isStreaming = false
      messages.value = [...messages.value]
      pushToast(t.value.imageAnalysisFailed, 'error')
    }
  } finally {
    isSending.value = false
    isAnalyzingImage.value = false
    pestImageFile.value = null
    pestImagePreview.value = null
  }
}

async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoints.health}`, {
      headers: getAuthHeaders(),
    })
    if (!response.ok) throw new Error('Health check failed')
    const data = await response.json()
    isConnected.value = true
    modelInfo.value = data
  } catch (error) {
    isConnected.value = false
  }
}

function mergeRagRuntimeInfo(data = {}) {
  modelInfo.value = {
    ...modelInfo.value,
    ...data,
    rag_enabled: data.rag_enabled ?? modelInfo.value.rag_enabled,
    rag_chunks: data.rag_chunks ?? data.chunks_count ?? modelInfo.value.rag_chunks ?? 0,
    rag_pdf_files: data.rag_pdf_files ?? data.pdf_count ?? modelInfo.value.rag_pdf_files ?? 0,
    rag_auto_chunks: data.rag_auto_chunks ?? data.auto_chunks ?? data.chunks_count ?? modelInfo.value.rag_auto_chunks ?? 0,
    rag_auto_last_sync: data.rag_auto_last_sync ?? data.last_sync ?? modelInfo.value.rag_auto_last_sync,
  }
}

async function uploadRagPdf(event) {
  const input = event?.target
  const file = input?.files?.[0]
  if (input) input.value = ''
  if (!file) return

  const isPdf = file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')
  if (!isPdf) {
    ragUploadStatus.value = t.value.ragUploadFailed
    pushToast(t.value.ragUploadFailed, 'error')
    return
  }

  isUploadingRagPdf.value = true
  ragUploadStatus.value = t.value.ragUploading

  try {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch(`${API_BASE_URL}${endpoints.ragUploadPdf}`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: formData,
    })
    const data = await response.json().catch(() => ({}))
    if (!response.ok) {
      throw new Error(data.detail || t.value.ragUploadFailed)
    }

    mergeRagRuntimeInfo(data)
    ragUploadStatus.value = `${t.value.ragUploadDone}: ${file.name}`
    pushToast(t.value.ragUploadDone, 'success')
    await checkHealth()
  } catch (error) {
    console.error('RAG PDF 上传失败:', error)
    ragUploadStatus.value = error?.message || t.value.ragUploadFailed
    pushToast(t.value.ragUploadFailed, 'error')
  } finally {
    isUploadingRagPdf.value = false
  }
}

async function rebuildRag() {
  if (isUploadingRagPdf.value) return

  isUploadingRagPdf.value = true
  ragUploadStatus.value = t.value.ragUploading

  try {
    const response = await fetch(`${API_BASE_URL}${endpoints.ragRebuild}`, {
      method: 'POST',
      headers: getAuthHeaders(),
    })
    const data = await response.json().catch(() => ({}))
    if (!response.ok) {
      throw new Error(data.detail || t.value.ragRebuildFailed)
    }

    mergeRagRuntimeInfo(data)
    ragUploadStatus.value = t.value.ragRebuildDone
    pushToast(t.value.ragRebuildDone, 'success')
    await checkHealth()
  } catch (error) {
    console.error('RAG 重建失败:', error)
    ragUploadStatus.value = error?.message || t.value.ragRebuildFailed
    pushToast(t.value.ragRebuildFailed, 'error')
  } finally {
    isUploadingRagPdf.value = false
  }
}

async function sendMessage(prefilled = '') {
  if (sendingLock || isSending.value) {
    return
  }

  const text = (prefilled || userInput.value).trim()
  const hasPendingImage = Boolean(pestImageFile.value)

  if (hasPendingImage) {
    if (!isConnected.value) {
      pushToast(t.value.errorConnection, 'error')
      return
    }
    sendingLock = true
    try {
      await classifyAndAnalyze(pestImageFile.value, pestImagePreview.value || originalImagePreview.value)
    } finally {
      sendingLock = false
    }
    return
  }

  if (!text) {
    pushToast(t.value.errorEmpty, 'warning')
    return
  }
  if (!isConnected.value) {
    pushToast(t.value.errorConnection, 'error')
    return
  }

  sendingLock = true
  isSending.value = true

  // 确保有对话 ID
  await createConversationIfNeeded(text)

  evidence.value = []
  evidenceQuality.value = 'none'
  evidenceExpanded.value = {}
  saveEvidenceCache(currentConversationId.value, evidence.value, evidenceQuality.value)

  // 输入语言用于筛选历史和控制回答语言：英文输入回英文，中文输入回中文。
  // 图片分析仍然按当前界面语言走，文字聊天按用户输入语言走。
  const detectedLanguage = detectInputLanguage(text)
  const answerLanguage = detectedLanguage

  // 过滤空消息，但保留图片分析产生的文本/隐藏上下文，支持后续追问。
  // 旧逻辑会把 messageType === 'image_analysis' 全部排除，导致“这张图严重吗/要不要喷药”没有上下文。
  const historyMessages = messages.value
    .filter((msg) => msg.content && msg.content.trim())
    .filter((msg) => {
      const content = msg.content || ''
      if (msg.active_context_card) return true
      if (content.includes('[Image context]') || content.includes('[Durian context card]')) return true
      return detectInputLanguage(content) === detectedLanguage
    })
    .slice(-6)
    .map((msg) => ({
      role: msg.role,
      content: normalizeModelOutput(msg.content),
      image_url: msg.image_url || null,
      active_context_card: msg.active_context_card || null,
    }))

  messages.value.push({ role: 'user', content: text })
  const assistantMessage = {
    role: 'assistant',
    content: streamMode.value ? getRuntimeText(answerLanguage, 'chatPreparing') : '',
    isStreaming: true,
    evidence: [],
    evidence_quality: 'none',
  }
  messages.value.push(assistantMessage)
  userInput.value = ''
  await nextTick()
  resizeComposerInput()
  await scrollToBottom(true)

  // 保存用户消息
  await saveMessageToConversation('user', text)

  console.log('当前 UI 语言:', language.value)
  console.log('输入检测语言:', detectedLanguage)
  console.log('回答语言:', answerLanguage)

  try {
    const endpoint = streamMode.value ? endpoints.chatStream : endpoints.chat
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify({
        messages: [...historyMessages, { role: 'user', content: text }],
        max_tokens: Number(maxTokens.value) || DEFAULT_CHAT_MAX_TOKENS,
        max_new_tokens: Number(maxTokens.value) || DEFAULT_CHAT_MAX_TOKENS,
        temperature: Number(temperature.value),
        top_p: Number(topP.value),
        use_rag: true,
        stream: Boolean(streamMode.value),
        response_language: answerLanguage,
      }),
    })

    if (!response.ok) {
      const errorText = await response.text().catch(() => '')
      throw new Error(`API error: ${response.status} ${errorText}`)
    }

    if (streamMode.value) {
      if (!response.body) {
        throw new Error('当前浏览器不支持 ReadableStream')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''
      let streamDone = false
      let receivedContent = false
      let lastWaitingUpdate = 0

      while (!streamDone) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const parsedBuffer = splitSSEEvents(buffer)
        buffer = parsedBuffer.rest

        for (const rawEvent of parsedBuffer.events) {
          const data = getSSEData(rawEvent)
          if (!data) continue

          if (data === '[DONE]') {
            streamDone = true
            break
          }

          let json = null
          try {
            json = JSON.parse(data)
          } catch (e) {
            console.error('SSE parse error:', e, 'raw data:', data, 'raw event:', rawEvent)
            continue
          }

          if (json.type === 'content') {
            if (!receivedContent) {
              assistantMessage.content = ''
              receivedContent = true
            }
            assistantMessage.content += normalizeStreamChunk(json.content || '')
            scheduleMessagesRender(false)
          } else if (json.type === 'metadata') {
            if (json.tokens_generated) {
              totalTokens.value += Number(json.tokens_generated)
            }
            if (json.active_context_card) {
              assistantMessage.active_context_card = json.active_context_card
            }
            const metadataEvidence = Array.isArray(json.evidence) ? json.evidence : []
            const metadataQuality = json.evidence_quality || 'none'
            assistantMessage.evidence = metadataEvidence
            assistantMessage.evidence_quality = metadataQuality
            evidence.value = metadataEvidence
            evidenceQuality.value = metadataQuality
            evidenceExpanded.value = {}
            saveEvidenceCache(currentConversationId.value, evidence.value, evidenceQuality.value)
          } else if (json.type === 'status' || json.type === 'ping') {
            const now = Date.now()
            if (!receivedContent && now - lastWaitingUpdate > 900) {
              lastWaitingUpdate = now
              assistantMessage.content = getRuntimeText(answerLanguage, 'chatGenerating')
              scheduleMessagesRender(true)
            }
          } else if (json.type === 'error') {
            throw new Error(json.error || t.value.errorConnection)
          }
        }
      }
    } else {
      // 非流式处理
      const data = await response.json()
      assistantMessage.content = normalizeModelOutput(data.response || '')
      assistantMessage.active_context_card = data.active_context_card || null

      // 更新 Token 计数
      totalTokens.value += Number(data.tokens_generated || 0)

      const responseEvidence = Array.isArray(data.evidence) ? data.evidence : []
      const responseQuality = data.evidence_quality || 'none'
      assistantMessage.evidence = responseEvidence
      assistantMessage.evidence_quality = responseQuality
      evidence.value = responseEvidence
      evidenceQuality.value = responseQuality
      evidenceExpanded.value = {}
      saveEvidenceCache(currentConversationId.value, evidence.value, evidenceQuality.value)

      messages.value = [...messages.value]
      await scrollToBottom(true)
    }

    assistantMessage.content = normalizeModelOutput(assistantMessage.content)
    assistantMessage.isStreaming = false
    messages.value = [...messages.value]
    await scrollToBottom(true)

    if (!assistantMessage.content) {
      throw new Error('模型返回内容为空')
    }

    // 保存助手消息
    await saveMessageToConversation('assistant', assistantMessage.content, null, assistantMessage.active_context_card || null)
  } catch (error) {
    console.error('发送消息失败:', error)
    assistantMessage.content = t.value.errorConnection
    assistantMessage.isStreaming = false
    messages.value = [...messages.value]
    pushToast(t.value.errorConnection, 'error')
  } finally {
    isSending.value = false
    sendingLock = false
  }
}

function startNewChat() {
  currentConversationId.value = null

  messages.value = []
  messageImages.value = {}

  evidence.value = []
  evidenceQuality.value = 'none'
  evidenceExpanded.value = {}

  userInput.value = ''

  pestImageFile.value = null
  pestImagePreview.value = null
  pestClassification.value = null
  originalImagePreview.value = null

  showCropperModal.value = false

  if (cropperInstance.value) {
    cropperInstance.value.destroy()
    cropperInstance.value = null
  }

  isSending.value = false
  totalTokens.value = 0
}

async function initializeLoggedInSession() {
  totalTokens.value = Number(localStorage.getItem(getUserStorageKey('totalTokens')) || 0)
  currentConversationId.value = null
  messages.value = []
  messageImages.value = {}
  conversations.value = []
  evidence.value = []
  evidenceQuality.value = 'none'
  evidenceExpanded.value = {}
  userInput.value = ''
  pestImageFile.value = null
  pestImagePreview.value = null
  pestClassification.value = null
  originalImagePreview.value = null
  ragUploadStatus.value = ''

  await checkHealth()
  await loadConversations()

  const savedConversationId = localStorage.getItem(getUserStorageKey('currentConversationId'))
  if (savedConversationId) {
    await loadConversation(savedConversationId)
  }
}

async function login() {
  const username = loginUsername.value.trim()
  const password = loginPassword.value
  const account = LOGIN_ACCOUNTS.find((item) => item.username === username && item.password === password)

  if (!account) {
    loginError.value = t.value.loginFailed
    return
  }

  currentUser.value = account.username
  localStorage.setItem(CURRENT_USER_KEY, account.username)
  loginUsername.value = account.username
  loginPassword.value = ''
  loginError.value = ''
  await initializeLoggedInSession()
}

function logout() {
  const previousUser = currentUser.value
  currentConversationId.value = null
  messages.value = []
  messageImages.value = {}
  evidence.value = []
  evidenceQuality.value = 'none'
  evidenceExpanded.value = {}
  userInput.value = ''
  pestImageFile.value = null
  pestImagePreview.value = null
  pestClassification.value = null
  originalImagePreview.value = null
  ragUploadStatus.value = ''
  conversations.value = []
  currentUser.value = ''
  localStorage.removeItem(CURRENT_USER_KEY)
  totalTokens.value = 0
  loginUsername.value = previousUser || 'admin'
  loginPassword.value = ''
  loginError.value = ''
}

function onKeyPress(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    if (sendingLock || isSending.value) return
    sendMessage()
  }
}

onMounted(async () => {
  document.documentElement.lang = language.value

  if (isLoggedIn.value) {
    await initializeLoggedInSession()
  }

  healthTimer = window.setInterval(() => {
    if (isLoggedIn.value) {
      checkHealth()
    }
  }, 30000)
})

onBeforeUnmount(() => {
  if (healthTimer) {
    clearInterval(healthTimer)
  }
  if (pendingMessageRender) {
    window.cancelAnimationFrame(pendingMessageRender)
    pendingMessageRender = 0
  }
})
</script>

<template>
  <div v-if="!isLoggedIn" class="login-shell">
    <form class="login-panel" @submit.prevent="login">
      <img src="/durian-logo.png" alt="Durian Logo" class="login-logo" />
      <h1>{{ t.loginTitle }}</h1>
      <p>{{ t.loginSubtitle }}</p>

      <label class="login-field">
        <span>{{ t.username }}</span>
        <select v-model="loginUsername">
          <option value="admin">admin</option>
          <option value="admin2">admin2</option>
        </select>
      </label>

      <label class="login-field">
        <span>{{ t.password }}</span>
        <input v-model="loginPassword" type="password" autocomplete="current-password" />
      </label>

      <div v-if="loginError" class="login-error">{{ loginError }}</div>
      <button class="btn btn-primary login-submit" type="submit">{{ t.login }}</button>
    </form>
  </div>

  <div
    v-else
    class="app-shell"
    :class="{ 'with-right-sidebar': !rightSidebarCollapsed }"
    :style="{ '--right-sidebar-width': `${rightSidebarWidth}px` }"
  >
    <aside class="sidebar left-sidebar">
      <div class="brand-panel">
        <img src="/durian-logo.png" alt="Durian Logo" class="brand-mark" />
        <div>
          <h1>{{ t.pageTitle }}</h1>
          <p>{{ t.subtitle }}</p>
        </div>
      </div>

      <div class="toolbar">
        <select v-model="language" class="select">
          <option value="zh">中文</option>
          <option value="en">English</option>
          <option value="ms">Bahasa Melayu</option>
          <option value="th">ไทย</option>
        </select>
        <button class="btn btn-primary" @click="startNewChat">{{ t.newChat }}</button>
        <div class="session-row">
          <span>{{ t.account }}: {{ currentUser }}</span>
          <button class="session-logout" type="button" @click="logout">{{ t.logout }}</button>
        </div>
      </div>

      <section class="rag-control-panel">
        <div class="section-title">RAG</div>
        <input
          ref="ragPdfInput"
          type="file"
          accept="application/pdf,.pdf"
          class="hidden-file-input"
          @change="uploadRagPdf"
        />
        <button class="btn btn-secondary" :disabled="isUploadingRagPdf" @click="ragPdfInput?.click()">
          {{ isUploadingRagPdf ? t.ragUploading : t.ragUploadPdf }}
        </button>
        <button class="btn btn-secondary" :disabled="isUploadingRagPdf" @click="rebuildRag">
          {{ t.ragRebuild }}
        </button>
        <div v-if="ragUploadStatus" class="rag-upload-status">{{ ragUploadStatus }}</div>
      </section>

      <section class="history-panel">
        <div class="section-title">{{ t.history }}</div>

        <div v-if="conversations.length === 0" class="empty-state">
          {{ t.noHistory }}
        </div>

        <div
          v-for="conv in conversations"
          :key="conv.id"
          class="history-item"
          :class="{ active: conv.id === currentConversationId }"
          @click="loadConversation(conv.id)"
        >
          <div class="history-content">
            <div class="history-title">{{ conv.title }}</div>
            <div class="history-meta">
              {{ conv.message_count || 0 }} {{ t.messageCountUnit }}
            </div>
          </div>
          <button
            class="history-delete-btn"
            :title="t.deleteConversation"
            @click.stop="deleteConversation(conv.id)"
          >
            ×
          </button>
        </div>
      </section>
    </aside>

    <main class="main-panel">
      <section v-if="messages.length === 0" class="welcome-panel">
        <div class="welcome-card">
          <div class="hero-badge">DurianGPT</div>
          <h2>{{ t.subtitle }}</h2>
        </div>
      </section>

      <section v-else ref="chatMessagesRef" class="messages-panel">
        <div
          v-for="(message, index) in messages"
          :key="`${message.role}-${index}`"
          class="message-row"
          :class="message.role"
        >
          <div class="avatar">{{ message.role === 'user' ? t.userAvatar : t.assistantAvatar }}</div>
          <div class="message-bubble" :class="message.role">
            <img v-if="message.imageKey && messageImages[message.imageKey]" :src="messageImages[message.imageKey]" class="message-image" alt="uploaded" />
            <div v-if="message.content" v-html="renderStructuredAnswer(message.content)" class="message-content" />
            <div
              v-if="message.role === 'assistant' && getMessageEvidence(message).length"
              class="message-evidence-panel"
            >
              <div class="message-evidence-title">
                {{ t.references }}
                <span v-if="message.evidence_quality"> · {{ getQualityLabel(message.evidence_quality) }}</span>
              </div>
              <div
                v-for="(item, evidenceIndex) in getMessageEvidence(message)"
                :key="`inline-evidence-${index}-${evidenceIndex}`"
                class="message-evidence-item"
              >
                <div class="message-evidence-meta">
                  <strong>[{{ t.evidenceLabel }}{{ evidenceIndex + 1 }}]</strong>
                  <span v-if="getEvidenceScore(item)">{{ getEvidenceScore(item) }}</span>
                </div>
                <div class="message-evidence-text">{{ getInlineEvidencePreview(item.text) }}</div>
                <div class="message-evidence-source">{{ getEvidenceSource(item) }}</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section class="composer-panel">
        <div class="composer-controls">
          <label class="switch">
            <input v-model="streamMode" type="checkbox" />
            <span>{{ t.streamMode }}</span>
          </label>
          <div class="control-sliders">
            <div class="range-group">
              <label>{{ t.tempLabel }}</label>
              <input v-model="temperature" type="range" min="0" max="1.5" step="0.1" />
              <span>{{ temperature }}</span>
            </div>
            <div class="range-group">
              <label>{{ t.topPLabel }}</label>
              <input v-model="topP" type="range" min="0" max="1" step="0.05" />
              <span>{{ topP }}</span>
            </div>
          </div>
        </div>

        <div v-if="pestImageFile && pestImagePreview" class="pending-image-tag">
          <img :src="pestImagePreview" class="pending-image-thumb" alt="selected image" />
          <div class="pending-image-meta">
            <span class="pending-image-title">{{ pestImageFile.name || t.selectedImage }}</span>
            <span class="pending-image-hint">图片已添加，输入问题后发送；不输入则直接分析图片</span>
          </div>
          <button class="pending-image-remove" type="button" :disabled="isSending" @click="clearPendingImage">×</button>
        </div>

        <div class="composer-input-area">
          <input
            ref="imageInput"
            type="file"
            accept="image/*"
            style="display: none"
            @change="onImageSelected"
          />
          <button class="btn btn-icon" :disabled="isSending" @click.prevent="$refs.imageInput?.click()" title="上传图片">
            ➕
          </button>
          <textarea
            ref="composerInputRef"
            v-model="userInput"
            class="composer-input"
            :placeholder="t.placeholder"
            rows="1"
            @input="resizeComposerInput"
            @focus="handleComposerFocus"
            @keypress="onKeyPress"
          />
          <button class="btn btn-primary send-btn" :disabled="isSending" @click="sendMessage()">
            {{ isSending ? '...' : t.send }}
          </button>
        </div>
      </section>
    </main>

    <aside class="sidebar right-sidebar" :class="{ collapsed: rightSidebarCollapsed }" :style="{ width: rightSidebarCollapsed ? '50px' : rightSidebarWidth + 'px' }">
      <button class="sidebar-toggle" @click="rightSidebarCollapsed = !rightSidebarCollapsed" :title="rightSidebarCollapsed ? '展开' : '折叠'">
        {{ rightSidebarCollapsed ? '◀' : '▶' }}
      </button>
      <div v-if="!rightSidebarCollapsed" class="resize-handle" @mousedown="startResizeRightSidebar" />
      <section class="reference-card">
        <div class="reference-header">
          <div>
            <div class="section-title">{{ t.references }}</div>
            <div class="reference-summary">
              {{ evidence.length }} {{ t.evidenceLabel }}
              <span v-if="evidence.length"> · {{ getQualityLabel(evidenceQuality) }}</span>
            </div>
          </div>
        </div>
        <div v-if="evidence.length === 0" class="empty-state">{{ t.noEvidence }}</div>
        <article v-for="(item, index) in evidence" :key="index" class="evidence-item">
          <div class="evidence-head">
            <strong>[{{ t.evidenceLabel }}{{ index + 1 }}]</strong>
            <span>{{ item.score ? `${(item.score * 100).toFixed(0)}%` : '0%' }}</span>
          </div>
          <div class="evidence-text">{{ getEvidencePreview(item.text, index) }}</div>
          <button
            v-if="item.text && item.text.length > 700"
            type="button"
            class="evidence-toggle"
            @click="toggleEvidence(index)"
          >
            {{ evidenceExpanded[index] ? t.collapseText : t.expandText }}
          </button>
          <div class="evidence-source">{{ item.doc || t.unknownSource }}</div>
        </article>
      </section>
    </aside>

    <div class="toast-stack">
      <div v-for="toast in toasts" :key="toast.id" class="toast" :class="toast.type">
        {{ toast.message }}
      </div>
    </div>

    <!-- 裁剪模态框 -->
    <div v-if="showCropperModal" class="cropper-modal-overlay" @click="closeCropperModal">
      <div class="cropper-modal" @click.stop>
        <div class="cropper-header">
          <h3>{{ t.cropImageTitle }}</h3>
          <button class="close-btn" @click="closeCropperModal">✕</button>
        </div>
        <div class="cropper-container">
          <img id="cropper-image" :src="originalImagePreview" alt="Cropper" />
        </div>
        <div class="cropper-footer">
          <button class="btn btn-secondary" @click="closeCropperModal">{{ t.cancel }}</button>
          <button class="btn btn-primary" @click="confirmCrop">{{ t.confirmCrop }}</button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.login-shell {
  display: flex;
  min-height: 100vh;
  min-height: 100dvh;
  align-items: center;
  justify-content: center;
  padding: 24px;
  background: #f4f7f4;
}

.login-panel {
  width: min(380px, 100%);
  padding: 24px;
  border: 1px solid #dfe7df;
  border-radius: 8px;
  background: white;
  box-shadow: 0 10px 30px rgba(33, 70, 42, 0.12);
}

.login-logo {
  width: 56px;
  height: 56px;
  object-fit: contain;
  border-radius: 6px;
}

.login-panel h1 {
  margin: 14px 0 6px;
  font-size: 22px;
  font-weight: 700;
  color: #1f2937;
}

.login-panel p {
  margin: 0 0 20px;
  color: #5b665d;
  font-size: 14px;
  line-height: 1.5;
}

.login-field {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 14px;
  color: #374151;
  font-size: 13px;
  font-weight: 600;
}

.login-field input,
.login-field select {
  width: 100%;
  box-sizing: border-box;
  padding: 10px 12px;
  border: 1px solid #ccd8ce;
  border-radius: 6px;
  font: inherit;
  color: #1f2937;
  background: white;
}

.login-error {
  margin: 2px 0 12px;
  color: #dc2626;
  font-size: 13px;
}

.login-submit {
  width: 100%;
  min-height: 40px;
}

.app-shell {
  display: grid;
  grid-template-columns: 250px 1fr;
  grid-template-rows: 1fr auto;
  height: 100vh;
  height: 100dvh;
  min-width: 0;
  gap: 1px;
  background: #f0f0f0;
}

.app-shell.with-right-sidebar {
  grid-template-columns: 250px minmax(0, 1fr) var(--right-sidebar-width, 420px);
}

.sidebar {
  background: white;
  border-right: 1px solid #e0e0e0;
  overflow-y: auto;
  padding: 20px;
}

.left-sidebar {
  grid-column: 1;
  grid-row: 1 / 3;
}

.right-sidebar {
  grid-column: 3;
  grid-row: 1 / 3;
  border-left: 1px solid #e0e0e0;
  border-right: none;
  position: relative;
  transition: all 0.3s ease;
  box-sizing: border-box;
  padding: 20px;
  overflow-y: auto;
}

.right-sidebar.collapsed {
  width: 50px;
  padding: 0;
  overflow: hidden;
}

.right-sidebar.collapsed .reference-card {
  display: none;
}

.sidebar-toggle {
  position: absolute;
  top: 20px;
  right: 10px;
  width: 30px;
  height: 30px;
  background: #f0f0f0;
  border: 1px solid #ddd;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10;
  transition: all 0.2s ease;
}

.sidebar-toggle:hover {
  background: #e0e0e0;
  border-color: #4CAF50;
}

.right-sidebar.collapsed .sidebar-toggle {
  right: 8px;
}

.main-panel {
  grid-column: 2;
  grid-row: 1 / 3;
  display: flex;
  min-width: 0;
  min-height: 0;
  flex-direction: column;
  background: white;
}

.brand-panel {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
  align-items: center;
}

.brand-mark {
  width: 48px;
  height: 48px;
  object-fit: contain;
  border-radius: 4px;
}

.brand-panel h1 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
}

.brand-panel p {
  margin: 4px 0 0 0;
  font-size: 12px;
  color: #666;
}

.toolbar {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.hidden-file-input {
  display: none;
}

.rag-upload-status {
  color: #56705a;
  font-size: 12px;
  line-height: 1.4;
  overflow-wrap: anywhere;
}

.rag-control-panel {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 12px;
  padding: 12px 0;
  border-top: 1px solid #e6ece6;
  border-bottom: 1px solid #e6ece6;
}

.rag-control-panel .section-title {
  margin-bottom: 0;
}

.rag-control-panel .btn {
  width: 100%;
}

.session-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  color: #566056;
  font-size: 12px;
}

.session-logout {
  padding: 4px 8px;
  border: 1px solid #d7dfd8;
  border-radius: 4px;
  background: white;
  color: #374151;
  cursor: pointer;
  font-size: 12px;
}

.session-logout:hover {
  background: #f2f6f2;
}

.select,
.btn {
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
  cursor: pointer;
}

.btn:disabled {
  cursor: not-allowed;
  opacity: 0.62;
}

.btn-primary {
  background: #4CAF50;
  color: white;
  border: none;
}

.btn-primary:hover:not(:disabled) {
  background: #45a049;
}

.welcome-panel {
  display: flex;
  align-items: center;
  justify-content: center;
  flex: 1;
}

.welcome-card {
  text-align: center;
}

.hero-badge {
  font-size: 48px;
  font-weight: bold;
  color: #4CAF50;
  margin-bottom: 20px;
}

.messages-panel {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  -webkit-overflow-scrolling: touch;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.composer-panel {
  border-top: 1px solid #e0e0e0;
  padding: 16px;
  background: white;
}

.composer-controls {
  display: flex;
  gap: 16px;
  margin-bottom: 12px;
  font-size: 12px;
}

.switch {
  display: flex;
  align-items: center;
  gap: 8px;
}

.switch input {
  cursor: pointer;
}

.control-sliders {
  display: flex;
  gap: 16px;
}

.range-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.range-group input {
  width: 80px;
}

.pending-image-tag {
  display: flex;
  align-items: center;
  gap: 10px;
  width: fit-content;
  max-width: min(520px, 100%);
  margin-bottom: 10px;
  padding: 8px 10px;
  border: 1px solid #d8ead8;
  border-radius: 10px;
  background: #f6fff6;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
}

.pending-image-thumb {
  width: 54px;
  height: 54px;
  object-fit: cover;
  border-radius: 8px;
  border: 1px solid #d7d7d7;
}

.pending-image-meta {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 2px;
}

.pending-image-title {
  max-width: 340px;
  overflow: hidden;
  color: #333;
  font-size: 13px;
  font-weight: 600;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.pending-image-hint {
  color: #666;
  font-size: 12px;
}

.pending-image-remove {
  width: 24px;
  height: 24px;
  border: none;
  border-radius: 50%;
  background: #e7e7e7;
  color: #333;
  cursor: pointer;
  font-size: 16px;
  line-height: 24px;
}

.pending-image-remove:hover:not(:disabled) {
  background: #d7d7d7;
}

.composer-input-area {
  display: flex;
  gap: 8px;
}

.composer-input {
  flex: 1;
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-family: inherit;
  resize: none;
}

.send-btn {
  padding: 8px 16px;
  background: #4CAF50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.send-btn:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.btn-icon {
  padding: 8px 12px;
  background: #f0f0f0;
  color: #333;
  border: 1px solid #ddd;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 40px;
}

.btn-icon:hover:not(:disabled) {
  background: #e0e0e0;
}

.btn-icon:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.reference-card {
  min-height: 100%;
}

.section-title {
  font-weight: 600;
  margin-bottom: 12px;
  font-size: 14px;
}

.reference-header {
  position: sticky;
  top: 0;
  z-index: 2;
  margin: -20px -20px 14px;
  padding: 18px 20px 12px;
  border-bottom: 1px solid #e7ece7;
  background: white;
}

.reference-header .section-title {
  margin-bottom: 4px;
  font-size: 16px;
}

.reference-summary {
  color: #6b7280;
  font-size: 12px;
}

.info-item {
  display: flex;
  justify-content: space-between;
  padding: 8px 0;
  font-size: 12px;
  border-bottom: 1px solid #f0f0f0;
}

.status-pill {
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 11px;
}

.status-pill.ok {
  background: #c8e6c9;
  color: #2e7d32;
}

.status-pill.bad {
  background: #ffcdd2;
  color: #c62828;
}

.empty-state {
  color: #999;
  font-size: 12px;
  text-align: center;
  padding: 20px 0;
}

.evidence-item {
  padding: 12px;
  border: 1px solid #dfe7df;
  border-radius: 8px;
  margin-bottom: 12px;
  background: #fbfdfb;
  font-size: 13px;
}

.evidence-head {
  display: flex;
  gap: 12px;
  justify-content: space-between;
  margin-bottom: 8px;
  font-weight: 600;
  color: #244428;
}

.evidence-text {
  margin-bottom: 8px;
  color: #333;
  line-height: 1.58;
  word-break: break-word;
  overflow-wrap: break-word;
  white-space: pre-wrap;
}

.evidence-source {
  padding-top: 8px;
  border-top: 1px solid #edf2ed;
  font-size: 12px;
  color: #6b7280;
  word-break: break-all;
}

.evidence-toggle {
  margin: 0 0 8px;
  padding: 6px 10px;
  border: 1px solid #cfd8cf;
  border-radius: 6px;
  background: white;
  color: #2e7d32;
  cursor: pointer;
  font-size: 12px;
}

.toast-stack {
  position: fixed;
  bottom: 20px;
  right: 20px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  z-index: 1000;
}

.toast {
  padding: 12px 16px;
  border-radius: 4px;
  color: white;
  font-size: 14px;
  animation: slideIn 0.3s ease-out;
}

.toast.info {
  background: #2196F3;
}

.toast.success {
  background: #4CAF50;
}

.toast.warning {
  background: #FF9800;
}

.toast.error {
  background: #f44336;
}

@keyframes slideIn {
  from {
    transform: translateX(400px);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

/* 裁剪模态框样式 */
.cropper-modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
}

.cropper-modal {
  background: white;
  border-radius: 8px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
  display: flex;
  flex-direction: column;
  width: 90%;
  max-width: 800px;
  max-height: 90vh;
  overflow: hidden;
}

.cropper-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid #e0e0e0;
}

.cropper-header h3 {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
}

.close-btn {
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: #999;
  padding: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.close-btn:hover {
  color: #333;
}

.cropper-container {
  flex: 1;
  overflow: auto;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f5f5f5;
  padding: 20px;
}

.cropper-container img {
  max-width: 100%;
  max-height: 100%;
}

.cropper-footer {
  display: flex;
  gap: 12px;
  justify-content: flex-end;
  padding: 16px 20px;
  border-top: 1px solid #e0e0e0;
}

.btn-secondary {
  background: #f0f0f0;
  color: #333;
  border: 1px solid #ddd;
}

.btn-secondary:hover:not(:disabled) {
  background: #e0e0e0;
}

/* Markdown 内容样式 - 使用 :deep() 确保 v-html 内容正常渲染 */
.message-content {
  width: 100%;
  max-width: 100%;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  overflow-wrap: anywhere;
  word-break: break-word;
}

.message-bubble.user .message-content,
.message-bubble.user .message-content :deep(*) {
  color: white;
}

.message-content :deep(p) {
  margin: 0 0 8px 0;
  line-height: 1.6;
  color: #333;
}

.message-content :deep(p:last-child) {
  margin-bottom: 0;
}

.message-content :deep(ul),
.message-content :deep(ol) {
  margin: 8px 0;
  padding-left: 22px;
  color: #333;
}

.message-content :deep(li) {
  margin-bottom: 6px;
  line-height: 1.6;
  color: #333;
}

.message-content :deep(h1),
.message-content :deep(h2),
.message-content :deep(h3),
.message-content :deep(h4),
.message-content :deep(h5),
.message-content :deep(h6) {
  margin: 12px 0 8px;
  line-height: 1.4;
  color: #333;
  font-weight: 600;
}

.message-content :deep(h1) {
  font-size: 18px;
}

.message-content :deep(h2) {
  font-size: 16px;
}

.message-content :deep(h3) {
  font-size: 14px;
}

.message-content :deep(table) {
  width: 100%;
  border-collapse: collapse;
  margin: 10px 0;
  font-size: 13px;
  table-layout: fixed;
  display: block;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
}

.message-content :deep(th),
.message-content :deep(td) {
  border: 1px solid #ddd;
  padding: 8px 10px;
  text-align: left;
  vertical-align: top;
  white-space: normal;
  word-break: break-word;
  overflow-wrap: break-word;
  min-width: 60px;
}

.message-content :deep(th) {
  background: #f3f4f6;
  font-weight: 600;
  color: #1f2937;
}

.message-content :deep(tbody tr:nth-child(even)) {
  background: #f9fafb;
}

.message-content :deep(tbody tr:hover) {
  background: #f3f4f6;
}

.message-content :deep(pre) {
  background: #1f2937;
  color: #f9fafb;
  padding: 12px;
  border-radius: 8px;
  overflow-x: auto;
  margin: 10px 0;
  font-size: 12px;
}

.message-content :deep(code) {
  background: rgba(0, 0, 0, 0.08);
  padding: 2px 4px;
  border-radius: 4px;
  font-family: Consolas, Monaco, monospace;
  font-size: 12px;
}

.message-content :deep(pre code) {
  background: transparent;
  padding: 0;
}

.message-content :deep(blockquote) {
  border-left: 4px solid #0ea5e9;
  padding-left: 12px;
  margin: 8px 0;
  color: #666;
  font-style: italic;
}

.message-content :deep(strong) {
  font-weight: 600;
  color: #333;
}

.message-content :deep(em) {
  font-style: italic;
}

.message-content :deep(a) {
  color: #0ea5e9;
  text-decoration: none;
}

.message-content :deep(a:hover) {
  text-decoration: underline;
}

/* 消息气泡改进 */
.message-bubble {
  max-width: 65%;
  padding: 12px 14px;
  border-radius: 12px;
  word-wrap: break-word;
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 0;
}

.message-bubble.user {
  background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
  color: white;
  border-radius: 12px 4px 12px 12px;
}

.message-bubble.assistant {
  background: #f5f5f5;
  color: #333;
  border-radius: 4px 12px 12px 12px;
  border: 1px solid #e0e0e0;
}

.message-image {
  max-width: 100%;
  max-height: 350px;
  border-radius: 8px;
  display: block;
  object-fit: contain;
  background: white;
  padding: 4px;
  border: 1px solid #ddd;
}

.message-bubble.user .message-image {
  border: 1px solid rgba(255, 255, 255, 0.3);
  padding: 2px;
}

/* 消息行改进 */
.message-row {
  display: flex;
  gap: 10px;
  margin-bottom: 16px;
  align-items: flex-start;
}

.message-row.user {
  justify-content: flex-end;
}

.message-row.user .avatar {
  order: 2;
}

.message-row.user .message-bubble {
  order: 1;
}

.avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: #4CAF50;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 600;
  flex-shrink: 0;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.message-row.assistant .avatar {
  background: #2196F3;
}

/* 历史列表样式 */
.history-panel {
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid #e0e0e0;
  max-height: calc(100vh - 400px);
  overflow-y: auto;
}

.history-item {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  width: 100%;
  padding: 10px 12px;
  margin-bottom: 8px;
  background: #f9f9f9;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  cursor: pointer;
  text-align: left;
  transition: all 0.2s ease;
}

.history-item:hover {
  background: #f0f0f0;
  border-color: #4CAF50;
}

.history-item.active {
  background: #e8f5e9;
  border-color: #4CAF50;
  font-weight: 600;
}

.history-content {
  flex: 1;
  min-width: 0;
}

.history-title {
  font-size: 13px;
  font-weight: 500;
  color: #333;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-bottom: 4px;
}

.history-meta {
  font-size: 11px;
  color: #999;
}

.history-delete-btn {
  width: 24px;
  height: 24px;
  border: none;
  border-radius: 50%;
  background: transparent;
  color: #999;
  cursor: pointer;
  font-size: 18px;
  line-height: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.history-delete-btn:hover {
  background: #fee2e2;
  color: #dc2626;
}

@media (max-width: 900px) {
  .app-shell,
  .app-shell.with-right-sidebar {
    display: block;
    min-height: 100vh;
    min-height: 100dvh;
    height: auto;
    width: 100vw;
    overflow-x: hidden;
    overflow-y: auto;
    -webkit-overflow-scrolling: touch;
  }

  .left-sidebar {
    display: none;
  }

  .brand-panel {
    margin-bottom: 10px;
  }

  .brand-mark {
    width: 36px;
    height: 36px;
  }

  .brand-panel h1 {
    font-size: 16px;
  }

  .brand-panel p {
    font-size: 11px;
  }

  .toolbar {
    flex-direction: row;
    align-items: center;
    flex-wrap: wrap;
  }

  .toolbar .select,
  .toolbar .btn {
    flex: 1;
    min-width: 0;
  }

  .toolbar .rag-upload-status {
    flex: 1 0 100%;
  }

  .toolbar .session-row {
    flex: 1 0 100%;
  }

  .rag-control-panel {
    margin-top: 8px;
    padding: 8px 0;
  }

  .history-panel {
    display: none;
  }

  .right-sidebar {
    display: none;
  }

  .main-panel {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    min-height: 100dvh;
    height: auto;
    width: 100%;
    overflow: visible;
  }

  .welcome-panel {
    padding: 20px;
  }

  .hero-badge {
    font-size: 32px;
  }

  .messages-panel {
    flex: 1 0 auto;
    min-height: calc(100dvh - 84px);
    height: auto;
    max-height: none;
    overflow: visible;
    overscroll-behavior-y: auto;
    touch-action: auto;
    padding: 12px;
    padding-bottom: calc(128px + env(safe-area-inset-bottom));
    gap: 8px;
  }

  .message-row {
    gap: 8px;
    margin-bottom: 12px;
  }

  .avatar {
    width: 30px;
    height: 30px;
    font-size: 11px;
  }

  .message-bubble {
    max-width: calc(100% - 42px);
    padding: 10px 12px;
    border-radius: 10px;
    overflow: visible;
    touch-action: auto;
  }

  .message-image {
    max-height: 260px;
  }

  .composer-panel {
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 1000;
    width: 100%;
    padding: 10px 12px;
    padding-bottom: calc(10px + env(safe-area-inset-bottom));
    pointer-events: auto;
    box-shadow: 0 -6px 16px rgba(0, 0, 0, 0.08);
  }

  .composer-controls {
    display: none;
  }

  .control-sliders {
    display: none;
  }

  .composer-input-area {
    display: flex;
    width: 100%;
    align-items: flex-end;
    gap: 8px;
    pointer-events: auto;
  }

  .composer-input-area button,
  .composer-input-area textarea {
    position: relative;
    z-index: 11;
    pointer-events: auto;
    touch-action: manipulation;
  }

  .composer-input {
    width: auto;
    min-width: 0;
    min-height: 48px;
    max-height: 140px;
    padding: 12px;
    line-height: 1.4;
    font-size: 16px;
    border-radius: 8px;
    overflow-y: auto;
    -webkit-appearance: none;
  }

  .send-btn {
    width: 76px !important;
    max-width: 76px;
    min-width: 76px;
    min-height: 48px;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 15px;
    flex: 0 0 76px;
  }

  .btn-icon {
    width: 48px;
    min-width: 48px;
    height: 48px;
    min-height: 48px;
    flex: 0 0 48px;
    padding: 0;
  }

  .welcome-panel {
    padding-bottom: calc(128px + env(safe-area-inset-bottom));
  }

  .toast-stack {
    left: 12px;
    right: 12px;
    bottom: 12px;
  }

  .toast {
    width: 100%;
    box-sizing: border-box;
  }

  .cropper-modal {
    width: 96vw;
    max-height: 88vh;
  }

  .cropper-container {
    padding: 10px;
  }
}

@media (max-width: 640px) {
  .login-shell {
    padding: 14px;
  }

  .login-panel {
    padding: 20px;
  }

  .left-sidebar {
    padding: 8px 10px;
  }

  .brand-panel {
    align-items: center;
  }

  .toolbar {
    gap: 6px;
  }

  .toolbar .select {
    flex: 1 0 100%;
  }

  .toolbar .btn {
    flex: 1 1 calc(50% - 6px);
    min-height: 36px;
    padding: 8px 10px;
    font-size: 13px;
    white-space: normal;
  }

  .messages-panel {
    padding: 10px;
  }

  .message-bubble {
    max-width: calc(100% - 38px);
  }

  .pending-image-tag {
    width: 100%;
    align-items: flex-start;
  }

  .pending-image-title {
    max-width: calc(100vw - 130px);
  }

  .composer-input-area {
    gap: 6px;
  }

  .btn-icon {
    width: 48px;
    min-width: 48px;
    height: 48px;
    min-height: 48px;
    padding: 0;
  }

  .send-btn {
    width: 72px !important;
    min-width: 72px;
    max-width: 72px;
    min-height: 48px;
    padding: 10px 12px;
    flex-basis: 72px;
  }
}

/* Inline references for assistant answers */
.message-evidence-panel {
  margin-top: 10px;
  padding: 10px;
  border: 1px solid #dfe7df;
  border-radius: 10px;
  background: #ffffff;
}

.message-evidence-title {
  margin-bottom: 8px;
  color: #244428;
  font-size: 12px;
  font-weight: 700;
}

.message-evidence-item {
  padding: 8px 0;
  border-top: 1px solid #edf2ed;
  font-size: 12px;
}

.message-evidence-item:first-of-type {
  border-top: none;
  padding-top: 0;
}

.message-evidence-meta {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 4px;
  color: #2e7d32;
  font-weight: 600;
}

.message-evidence-text {
  color: #374151;
  line-height: 1.5;
  overflow-wrap: anywhere;
  word-break: break-word;
}

.message-evidence-source {
  margin-top: 4px;
  color: #6b7280;
  font-size: 11px;
  overflow-wrap: anywhere;
  word-break: break-all;
}

/* 手机横屏：显示左右两侧栏；手机竖屏仍只保留对话主体 */
@media (max-width: 900px) and (orientation: landscape) {
  .app-shell,
  .app-shell.with-right-sidebar {
    display: grid !important;
    grid-template-columns: minmax(150px, 20vw) minmax(0, 1fr) minmax(210px, 26vw) !important;
    grid-template-rows: 1fr !important;
    height: 100vh !important;
    height: 100dvh !important;
    min-height: 100vh !important;
    min-height: 100dvh !important;
    width: 100vw !important;
    overflow: hidden !important;
  }

  .left-sidebar {
    display: block !important;
    grid-column: 1 !important;
    grid-row: 1 !important;
    padding: 10px !important;
    overflow-y: auto !important;
  }

  .right-sidebar,
  .right-sidebar.collapsed {
    display: block !important;
    grid-column: 3 !important;
    grid-row: 1 !important;
    width: auto !important;
    min-width: 0 !important;
    padding: 10px !important;
    overflow-y: auto !important;
    position: relative !important;
  }

  .right-sidebar.collapsed .reference-card {
    display: block !important;
  }

  .sidebar-toggle,
  .resize-handle {
    display: none !important;
  }

  .main-panel {
    grid-column: 2 !important;
    grid-row: 1 !important;
    display: flex !important;
    height: 100dvh !important;
    min-height: 0 !important;
    overflow: hidden !important;
  }

  .messages-panel {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    height: auto !important;
    overflow-y: auto !important;
    padding: 10px !important;
    padding-bottom: 10px !important;
  }

  .composer-panel {
    position: static !important;
    width: auto !important;
    padding: 8px !important;
    box-shadow: none !important;
  }

  .composer-controls,
  .control-sliders {
    display: none !important;
  }

  .composer-input {
    min-height: 44px !important;
    max-height: 88px !important;
    font-size: 16px !important;
  }

  .send-btn {
    width: 68px !important;
    min-width: 68px !important;
    max-width: 68px !important;
    min-height: 44px !important;
    padding: 8px 10px !important;
  }

  .btn-icon {
    width: 44px !important;
    min-width: 44px !important;
    height: 44px !important;
    min-height: 44px !important;
  }

  .message-bubble {
    max-width: calc(100% - 38px) !important;
  }

  .brand-panel {
    margin-bottom: 8px !important;
  }

  .history-panel {
    display: block !important;
    max-height: calc(100dvh - 220px) !important;
    overflow-y: auto !important;
  }

  .rag-control-panel {
    display: none !important;
  }

  .reference-header {
    position: static !important;
    margin: 0 0 10px !important;
    padding: 0 0 10px !important;
  }

  .evidence-item {
    padding: 8px !important;
    font-size: 12px !important;
  }
}

@media (max-width: 900px) and (orientation: portrait) {
  .left-sidebar,
  .right-sidebar {
    display: none !important;
  }

  .app-shell,
  .app-shell.with-right-sidebar {
    display: block !important;
  }

  .main-panel {
    width: 100% !important;
  }
}

</style>
