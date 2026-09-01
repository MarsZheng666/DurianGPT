# DurianGPT Vue 前端

用 Vue 3 + Vite 重构的现代化前端版本。

## 功能特性

- 响应式 Vue 3 组件架构
- Vite 快速开发服务器
- 流式对话支持
- RAG 检索与证据展示
- 对话历史管理
- 多语言支持（中文、英文、马来语）
- 实时模型信息显示
-现代化 UI 设计

## 快速开始

### 1. 安装依赖

```bash
cd frontend-vue
npm install
```

### 2. 启动开发服务器

```bash
npm run dev
```

服务器将在 `http://localhost:8080` 启动

### 3. 构建生产版本

```bash
npm run build
```

输出文件在 `dist/` 目录

## 项目结构

```
frontend-vue/
├── index.html           # HTML 入口
├── package.json         # 项目配置
├── vite.config.js       # Vite 配置
├── src/
│   ├── main.js         # Vue 应用入口
│   ├── App.vue         # 主应用组件
│   └── style.css       # 全局样式
└── README.md           # 本文件
```

## 配置

### API 地址

编辑 `src/App.vue` 中的 `API_BASE_URL`：

```javascript
const API_BASE_URL = 'http://localhost:8000'
```

### 后端要求

确保后端服务运行在 `http://localhost:8000`，提供以下接口：

- `GET /health` - 健康检查
- `POST /chat` - 聊天接口（支持流式）
- `GET /conversations` - 获取对话列表
- `POST /conversations` - 创建新对话
- `GET /conversations/{id}` - 获取对话详情
- `POST /conversations/{id}/messages` - 保存消息

## 功能说明

### 聊天界面

- 左侧：对话历史和语言选择
- 中间：聊天消息展示和输入框
- 右侧：模型信息和检索证据

### 参数调整

- **Temperature**: 控制输出随机性（0-1.5）
- **Top P**: 核采样参数（0-1）
- **Max Tokens**: 最大生成长度（64-2048）
- **Stream Mode**: 启用流式输出

### 多语言支持

支持三种语言：
- 中文 (zh)
- English (en)
- Bahasa Melayu (ms)

语言选择会自动保存到本地存储。

## 开发

### 依赖

- Vue 3.3.4
- Vite 4.3.9
- Axios 1.4.0
- Marked 9.0.0

### 代码风格

使用 Vue 3 Composition API + `<script setup>` 语法。

## 与原版本的区别

| 特性 | 原版本 | Vue 版本 |
|------|--------|---------|
| 框架 | 原生 JS | Vue 3 |
| 构建工具 | 无 | Vite |
| 状态管理 | 全局对象 | Ref/Reactive |
| 组件化 | 无 | 完整组件 |
| 开发体验 | 基础 | 现代化 |
| 性能 | 良好 | 优秀 |

## 故障排除

### 连接失败

确保后端服务正在运行：
```bash
python durian_gpt_inference_api.py
```

### 端口被占用

修改 `vite.config.js` 中的端口：
```javascript
server: {
  port: 8081,  // 改为其他端口
}
```

### 样式不显示

清除浏览器缓存或使用无痕模式重新加载。

## 许可证

MIT
