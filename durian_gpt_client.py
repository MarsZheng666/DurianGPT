#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
榴莲GPT - Python 客户端
用于与推理服务交互的命令行工具
"""

import requests
import json
import sys
from typing import List, Dict, Optional
from datetime import datetime

class DurianGPTClient:
    """榴莲GPT 客户端"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.messages: List[Dict] = []
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
        })
    
    def health_check(self) -> Dict:
        """检查服务健康状态"""
        try:
            response = self.session.get(f"{self.api_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "unhealthy"}
    
    def infer(self, 
              messages: List[Dict],
              max_new_tokens: int = 512,
              temperature: float = 0.3,
              top_p: float = 0.85,
              top_k: int = 40,
              repetition_penalty: float = 1.08) -> Dict:
        """基础推理"""
        try:
            payload = {
                "messages": messages,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "stream": False
            }
            
            response = self.session.post(
                f"{self.api_url}/infer",
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def chat(self,
             messages: List[Dict],
             max_new_tokens: int = 512,
             temperature: float = 0.3,
             top_p: float = 0.85,
             top_k: int = 40,
             repetition_penalty: float = 1.08) -> Dict:
        """聊天（带 RAG）"""
        try:
            payload = {
                "messages": messages,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "stream": False
            }
            
            response = self.session.post(
                f"{self.api_url}/chat",
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def add_message(self, role: str, content: str):
        """添加消息到历史"""
        self.messages.append({
            "role": role,
            "content": content
        })
    
    def clear_history(self):
        """清空对话历史"""
        self.messages = []
    
    def interactive_chat(self):
        """交互式聊天"""
        print("=" * 60)
        print("榴莲GPT - 交互式聊天")
        print("=" * 60)
        
        # 检查服务
        health = self.health_check()
        if "error" in health:
            print(f"❌ 服务连接失败: {health['error']}")
            return
        
        print(f"✓ 服务状态: {health.get('status', 'unknown')}")
        print(f"✓ 模型: {health.get('model_path', 'unknown')}")
        print(f"✓ RAG 启用: {health.get('rag_enabled', False)}")
        print(f"✓ RAG 数据块: {health.get('rag_chunks', 0)}")
        print()
        print("输入 'quit' 退出，'clear' 清空历史，'help' 查看帮助")
        print("=" * 60)
        print()
        
        while True:
            try:
                user_input = input("你: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print("再见！")
                    break
                
                if user_input.lower() == 'clear':
                    self.clear_history()
                    print("✓ 对话历史已清空")
                    continue
                
                if user_input.lower() == 'help':
                    print("""
命令列表:
  quit   - 退出程序
  clear  - 清空对话历史
  help   - 显示此帮助信息
  
参数调整:
  /temp <值>  - 设置温度 (0.0-2.0)
  /top_p <值> - 设置 top_p (0.0-1.0)
  /top_k <值> - 设置 top_k (0-100)
  /tokens <值> - 设置最大生成 token 数
                    """)
                    continue
                
                # 添加用户消息
                self.add_message("user", user_input)
                
                # 调用 API
                print("\n榴莲GPT: ", end="", flush=True)
                result = self.chat(self.messages)
                
                if "error" in result:
                    print(f"❌ 错误: {result['error']}")
                    self.messages.pop()  # 移除失败的消息
                    continue
                
                response = result.get("response", "")
                print(response)
                
                # 添加助手消息
                self.add_message("assistant", response)
                
                # 显示证据
                evidence = result.get("evidence", [])
                if evidence:
                    quality = result.get("evidence_quality", "none")
                    print(f"\n📚 参考资料 (质量: {quality}):")
                    for i, ev in enumerate(evidence[:3], 1):
                        score = ev.get("score", 0)
                        text = ev.get("text", "")[:100]
                        print(f"  [{i}] (相似度: {score:.2f}) {text}...")
                
                print()
                
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
                continue
    
    def batch_infer(self, queries: List[str]) -> List[Dict]:
        """批量推理"""
        results = []
        for i, query in enumerate(queries, 1):
            print(f"处理 {i}/{len(queries)}: {query[:50]}...")
            
            messages = [{"role": "user", "content": query}]
            result = self.chat(messages)
            results.append({
                "query": query,
                "response": result.get("response", ""),
                "evidence_quality": result.get("evidence_quality", "none"),
                "tokens": result.get("tokens_generated", 0)
            })
        
        return results
    
    def save_conversation(self, filepath: str):
        """保存对话"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "messages": self.messages
            }, f, ensure_ascii=False, indent=2)
        print(f"✓ 对话已保存到: {filepath}")
    
    def load_conversation(self, filepath: str):
        """加载对话"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.messages = data.get("messages", [])
        print(f"✓ 对话已加载: {len(self.messages)} 条消息")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="榴莲GPT 客户端")
    parser.add_argument("--url", default="http://localhost:8000", help="API 服务地址")
    parser.add_argument("--mode", choices=["chat", "infer", "health"], default="chat", help="运行模式")
    parser.add_argument("--query", help="查询文本")
    parser.add_argument("--file", help="输入文件（每行一个查询）")
    parser.add_argument("--output", help="输出文件")
    
    args = parser.parse_args()
    
    client = DurianGPTClient(args.url)
    
    if args.mode == "health":
        health = client.health_check()
        print(json.dumps(health, indent=2, ensure_ascii=False))
    
    elif args.mode == "infer":
        if not args.query:
            print("❌ 请提供 --query 参数")
            sys.exit(1)
        
        messages = [{"role": "user", "content": args.query}]
        result = client.infer(messages)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif args.mode == "chat":
        if args.query:
            # 单次查询
            messages = [{"role": "user", "content": args.query}]
            result = client.chat(messages)
            print(result.get("response", ""))
        elif args.file:
            # 批量推理
            with open(args.file, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip()]
            
            results = client.batch_infer(queries)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"✓ 结果已保存到: {args.output}")
            else:
                print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            # 交互式聊天
            client.interactive_chat()


if __name__ == "__main__":
    main()
