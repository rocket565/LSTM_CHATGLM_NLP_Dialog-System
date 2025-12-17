"""
LSTM + ChatGLM-6B 混合问答系统
- LSTM: 用于理解问题并从知识库检索相关上下文
- ChatGLM: 基于检索的上下文生成答案
"""

import torch
import torch.nn as nn
import json
import jieba
from typing import List, Dict, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModel
from collections import Counter


class LSTMRetriever(nn.Module):
    """LSTM检索模型：根据问题检索相关知识库片段"""

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super(LSTMRetriever, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2, num_heads=4, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, question_ids, question_mask=None):
        embedded = self.embedding(question_ids)  # [batch_size, seq_len, embedding_dim]
        lstm_out, (h_n, c_n) = self.lstm(
            embedded
        )  # [batch_size, seq_len, hidden_dim*2]

        # Self-attention
        if question_mask is not None:
            attn_mask = question_mask.unsqueeze(1).repeat(1, question_ids.size(1), 1)
            attn_out, _ = self.attention(
                lstm_out, lstm_out, lstm_out, key_padding_mask=~question_mask.bool()
            )
        else:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # 池化：取平均
        if question_mask is not None:
            mask_expanded = question_mask.unsqueeze(-1).expand(attn_out.size())
            sum_out = torch.sum(attn_out * mask_expanded, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1)
            pooled = sum_out / (sum_mask + 1e-9)
        else:
            pooled = torch.mean(attn_out, dim=1)

        # 全连接层
        output = self.fc(pooled)  # [batch_size, hidden_dim]
        return output

    def encode_context(self, context_ids, context_mask=None):
        """编码知识库上下文 - 使用与forward相同的逻辑"""
        embedded = self.embedding(context_ids)
        lstm_out, _ = self.lstm(embedded)

        # 使用attention，与forward方法保持一致
        if context_mask is not None:
            attn_out, _ = self.attention(
                lstm_out, lstm_out, lstm_out, key_padding_mask=~context_mask.bool()
            )
        else:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # 池化：取平均
        if context_mask is not None:
            mask_expanded = context_mask.unsqueeze(-1).expand(attn_out.size())
            sum_out = torch.sum(attn_out * mask_expanded, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1)
            pooled = sum_out / (sum_mask + 1e-9)
        else:
            pooled = torch.mean(attn_out, dim=1)

        output = self.fc(pooled)  # 移除ReLU
        return output


class HybridQASystem:
    """LSTM + ChatGLM 混合问答系统"""

    def __init__(
        self,
        lstm_model_path,
        chatglm_model_path,
        vocab_path,
        knowledge_base_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device

        # 加载词汇表
        print("加载词汇表...")
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.word2idx = self.vocab["word2idx"]
        self.idx2word = {int(k): v for k, v in self.vocab["idx2word"].items()}

        # 加载LSTM检索模型
        print("加载LSTM检索模型...")
        self.lstm_retriever = LSTMRetriever(
            vocab_size=len(self.word2idx),
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
        ).to(device)

        if lstm_model_path:
            checkpoint = torch.load(lstm_model_path, map_location=device)
            if "model_state_dict" in checkpoint:
                self.lstm_retriever.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.lstm_retriever.load_state_dict(checkpoint)
            print("LSTM模型加载成功！")
            print(
                f"DEBUG - Embedding权重样本: {self.lstm_retriever.embedding.weight.data[:2, :5]}"
            )

        self.lstm_retriever.eval()

        self.lstm_retriever.eval()

        # 加载ChatGLM模型
        print("加载ChatGLM-6B模型...")
        self.chatglm_tokenizer = AutoTokenizer.from_pretrained(
            chatglm_model_path, trust_remote_code=True
        )
        self.chatglm_model = (
            AutoModel.from_pretrained(chatglm_model_path, trust_remote_code=True)
            .half()
            .to(device)
        )
        self.chatglm_model.eval()
        print("ChatGLM模型加载成功！")

        # 加载知识库
        print("加载知识库...")
        with open(knowledge_base_path, "r", encoding="utf-8") as f:
            self.knowledge_base = json.load(f)

        # 预编码知识库
        print("预编码知识库...")
        self.context_embeddings = []
        self.contexts = []
        if "knowledge" in self.knowledge_base:
            self.contexts = [
                item["context"] for item in self.knowledge_base["knowledge"]
            ]
        else:
            # 如果是训练数据格式（直接是列表）
            for item in self.knowledge_base:
                if isinstance(item, dict) and "context" in item:
                    if item["context"] not in self.contexts:
                        self.contexts.append(item["context"])

        self._encode_knowledge_base()
        print(f"知识库加载完成，共 {len(self.contexts)} 条记录")

    def _encode_knowledge_base(self):
        """预编码所有知识库上下文"""
        self.context_embeddings = []
        batch_size = 32

        with torch.no_grad():
            for i in range(0, len(self.contexts), batch_size):
                batch_contexts = self.contexts[i : i + batch_size]
                batch_ids, batch_masks = self._tokenize_batch(batch_contexts)
                batch_ids = batch_ids.to(self.device)
                batch_masks = batch_masks.to(self.device)

                embeddings = self.lstm_retriever.encode_context(batch_ids, batch_masks)
                self.context_embeddings.append(embeddings.cpu())

        self.context_embeddings = torch.cat(self.context_embeddings, dim=0)
        print(f"知识库编码完成，shape: {self.context_embeddings.shape}")

    def _tokenize_text(self, text, max_len=512):
        """将文本转换为token ids"""
        words = list(jieba.cut(text))
        ids = [self.word2idx.get(w, self.word2idx.get("<UNK>", 1)) for w in words]

        # 截断或填充
        if len(ids) > max_len:
            ids = ids[:max_len]

        mask = [1] * len(ids)

        # 填充
        padding_len = max_len - len(ids)
        ids.extend([0] * padding_len)
        mask.extend([0] * padding_len)

        return ids, mask

    def _tokenize_batch(self, texts, max_len=512):
        """批量tokenize"""
        all_ids = []
        all_masks = []

        for text in texts:
            ids, mask = self._tokenize_text(text, max_len)
            all_ids.append(ids)
            all_masks.append(mask)

        return torch.tensor(all_ids), torch.tensor(all_masks)

    def retrieve_contexts(
        self, question: str, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        使用LSTM检索最相关的上下文

        Args:
            question: 问题文本
            top_k: 返回top-k个最相关的上下文

        Returns:
            List of (context, score) tuples
        """
        # Tokenize问题
        question_ids, question_mask = self._tokenize_text(question)
        question_ids = torch.tensor([question_ids]).to(self.device)
        question_mask = torch.tensor([question_mask]).to(self.device)

        # 获取问题embedding
        with torch.no_grad():
            question_embedding = self.lstm_retriever(
                question_ids, question_mask
            )  # [1, hidden_dim]

        # 计算相似度
        question_embedding = question_embedding.cpu()
        # question_embedding: [1, hidden_dim]
        # context_embeddings: [num_contexts, hidden_dim]
        # 使用广播计算余弦相似度
        similarities = torch.cosine_similarity(
            question_embedding,  # [1, hidden_dim]
            self.context_embeddings,  # [num_contexts, hidden_dim]
            dim=1,
        )  # [num_contexts]

        # 获取top-k
        top_k_scores, top_k_indices = torch.topk(
            similarities, min(top_k, len(similarities))
        )

        results = []
        for idx, score in zip(top_k_indices.tolist(), top_k_scores.tolist()):
            results.append((self.contexts[idx], score))

        return results

    def generate_answer(
        self, question: str, contexts: List[str], max_length: int = 512
    ) -> str:
        """
        使用ChatGLM生成答案

        Args:
            question: 问题
            contexts: 检索到的相关上下文列表
            max_length: 最大生成长度

        Returns:
            生成的答案
        """
        # 确保question和contexts都是字符串
        question = str(question)
        contexts = [str(ctx) for ctx in contexts]

        # 构建提示词
        context_text = "\n\n".join(
            [f"参考资料{i+1}：{ctx}" for i, ctx in enumerate(contexts)]
        )
        prompt = f"""请根据以下参考资料回答问题。如果参考资料中没有相关信息，请回答"根据提供的资料无法回答该问题"。

{context_text}

问题：{question}

答案："""

        # 确保prompt是字符串
        prompt = str(prompt)

        # 使用ChatGLM生成答案
        with torch.no_grad():
            # 方式1：尝试使用chat方法，禁用cache
            try:
                response, history = self.chatglm_model.chat(
                    self.chatglm_tokenizer,
                    prompt,
                    max_length=2048,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    use_cache=False,  # 禁用cache
                )
            except Exception as e:
                # 方式2：如果失败，使用简单的生成方式
                print(f"Chat方法失败，尝试直接生成: {e}")
                inputs = self.chatglm_tokenizer(prompt, return_tensors="pt").to(
                    self.device
                )
                outputs = self.chatglm_model.generate(
                    **inputs,
                    max_length=2048,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    use_cache=False,
                    pad_token_id=self.chatglm_tokenizer.pad_token_id,
                    eos_token_id=self.chatglm_tokenizer.eos_token_id,
                )
                response = self.chatglm_tokenizer.decode(
                    outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
                )

        return response.strip()

    def answer_question(
        self,
        question: str,
        top_k: int = 3,
        max_length: int = 512,
        return_contexts: bool = False,
    ) -> Dict:
        """
        完整的问答流程

        Args:
            question: 问题
            top_k: 检索top-k个相关上下文
            max_length: 最大生成长度
            return_contexts: 是否返回检索的上下文

        Returns:
            包含答案和其他信息的字典
        """
        # Step 1: LSTM检索相关上下文
        print(f"\n问题: {question}")
        print("正在检索相关上下文...")
        retrieved = self.retrieve_contexts(question, top_k)

        contexts = [ctx for ctx, score in retrieved]
        scores = [score for ctx, score in retrieved]

        print(f"检索到 {len(contexts)} 个相关上下文")
        for i, (ctx, score) in enumerate(retrieved):
            print(f"  上下文{i+1} (相似度: {score:.4f}): {ctx[:100]}...")

        # Step 2: ChatGLM生成答案
        print("正在生成答案...")
        answer = self.generate_answer(question, contexts, max_length)

        result = {"question": question, "answer": answer, "num_contexts": len(contexts)}

        if return_contexts:
            result["contexts"] = contexts
            result["scores"] = scores

        return result


def main():
    """主函数 - 支持命令行运行"""
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(
        description="LSTM + ChatGLM 混合问答系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  1. 交互式问答:
     python lstm_chatglm_hybrid.py
  
  2. 单个问题:
     python lstm_chatglm_hybrid.py --question "你的问题？"
  
  3. 批量处理:
     python lstm_chatglm_hybrid.py --batch_file data/test.json --output answers.json
        """,
    )

    parser.add_argument(
        "--lstm_model",
        type=str,
        default="lstm_retriever_model/best_model.pt",
        help="LSTM模型路径 (默认: lstm_retriever_model/best_model.pt)",
    )
    parser.add_argument(
        "--chatglm_model",
        type=str,
        default="zai-org/chatglm-6b",
        help="ChatGLM模型路径 (默认: zai-org/chatglm-6b)",
    )
    parser.add_argument(
        "--vocab", type=str, default="vocab.json", help="词汇表路径 (默认: vocab.json)"
    )
    parser.add_argument(
        "--knowledge_base",
        type=str,
        default="data/knowledge_base.json",
        help="知识库路径 (默认: data/knowledge_base.json)",
    )
    parser.add_argument(
        "--top_k", type=int, default=3, help="检索top-k个上下文 (默认: 3)"
    )
    parser.add_argument(
        "--question", type=str, default=None, help="要回答的问题（单个问题模式）"
    )
    parser.add_argument(
        "--batch_file", type=str, default=None, help="批量处理的问题文件（JSON格式）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="answers.json",
        help="输出文件路径 (默认: answers.json)",
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="最大生成长度 (默认: 512)"
    )
    parser.add_argument("--no_context", action="store_true", help="不显示检索的上下文")

    args = parser.parse_args()

    # 打印欢迎信息
    print("=" * 70)
    print("LSTM + ChatGLM-6B 混合问答系统".center(70))
    print("=" * 70)

    # 检查必要文件
    print("\n检查必要文件...")
    required_files = {
        args.vocab: "词汇表",
        args.knowledge_base: "知识库",
        args.chatglm_model: "ChatGLM模型",
    }

    missing_files = []
    for file_path, desc in required_files.items():
        if os.path.exists(file_path):
            print(f"  {desc}: {file_path}")
        else:
            print(f"  {desc}: {file_path} (缺失)")
            missing_files.append(file_path)

    if missing_files:
        print(f"\n错误: 缺少必要文件")
        print("请确保以下文件存在:")
        for f in missing_files:
            print(f"  - {f}")
        if args.vocab in missing_files:
            print("\n提示: 请先运行训练脚本生成词汇表:")
            print("  python train_hybrid_system.py")
        sys.exit(1)

    # LSTM模型是可选的
    if os.path.exists(args.lstm_model):
        print(f"  LSTM模型: {args.lstm_model}")
    else:
        print(f"  LSTM模型: {args.lstm_model} (未找到，将使用随机初始化)")
        print("    建议先训练模型以获得更好效果: python train_hybrid_system.py")
        user_input = input("\n是否继续？(y/n): ").strip().lower()
        if user_input != "y":
            print("已取消")
            sys.exit(0)
        args.lstm_model = None

    # 初始化混合问答系统
    print("\n" + "=" * 70)
    print("初始化系统...".center(70))
    print("=" * 70)

    try:
        qa_system = HybridQASystem(
            lstm_model_path=args.lstm_model,
            chatglm_model_path=args.chatglm_model,
            vocab_path=args.vocab,
            knowledge_base_path=args.knowledge_base,
        )
    except Exception as e:
        print(f"\n初始化失败: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n系统初始化完成！")
    print("=" * 70)

    # 单个问题
    if args.question:
        print(f"\n单个问题模式")
        print("-" * 70)
        try:
            result = qa_system.answer_question(
                args.question,
                top_k=args.top_k,
                max_length=args.max_length,
                return_contexts=not args.no_context,
            )
            print(f"\n{'='*70}")
            print(f"答案: {result['answer']}")
            print(f"{'='*70}")

            if not args.no_context and "contexts" in result:
                print(f"\n参考上下文 (共{len(result['contexts'])}个):")
                for i, (ctx, score) in enumerate(
                    zip(result["contexts"], result["scores"]), 1
                ):
                    print(f"\n  [{i}] 相似度: {score:.4f}")
                    print(f"      {ctx[:150]}..." if len(ctx) > 150 else f"      {ctx}")
        except Exception as e:
            print(f"\n处理问题时出错: {str(e)}")
            import traceback

            traceback.print_exc()

    # 批量处理
    elif args.batch_file:
        print(f"\n批量处理模式")
        print("-" * 70)

        if not os.path.exists(args.batch_file):
            print(f"错误: 文件不存在 - {args.batch_file}")
            sys.exit(1)

        try:
            print(f"读取问题文件: {args.batch_file}")
            with open(args.batch_file, "r", encoding="utf-8") as f:
                questions_data = json.load(f)

            print(f"共 {len(questions_data)} 个问题")

            results = []
            for i, item in enumerate(questions_data):
                question = item.get("question", "")
                if not question:
                    print(f"  跳过问题 {i+1}: 空问题")
                    continue

                print(f"\n[{i+1}/{len(questions_data)}] 处理: {question[:50]}...")

                try:
                    result = qa_system.answer_question(
                        question,
                        top_k=args.top_k,
                        max_length=args.max_length,
                        return_contexts=False,
                    )
                    result["id"] = item.get("id", f"Q{i+1}")
                    results.append(result)
                    print(f"    完成")
                except Exception as e:
                    print(f"    失败: {str(e)}")
                    results.append(
                        {
                            "id": item.get("id", f"Q{i+1}"),
                            "question": question,
                            "answer": f"错误: {str(e)}",
                            "error": True,
                        }
                    )

            # 保存结果
            print(f"\n保存结果到: {args.output}")
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"\n{'='*70}")
            print(f"批量处理完成！")
            print(f"  - 总问题数: {len(questions_data)}")
            print(
                f"  - 成功处理: {len([r for r in results if not r.get('error', False)])}"
            )
            print(f"  - 失败: {len([r for r in results if r.get('error', False)])}")
            print(f"  - 结果文件: {args.output}")
            print(f"{'='*70}")

        except Exception as e:
            print(f"\n批量处理失败: {str(e)}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    # 交互式问答
    else:
        print(f"\n交互式问答模式")
        print("-" * 70)
        print("提示:")
        print("  - 输入问题后按回车")
        print("  - 输入 'quit', 'exit' 或 'q' 退出")
        print("  - 输入 'help' 查看帮助")
        print("=" * 70)

        while True:
            try:
                question = input("\n请输入问题: ").strip()

                if not question:
                    continue

                if question.lower() in ["quit", "exit", "q"]:
                    print("\n再见！")
                    break

                if question.lower() == "help":
                    print("\n帮助信息:")
                    print("  - 直接输入问题即可获得答案")
                    print("  - 系统会先从知识库检索相关信息")
                    print("  - 然后使用ChatGLM生成答案")
                    print(f"  - 当前检索上下文数: {args.top_k}")
                    continue

                print("\n" + "-" * 70)
                result = qa_system.answer_question(
                    question,
                    top_k=args.top_k,
                    max_length=args.max_length,
                    return_contexts=not args.no_context,
                )

                print(f"\n答案:")
                print(f"  {result['answer']}")

                print("-" * 70)

            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"\n错误: {str(e)}")
                import traceback

                traceback.print_exc()


if __name__ == "__main__":
    main()
