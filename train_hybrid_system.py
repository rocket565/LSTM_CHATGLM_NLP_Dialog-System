import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import jieba
from collections import Counter
from tqdm import tqdm
import os
import argparse
from lstm_chatglm_hybrid import LSTMRetriever


class QADataset(Dataset):
    """问答数据集"""

    def __init__(self, data_path, vocab, max_len=512):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.vocab = vocab
        self.word2idx = vocab["word2idx"]
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def tokenize(self, text):
        """文本转token ids"""
        words = list(jieba.cut(text))
        ids = [self.word2idx.get(w, self.word2idx.get("<UNK>", 1)) for w in words]

        if len(ids) > self.max_len:
            ids = ids[: self.max_len]

        mask = [1] * len(ids)
        padding_len = self.max_len - len(ids)
        ids.extend([0] * padding_len)
        mask.extend([0] * padding_len)

        return torch.tensor(ids), torch.tensor(mask)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        context = item["context"]

        q_ids, q_mask = self.tokenize(question)
        c_ids, c_mask = self.tokenize(context)

        return {
            "question_ids": q_ids,
            "question_mask": q_mask,
            "context_ids": c_ids,
            "context_mask": c_mask,
            "label": 1.0,  # 正样本
        }


def build_vocab(data_paths, min_freq=2, max_vocab_size=50000):
    """构建词汇表"""
    print("构建词汇表...")
    word_counter = Counter()

    for data_path in data_paths:
        print(f"  处理 {data_path}...")
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 如果是知识库格式
        if isinstance(data, dict) and "knowledge" in data:
            data = data["knowledge"]

        for item in tqdm(data):
            # 问题
            if "question" in item:
                words = jieba.cut(item["question"])
                word_counter.update(words)

            # 上下文
            if "context" in item:
                words = jieba.cut(item["context"])
                word_counter.update(words)

    # 过滤低频词
    words = [w for w, c in word_counter.most_common() if c >= min_freq]
    words = words[: max_vocab_size - 4]  # 保留特殊token的位置

    # 构建词汇表
    word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}

    for i, word in enumerate(words):
        word2idx[word] = i + 4

    idx2word = {v: k for k, v in word2idx.items()}

    vocab = {"word2idx": word2idx, "idx2word": idx2word}

    print(f"词汇表大小: {len(word2idx)}")
    return vocab


class ContrastiveLoss(nn.Module):
    """改进的对比学习损失：使用InfoNCE风格的损失"""

    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, question_emb, context_emb):
        """
        使用in-batch negatives的对比学习损失

        Args:
            question_emb: [batch_size, hidden_dim]
            context_emb: [batch_size, hidden_dim]

        Returns:
            loss: scalar
        """
        batch_size = question_emb.size(0)

        # 归一化embeddings
        question_emb = torch.nn.functional.normalize(question_emb, p=2, dim=1)
        context_emb = torch.nn.functional.normalize(context_emb, p=2, dim=1)

        # 计算所有question-context对的相似度 [batch_size, batch_size]
        similarity_matrix = (
            torch.matmul(question_emb, context_emb.t()) / self.temperature
        )

        # 对角线是正样本，其他都是负样本
        labels = torch.arange(batch_size).to(question_emb.device)

        # 使用交叉熵损失
        loss = torch.nn.functional.cross_entropy(similarity_matrix, labels)

        return loss


def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        question_ids = batch["question_ids"].to(device)
        question_mask = batch["question_mask"].to(device)
        context_ids = batch["context_ids"].to(device)
        context_mask = batch["context_mask"].to(device)

        # 前向传播
        question_emb = model(question_ids, question_mask)
        context_emb = model.encode_context(context_ids, context_mask)

        # 计算损失（使用in-batch negatives）
        loss = criterion(question_emb, context_emb)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            question_ids = batch["question_ids"].to(device)
            question_mask = batch["question_mask"].to(device)
            context_ids = batch["context_ids"].to(device)
            context_mask = batch["context_mask"].to(device)

            question_emb = model(question_ids, question_mask)
            context_emb = model.encode_context(context_ids, context_mask)

            loss = criterion(question_emb, context_emb)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def evaluate_retrieval(model, test_data, knowledge_base, vocab, device, top_k=3):
    """
    评估检索性能
    
    Args:
        model: 训练好的模型
        test_data: 测试数据集
        knowledge_base: 知识库
        vocab: 词汇表
        device: 设备
        top_k_list: 要评估的k值列表
    
    Returns:
        dict: 包含各种评估指标的字典
    """
    model.eval()
    
    # 预编码所有知识库上下文
    print("预编码知识库...")
    contexts = [item["context"] for item in knowledge_base["knowledge"]]
    context_embeddings = []
    
    batch_size = 32
    for i in tqdm(range(0, len(contexts), batch_size), desc="Encoding contexts"):
        batch_contexts = contexts[i:i+batch_size]
        batch_ids = []
        batch_masks = []
        
        for ctx in batch_contexts:
            tokens = list(jieba.cut(ctx))
            token_ids = [vocab.get(token, vocab.get("<UNK>", 1)) for token in tokens]
            
            max_len = 128
            if len(token_ids) > max_len:
                token_ids = token_ids[:max_len]
            
            mask = [1] * len(token_ids) + [0] * (max_len - len(token_ids))
            token_ids = token_ids + [0] * (max_len - len(token_ids))
            
            batch_ids.append(token_ids)
            batch_masks.append(mask)
        
        batch_ids = torch.tensor(batch_ids).to(device)
        batch_masks = torch.tensor(batch_masks).to(device)
        
        with torch.no_grad():
            emb = model.encode_context(batch_ids, batch_masks)
            context_embeddings.append(emb.cpu())
    
    context_embeddings = torch.cat(context_embeddings, dim=0)
    
    # 评估检索性能
    print(f"评估检索性能（测试集大小: {len(test_data)}）...")
    
    correct_retrievals = 0
    total_samples = 0
    
    for item in tqdm(test_data, desc="Evaluating retrieval"):
        question = item["question"]
        correct_context = item["context"]
        
        # 找到正确答案在知识库中的索引
        try:
            correct_idx = contexts.index(correct_context)
        except ValueError:
            # 如果测试集的context不在知识库中，跳过
            continue
        
        # 编码问题
        tokens = list(jieba.cut(question))
        token_ids = [vocab.get(token, vocab.get("<UNK>", 1)) for token in tokens]
        
        max_len = 128
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        
        mask = [1] * len(token_ids) + [0] * (max_len - len(token_ids))
        token_ids = token_ids + [0] * (max_len - len(token_ids))
        
        question_ids = torch.tensor([token_ids]).to(device)
        question_mask = torch.tensor([mask]).to(device)
        
        with torch.no_grad():
            question_emb = model(question_ids, question_mask).cpu()
        
        # 计算相似度
        similarities = torch.cosine_similarity(
            question_emb,
            context_embeddings,
            dim=1
        )
        
        # 获取top-k
        top_k_indices = torch.topk(similarities, k=top_k)[1]
        
        # 检查正确答案是否在top-k中
        if correct_idx in top_k_indices:
            correct_retrievals += 1
        
        total_samples += 1
    
    # 计算Recall@K
    recall = correct_retrievals / total_samples if total_samples > 0 else 0
    
    return {
        f"Recall@{top_k}": recall,
        "Total_Samples": total_samples,
        "Correct_Retrievals": correct_retrievals
    }


def main():
    parser = argparse.ArgumentParser(description="训练LSTM检索模型")
    parser.add_argument(
        "--train_data", type=str, default="data/train.json", help="训练数据路径"
    )
    parser.add_argument(
        "--val_data", type=str, default="data/validation.json", help="验证数据路径"
    )
    parser.add_argument(
        "--knowledge_base",
        type=str,
        default="data/knowledge_base.json",
        help="知识库路径",
    )
    parser.add_argument(
        "--vocab_path", type=str, default="vocab.json", help="词汇表保存路径"
    )
    parser.add_argument(
        "--output_dir", type=str, default="lstm_retriever_model", help="模型保存目录"
    )
    parser.add_argument("--embedding_dim", type=int, default=128, help="词嵌入维度")
    parser.add_argument("--hidden_dim", type=int, default=256, help="LSTM隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=2, help="LSTM层数")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--max_len", type=int, default=512, help="最大序列长度")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 构建或加载词汇表
    if os.path.exists(args.vocab_path):
        print(f"加载已有词汇表: {args.vocab_path}")
        with open(args.vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
    else:
        vocab = build_vocab(
            [args.train_data, args.val_data, args.knowledge_base],
            min_freq=2,
            max_vocab_size=50000,
        )
        with open(args.vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        print(f"词汇表已保存到: {args.vocab_path}")

    # 创建数据集
    print("加载数据集...")
    train_dataset = QADataset(args.train_data, vocab, max_len=args.max_len)
    val_dataset = QADataset(args.val_data, vocab, max_len=args.max_len)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 创建模型
    model = LSTMRetriever(
        vocab_size=len(vocab["word2idx"]),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = ContrastiveLoss(temperature=0.07)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    # 加载测试数据和知识库用于评估
    test_data_path = "data/test.json"
    test_data = None
    knowledge_base_data = None
    if os.path.exists(test_data_path):
        with open(test_data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        with open(args.knowledge_base, "r", encoding="utf-8") as f:
            knowledge_base_data = json.load(f)
        print(f"测试集大小: {len(test_data)}")
    
    # 训练
    best_val_loss = float("inf")
    best_recall = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # 每个epoch评估检索性能
        if test_data is not None:
            print("评估检索性能...")
            metrics = evaluate_retrieval(
                model, 
                test_data[:100],  # 只用100个样本快速评估
                knowledge_base_data, 
                vocab["word2idx"], 
                device, 
                top_k=3
            )
            recall_at_3 = metrics['Recall@3']
            print(f"Recall@3: {recall_at_3:.4f} ({recall_at_3*100:.2f}%)")
            
            if recall_at_3 > best_recall:
                best_recall = recall_at_3

        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
            }
            torch.save(checkpoint, os.path.join(args.output_dir, "best_model.pt"))
            print(f"最佳模型已保存！")

        # 保存检查点
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "args": vars(args),
        }
        torch.save(
            checkpoint, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
        )

    print("\n训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳Recall@3: {best_recall:.4f} ({best_recall*100:.2f}%)")
    
    # 评估检索性能
    print("\n" + "="*70)
    print("评估检索性能...")
    print("="*70)
    
    # 加载最佳模型
    model_save_path = os.path.join(args.output_dir, "best_model.pt")
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # 在完整测试集上评估
    if test_data is not None:
        
        # 在完整测试集上评估
        print("在完整测试集上评估...")
        metrics = evaluate_retrieval(
            model, 
            test_data, 
            knowledge_base_data, 
            vocab["word2idx"], 
            device, 
            top_k=3
        )
        
        print("\n检索性能指标:")
        print(f"  测试样本数: {metrics['Total_Samples']}")
        print(f"  正确检索数: {metrics['Correct_Retrievals']}")
        print(f"  Recall@3: {metrics['Recall@3']:.4f} ({metrics['Recall@3']*100:.2f}%)")
        print(f"\n解释: Recall@3表示正确答案出现在前3个检索结果中的比例")
        
        # 保存评估结果
        import datetime
        eval_result = {
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": metrics,
            "config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "embedding_dim": args.embedding_dim,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
            }
        }
        
        eval_save_path = os.path.join(args.output_dir, "evaluation_results.json")
        with open(eval_save_path, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, indent=2, ensure_ascii=False)
        print(f"\n评估结果已保存到: {eval_save_path}")
    else:
        print(f"\n警告: 测试数据文件 {test_data_path} 不存在，跳过评估")


if __name__ == "__main__":
    main()
