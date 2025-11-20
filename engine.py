# 这个里面主要使我们的训练流程需要的一些东西
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, recall_score, precision_score, matthews_corrcoef, average_precision_score
import time
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from utils import EarlyStopping
from abc import ABC, abstractmethod

class AbstractTrainer(ABC):
    @abstractmethod
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        pass

    @abstractmethod
    def evaluate_epoch(self, test_loader):
        """评估一个epoch"""
        pass

    @abstractmethod
    def train(self, train_loader, test_loader):
        """主训练循环 - 从头开始"""
        pass

    @abstractmethod
    def train_from_epoch(self, train_loader, test_loader, start_epoch=0):
        """从指定epoch开始训练"""
        pass




class BaseTrainer(AbstractTrainer):
    def __init__(self, model, cfg):
        """
        这是我们核心的训练类
        
        :param self: 说明
        :param model: 说明
        :param cfg: 说明
        :param model_path: 说明
        """
        self.model = model
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.lr = cfg.training_loop.learning_rate
        self.epochs = cfg.training_loop.epochs
        self.weight_decay = cfg.training_loop.get('weight_decay', 1e-5)
        self.task_type = cfg.training_loop.get('task_type', 'classification')
        self.best_model_path = cfg.best_model_path
        # 优化器和损失函数
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        if self.task_type == 'regression':
            self.criterion = nn.MSELoss()  # 回归用MSE
        elif self.task_type == 'Multiclass':
            self.criterion = nn.CrossEntropyLoss()  # 多分类用CrossEntropy
        else:
            self.criterion = nn.BCEWithLogitsLoss()  # 分类用BCE
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #         self.optimizer, T_max=self.epochs, eta_min=1e-5
        #     )
        # 早停
        self.early_stopping = EarlyStopping(patience=20, delta=1e-6)
        
        # 日志记录
        self.setup_logging(cfg)
        self.training_history = []
        
    def setup_logging(self,cfg):
        """设置日志记录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(cfg.log_path)
        log_dir.mkdir(exist_ok=True)
        
        self.log_file = log_dir / f"training_{timestamp}.log"
        self.history_file = log_dir / f"history_{timestamp}.csv"
        
        # 记录配置信息
        config_str = self.format_config()
        self.log_and_print(config_str)
        self.log_and_print("="*80)
        
    def format_config(self):
        config_info = f"""
                        训练配置信息:
                        模型类型: {self.model.name}
                        学习率: {self.lr}
                        训练轮数: {self.epochs}
                        权重衰减: {self.weight_decay}
                        设备: {self.device}
                        开始时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                        """
        return config_info
    
    def log_and_print(self, message):
        """同时打印和记录日志"""
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{message}\n")
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        all_preds = []
        all_labels = []

        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            for key in batch:
                batch[key] = batch[key].to(self.device)
            # 前向传播
            outputs = self.model(batch)
            logits = outputs['logits']
            # 计算损失
            loss = self.criterion(logits, batch['label'])
            total_loss_batch = loss
            # 反向传播
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            # 累积损失
            total_loss += total_loss_batch.item()
            num_batches += 1
            # 收集预测结果用于计算指标
            with torch.no_grad():
                if self.task_type == 'regression':
                    probs = logits.cpu().numpy()
                elif self.task_type == 'Multiclass':
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                else:
                    probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(batch['label'].cpu().numpy())

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}'
            })
        
        # 计算平均损失
        avg_loss = total_loss / num_batches

        
        # 计算训练指标
        train_metrics = self.calculate_metrics(
            all_preds, all_labels
        )
        
        return {
            'loss': avg_loss,
            'metrics': train_metrics
        }
    
    def evaluate_epoch(self, test_loader):
        """评估一个epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        all_preds = []
        all_labels = []

        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Evaluating")
            for batch in pbar:
                # 移动数据到设备
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(self.device)
                
                # 前向传播
                outputs = self.model(batch)
                logits = outputs['logits']
        
                # 计算损失
                loss = self.criterion(logits, batch['label'])
                total_loss_batch = loss 
                # 累积损失
                total_loss += total_loss_batch.item()
                num_batches += 1
                
                if self.task_type == 'regression':
                    probs = logits.cpu().numpy()
                elif self.task_type == 'Multiclass':
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                else:
                    probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(batch['label'].cpu().numpy())
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{total_loss_batch.item():.4f}',
                })
        
        # 计算平均损失
        avg_loss = total_loss / num_batches

        
        # 计算评估指标
        eval_metrics = self.calculate_metrics(
            all_preds, all_labels
        )
        
        return {
            'loss': avg_loss,
            'metrics': eval_metrics
        }

    def calculate_metrics(self, preds, labels):
        """计算各种指标"""
        if self.task_type == 'Multiclass':
            return self.calculate_metrics_mutil(preds, labels)
        if self.task_type == 'regression':
            return self.calculate_metrics_regression(preds, labels)
        preds = np.array(preds)
        labels = np.array(labels)
        pred_binary = (preds > 0.5).astype(int)
        metrics = {}
        # 司机任务指标
        metrics['acc'] = accuracy_score(labels, pred_binary)
        metrics['f1'] = f1_score(labels, pred_binary, average='binary', zero_division=0)
        metrics['recall'] = recall_score(labels, pred_binary, average='binary', zero_division=0)
        metrics['precision'] = precision_score(labels, pred_binary, average='binary', zero_division=0)
        metrics['mcc'] = matthews_corrcoef(labels, pred_binary)
        try:
            metrics['auc'] = roc_auc_score(labels, preds)
        except:
            metrics['auc'] = 0.5
        try:
            metrics['auc_pr'] = average_precision_score(labels, preds)
        except:
            metrics['auc_pr'] = 0.5
        return metrics

    def calculate_metrics_mutil(self, preds, labels):
        """计算各种指标"""
        preds = np.array(preds)
        labels = np.array(labels)
        pred_binary = np.argmax(preds, axis=1)
        metrics = {}
        metrics['acc'] = accuracy_score(labels, pred_binary)
        metrics['f1'] = f1_score(labels, pred_binary, average='weighted', zero_division=0)
        try:
            metrics['auc'] = roc_auc_score(labels, preds, multi_class='ovr', average='weighted')
        except:
            metrics['auc'] = 0.5

        
        return metrics
    def calculate_metrics_regression(self, preds, labels):
        """计算回归任务的各种指标"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        preds = np.array(preds)
        labels = np.array(labels)
        metrics = {}
        metrics['mse'] = mean_squared_error(labels, preds)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(labels, preds)
        try:
            metrics['r2'] = r2_score(labels, preds)
        except:
            metrics['r2'] = 0.0
        return metrics
    
    def save_history(self, epoch, train_results, eval_results):
        """保存训练历史"""
        history_entry = {
            'epoch': epoch + 1,
            'train_loss': train_results['loss'],
            'val_loss': eval_results['loss'],
            **{f'train_{k}': v for k, v in train_results['metrics'].items()},
            **{f'val_{k}': v for k, v in eval_results['metrics'].items()},
            'lr': self.optimizer.param_groups[0]['lr'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.training_history.append(history_entry)
        # 保存到CSV
        df = pd.DataFrame(self.training_history)
        df.to_csv(self.history_file, index=False)
    
    def print_epoch_results(self, epoch, train_results, eval_results, epoch_time):
        """打印epoch结果 - 根据任务类型动态调整"""
        
        # 基础信息
        base_message = f"""
        Epoch {epoch + 1}/{self.epochs} - 耗时: {epoch_time:.2f}s
        训练 - 总损失: {train_results['loss']:.4f} 
        验证 - 总损失: {eval_results['loss']:.4f} 
        """
        
        # 根据任务类型打印不同指标
        if self.task_type == 'regression':
            metrics_message = f"""
        训练指标:
        任务 - MSE: {train_results['metrics']['mse']:.4f} | RMSE: {train_results['metrics']['rmse']:.4f} | 
                MAE: {train_results['metrics']['mae']:.4f} | R²: {train_results['metrics']['r2']:.4f}

        验证指标:
        任务 - MSE: {eval_results['metrics']['mse']:.4f} | RMSE: {eval_results['metrics']['rmse']:.4f} | 
                MAE: {eval_results['metrics']['mae']:.4f} | R²: {eval_results['metrics']['r2']:.4f}
            """
        
        elif self.task_type == 'Multiclass':
            metrics_message = f"""
        训练指标:
        任务 - ACC: {train_results['metrics']['acc']:.4f} | F1: {train_results['metrics']['f1']:.4f} | 
                AUC: {train_results['metrics']['auc']:.4f}
        验证指标:
        任务 - ACC: {eval_results['metrics']['acc']:.4f} | F1: {eval_results['metrics']['f1']:.4f} | 
                AUC: {eval_results['metrics']['auc']:.4f}

            """
        else:  # classification (二分类)
            metrics_message = f"""
        训练指标:
        任务 - ACC: {train_results['metrics']['acc']:.4f} | F1: {train_results['metrics']['f1']:.4f} | 
                Recall: {train_results['metrics']['recall']:.4f} | Precision: {train_results['metrics']['precision']:.4f} | 
                MCC: {train_results['metrics']['mcc']:.4f} | AUC: {train_results['metrics']['auc']:.4f} | 
                AUC-PR: {train_results['metrics']['auc_pr']:.4f}
        验证指标:
        任务 - ACC: {eval_results['metrics']['acc']:.4f} | F1: {eval_results['metrics']['f1']:.4f} | 
                Recall: {eval_results['metrics']['recall']:.4f} | Precision: {eval_results['metrics']['precision']:.4f} | 
                MCC: {eval_results['metrics']['mcc']:.4f} | AUC: {eval_results['metrics']['auc']:.4f} | 
                AUC-PR: {eval_results['metrics']['auc_pr']:.4f}
            """
        
        # 组合完整消息
        message = base_message + metrics_message + f"""
        学习率: {self.optimizer.param_groups[0]['lr']:.6f}
        {'='*80}
        """
        
        self.log_and_print(message)
    
    def train_from_epoch(self, train_loader, test_loader, start_epoch=0):
        """从指定epoch开始训练"""
        self.log_and_print(f"从第 {start_epoch + 1} 轮继续训练，总共 {self.epochs} 个epoch")
        self.log_and_print(f"训练集批次数: {len(train_loader)}")
        self.log_and_print(f"验证集批次数: {len(test_loader)}")
        
        best_val_loss = float('inf')
        best_model_path = self.best_model_path 
        # 如果有历史记录，尝试获取最佳验证损失
        if self.training_history:
            best_val_loss = min([h['val_loss'] for h in self.training_history])
            self.log_and_print(f"当前最佳验证损失: {best_val_loss:.4f}")
        for epoch in range(start_epoch, self.epochs):
            epoch_start_time = time.time()
            
            # 训练
            train_results = self.train_epoch(train_loader)
            
            # 评估
            eval_results = self.evaluate_epoch(test_loader)
            
            # 学习率调度
            self.scheduler.step(eval_results['loss'])
            
            # 记录历史
            epoch_time = time.time() - epoch_start_time
            self.save_history(epoch, train_results, eval_results)
            self.print_epoch_results(epoch, train_results, eval_results, epoch_time)
            
            # 保存最佳模型
            if eval_results['loss'] < best_val_loss:
                best_val_loss = eval_results['loss']
                Path(self.best_model_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),  # 也保存调度器状态
                    'val_loss': best_val_loss,
                    'training_history': self.training_history,  # 保存训练历史
                }, best_model_path)
                self.log_and_print(f"保存最佳模型到: {best_model_path}")
            
            # 早停检查
            if self.early_stopping.step(eval_results['loss']):
                self.log_and_print(f"早停触发，在第 {epoch + 1} 轮停止训练")
                break
        
        self.log_and_print("训练完成！")
        self.log_and_print(f"日志保存在: {self.log_file}")
        self.log_and_print(f"训练历史保存在: {self.history_file}")
        self.log_and_print(f"最佳模型保存在: {best_model_path}")
    def train(self, train_loader, test_loader):
        """主训练循环 - 从头开始"""
        return self.train_from_epoch(train_loader, test_loader, start_epoch=0)