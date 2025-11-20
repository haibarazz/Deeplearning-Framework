import torch
import os
os.environ["HYDRA_FULL_ERROR"] = "1"  # 这个是为了我们用hydra调试时能看到完整的错误信息
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from engine import BaseTrainer
from Data_pre import load_and_preprocess_data, prepare_data
from Feature_selected import FEATURE


def create_model(cfg: DictConfig):
    """
    动态计算输入维度并实例化模型
    """
    # 基础特征维度：顺序数值特征 + 顺序离散特征 + 其他数值特征（占位）
    basic_features = FEATURE.num_order_feat + FEATURE.cat_order_feat
    input_dim = len(basic_features)
    # 注意这里，就是有些参数是我们无法事先知道的时候，我们就在这里动态计算，然后按照key的方式设置。
    model = hydra.utils.instantiate(
        cfg.model,
        input_dim=input_dim,
        task_type=cfg.training_loop.task_type,
    )
    return model

def load_training(checkpoint_path, cfg, train_loader, test_loader):
    """
    load_training 的 Docstring
    
    :param checkpoint_path: 断点恢复地址，可以在我们的yaml配置文件中指定
    :param cfg: 配置文件
    :param train_loader: 训练数据加载器
    :param test_loader: 测试数据加载器
    """
    print(f"从检查点加载模型: {checkpoint_path}")
    # 检查文件是否存在
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    # 创建模型
    model = create_model(cfg)
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"成功加载模型参数，来自第 {checkpoint['epoch'] + 1} 轮")
    print(f"最佳验证损失: {checkpoint['val_loss']:.4f}")
    # 创建训练器
    trainer = BaseTrainer(model, cfg)
    # 加载优化器参数
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("成功加载优化器参数")
    # 调整学习率调度器的起始点
    start_epoch = checkpoint['epoch'] + 1
    if hasattr(trainer.scheduler, 'last_epoch'):
        trainer.scheduler.last_epoch = start_epoch
    print(f"将从第 {start_epoch + 1} 轮开始继续训练")
    # 继续训练
    trainer.train_from_epoch(train_loader, test_loader, start_epoch=start_epoch)





@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print("开始训练流程...")
    df = load_and_preprocess_data(cfg)
    train_loader, test_loader = prepare_data(df, cfg)
    
    # 断点继续跑
    # checkpoint_path = "checkpoints/best_lstm_model.pth"
    # load_training(checkpoint_path, cfg, train_loader, test_loader)

    model = create_model(cfg)
    trainer = BaseTrainer(model, cfg)
    # 开始训练
    trainer.train(train_loader, test_loader)


if __name__ == "__main__":
    main()
