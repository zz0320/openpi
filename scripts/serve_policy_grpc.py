#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenPi gRPC 推理服务入口脚本

使用方法:
    # 启动服务器 (等待 Client 配置)
    python scripts/serve_policy_grpc.py --port 50051
    
    # 预加载模型
    python scripts/serve_policy_grpc.py --port 50051 \
        --config pi05_astribot_lora \
        --checkpoint checkpoints/pi05_astribot_lora/astribot_lora_exp1/50000 \
        --prompt "clear up the desktop"
    
    # 使用 EnvMode 加载默认配置 (类似 serve_policy.py)
    python scripts/serve_policy_grpc.py --env aloha_sim --port 50051
"""

import dataclasses
import enum
import logging
import socket
import sys

import tyro


class EnvMode(enum.Enum):
    """支持的环境模式"""
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    ASTRIBOT = "astribot"


@dataclasses.dataclass
class Checkpoint:
    """从训练好的 checkpoint 加载策略"""
    # 训练配置名称 (e.g., "pi05_astribot_lora")
    config: str
    # Checkpoint 目录 (e.g., "checkpoints/pi05_astribot_lora/exp/50000")
    dir: str


@dataclasses.dataclass
class Default:
    """使用环境的默认策略"""


@dataclasses.dataclass
class Args:
    """gRPC 推理服务参数"""
    
    # 环境模式 (使用默认策略时)
    env: EnvMode = EnvMode.ASTRIBOT
    
    # 默认 prompt (如果数据中没有 prompt)
    default_prompt: str | None = None
    
    # gRPC 端口
    port: int = 50051
    # 监听地址
    host: str = "0.0.0.0"
    # 工作线程数
    workers: int = 10
    
    # PyTorch 设备
    device: str | None = None
    
    # 策略加载方式
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# 默认 checkpoints
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
    EnvMode.ASTRIBOT: Checkpoint(
        config="pi05_astribot_lora",
        dir="checkpoints/pi05_astribot_lora/astribot_lora_exp1/79999",
    ),
}


def main(args: Args) -> None:
    from openpi.serving import grpc_policy_server
    
    # 确定配置和 checkpoint
    config_name = None
    checkpoint_dir = None
    
    match args.policy:
        case Checkpoint():
            config_name = args.policy.config
            checkpoint_dir = args.policy.dir
        case Default():
            if checkpoint := DEFAULT_CHECKPOINT.get(args.env):
                config_name = checkpoint.config
                checkpoint_dir = checkpoint.dir
            else:
                logging.warning(f"未找到环境 {args.env} 的默认配置，服务器将以空闲模式启动")
    
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(f"创建 gRPC 服务器 (host: {hostname}, ip: {local_ip})")
    logging.info(f"  端口: {args.port}")
    if config_name:
        logging.info(f"  配置: {config_name}")
        logging.info(f"  Checkpoint: {checkpoint_dir}")
    else:
        logging.info("  模式: 空闲模式 (等待 Client 配置)")
    
    # 启动服务器
    grpc_policy_server.run_server(
        host=args.host,
        port=args.port,
        max_workers=args.workers,
        config_name=config_name,
        checkpoint_dir=checkpoint_dir,
        default_prompt=args.default_prompt,
        pytorch_device=args.device,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

