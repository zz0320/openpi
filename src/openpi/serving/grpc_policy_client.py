# -*- coding: utf-8 -*-
"""
OpenPi gRPC 推理客户端

运行环境: 机器人侧 (任何 Python 环境，只需 grpcio)

通过 gRPC 连接远程 OpenPi 推理服务器获取 action

使用方法:
    from openpi.serving.grpc_policy_client import OpenPiClient

    # 连接服务器
    client = OpenPiClient("localhost:50051")
    
    # 配置模型 (如果服务器未预加载)
    client.configure(
        config_name="pi05_astribot_lora",
        checkpoint_dir="checkpoints/pi05_astribot_lora/exp/50000"
    )
    
    # 推理
    action = client.predict(
        state=[0.0] * 16,
        images=[{"name": "head", "data": jpeg_bytes, "encoding": "jpeg"}],
        prompt="pick up the cup"
    )
"""

import io
import json
import logging
import time
from collections import deque
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

import grpc

# 导入生成的 protobuf 代码
try:
    from openpi.serving.generated import openpi_inference_pb2 as pb2
    from openpi.serving.generated import openpi_inference_pb2_grpc as pb2_grpc
except ImportError:
    pb2 = None
    pb2_grpc = None
    print("警告: 未找到 protobuf 生成文件，请先运行 scripts/generate_proto.sh")


logger = logging.getLogger("openpi.grpc_client")


class ActionChunkManager:
    """
    Action Chunk 管理器
    
    在 Client 端管理 action queue，实现：
    1. 从 Server 获取完整的 action chunk
    2. 在本地逐步消费 action
    3. 当 queue 用完时，自动请求新的 chunk
    
    适用于 action chunking 策略 (Pi0/Pi0.5)
    
    Example:
        >>> client = OpenPiClient("localhost:50051")
        >>> chunk_manager = ActionChunkManager(client, n_action_steps=10)
        >>> 
        >>> for frame_idx in range(1000):
        ...     action = chunk_manager.get_action(
        ...         state=current_state,
        ...         images=images,
        ...         prompt="pick up the cup"
        ...     )
        ...     if action is None:
        ...         break
        ...     # 发送到机器人...
    """
    
    def __init__(
        self,
        client: "OpenPiClient",
        n_action_steps: Optional[int] = None,
        auto_refill_threshold: float = 0.0
    ):
        """
        初始化 Action Chunk 管理器
        
        Args:
            client: OpenPiClient 实例
            n_action_steps: 每个 chunk 实际使用的 action 数量
                           如果为 None，使用 Server 返回的完整 chunk
            auto_refill_threshold: 自动补充阈值 (0.0-1.0)
                                   0.0 表示用完才请求
        """
        self.client = client
        self.n_action_steps = n_action_steps
        self.auto_refill_threshold = auto_refill_threshold
        
        # Action queue
        self._action_queue: deque = deque()
        self._chunk_size = 0
        self._action_dim = 0
        
        # 状态
        self._current_chunk_start_frame = 0
        self._actions_consumed = 0
        self._total_actions_consumed = 0
        self._is_terminal = False
        self._last_action_triggered_inference = False
    
    @property
    def queue_size(self) -> int:
        """当前 queue 中的 action 数量"""
        return len(self._action_queue)
    
    @property
    def chunk_size(self) -> int:
        """Server 返回的 chunk 大小"""
        return self._chunk_size
    
    @property
    def action_dim(self) -> int:
        """Action 维度"""
        return self._action_dim
    
    @property
    def is_empty(self) -> bool:
        """Queue 是否为空"""
        return len(self._action_queue) == 0
    
    @property
    def is_terminal(self) -> bool:
        """Episode 是否结束"""
        return self._is_terminal and self.is_empty
    
    @property
    def last_action_triggered_inference(self) -> bool:
        """上次 get_action() 是否触发了新的模型推理"""
        return self._last_action_triggered_inference
    
    def reset(self):
        """重置状态"""
        self._action_queue.clear()
        self._actions_consumed = 0
        self._total_actions_consumed = 0
        self._is_terminal = False
        self._current_chunk_start_frame = 0
        self._last_action_triggered_inference = False
    
    def _should_refill(self) -> bool:
        """检查是否需要补充 action"""
        if self._is_terminal:
            return False
        if self._chunk_size == 0:
            return True
        
        effective_size = self.n_action_steps or self._chunk_size
        remaining_ratio = len(self._action_queue) / effective_size
        return remaining_ratio <= self.auto_refill_threshold
    
    def _fetch_chunk(
        self,
        state: List[float],
        images: Optional[List[dict]] = None,
        prompt: Optional[str] = None,
        frame_index: int = 0
    ) -> bool:
        """从 Server 获取新的 action chunk"""
        try:
            chunk_response = self.client.predict_chunk(
                state=state,
                images=images,
                prompt=prompt,
                frame_index=frame_index
            )
            
            if chunk_response.status == pb2.EPISODE_END:
                self._is_terminal = True
                return False
            
            if chunk_response.status != pb2.OK:
                logger.error(f"获取 chunk 失败: {chunk_response.error_message}")
                return False
            
            # 更新 chunk 信息
            self._chunk_size = chunk_response.chunk_size
            self._action_dim = chunk_response.action_dim
            self._current_chunk_start_frame = frame_index
            
            # 清空旧的 queue
            self._action_queue.clear()
            self._actions_consumed = 0
            
            # 确定实际使用的 action 数量
            n_to_use = self.n_action_steps if self.n_action_steps else self._chunk_size
            n_to_use = min(n_to_use, len(chunk_response.actions))
            
            for i in range(n_to_use):
                action = list(chunk_response.actions[i].values)
                self._action_queue.append(action)
            
            return True
            
        except Exception as e:
            logger.error(f"获取 chunk 异常: {e}")
            return False
    
    def get_action(
        self,
        state: List[float],
        images: Optional[List[dict]] = None,
        prompt: Optional[str] = None,
        frame_index: int = 0
    ) -> Optional[List[float]]:
        """
        获取下一个 action
        
        自动管理 chunk 请求
        
        Returns:
            action 列表，如果 episode 结束返回 None
        """
        self._last_action_triggered_inference = False
        
        # 检查是否需要获取新 chunk
        if self.is_empty or self._should_refill():
            chunk_frame = self._current_chunk_start_frame + self._actions_consumed
            if self.is_empty:
                chunk_frame = frame_index
            
            if self._fetch_chunk(state, images, prompt, chunk_frame):
                self._last_action_triggered_inference = True
            else:
                if self._is_terminal and not self.is_empty:
                    pass
                elif self.is_empty:
                    return None
        
        if self.is_empty:
            return None
        
        action = self._action_queue.popleft()
        self._actions_consumed += 1
        self._total_actions_consumed += 1
        
        return action


class OpenPiClient:
    """
    OpenPi gRPC 推理客户端
    
    负责与远程推理服务器通信
    
    Example:
        >>> client = OpenPiClient("localhost:50051")
        >>> 
        >>> # 配置模型
        >>> client.configure(
        ...     config_name="pi05_astribot_lora",
        ...     checkpoint_dir="checkpoints/pi05_astribot_lora/exp/50000",
        ...     default_prompt="pick up the cup"
        ... )
        >>> 
        >>> # 单次推理
        >>> action = client.predict(state=[0.0] * 16)
        >>> print(action.values)
        >>> 
        >>> # Chunk 推理
        >>> chunk = client.predict_chunk(state=[0.0] * 16)
        >>> for step in chunk.actions:
        ...     print(step.values)
    """
    
    def __init__(
        self,
        server_address: str = "localhost:50051",
        timeout: float = 30.0
    ):
        """
        初始化客户端
        
        Args:
            server_address: 服务器地址 (host:port)
            timeout: 连接超时时间 (秒)
        """
        if pb2 is None or pb2_grpc is None:
            raise RuntimeError("未找到 protobuf 生成文件，请先运行 scripts/generate_proto.sh")
        
        self.server_address = server_address
        self.timeout = timeout
        self.channel = None
        self.stub = None
        self._connected = False
        self._metadata: Dict[str, Any] = {}
        
        self._connect()
    
    def _connect(self):
        """连接到服务器"""
        logger.info(f"连接推理服务器: {self.server_address}")
        
        self.channel = grpc.insecure_channel(
            self.server_address,
            options=[
                ('grpc.max_send_message_length', 100 * 1024 * 1024),
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                ('grpc.keepalive_time_ms', 10000),
                ('grpc.keepalive_timeout_ms', 5000),
            ]
        )
        self.stub = pb2_grpc.OpenPiInferenceServiceStub(self.channel)
        
        try:
            grpc.channel_ready_future(self.channel).result(timeout=self.timeout)
            self._connected = True
            logger.info("已连接到推理服务器")
        except grpc.FutureTimeoutError:
            raise ConnectionError(f"无法连接到服务器: {self.server_address}")
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    @property
    def server_metadata(self) -> Dict[str, Any]:
        """获取服务器元数据"""
        return self._metadata
    
    def get_status(self) -> "pb2.ServiceStatus":
        """获取服务状态"""
        status = self.stub.GetStatus(pb2.Empty())
        if status.metadata_json:
            try:
                self._metadata = json.loads(status.metadata_json)
            except:
                pass
        return status
    
    def configure(
        self,
        config_name: str,
        checkpoint_dir: str,
        default_prompt: Optional[str] = None,
        device: Optional[str] = None
    ) -> "pb2.ServiceStatus":
        """
        配置 Server 使用的模型
        
        Args:
            config_name: 训练配置名称 (e.g., "pi05_astribot_lora")
            checkpoint_dir: Checkpoint 目录
            default_prompt: 默认语言指令
            device: PyTorch 设备 (e.g., "cuda")
        """
        config = pb2.PolicyConfig(
            config_name=config_name,
            checkpoint_dir=checkpoint_dir,
            default_prompt=default_prompt or "",
            device=device or ""
        )
        
        status = self.stub.Configure(config)
        if status.metadata_json:
            try:
                self._metadata = json.loads(status.metadata_json)
            except:
                pass
        return status
    
    @staticmethod
    def encode_image(
        image,
        camera_name: str = "head",
        encoding: str = "jpeg",
        quality: int = 85
    ) -> dict:
        """
        将图像编码为可发送的格式
        
        Args:
            image: PIL Image, numpy array (H, W, C), 或 bytes
            camera_name: 相机名称 (e.g., "head", "wrist_left", "wrist_right")
            encoding: 编码格式 ("jpeg", "png", "raw")
            quality: JPEG 质量 (1-100)
            
        Returns:
            dict: 可传递给 predict() 的图像字典
        """
        from PIL import Image as PILImage
        
        # 转换为 PIL Image
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_image = PILImage.fromarray(image)
        elif isinstance(image, bytes):
            return {
                'name': camera_name,
                'data': image,
                'width': 0,
                'height': 0,
                'encoding': encoding
            }
        elif hasattr(image, 'mode'):
            pil_image = image
        else:
            raise ValueError(f"不支持的图像类型: {type(image)}")
        
        width, height = pil_image.size
        
        # 编码
        if encoding.lower() in ['jpeg', 'jpg']:
            buffer = io.BytesIO()
            pil_image.convert('RGB').save(buffer, format='JPEG', quality=quality)
            data = buffer.getvalue()
        elif encoding.lower() == 'png':
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            data = buffer.getvalue()
        elif encoding.lower() == 'raw':
            data = pil_image.convert('RGB').tobytes()
        else:
            raise ValueError(f"不支持的编码格式: {encoding}")
        
        return {
            'name': camera_name,
            'data': data,
            'width': width,
            'height': height,
            'encoding': encoding
        }
    
    def _build_observation(
        self,
        state: List[float],
        images: Optional[List[dict]] = None,
        prompt: Optional[str] = None,
        episode_id: int = 0,
        frame_index: int = 0
    ) -> "pb2.Observation":
        """构建观测消息"""
        obs = pb2.Observation(
            state=state,
            timestamp=time.time(),
            episode_id=episode_id,
            frame_index=frame_index,
            prompt=prompt or ""
        )
        
        if images:
            for img in images:
                obs.images.append(pb2.ImageData(
                    camera_name=img.get('name', 'head'),
                    data=img.get('data', b''),
                    width=img.get('width', 0),
                    height=img.get('height', 0),
                    encoding=img.get('encoding', 'jpeg')
                ))
        
        return obs
    
    def predict(
        self,
        state: List[float],
        images: Optional[List[dict]] = None,
        prompt: Optional[str] = None,
        episode_id: int = 0,
        frame_index: int = 0
    ) -> "pb2.Action":
        """
        单次推理 - 返回第一个 action
        
        Args:
            state: 状态向量 (Astribot: 16维)
            images: 图像列表，使用 encode_image() 编码
            prompt: 语言指令
            episode_id: episode 索引
            frame_index: 帧索引
            
        Returns:
            Action 响应
        """
        obs = self._build_observation(state, images, prompt, episode_id, frame_index)
        return self.stub.Predict(obs)
    
    def predict_chunk(
        self,
        state: List[float],
        images: Optional[List[dict]] = None,
        prompt: Optional[str] = None,
        episode_id: int = 0,
        frame_index: int = 0
    ) -> "pb2.ActionChunk":
        """
        Chunk 推理 - 返回完整的 action chunk
        
        Args:
            state: 状态向量
            images: 图像列表
            prompt: 语言指令
            episode_id: episode 索引
            frame_index: 帧索引
            
        Returns:
            ActionChunk 响应
        """
        obs = self._build_observation(state, images, prompt, episode_id, frame_index)
        return self.stub.PredictChunk(obs)
    
    def stream_predict(
        self,
        observation_generator
    ) -> Iterator["pb2.Action"]:
        """流式推理"""
        return self.stub.StreamPredict(observation_generator())
    
    def reset(self) -> "pb2.ServiceStatus":
        """重置推理状态"""
        return self.stub.Reset(pb2.Empty())
    
    def set_episode(self, episode: int) -> "pb2.ServiceStatus":
        """设置当前 episode"""
        cmd = pb2.ControlCommand(
            type=pb2.CMD_SET_EPISODE,
            params={"episode": str(episode)}
        )
        return self.stub.Control(cmd)
    
    def set_prompt(self, prompt: str) -> "pb2.ServiceStatus":
        """设置当前 prompt"""
        cmd = pb2.ControlCommand(
            type=pb2.CMD_SET_PROMPT,
            params={"prompt": prompt}
        )
        return self.stub.Control(cmd)
    
    def close(self):
        """关闭连接"""
        if self.channel:
            self.channel.close()
            self._connected = False
        logger.info("已断开推理服务器连接")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# 为兼容 lerobot_grpc_inference 提供的别名
InferenceClient = OpenPiClient


def main():
    """测试客户端"""
    import argparse
    
    parser = argparse.ArgumentParser(description='OpenPi gRPC 客户端测试')
    parser.add_argument('--server', default='localhost:50051', help='服务器地址')
    parser.add_argument('--config', type=str, help='配置名称')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint 目录')
    parser.add_argument('--prompt', type=str, default='pick up the cup', help='测试 prompt')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    client = OpenPiClient(args.server)
    
    try:
        # 获取状态
        status = client.get_status()
        print(f"服务器状态: ready={status.is_ready}, model={status.model_name}")
        
        # 配置模型 (如果需要)
        if args.config and args.checkpoint:
            print(f"配置模型: {args.config}")
            status = client.configure(
                config_name=args.config,
                checkpoint_dir=args.checkpoint,
                default_prompt=args.prompt
            )
            print(f"配置结果: {status.message}")
        
        # 测试推理
        if status.is_ready:
            print("\n测试单次推理...")
            action = client.predict(
                state=[0.0] * 16,
                prompt=args.prompt
            )
            print(f"Action 状态: {action.status}")
            print(f"Action 值: {action.values[:5]}... (共 {len(action.values)} 维)")
            
            print("\n测试 Chunk 推理...")
            chunk = client.predict_chunk(
                state=[0.0] * 16,
                prompt=args.prompt
            )
            print(f"Chunk 状态: {chunk.status}")
            print(f"Chunk 大小: {chunk.chunk_size}, Action 维度: {chunk.action_dim}")
            if chunk.actions:
                print(f"第一个 Action: {chunk.actions[0].values[:5]}...")
        else:
            print("服务器未就绪，请先配置模型")
            
    finally:
        client.close()


if __name__ == '__main__':
    main()

