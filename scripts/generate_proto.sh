#!/bin/bash
# 生成 gRPC proto 代码

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROTO_DIR="$PROJECT_ROOT/proto"
OUTPUT_DIR="$PROJECT_ROOT/src/openpi/serving/generated"

echo "=== OpenPi gRPC Proto 代码生成 ==="
echo "Proto 目录: $PROTO_DIR"
echo "输出目录: $OUTPUT_DIR"

# 检查 proto 文件
if [ ! -f "$PROTO_DIR/openpi_inference.proto" ]; then
    echo "错误: 未找到 proto 文件: $PROTO_DIR/openpi_inference.proto"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查 grpcio-tools 是否安装
if ! python -c "import grpc_tools.protoc" 2>/dev/null; then
    echo "安装 grpcio-tools..."
    pip install grpcio-tools
fi

# 生成 Python 代码
echo "生成 Python gRPC 代码..."
python -m grpc_tools.protoc \
    -I"$PROTO_DIR" \
    --python_out="$OUTPUT_DIR" \
    --grpc_python_out="$OUTPUT_DIR" \
    "$PROTO_DIR/openpi_inference.proto"

# 创建 __init__.py
cat > "$OUTPUT_DIR/__init__.py" << 'EOF'
# Auto-generated gRPC code for OpenPi inference service
from . import openpi_inference_pb2
from . import openpi_inference_pb2_grpc
EOF

# 修复导入路径 (grpc_tools 生成的代码使用相对导入可能有问题)
sed -i 's/import openpi_inference_pb2/from . import openpi_inference_pb2/g' "$OUTPUT_DIR/openpi_inference_pb2_grpc.py"

echo "=== 代码生成完成 ==="
echo "生成的文件:"
ls -la "$OUTPUT_DIR"

echo ""
echo "使用方法:"
echo "  from openpi.serving.generated import openpi_inference_pb2 as pb2"
echo "  from openpi.serving.generated import openpi_inference_pb2_grpc as pb2_grpc"

