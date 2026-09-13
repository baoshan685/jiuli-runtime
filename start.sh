#!/usr/bin/env bash
# 酒醴 Web UI 一键启动（macOS / Linux）
# 默认离线演示模式。接真实模型：
#   ./start.sh --api-base https://api.xxx.com/v1 --api-key sk-xxx --model your-model
cd "$(dirname "$0")"
PKG="${PKG:-examples/demo-pkg}"
python3 -m jiuli.server --pkg "$PKG" --db jiuli.web.db --port 8770 "$@"
