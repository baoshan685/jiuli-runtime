@echo off
rem 酒醴 Web UI 一键启动（Windows）
rem 默认离线演示模式（无需 API key）。要接真实模型，先设置环境变量：
rem   set JIULI_API_KEY=sk-xxx  （可选 JIULI_BASE_URL / JIULI_MODEL）
rem 然后运行： start.bat --api-base %JIULI_BASE_URL% --api-key %JIULI_API_KEY% --model %JIULI_MODEL%
set PKG=examples\demo-pkg

cd /d "%~dp0"
python -m jiuli.server --pkg "%PKG%" --db jiuli.web.db --port 8770 %*
pause
