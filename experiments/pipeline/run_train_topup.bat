@echo off
REM ============================================================================
REM  PCFM Heat 续训 —— 独立进程启动器
REM  与 DSH 解耦：本脚本启动的 python 不依附 DSH，重启 DSH 不会中断训练。
REM  用法：双击，或在命令行执行  run_train_topup.bat
REM ============================================================================
setlocal EnableExtensions
set "PY=C:\Users\Jessewen_Wang\.conda\envs\MLGener\python.exe"
set "WORK=C:\Resource\Code\DSH\paperResearch\work\repro"

cd /d "%WORK%" || ( echo [FATAL] cd failed & pause & exit /b 1 )
if not exist "%PY%" ( echo [FATAL] python not found: %PY% & pause & exit /b 9009 )

REM ── 显存自检：batch 64 约需 6.2 GB，不足则拒绝启动 ───────────────────────
echo [1/3] 检查显存...
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader
echo     若 free 小于 7000 MiB，请先重启电脑释放显存，或把下方 BATCH 改为 32。

REM ── 断点续训：从最新 checkpoint 继续 ─────────────────────────────────────
set "CKPT=out\heat_30000.pt"
if not exist "%CKPT%" (
  echo [FATAL] checkpoint not found: %CKPT%
  pause & exit /b 1
)
echo [2/3] 从 %CKPT% 续训

REM 论文总样本量 256*20000=5.12M；batch=64 时需 80000 步。
REM 已完成 30000 步，故补 50000 步。若要改为 batch 32，把步数改为 100000。
set "STEPS=50000"
set "BATCH=64"

echo [3/3] 启动训练（日志写入 out\train_resume.log，可关闭本窗口，训练继续）
echo.
"%PY%" run_heat_official.py train ^
  --resume "%CKPT%" --steps %STEPS% --batch %BATCH% --lr 3e-5 ^
  --save-every 5000 --log-every 1000 ^
  > out\train_resume.log 2>&1

echo.
echo [DONE] 训练结束，退出码 %errorlevel%
echo 日志：%WORK%\out\train_resume.log
pause
endlocal
