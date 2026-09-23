P81 回滚点。注意：以下三处 P81 编辑在备份前已落盘，本目录 = P81 局部应用态。
已应用：
 1. tiling.h SFA_STAGE_MAX_CUBE=64（新增常量）
 2. kernel Init: krBuf_/kfBuf_/krfBuf_ 在 cubeOn_ 时不分配
 3. kernel 成员 stageBeg_/stageLen_ 尺寸 SFA_STAGE_MAX -> SFA_STAGE_MAX_CUBE
 4. kernel Init 自证门补 nBlk_ <= SFA_STAGE_MAX_CUBE
未应用：host CalcUbNeed 的 cube 开关 + CubeGate 段钳
P79 回滚点 = pre_p79_aivprefetch_20260923_095724（pre-P79）+ §15.73(i) 的 V-gather 前移
