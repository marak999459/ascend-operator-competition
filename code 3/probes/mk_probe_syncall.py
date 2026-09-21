#!/usr/bin/env python3
"""生成 **只给远端副本用** 的 SyncAll 探针：在 Process() 的单元循环后注入一次
`SyncAll();`，用来回答 P11（沿 KV 轴切核 + 跨核归并）的前置问题 ——

  1) `SyncAll()` 在 __NPU_ARCH__==2201 / AIV-only / CANN 9.0.0 下能不能编过；
  2) 我们的 kernel 是不是**真的每次调用都恰好过一次 barrier**（题2 的死锁根因就是
     各核 `SyncAll` 次数不齐，见 `op_kernel/sparse_flash_attention.cpp` 顶部警告）；
  3) blockDim 只有部分核有活儿时（r1_min：1 个单元 / 40 个块）barrier 会不会挂。
     ⇒ host 一直 `SetBlockDim(GetCoreNumAiv())`，40 块全常驻，所以理论上安全，
        但这条**必须实测**，不能靠读 API 文档。

模式：off（不打补丁，自报 md5 必须是本地干净源）/ bar（注入 SyncAll）
⛔ 探针只写远端 `~/sfa_real/code/`，本地提交源永不落探针（§15.14(e)）。
"""
import sys

SRC = "/home/fszqsn/ops_comp/ascend-operator-competition/code 3/code/op_kernel/sparse_flash_attention.cpp"
CLEAN_MD5 = "3d366c529adf6d005bd73fab023df95d"

ANCHOR = """            ProcessToken(b, s, hb);
        }
    }
"""
PATCH = """            ProcessToken(b, s, hb);
        }
        SyncAll();   // SFA_PROBE_SYNCALL
    }
"""


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "off"
    src = open(SRC, encoding="utf-8").read()
    import hashlib
    if hashlib.md5(src.encode()).hexdigest() != CLEAN_MD5:
        print(f"!! 本地源 md5 已变，CLEAN_MD5 常量要更新（当前 {hashlib.md5(src.encode()).hexdigest()}）",
              file=sys.stderr)
    if mode == "off":
        sys.stdout.write(src)
        return 0
    if mode == "bar":
        if src.count(ANCHOR) != 1:
            print(f"!! 锚点命中 {src.count(ANCHOR)} 次，不是 1 次 —— 别猜，先看源码", file=sys.stderr)
            return 2
        sys.stdout.write(src.replace(ANCHOR, PATCH))
        return 0
    print("用法: mk_probe_syncall.py [off|bar]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
