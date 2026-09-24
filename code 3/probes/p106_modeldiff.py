#!/usr/bin/env python3
# 把 op_host 的【纯模型区】原样抽出来，做两份 CalcBlocking（P105 / P38），编成一个
# 不碰设备的 g++ 程序。零手抄：模型区是从 host 源里按行切的，只在 ks 那一档上替换。
import pathlib, re, sys

HOST = pathlib.Path(__file__).resolve().parents[1] / "code/op_host/sparse_flash_attention.cpp"
src = HOST.read_text().splitlines(True)

def find(pat, lo=0):
    for i in range(lo, len(src)):
        if re.search(pat, src[i]):
            return i
    sys.exit(f"找不到 {pat}")

a = find(r"^constexpr uint32_t HQ_DIM")
b = find(r"^static bool CalcBlocking")
c = find(r"^// 变长长度数组的元素个数", b)
model = "".join(src[a:b]) + "".join(src[c:c])   # 丢掉 LenArraySize
# CalcBlocking 整体改名两份：先取出函数体
calc = "".join(src[b:c])

# P38 的 ks 那一档：把 P105 的 KS_CAND 循环换回"只有 ks=2 + 税按 ×108/100"
def to_p38(text):
    lines = text.splitlines(True)
    out, i = [], 0
    while i < len(lines):
        if "if (KsAllowed(units, coreNum, lseElems, sparseCount, nBlk, 2ULL)) {" in lines[i]:
            indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
            out.append(indent + "if (KsAllowed(units, coreNum, lseElems, sparseCount, nBlk, 2ULL)) {\n")
            out.append(indent + "    const uint64_t units2 = units * 2ULL;\n")
            out.append(indent + "    const uint64_t waves2 = (coreNum == 0)\n")
            out.append(indent + "        ? units2 : (units2 + coreNum - 1ULL) / coreNum;\n")
            out.append(indent + "    uint64_t chunks2 = (toks / 2ULL + nBlk - 1ULL) / nBlk;\n")
            out.append(indent + "    if (chunks2 == 0ULL) { chunks2 = 1ULL; }\n")
            out.append(indent + "    const uint64_t cost2 = ((waves2 * calls * chunks2) * 108ULL) / 100ULL + 1ULL;\n")
            out.append(indent + "    if (cost2 < cost) { cost = cost2; ks = 2U; }\n")
            out.append(indent + "}\n")
            depth = 0; started = False
            while i < len(lines):
                depth += lines[i].count("{") - lines[i].count("}")
                if "{" in lines[i]: started = True
                i += 1
                if started and depth == 0: break
            continue
        out.append(lines[i]); i += 1
    return "".join(out)

hdr = '''// 自动生成：probes/p106_modeldiff.py —— 不要手改，改模型区请改 op_host 后重跑
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
constexpr uint32_t SFA_STAGE_MAX = 48;
'''

# 让两份 CalcBlocking 把 bestCost 也带出来：只改签名 + 赋值处，模型本体不动
def expose_cost(text):
    text = text.replace("uint32_t &nbOut, uint32_t &nBlkOut, uint32_t &ksOut)",
                        "uint32_t &nbOut, uint32_t &nBlkOut, uint32_t &ksOut, uint64_t &costOut)", 1)
    text = text.replace("    ksOut = bestKs;\n    return true;",
                        "    ksOut = bestKs;\n    costOut = bestCost;\n    return true;", 1)
    return text

body_p105 = expose_cost(calc)
body_p38 = expose_cost(to_p38(calc)).replace("static bool CalcBlocking", "static bool CalcBlockingP38")
assert "cost2" in body_p38 and "KS_CAND" not in body_p38.split("CalcBlockingP38")[1]

main = r'''
struct Row { uint32_t rows, qN, sbs; uint64_t count, kvS; };
static void emit(const Row &r, uint32_t elemSize, bool quietSame)
{
    uint64_t lse = (uint64_t)r.rows * r.qN;
    uint32_t nb, k, ks, nb2, k2, ks2;
    uint64_t c105 = 0, c38 = 0;
    bool ok  = CalcBlocking(r.sbs, 186568, 512, 64, r.qN, elemSize, 40, r.rows, r.count, lse, r.kvS, nb, k, ks, c105);
    bool ok2 = CalcBlockingP38(r.sbs, 186568, 512, 64, r.qN, elemSize, 40, r.rows, r.count, lse, r.kvS, nb2, k2, ks2, c38);
    if (!ok || !ok2) { printf("FAIL  "); return; }
    bool same = (nb == nb2 && k == k2 && ks == ks2);
    if (quietSame && same) return;
    printf("rows=%-4u qN=%-2u sbs=%-4u count=%-6llu %-4s | P38 nb=%-2u k=%-3u ks=%-2u  ->  P105 nb=%-2u k=%-3u ks=%-2u"
           "  模型预测加速 %5.2fx %s\n",
           r.rows, r.qN, r.sbs, (unsigned long long)r.count, elemSize == 2 ? "fp16" : "fp32",
           nb2, k2, ks2, nb, k, ks, (double)c38 / (double)c105, same ? "" : "  <<< 不同档");
}
int main(int argc, char **argv)
{
    const bool big = (argc > 1 && strcmp(argv[1], "big") == 0);
    const std::vector<uint32_t> ROWS = {2,4,8,16,32,64,128,256,512,1024,4096};
    const std::vector<uint32_t> QN   = {1,2,4,8,16};
    const std::vector<uint64_t> CNT  = big ? std::vector<uint64_t>{2048,8192,32768,65536,262144}
                                           : std::vector<uint64_t>{256,1024,2048,4096,16384,65536};
    const std::vector<uint32_t> SBS  = big ? std::vector<uint32_t>{1} : std::vector<uint32_t>{1,16,64,128};
    for (uint32_t rows : ROWS) for (uint32_t qn : QN) for (uint64_t cnt : CNT) for (uint32_t sbs : SBS) {
        Row r{rows, qn, sbs, cnt, 1ULL << 30};
        emit(r, 2, big);
    }
    return 0;
}
'''

out = hdr + model + "\n" + body_p105 + "\n" + body_p38 + main
p = pathlib.Path(__file__).resolve().parent / "p106_modeldiff.cpp"
p.write_text(out)
print("wrote", p, len(out.splitlines()), "lines")
