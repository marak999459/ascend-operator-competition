// ============================================================================
// SparseFlashAttention —— 真机（NPU）对拍 harness
//
// ⚠️⚠️ 这是【重建版】，不是原始文件。
//   原始 test_sfa_real.cpp 在 2026-09-19 的目录整理中被误删（无备份）。
//   本文件依据以下**仍然可靠的信息**重建：
//     1) 用例文件格式：refs/sfa/sfa_ref.py 的 write_case()（_MAGIC = "SFA_CASE 2\n"）
//     2) 用法：refs/sfa/run.sh —— `./test_sfa <case.bin>`，退出码 0=PASS
//     3) 输出格式：run.sh 用 grep -aE "最大相对误差|超差元素|墙钟" 提取
//     4) 输入契约与属性顺序：见根目录 official_problem_statement.md 与
//        code 3/code/op_host/sparse_flash_attention.cpp 的 OpDef
//
// ⛔ 本文件【尚未在真机上编译/运行验证】。首次使用前请先跑一个已知用例
//    确认能出 PASS，再信任它的结论。
//
// ⚠️ 已知遗留问题：refs/sfa 下的 .bin 用例头是 `SFA_CASE 1`（旧版生成），
//    其 payload 比当前 sfa_ref.py 的 write_case()（`SFA_CASE 2`）**少 16 字节**，
//    差值来源未确定。因此本 harness 在加载时做**精确字节数自校验**：
//    不吻合就报 FAIL 并提示重新生成用例，**绝不静默读成垃圾数据**。
//    最稳的用法：先用当前 sfa_ref.py 重新生成用例，再跑本 harness。
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn_sparse_flash_attention.h"

// ---------------------------------------------------------------- fp16 解码
// ⚠️ 不要用 (1u << (exp-15)) 之类的移位写法：exp<15 时移位为负 = UB
//    （本项目曾因此把正确的 0.125 读成 5.37e8，误判 kernel 有 bug）
static float HalfToFloat(uint16_t h)
{
    const uint32_t sign = static_cast<uint32_t>(h >> 15) & 0x1u;
    const uint32_t exp  = static_cast<uint32_t>(h >> 10) & 0x1Fu;
    const uint32_t man  = static_cast<uint32_t>(h) & 0x3FFu;

    float v;
    if (exp == 0) {
        v = (man == 0) ? 0.0f : std::ldexp(static_cast<float>(man), -24);
    } else if (exp == 31) {
        v = 0.0f;  // Inf / NaN 一律当 0（本题不用）
    } else {
        v = (1.0f + static_cast<float>(man) / 1024.0f) * std::ldexp(1.0f, static_cast<int>(exp) - 15);
    }
    return sign ? -v : v;
}

// ---------------------------------------------------------------- 用例结构
struct Case {
    long   B = 0, S1 = 0, S2 = 0, N1 = 0, D = 512;
    long   SBS = 1, COUNT = 2048, MODE = 3, LSE = 1;
    double scale = 0.0;

    std::vector<uint16_t> query, key, value, qrope, krope, expect;
    std::vector<int32_t>  idx, asq, ask;
    std::vector<float>    expect_max, expect_sum;
};

static bool ReadLine(FILE *fp, std::string &out)
{
    char buf[512];
    if (!std::fgets(buf, sizeof(buf), fp)) { return false; }
    out = buf;
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) { out.pop_back(); }
    return true;
}

template <typename T>
static bool ReadVec(FILE *fp, std::vector<T> &v, size_t n)
{
    v.resize(n);
    if (n == 0) { return true; }
    return std::fread(v.data(), sizeof(T), n, fp) == n;
}

static bool LoadCase(const char *path, Case &c)
{
    FILE *fp = std::fopen(path, "rb");
    if (!fp) { std::printf("[FAIL] cannot open %s\n", path); return false; }

    std::string line;
    if (!ReadLine(fp, line) || line.compare(0, 8, "SFA_CASE") != 0) {
        std::printf("[FAIL] bad magic (expect SFA_CASE <ver>)\n");
        std::fclose(fp);
        return false;
    }
    std::printf("  magic = [%s]\n", line.c_str());
    if (line.compare(0, 11, "SFA_CASE 2") != 0) {
        // ⚠️ 已知 refs/sfa 下的 .bin 是 "SFA_CASE 1"（旧版生成），其 payload 比
        //    当前 sfa_ref.py 的 write_case() 少 16 字节，差值来源未确定。
        //    下面的自校验会拦住这种不吻合，避免静默读成垃圾数据。
        std::printf("  [WARN] magic 不是 SFA_CASE 2 —— 可能是旧版生成的用例，布局或有差异\n");
    }

    // 文本头：每行 "<KEY> <VALUE>"，共 10 行（见 sfa_ref.write_case）
    const char *keys[10] = {"B", "S1", "S2", "N1", "D", "SBS", "COUNT", "SCALE", "MODE", "LSE"};
    for (int i = 0; i < 10; ++i) {
        if (!ReadLine(fp, line)) { std::printf("[FAIL] header truncated at %s\n", keys[i]); std::fclose(fp); return false; }
        char k[64] = {0};
        char v[128] = {0};
        if (std::sscanf(line.c_str(), "%63s %127s", k, v) != 2) {
            std::printf("[FAIL] bad header line: %s\n", line.c_str()); std::fclose(fp); return false;
        }
        if (std::strcmp(k, "B") == 0)          { c.B = std::atol(v); }
        else if (std::strcmp(k, "S1") == 0)    { c.S1 = std::atol(v); }
        else if (std::strcmp(k, "S2") == 0)    { c.S2 = std::atol(v); }
        else if (std::strcmp(k, "N1") == 0)    { c.N1 = std::atol(v); }
        else if (std::strcmp(k, "D") == 0)     { c.D = std::atol(v); }
        else if (std::strcmp(k, "SBS") == 0)   { c.SBS = std::atol(v); }
        else if (std::strcmp(k, "COUNT") == 0) { c.COUNT = std::atol(v); }
        else if (std::strcmp(k, "SCALE") == 0) { c.scale = std::atof(v); }
        else if (std::strcmp(k, "MODE") == 0)  { c.MODE = std::atol(v); }
        else if (std::strcmp(k, "LSE") == 0)   { c.LSE = std::atol(v); }
    }

    const size_t nq   = static_cast<size_t>(c.B) * c.S1 * c.N1 * c.D;
    const size_t nkv  = static_cast<size_t>(c.B) * c.S2 * c.D;            // KV_N = 1
    const size_t nrp  = static_cast<size_t>(c.B) * c.S1 * c.N1 * 64;      // Dr = 64
    const size_t nkrp = static_cast<size_t>(c.B) * c.S2 * 64;
    const size_t nidx = static_cast<size_t>(c.B) * c.S1 * c.COUNT;
    const size_t nlen = static_cast<size_t>(c.B);
    const size_t nlse = static_cast<size_t>(c.B) * c.S1 * c.N1;           // (B,1,Q_S,Q_N)

    // ⭐ 自校验：payload 字节数必须与文件剩余字节精确吻合，否则布局假设是错的
    const size_t want = nq * 2 + nkv * 2 + nkv * 2 + nrp * 2 + nkrp * 2
                      + nidx * 4 + nlen * 4 + nlen * 4 + nq * 2
                      + (c.LSE ? (nlse * 4 + nlse * 4) : 0u);
    const long pos  = std::ftell(fp);
    std::fseek(fp, 0, SEEK_END);
    const long total = std::ftell(fp);
    const long have  = total - pos;
    std::fseek(fp, pos, SEEK_SET);
    if (static_cast<size_t>(have) != want) {
        std::printf("[FAIL] payload 布局不吻合：文件剩余 %ld B，按本文件的格式假设应为 %zu B（差 %ld）\n",
                    have, want, have - static_cast<long>(want));
        std::printf("       → 这不是断言失败，而是**用例文件格式与本 harness 不一致**。\n");
        std::printf("         请用 refs/sfa/sfa_ref.py 的 write_case() 重新生成用例后再跑：\n");
        std::printf("           python3 sfa_ref.py gen <B> <S1> <S2> <N1> <D> <SBS> ...\n");
        std::printf("         或修正本文件的段布局（见 sfa_ref.write_case）。\n");
        std::fclose(fp);
        return false;
    }

    bool ok = ReadVec(fp, c.query, nq)    && ReadVec(fp, c.key, nkv)   && ReadVec(fp, c.value, nkv)
           && ReadVec(fp, c.qrope, nrp)   && ReadVec(fp, c.krope, nkrp) && ReadVec(fp, c.idx, nidx)
           && ReadVec(fp, c.asq, nlen)    && ReadVec(fp, c.ask, nlen)   && ReadVec(fp, c.expect, nq);
    if (ok) {
        if (c.LSE) { ok = ReadVec(fp, c.expect_max, nlse) && ReadVec(fp, c.expect_sum, nlse); }
    }
    std::fclose(fp);
    if (!ok) { std::printf("[FAIL] payload 读取失败\n"); return false; }
    return true;
}

// ---------------------------------------------------------------- main
int main(int argc, char **argv)
{
    if (argc < 2) {
        std::printf("usage: %s <case.bin>\n", argv[0]);
        return 2;
    }
    Case c;
    if (!LoadCase(argv[1], c)) { return 2; }

    std::printf("case=%s\n", argv[1]);
    std::printf("  B=%ld S1=%ld S2=%ld N1=%ld D=%ld SBS=%ld COUNT=%ld MODE=%ld LSE=%ld scale=%.8f\n",
                c.B, c.S1, c.S2, c.N1, c.D, c.SBS, c.COUNT, c.MODE, c.LSE, c.scale);

    // ---- ACL 初始化 ----
    if (aclInit(nullptr) != ACL_SUCCESS) { std::printf("[FAIL] aclInit\n"); return 2; }
    if (aclrtSetDevice(0) != ACL_SUCCESS) { std::printf("[FAIL] aclrtSetDevice\n"); return 2; }
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) { std::printf("[FAIL] aclrtCreateStream\n"); return 2; }

    const size_t sz_h  = sizeof(uint16_t);
    const size_t sz_i  = sizeof(int32_t);
    const size_t sz_f  = sizeof(float);
    const size_t nq    = static_cast<size_t>(c.B) * c.S1 * c.N1 * c.D;
    const size_t nkv   = static_cast<size_t>(c.B) * c.S2 * c.D;
    const size_t nrp   = static_cast<size_t>(c.B) * c.S1 * c.N1 * 64;
    const size_t nkrp  = static_cast<size_t>(c.B) * c.S2 * 64;
    const size_t nidx  = static_cast<size_t>(c.B) * c.S1 * c.COUNT;
    const size_t nlen  = static_cast<size_t>(c.B);
    const size_t nlse  = static_cast<size_t>(c.B) * c.S1 * c.N1;

    auto DevAlloc = [](size_t bytes, void **p) -> bool {
        return aclrtMalloc(p, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    };
    auto H2D = [&](void *d, const void *h, size_t bytes) -> bool {
        return aclrtMemcpy(d, bytes, h, bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    };
    auto D2H = [&](void *h, const void *d, size_t bytes) -> bool {
        return aclrtMemcpy(h, bytes, d, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
    };

    void *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_idx = nullptr;
    void *d_qr = nullptr, *d_kr = nullptr, *d_asq = nullptr, *d_ask = nullptr;
    void *d_out = nullptr, *d_max = nullptr, *d_sum = nullptr;

    bool ok = DevAlloc(nq * sz_h, &d_q)     && DevAlloc(nkv * sz_h, &d_k)   && DevAlloc(nkv * sz_h, &d_v)
           && DevAlloc(nidx * sz_i, &d_idx) && DevAlloc(nrp * sz_h, &d_qr)  && DevAlloc(nkrp * sz_h, &d_kr)
           && DevAlloc(nlen * sz_i, &d_asq) && DevAlloc(nlen * sz_i, &d_ask)
           && DevAlloc(nq * sz_h, &d_out)   && DevAlloc(nlse * sz_f, &d_max) && DevAlloc(nlse * sz_f, &d_sum);
    if (!ok) { std::printf("[FAIL] aclrtMalloc\n"); return 2; }

    ok = H2D(d_q, c.query.data(), nq * sz_h)     && H2D(d_k, c.key.data(), nkv * sz_h)
      && H2D(d_v, c.value.data(), nkv * sz_h)    && H2D(d_idx, c.idx.data(), nidx * sz_i)
      && H2D(d_qr, c.qrope.data(), nrp * sz_h)   && H2D(d_kr, c.krope.data(), nkrp * sz_h)
      && H2D(d_asq, c.asq.data(), nlen * sz_i)   && H2D(d_ask, c.ask.data(), nlen * sz_i);
    if (!ok) { std::printf("[FAIL] H2D\n"); return 2; }

    // ---- 张量描述符（BSND，fp16，紧凑连续）----
    int64_t qDims[4]  = {c.B, c.S1, c.N1, c.D};
    int64_t kvDims[4] = {c.B, c.S2, 1, c.D};
    int64_t rDims[4]  = {c.B, c.S1, c.N1, 64};
    int64_t krDims[4] = {c.B, c.S2, 1, 64};
    int64_t iDims[4]  = {c.B, c.S1, 1, c.COUNT};
    int64_t lenDims[1] = {c.B};
    int64_t oDims[4]  = {c.B, c.S1, c.N1, c.D};

    int64_t qStr[4]  = {c.S1 * c.N1 * c.D, c.N1 * c.D, c.D, 1};
    int64_t kvStr[4] = {c.S2 * c.D, c.D, c.D, 1};
    int64_t rStr[4]  = {c.S1 * c.N1 * 64, c.N1 * 64, 64, 1};
    int64_t krStr[4] = {c.S2 * 64, 64, 64, 1};
    int64_t iStr[4]  = {c.S1 * c.COUNT, c.COUNT, c.COUNT, 1};
    int64_t lenStr[1] = {1};
    int64_t oStr[4]  = {c.S1 * c.N1 * c.D, c.N1 * c.D, c.D, 1};

    // lseDims 与 kernel 的 lseOff=(b*S1+s)*N1+n 一致 → 逻辑 (B,1,Q_S,N1)，物理紧凑
    int64_t lseDims[4] = {c.B, 1, c.S1, c.N1};
    int64_t lseStr[4]  = {c.S1 * c.N1, 0, c.N1, 1};

    aclTensor *tq = aclCreateTensor(qDims, 4, ACL_FLOAT16, qStr, 0, ACL_FORMAT_ND, qDims, 4, d_q);
    aclTensor *tk = aclCreateTensor(kvDims, 4, ACL_FLOAT16, kvStr, 0, ACL_FORMAT_ND, kvDims, 4, d_k);
    aclTensor *tv = aclCreateTensor(kvDims, 4, ACL_FLOAT16, kvStr, 0, ACL_FORMAT_ND, kvDims, 4, d_v);
    aclTensor *ti = aclCreateTensor(iDims, 4, ACL_INT32, iStr, 0, ACL_FORMAT_ND, iDims, 4, d_idx);
    aclTensor *tqr = aclCreateTensor(rDims, 4, ACL_FLOAT16, rStr, 0, ACL_FORMAT_ND, rDims, 4, d_qr);
    aclTensor *tkr = aclCreateTensor(krDims, 4, ACL_FLOAT16, krStr, 0, ACL_FORMAT_ND, krDims, 4, d_kr);
    aclTensor *tasq = aclCreateTensor(lenDims, 1, ACL_INT32, lenStr, 0, ACL_FORMAT_ND, lenDims, 1, d_asq);
    aclTensor *task = aclCreateTensor(lenDims, 1, ACL_INT32, lenStr, 0, ACL_FORMAT_ND, lenDims, 1, d_ask);
    aclTensor *to = aclCreateTensor(oDims, 4, ACL_FLOAT16, oStr, 0, ACL_FORMAT_ND, oDims, 4, d_out);
    aclTensor *tmax = (c.LSE != 0) ? aclCreateTensor(lseDims, 4, ACL_FLOAT, lseStr, 0, ACL_FORMAT_ND, lseDims, 4, d_max) : nullptr;
    aclTensor *tsum = (c.LSE != 0) ? aclCreateTensor(lseDims, 4, ACL_FLOAT, lseStr, 0, ACL_FORMAT_ND, lseDims, 4, d_sum) : nullptr;

    // ---- 调算子（属性顺序与 OpDef 声明一致）----
    uint64_t wsSize = 0;
    aclOpExecutor *executor = nullptr;
    // scaleValue 是 double 入参（官方：接口传 double，内部按 fp16 精度处理）
    const double scale = (c.scale != 0.0) ? c.scale : (1.0 / std::sqrt(static_cast<double>(c.D)));
    aclnnStatus st = aclnnSparseFlashAttentionGetWorkspaceSize(
        tq, tk, tv, ti, tasq, task, tqr, tkr,
        scale, c.SBS, c.MODE, /*attentionMode=*/2, c.LSE != 0,
        to, tmax, tsum, &wsSize, &executor);
    if (st != ACL_SUCCESS) {
        std::printf("[FAIL] aclnnSparseFlashAttentionGetWorkspaceSize ret=%d\n", static_cast<int>(st));
        return 2;
    }
    void *ws = nullptr;
    if (wsSize > 0) {
        if (aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            std::printf("[FAIL] workspace malloc\n"); return 2;
        }
    }
    st = aclnnSparseFlashAttention(ws, wsSize, executor, stream);
    if (st != ACL_SUCCESS) { std::printf("[FAIL] aclnnSparseFlashAttention ret=%d\n", static_cast<int>(st)); return 2; }

    // ⚠️ 必须同步，否则读到半成品（本项目踩过"父进程读太早"的坑）
    aclError se = aclrtSynchronizeStream(stream);
    std::printf("  stream status=%d\n", static_cast<int>(se));
    if (se != ACL_SUCCESS) { std::printf("[FAIL] SynchronizeStream\n"); return 2; }

    // ---- 回读 + 对拍 ----
    std::vector<uint16_t> got(nq);
    if (!D2H(got.data(), d_out, nq * sz_h)) { std::printf("[FAIL] D2H out\n"); return 2; }

    // 判据与 run.sh 的 grep 对齐：相对误差 max(|got-exp|/max(|exp|,eps))
    const float kEpsAbs = 1e-6f;
    double maxRel = 0.0;
    size_t bad = 0;
    double maxAbs = 0.0;
    for (size_t i = 0; i < nq; ++i) {
        const float e = HalfToFloat(c.expect[i]);
        const float g = HalfToFloat(got[i]);
        const float ad = std::fabs(g - e);
        if (ad > maxAbs) { maxAbs = ad; }
        const float den = (std::fabs(e) > kEpsAbs) ? std::fabs(e) : kEpsAbs;
        const double rel = static_cast<double>(ad) / den;
        if (rel > maxRel) { maxRel = rel; }
        // 阈值：官方容差口径未知，此处用 1e-2 相对（比 run.sh 的历史口径宽松保守）
        if (rel > 1e-2) { ++bad; }
    }
    std::printf("  最大相对误差 = %.6e   最大绝对误差 = %.6e\n", maxRel, maxAbs);
    std::printf("  超差元素 = %zu / %zu\n", bad, nq);

    // ---- LSE 对拍（softmax_max / softmax_sum）----
    bool lse_ok = true;
    if (c.LSE != 0) {
        std::vector<float> gmax(nlse), gsum(nlse);
        if (!D2H(gmax.data(), d_max, nlse * sz_f) || !D2H(gsum.data(), d_sum, nlse * sz_f)) {
            std::printf("[FAIL] D2H lse\n"); return 2;
        }
        size_t bad_max = 0, bad_sum = 0, zero_max = 0;
        double max_rel_max = 0.0;
        for (size_t i = 0; i < nlse; ++i) {
            if (gmax[i] == 0.0f) { ++zero_max; }
            const float em = c.expect_max[i];
            const float den = (std::fabs(em) > kEpsAbs) ? std::fabs(em) : kEpsAbs;
            const double rel = static_cast<double>(std::fabs(gmax[i] - em)) / den;
            if (rel > max_rel_max) { max_rel_max = rel; }
            if (rel > 1e-2) { ++bad_max; }
            const float es = c.expect_sum[i];
            const float dens = (std::fabs(es) > kEpsAbs) ? std::fabs(es) : kEpsAbs;
            if (static_cast<double>(std::fabs(gsum[i] - es)) / dens > 1e-2) { ++bad_sum; }
        }
        std::printf("  LSE: max 最大相对误差 = %.6e  超差 %zu/%zu  (got_max 为 0 的个数 = %zu)\n",
                    max_rel_max, bad_max, nlse, zero_max);
        std::printf("  LSE: sum 超差 %zu/%zu\n", bad_sum, nlse);
        lse_ok = (bad_max == 0 && bad_sum == 0);
    }

    const bool pass = (bad == 0) && lse_ok;
    std::printf("==> %s\n", pass ? "PASS" : "FAIL");

    aclDestroyTensor(tq); aclDestroyTensor(tk); aclDestroyTensor(tv); aclDestroyTensor(ti);
    aclDestroyTensor(tqr); aclDestroyTensor(tkr); aclDestroyTensor(tasq); aclDestroyTensor(task);
    aclDestroyTensor(to);
    if (tmax) { aclDestroyTensor(tmax); }
    if (tsum) { aclDestroyTensor(tsum); }
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return pass ? 0 : 1;
}
