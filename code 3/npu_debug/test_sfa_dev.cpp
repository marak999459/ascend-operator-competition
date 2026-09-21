// SFA 真机开发用 harness：对拍(allclose) + 逐位 golden 差分 + 计时
//
//   与 refs/sfa/test_sfa_real.cpp 的区别（为什么另建一份而不改 refs）：
//     1) 判据用 torch 口径 allclose(|g-e| <= atol + rtol*|e|)，而不是
//        rel = |g-e|/max(|e|,1e-6) > 1e-2 —— 后者在 |e|≈0 的 fp16 输出上
//        会把 1 个 ULP 判成"超差 2.0"，噪声盖过真回归。
//     2) 能把一次运行的输出落成 golden 二进制，下一版**逐位**比对。
//        这是 P1「只换搬运、数值不变」唯一的硬证据形式。
//     3) 带 reps 计时，直接给 ms / GFLOP/s。
//
// 用法: test_sfa_dev <case.bin> [reps=1] [act=diff|write|none]
//   act=write  把本次输出写进 golden 目录（建立基线）
//   act=diff   与 golden 逐位比对（golden 缺失则跳过并提示）
//   act=none   只对拍 + 计时
// 退出码: 0=PASS 1=FAIL 2=用法/加载错
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <string>
#include <vector>

#include <sys/stat.h>

#include "acl/acl.h"
#include "aclnn_sparse_flash_attention.h"

static float HalfToFloat(uint16_t h)
{
    const uint32_t sign = static_cast<uint32_t>(h >> 15) & 0x1u;
    const uint32_t exp  = static_cast<uint32_t>(h >> 10) & 0x1Fu;
    const uint32_t man  = static_cast<uint32_t>(h) & 0x3FFu;
    float v;
    if (exp == 0)      { v = (man == 0) ? 0.0f : std::ldexp(static_cast<float>(man), -24); }
    else if (exp == 31){ v = 0.0f; }
    else               { v = (1.0f + static_cast<float>(man) / 1024.0f) * std::ldexp(1.0f, static_cast<int>(exp) - 15); }
    return sign ? -v : v;
}

struct Case {
    long B = 0, S1 = 0, S2 = 0, N1 = 0, D = 512;
    long SBS = 1, COUNT = 2048, MODE = 3, LSE = 1;
    double scale = 0.0;
    std::vector<uint16_t> query, key, value, qrope, krope, expect;
    std::vector<int32_t> idx, asq, ask;
    std::vector<float> expect_max, expect_sum;
    std::string name;
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
    return n == 0 || std::fread(v.data(), sizeof(T), n, fp) == n;
}

static bool LoadCase(const char *path, Case &c)
{
    FILE *fp = std::fopen(path, "rb");
    if (!fp) { std::printf("[FAIL] cannot open %s\n", path); return false; }
    std::string line;
    if (!ReadLine(fp, line) || line.compare(0, 8, "SFA_CASE") != 0) {
        std::printf("[FAIL] bad magic\n"); std::fclose(fp); return false;
    }
    const char *keys[10] = {"B", "S1", "S2", "N1", "D", "SBS", "COUNT", "SCALE", "MODE", "LSE"};
    for (int i = 0; i < 10; ++i) {
        if (!ReadLine(fp, line)) { std::printf("[FAIL] header truncated at %s\n", keys[i]); std::fclose(fp); return false; }
        char k[64] = {0}, v[128] = {0};
        if (std::sscanf(line.c_str(), "%63s %127s", k, v) != 2) { std::printf("[FAIL] bad line %s\n", line.c_str()); std::fclose(fp); return false; }
        const std::string ks(k);
        if      (ks == "S1")  c.S1  = std::atol(v);
        else if (ks == "S2")  c.S2  = std::atol(v);
        else if (ks == "N1")  c.N1  = std::atol(v);
        else if (ks == "D")   c.D   = std::atol(v);
        else if (ks == "SBS") c.SBS = std::atol(v);
        else if (ks == "COUNT") c.COUNT = std::atol(v);
        else if (ks == "SCALE") c.scale = std::atof(v);
        else if (ks == "MODE")  c.MODE  = std::atol(v);
        else if (ks == "LSE")   c.LSE   = std::atol(v);
        else                    c.B     = std::atol(v);
    }
    const size_t nq   = static_cast<size_t>(c.B) * c.S1 * c.N1 * c.D;
    const size_t nkv  = static_cast<size_t>(c.B) * c.S2 * c.D;
    const size_t nrp  = static_cast<size_t>(c.B) * c.S1 * c.N1 * 64;
    const size_t nkrp = static_cast<size_t>(c.B) * c.S2 * 64;
    const size_t nidx = static_cast<size_t>(c.B) * c.S1 * c.COUNT;
    const size_t nlen = static_cast<size_t>(c.B);
    const size_t nlse = static_cast<size_t>(c.B) * c.S1 * c.N1;
    const size_t want = nq*2 + nkv*2 + nkv*2 + nrp*2 + nkrp*2 + nidx*4 + nlen*4 + nlen*4 + nq*2
                      + (c.LSE ? (nlse*4 + nlse*4) : 0u);
    const long pos = std::ftell(fp);
    std::fseek(fp, 0, SEEK_END);
    const long have = std::ftell(fp) - pos;
    std::fseek(fp, pos, SEEK_SET);
    if (static_cast<size_t>(have) != want) {
        std::printf("[FAIL] payload 布局不吻合：剩余 %ld B，应为 %zu B → 重新生成用例\n", have, want);
        std::fclose(fp); return false;
    }
    bool ok = ReadVec(fp, c.query, nq) && ReadVec(fp, c.key, nkv) && ReadVec(fp, c.value, nkv)
           && ReadVec(fp, c.qrope, nrp) && ReadVec(fp, c.krope, nkrp) && ReadVec(fp, c.idx, nidx)
           && ReadVec(fp, c.asq, nlen) && ReadVec(fp, c.ask, nlen) && ReadVec(fp, c.expect, nq);
    if (ok && c.LSE) { ok = ReadVec(fp, c.expect_max, nlse) && ReadVec(fp, c.expect_sum, nlse); }
    std::fclose(fp);
    if (!ok) { std::printf("[FAIL] payload 读取失败\n"); return false; }
    const std::string p(path);
    c.name = p.substr(p.find_last_of('/') + 1);
    const size_t dot = c.name.find_last_of('.');
    if (dot != std::string::npos) { c.name.erase(dot); }
    return true;
}

// ---------------- golden ----------------
static std::string Gdir;
static bool Io(const std::string &f, const void *p, size_t bytes, bool write)
{
    if (write) {
        FILE *fp = std::fopen(f.c_str(), "wb");
        if (!fp) { return false; }
        const size_t w = std::fwrite(p, 1, bytes, fp);
        std::fclose(fp);
        return w == bytes;
    }
    std::vector<uint8_t> buf(bytes);
    FILE *fp = std::fopen(f.c_str(), "rb");
    if (!fp) { std::printf("  golden 缺失 %s → 先跑 act=write\n", f.c_str()); return false; }
    const size_t r = std::fread(buf.data(), 1, bytes, fp);
    std::fclose(fp);
    if (r != bytes) { std::printf("  golden 长度不符 %s (%zu vs %zu)\n", f.c_str(), r, bytes); return false; }
    if (std::memcmp(buf.data(), p, bytes) != 0) {
        size_t first = 0, ndiff = 0;
        for (size_t i = 0; i < bytes; ++i) {
            if (buf[i] != static_cast<const uint8_t *>(p)[i]) { if (ndiff == 0) { first = i; } ++ndiff; }
        }
        std::printf("  ⚠️ 与 golden **不逐位一致**：%s 有 %zu/%zu 字节不同，首个在 byte %zu\n",
                    f.c_str(), ndiff, bytes, first);
        return false;
    }
    std::printf("  逐位一致 %s (%zu B)\n", f.c_str(), bytes);
    return true;
}

static void MkdirP(const std::string &d)
{
    std::string cur;
    for (size_t i = 0; i < d.size(); ++i) {
        cur += d[i];
        if (d[i] == '/' && i + 1 < d.size()) { ::mkdir(cur.c_str(), 0775); }
    }
    ::mkdir(d.c_str(), 0775);
}

// allclose：|g-e| <= atol + rtol*|e|
struct Stat { size_t bad = 0; double maxAbs = 0.0; double maxAbsAt = 0.0; };
static Stat CmpF(const float *g, const float *e, size_t n, double atol, double rtol)
{
    Stat s;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs(static_cast<double>(g[i]) - static_cast<double>(e[i]));
        if (d > s.maxAbs) { s.maxAbs = d; s.maxAbsAt = std::fabs(e[i]); }
        if (d > atol + rtol * std::fabs(e[i])) { ++s.bad; }
    }
    return s;
}
static Stat CmpH(const uint16_t *g, const uint16_t *e, size_t n, double atol, double rtol)
{
    Stat s;
    for (size_t i = 0; i < n; ++i) {
        const double a = HalfToFloat(g[i]), b = HalfToFloat(e[i]);
        const double d = std::fabs(a - b);
        if (d > s.maxAbs) { s.maxAbs = d; s.maxAbsAt = std::fabs(b); }
        if (d > atol + rtol * std::fabs(b)) { ++s.bad; }
    }
    return s;
}

// OpDef 的输入 dtype 列表是 {fp16, fp32}，但用例文件与真机矩阵一直是 fp16 ⇒
// SFA_F32=1 把同一批用例的输入按 float 送进去，专门走 DT_QUERY=float 那条模板实例。
static bool F32In() { const char *e = std::getenv("SFA_F32"); return e != nullptr && e[0] == '1'; }
static std::vector<float> ToF(const std::vector<uint16_t> &h)
{
    std::vector<float> f(h.size());
    for (size_t i = 0; i < h.size(); ++i) { f[i] = HalfToFloat(h[i]); }
    return f;
}

int main(int argc, char **argv)
{
    if (argc < 2) { std::printf("usage: %s <case.bin> [reps] [write|diff|none] [atol] [rtol] [golddir]\n", argv[0]); return 2; }
    Case c;
    if (!LoadCase(argv[1], c)) { return 2; }
    const int reps = (argc > 2) ? std::atoi(argv[2]) : 1;
    const char *act = (argc > 3) ? argv[3] : "diff";
    const double atol = (argc > 4) ? std::atof(argv[4]) : 2e-3;   // fp16 在 1.0 附近 ULP≈9.8e-4
    const double rtol = (argc > 5) ? std::atof(argv[5]) : 1e-2;
    Gdir = (argc > 6) ? argv[6] : "golden";
    MkdirP(Gdir);

    const bool   f32  = F32In();
    const size_t esz  = f32 ? 4u : 2u;
    const aclDataType dtq = f32 ? ACL_FLOAT : ACL_FLOAT16;
    std::vector<float> fq, fk, fv, fqr, fkr, fexp;
    if (f32) {
        fq = ToF(c.query); fk = ToF(c.key); fv = ToF(c.value);
        fqr = ToF(c.qrope); fkr = ToF(c.krope); fexp = ToF(c.expect);
    }

    std::printf("case=%s B=%ld S1=%ld S2=%ld N1=%ld D=%ld SBS=%ld COUNT=%ld MODE=%ld LSE=%ld\n",
                c.name.c_str(), c.B, c.S1, c.S2, c.N1, c.D, c.SBS, c.COUNT, c.MODE, c.LSE);

    if (aclInit(nullptr) != ACL_SUCCESS) { std::printf("[FAIL] aclInit\n"); return 2; }
    if (aclrtSetDevice(0) != ACL_SUCCESS) { std::printf("[FAIL] aclrtSetDevice\n"); return 2; }
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) { std::printf("[FAIL] createStream\n"); return 2; }

    const size_t nq   = static_cast<size_t>(c.B) * c.S1 * c.N1 * c.D;
    const size_t nkv  = static_cast<size_t>(c.B) * c.S2 * c.D;
    const size_t nrp  = static_cast<size_t>(c.B) * c.S1 * c.N1 * 64;
    const size_t nkrp = static_cast<size_t>(c.B) * c.S2 * 64;
    const size_t nidx = static_cast<size_t>(c.B) * c.S1 * c.COUNT;
    const size_t nlen = static_cast<size_t>(c.B);
    const size_t nlse = static_cast<size_t>(c.B) * c.S1 * c.N1;

    void *d_q, *d_k, *d_v, *d_idx, *d_qr, *d_kr, *d_asq, *d_ask, *d_out, *d_max, *d_sum;
    bool ok = aclrtMalloc(&d_q, nq*esz, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS
           && aclrtMalloc(&d_k, nkv*esz, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS
           && aclrtMalloc(&d_v, nkv*esz, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS
           && aclrtMalloc(&d_idx, nidx*4, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS
           && aclrtMalloc(&d_qr, nrp*esz, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS
           && aclrtMalloc(&d_kr, nkrp*esz, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS
           && aclrtMalloc(&d_asq, nlen*4, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS
           && aclrtMalloc(&d_ask, nlen*4, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS
           && aclrtMalloc(&d_out, nq*esz, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS
           && aclrtMalloc(&d_max, nlse*4, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS
           && aclrtMalloc(&d_sum, nlse*4, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    if (!ok) { std::printf("[FAIL] malloc\n"); return 2; }
    auto H2D = [](void *d, const void *h, size_t b) { return aclrtMemcpy(d, b, h, b, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS; };
    const void *hq = f32 ? (const void *)fq.data()  : (const void *)c.query.data();
    const void *hk = f32 ? (const void *)fk.data()  : (const void *)c.key.data();
    const void *hv = f32 ? (const void *)fv.data()  : (const void *)c.value.data();
    const void *hqr = f32 ? (const void *)fqr.data() : (const void *)c.qrope.data();
    const void *hkr = f32 ? (const void *)fkr.data() : (const void *)c.krope.data();
    ok = H2D(d_q, hq, nq*esz) && H2D(d_k, hk, nkv*esz) && H2D(d_v, hv, nkv*esz)
      && H2D(d_idx, c.idx.data(), nidx*4) && H2D(d_qr, hqr, nrp*esz) && H2D(d_kr, hkr, nkrp*esz)
      && H2D(d_asq, c.asq.data(), nlen*4) && H2D(d_ask, c.ask.data(), nlen*4);
    if (!ok) { std::printf("[FAIL] H2D\n"); return 2; }

    int64_t qDims[4]  = {c.B, c.S1, c.N1, c.D};
    int64_t kvDims[4] = {c.B, c.S2, 1, c.D};
    int64_t rDims[4]  = {c.B, c.S1, c.N1, 64};
    int64_t krDims[4] = {c.B, c.S2, 1, 64};
    int64_t iDims[4]  = {c.B, c.S1, 1, c.COUNT};
    int64_t lenDims[1] = {c.B};
    int64_t oDims[4]  = {c.B, c.S1, c.N1, c.D};
    int64_t qStr[4]  = {c.S1*c.N1*c.D, c.N1*c.D, c.D, 1};
    int64_t kvStr[4] = {c.S2*c.D, c.D, c.D, 1};
    int64_t rStr[4]  = {c.S1*c.N1*64, c.N1*64, 64, 1};
    int64_t krStr[4] = {c.S2*64, 64, 64, 1};
    int64_t iStr[4]  = {c.S1*c.COUNT, c.COUNT, c.COUNT, 1};
    int64_t lenStr[1] = {1};
    int64_t oStr[4]  = {c.S1*c.N1*c.D, c.N1*c.D, c.D, 1};
    int64_t lseDims[4] = {c.B, 1, c.S1, c.N1};
    int64_t lseStr[4]  = {c.S1*c.N1, 0, c.N1, 1};

    aclTensor *tq  = aclCreateTensor(qDims, 4, dtq, qStr, 0, ACL_FORMAT_ND, qDims, 4, d_q);
    aclTensor *tk  = aclCreateTensor(kvDims, 4, dtq, kvStr, 0, ACL_FORMAT_ND, kvDims, 4, d_k);
    aclTensor *tv  = aclCreateTensor(kvDims, 4, dtq, kvStr, 0, ACL_FORMAT_ND, kvDims, 4, d_v);
    aclTensor *ti  = aclCreateTensor(iDims, 4, ACL_INT32, iStr, 0, ACL_FORMAT_ND, iDims, 4, d_idx);
    aclTensor *tqr = aclCreateTensor(rDims, 4, dtq, rStr, 0, ACL_FORMAT_ND, rDims, 4, d_qr);
    aclTensor *tkr = aclCreateTensor(krDims, 4, dtq, krStr, 0, ACL_FORMAT_ND, krDims, 4, d_kr);
    aclTensor *tasq = aclCreateTensor(lenDims, 1, ACL_INT32, lenStr, 0, ACL_FORMAT_ND, lenDims, 1, d_asq);
    aclTensor *task = aclCreateTensor(lenDims, 1, ACL_INT32, lenStr, 0, ACL_FORMAT_ND, lenDims, 1, d_ask);
    aclTensor *to   = aclCreateTensor(oDims, 4, dtq, oStr, 0, ACL_FORMAT_ND, oDims, 4, d_out);
    aclTensor *tmax = aclCreateTensor(lseDims, 4, ACL_FLOAT, lseStr, 0, ACL_FORMAT_ND, lseDims, 4, d_max);
    aclTensor *tsum = aclCreateTensor(lseDims, 4, ACL_FLOAT, lseStr, 0, ACL_FORMAT_ND, lseDims, 4, d_sum);

    const double scale = (c.scale != 0.0) ? c.scale : (1.0 / std::sqrt(static_cast<double>(c.D)));
    uint64_t wsSize = 0;
    aclOpExecutor *ex = nullptr;
    if (aclnnSparseFlashAttentionGetWorkspaceSize(tq, tk, tv, ti, tasq, task, tqr, tkr,
            scale, c.SBS, c.MODE, 2, c.LSE != 0, to, tmax, tsum, &wsSize, &ex) != ACL_SUCCESS) {
        std::printf("[FAIL] GetWorkspaceSize（tiling/so 版本不匹配？）\n"); return 2;
    }
    void *ws = nullptr;
    if (wsSize > 0 && aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) { std::printf("[FAIL] ws\n"); return 2; }

    auto RunOnce = [&]() { return aclnnSparseFlashAttention(ws, wsSize, ex, stream) == ACL_SUCCESS; };
    if (!RunOnce()) { std::printf("[FAIL] launch\n"); return 2; }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) { std::printf("[FAIL] sync（kernel 挂了？）\n"); return 2; }

    double avg = 0.0, mn = 1e30, mx = 0.0;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!RunOnce()) { std::printf("[FAIL] launch rep%d\n", r); return 2; }
        if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) { std::printf("[FAIL] sync rep%d\n", r); return 2; }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        avg += ms; if (ms < mn) { mn = ms; } if (ms > mx) { mx = ms; }
    }
    if (reps > 0) { avg /= reps; }

    // ---- 批量发射计时：连发 BATCH 次、只在最后 sync 一次 ----
    // 上面那个"每次 launch + 单独 sync"的墙钟里混着 host 侧 launch/同步开销，小用例
    // （r1_min 只有 512 个输出元素）也要 27 µs ⇒ 分不清是 kernel 慢还是 launch 慢。
    // 连发把固定开销摊掉：若平均显著下降 ⇒ 该用例是 launch-bound，kernel 时间以本行为准。
    {
        constexpr int BATCH = 20;
        double bsum = 0.0, bmn = 1e30, bmx = 0.0;
        constexpr int WARNS = 3;
        for (int w = 0; w < WARNS; ++w) {
            const auto t0 = std::chrono::steady_clock::now();
            bool ok = true;
            for (int r = 0; r < BATCH && ok; ++r) { ok = RunOnce(); }
            if (!ok) { std::printf("[FAIL] batch launch\n"); return 2; }
            if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) { std::printf("[FAIL] batch sync\n"); return 2; }
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0).count() / BATCH;
            bsum += ms; if (ms < bmn) { bmn = ms; } if (ms > bmx) { bmx = ms; }
        }
        std::printf("  批量口径(连发%d只sync一次, 摊薄 launch): 平均 %.4f ms  最小 %.4f  最大 %.4f"
                    "   [单发/批量 = %.2f]\n", BATCH, bsum / WARNS, bmn, bmx,
                    (bsum / WARNS > 0) ? avg / (bsum / WARNS) : 0.0);
    }

    std::vector<uint8_t> got(nq*esz);
    std::vector<float> gmax(nlse), gsum(nlse);
    if (aclrtMemcpy(got.data(), nq*esz, d_out, nq*esz, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(gmax.data(), nlse*4, d_max, nlse*4, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(gsum.data(), nlse*4, d_sum, nlse*4, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        std::printf("[FAIL] D2H\n"); return 2;
    }

    Stat so;
    if (f32) {
        so = CmpF(reinterpret_cast<const float *>(got.data()), fexp.data(), nq, atol, rtol);
    } else {
        so = CmpH(reinterpret_cast<const uint16_t *>(got.data()), c.expect.data(), nq, atol, rtol);
    }
    std::printf("  out : dtype=%s 超差 %zu/%zu  maxAbs=%.3e (该处|exp|=%.3e)\n",
                f32 ? "fp32" : "fp16", so.bad, nq, so.maxAbs, so.maxAbsAt);
    bool pass = (so.bad == 0);
    if (c.LSE) {
        const Stat sm = CmpF(gmax.data(), c.expect_max.data(), nlse, atol, rtol);
        const Stat ss = CmpF(gsum.data(), c.expect_sum.data(), nlse, atol, rtol);
        std::printf("  LSE : max 超差 %zu/%zu maxAbs=%.3e | sum 超差 %zu/%zu maxAbs=%.3e\n",
                    sm.bad, nlse, sm.maxAbs, ss.bad, nlse, ss.maxAbs);
        if (sm.bad || ss.bad) { pass = false; }
    }

    const std::string base = Gdir + "/" + c.name + (f32 ? ".f32" : "");
    if (std::strcmp(act, "write") == 0) {
        MkdirP(Gdir);
        bool w = Io(base + ".out", got.data(), nq*esz, true)
              && Io(base + ".max", gmax.data(), nlse*4, true)
              && Io(base + ".sum", gsum.data(), nlse*4, true);
        std::printf("  golden 写入 %s: %s\n", base.c_str(), w ? "OK" : "FAIL");
        if (!w) { pass = false; }
    } else if (std::strcmp(act, "diff") == 0) {
        bool g1 = Io(base + ".out", got.data(), nq*esz, false);
        bool g2 = Io(base + ".max", gmax.data(), nlse*4, false);
        bool g3 = Io(base + ".sum", gsum.data(), nlse*4, false);
        if (!g1 || !g2 || !g3) { pass = false; }
    }

    if (reps > 0 && avg > 0) {
        const double macs = static_cast<double>(c.B) * c.S1 * c.N1 *
                            static_cast<double>(c.SBS) * std::min(c.COUNT, c.S2) * (c.D + 64.0 + c.D);
        std::printf("  时间: 平均 %.4f ms  最小 %.4f  最大 %.4f  (reps=%d)\n", avg, mn, mx, reps);
        std::printf("  量级: ~%.3e MAC → %.2f GFLOP/s（按 2*MAC，粗口径）\n", macs, 2.0*macs/(avg*1e6));
    }
    std::printf("==> %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
