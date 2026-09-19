// SFA 真机耗时测量（多次重复，排除初始化开销）
// 用法: ./bench <case.bin> [重复次数] [devid]
#define _GNU_SOURCE
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include "aclnn_sparse_flash_attention.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <ctime>

#define CHK(expr, msg)                                                        \
    do {                                                                      \
        aclError _e = (expr);                                                 \
        if (_e != ACL_SUCCESS) { printf("[FAIL] %s -> %d\n", msg, (int)_e); return 1; } \
    } while (0)

struct Case {
    long B, S1, S2, N1, D, SBS, COUNT, MODE, LSE;
    double scale;
    std::vector<uint16_t> query, key, value, qrope, krope, expect;
    std::vector<int32_t> idx;
};

static bool load_case(const char* path, Case& c)
{
    FILE* fp = fopen(path, "rb");
    if (!fp) { printf("[FAIL] open %s\n", path); return false; }
    char line[256];
    fgets(line, sizeof(line), fp);
    if (strncmp(line, "SFA_CASE", 8) != 0) { fclose(fp); return false; }
    const char* keys[8] = {"B","S1","S2","N1","D","SBS","COUNT","SCALE"};
    long* dst[7] = {&c.B,&c.S1,&c.S2,&c.N1,&c.D,&c.SBS,&c.COUNT};
    for (int i = 0; i < 8; ++i) {
        fgets(line, sizeof(line), fp);
        char k[32]; double v; sscanf(line, "%31s %lf", k, &v);
        if (!strcmp(k, "SCALE")) { c.scale = v; continue; }
        *dst[i] = (long)v;
    }
    for (int i = 0; i < 2; ++i) {
        fgets(line, sizeof(line), fp);
        char k[32]; long v; sscanf(line, "%31s %ld", k, &v);
        if (!strcmp(k, "MODE")) c.MODE = v; else if (!strcmp(k, "LSE")) c.LSE = v;
    }
    const size_t nQ = (size_t)c.B*c.S1*c.N1*c.D, nK = (size_t)c.B*c.S2*c.D;
    const size_t nR = (size_t)c.B*c.S1*c.N1*64, nKR = (size_t)c.B*c.S2*64;
    const size_t nI = (size_t)c.B*c.S1*c.COUNT;
    c.query.resize(nQ); c.key.resize(nK); c.value.resize(nK);
    c.qrope.resize(nR); c.krope.resize(nKR); c.idx.resize(nI); c.expect.resize(nQ);
    bool ok = true;
    ok &= fread(c.query.data(),2,nQ,fp)==nQ;  ok &= fread(c.key.data(),2,nK,fp)==nK;
    ok &= fread(c.value.data(),2,nK,fp)==nK;  ok &= fread(c.qrope.data(),2,nR,fp)==nR;
    ok &= fread(c.krope.data(),2,nKR,fp)==nKR;ok &= fread(c.idx.data(),4,nI,fp)==nI;
    ok &= fread(c.expect.data(),2,nQ,fp)==nQ;
    fclose(fp);
    return ok;
}

static double now_ms()
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

int main(int argc, char** argv)
{
    if (argc < 2) { printf("usage: %s <case.bin> [reps] [devid]\n", argv[0]); return 1; }
    const int reps = (argc > 2) ? atoi(argv[2]) : 20;
    const int32_t devId = (argc > 3) ? atoi(argv[3]) : 0;
    Case c;
    if (!load_case(argv[1], c)) return 1;

    const int64_t B=c.B,S1=c.S1,S2=c.S2,N1=c.N1,D=c.D,CN=c.COUNT;
    const size_t nQ=(size_t)(B*S1*N1*D), nK=(size_t)(B*S2*D);
    const size_t nR=(size_t)(B*S1*N1*64), nKR=(size_t)(B*S2*64);
    const size_t nI=(size_t)(B*S1*CN), nL=(size_t)(B*S1*N1);

    CHK(aclInit(nullptr), "aclInit");
    CHK(aclrtSetDevice(devId), "setDevice");
    aclrtStream stream=nullptr;
    CHK(aclrtCreateStream(&stream), "createStream");

    void *dq,*dk,*dv,*di,*dqr,*dkr,*dout,*dmax,*dsum;
    CHK(aclrtMalloc(&dq,nQ*2,ACL_MEM_MALLOC_HUGE_FIRST),"mq");
    CHK(aclrtMalloc(&dk,nK*2,ACL_MEM_MALLOC_HUGE_FIRST),"mk");
    CHK(aclrtMalloc(&dv,nK*2,ACL_MEM_MALLOC_HUGE_FIRST),"mv");
    CHK(aclrtMalloc(&di,nI*4,ACL_MEM_MALLOC_HUGE_FIRST),"mi");
    CHK(aclrtMalloc(&dqr,nR*2,ACL_MEM_MALLOC_HUGE_FIRST),"mqr");
    CHK(aclrtMalloc(&dkr,nKR*2,ACL_MEM_MALLOC_HUGE_FIRST),"mkr");
    CHK(aclrtMalloc(&dout,nQ*2,ACL_MEM_MALLOC_HUGE_FIRST),"mo");
    CHK(aclrtMalloc(&dmax,nL*4,ACL_MEM_MALLOC_HUGE_FIRST),"mm");
    CHK(aclrtMalloc(&dsum,nL*4,ACL_MEM_MALLOC_HUGE_FIRST),"ms");
    CHK(aclrtMemcpy(dq,nQ*2,c.query.data(),nQ*2,ACL_MEMCPY_HOST_TO_DEVICE),"h2d q");
    CHK(aclrtMemcpy(dk,nK*2,c.key.data(),nK*2,ACL_MEMCPY_HOST_TO_DEVICE),"h2d k");
    CHK(aclrtMemcpy(dv,nK*2,c.value.data(),nK*2,ACL_MEMCPY_HOST_TO_DEVICE),"h2d v");
    CHK(aclrtMemcpy(di,nI*4,c.idx.data(),nI*4,ACL_MEMCPY_HOST_TO_DEVICE),"h2d i");
    CHK(aclrtMemcpy(dqr,nR*2,c.qrope.data(),nR*2,ACL_MEMCPY_HOST_TO_DEVICE),"h2d qr");
    CHK(aclrtMemcpy(dkr,nKR*2,c.krope.data(),nKR*2,ACL_MEMCPY_HOST_TO_DEVICE),"h2d kr");

    int64_t qD[4]={B,S1,N1,D},        qS[4]={S1*N1*D,N1*D,D,1};
    int64_t kD[4]={B,S2,1,D},         kS[4]={S2*D,D,D,1};
    int64_t iD[4]={B,S1,1,CN},        iS[4]={S1*CN,CN,CN,1};
    int64_t rD[4]={B,S1,N1,64},       rS[4]={S1*N1*64,N1*64,64,1};
    int64_t xD[4]={B,S2,1,64},        xS[4]={S2*64,64,64,1};
    int64_t lD[4]={B,1,S1,N1},        lS[4]={S1*N1,S1*N1,N1,1};

    auto tq=aclCreateTensor(qD,4,ACL_FLOAT16,qS,0,ACL_FORMAT_ND,qD,4,dq);
    auto tk=aclCreateTensor(kD,4,ACL_FLOAT16,kS,0,ACL_FORMAT_ND,kD,4,dk);
    auto tv=aclCreateTensor(kD,4,ACL_FLOAT16,kS,0,ACL_FORMAT_ND,kD,4,dv);
    auto ti=aclCreateTensor(iD,4,ACL_INT32,iS,0,ACL_FORMAT_ND,iD,4,di);
    auto tqr=aclCreateTensor(rD,4,ACL_FLOAT16,rS,0,ACL_FORMAT_ND,rD,4,dqr);
    auto tkr=aclCreateTensor(xD,4,ACL_FLOAT16,xS,0,ACL_FORMAT_ND,xD,4,dkr);
    auto to=aclCreateTensor(qD,4,ACL_FLOAT16,qS,0,ACL_FORMAT_ND,qD,4,dout);
    auto tmx=aclCreateTensor(lD,4,ACL_FLOAT,lS,0,ACL_FORMAT_ND,lD,4,dmax);
    auto tsm=aclCreateTensor(lD,4,ACL_FLOAT,lS,0,ACL_FORMAT_ND,lD,4,dsum);

    uint64_t wsSize=0; aclOpExecutor* ex=nullptr;
    aclnnStatus st = aclnnSparseFlashAttentionGetWorkspaceSize(
        tq,tk,tv,ti,nullptr,nullptr,tqr,tkr,c.scale,(int64_t)c.SBS,(int64_t)c.MODE,2,
        (c.LSE!=0),to,tmx,tsm,&wsSize,&ex);
    if (st!=0) { printf("[FAIL] GetWorkspaceSize %d\n",(int)st); return 1; }
    void* ws=nullptr;
    if (wsSize>0) CHK(aclrtMalloc(&ws,wsSize,ACL_MEM_MALLOC_HUGE_FIRST),"ws");

    // 预热
    for (int i=0;i<3;i++){ aclnnSparseFlashAttention(ws,wsSize,ex,stream); }
    CHK(aclrtSynchronizeStream(stream),"warmup sync");

    // 逐次计时
    std::vector<double> ts;
    for (int i=0;i<reps;i++){
        double t0=now_ms();
        aclnnSparseFlashAttention(ws,wsSize,ex,stream);
        aclrtSynchronizeStream(stream);
        ts.push_back(now_ms()-t0);
    }
    double sum=0,mn=1e18,mx=0;
    for (double t:ts){ sum+=t; if(t<mn)mn=t; if(t>mx)mx=t; }
    const double avg = sum/reps;

    printf("B=%ld S1=%ld S2=%ld N1=%ld SBS=%ld MODE=%ld | reps=%d\n",
           c.B,c.S1,c.S2,c.N1,c.SBS,c.MODE,reps);
    printf("  平均 = %.4f ms   最小 = %.4f ms   最大 = %.4f ms\n", avg, mn, mx);

    // 粗略算力估算（MAC 数）
    // 每个 (b,s) 行：N1 头 × m_token × 576 次 MAC + N1 × m_token × 512 次 MAC
    double macs = 0;
    {
        // 从 idx 统计每行有效 token 数
        for (long b=0;b<c.B;b++) for (long s=0;s<c.S1;s++){
            long base=(b*c.S1+s)*c.COUNT; long toks=0;
            for (long i=0;i<c.COUNT;i++){
                int32_t v=c.idx[base+i];
                if (v<0) break;
                long begin=(long)v*c.SBS;
                long thr = (c.MODE==3) ? (c.S2 - c.S1 + s + 1) : c.S2;
                if (begin>=thr) continue;
                long e=begin+c.SBS; if(e>thr)e=thr;
                toks += (e-begin);
            }
            macs += (double)c.N1*toks*(576.0+512.0);
        }
    }
    printf("  估算 MAC = %.3e   ->  %.2f GFLOP/s (按 2*MAC)\n",
           macs, 2.0*macs/(avg*1e6));
    printf("=== DONE ===\n");
    return 0;
}
