// [PROBE P1] 单核受控裁定 arch22 上 DataCopyExtParams 的字段口径（code1.md §14.5 疑案）
// 替换 op_kernel/mhc_expand.cpp 里 ForwardOneBlock 的逐副本 for k 循环。
// 口径假设只有三种：dst 侧字段 = 块间隔(字节) / 块首距(字节) / 块首距(32B 块)。
// 让 m 取值决定填进去的数字，使得**恰好一种**解释会把 m 个副本写到正确位置：
//   m=2 -> 字段 0        ：只有"gap/字节"成立时 stride=rb（正确）；"stride=字节"解释成 0 ⇒ 第二副本没写
//   m=3 -> 字段 rb(字节) ：只有"stride/字节"成立时正确；"gap"解释 ⇒ 每副本间隔 2rb（越界进下一行）
//   m=4 -> 字段 rb/32    ：只有"stride/32B 块"成立时正确；"字节"解释 ⇒ 16 字节互相覆盖
// 三者互斥 ⇒ 只有一条会 PASS，PASS 的那条直接给出单位。m=8 走原逐副本循环 = 正对照。
        const uint32_t mm = tiling_.m;
        const uint32_t rb = cur_h * elem_size_;
        const bool one_shot = (tiling_.dTileNum == 1) && (k_limit == mm) && k_begin == 0 &&
                              (mm == 2 || mm == 3 || mm == 4);
        if (one_shot) {
            const int64_t dst0 = static_cast<int64_t>(i) * mm * tiling_.D + jt * tiling_.dTileLen;
            const uint32_t ds = (mm == 2) ? 0u : (mm == 3) ? rb : (rb / 32u);
            DataCopyExtParams pb{static_cast<uint16_t>(mm), rb, 0u, ds, 0u};
            DataCopyPad(o_gm_[dst0], x_local, pb);
        } else {
            for (uint32_t k = k_begin; k < k_end; ++k) {
                const int64_t dst_off = static_cast<int64_t>(i) * tiling_.m * tiling_.D +
                                        static_cast<int64_t>(k) * tiling_.D + jt * tiling_.dTileLen;
                DataCopyPad(o_gm_[dst_off], x_local, cp);
            }
        }
