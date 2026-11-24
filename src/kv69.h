/* kv69.h -- the HVX conversions v73 brought, made for v69 (Snapdragon 8 Gen 1) from integer operations alone (qfloat's add and
 * subtract are not IEEE-exact: magic numbers through them came out 1 off). On v73 and later these stand unused.
 * The exponent from the count of leading zeros, the significand moved by per-lane shifts (vasl with a negative count shifts right).
 *   Q6_Vhf_equals_Vh   int16 -> fp16, exact for |x| < 2048 (the engine's uses: dequantized weights, -128..127)
 *   Q6_Vh_equals_Vhf   fp16 -> int16, rounded half up, |x| < 32768 (the softmax's t * 1024 down to -15360)
 *   Q6_Vsf_equals_Vw   int32 -> fp32, exact for |x| < 2^24
 *   Q6_Vw_equals_Vsf   fp32 -> int32, rounded half up, |x| < 2^31 */
#ifndef KV69_H
#define KV69_H
static inline HVX_Vector k69_hf_from_h(HVX_Vector x)
{
    const HVX_Vector a = Q6_Vh_vabs_Vh(x), lz = Q6_Vuh_vcl0_Vuh(a), m = Q6_Vh_vasl_VhVh(a, lz);    /* the top bit at 15 */
    const HVX_Vector man = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(m, 5), Q6_Vh_vsplat_R(0x3FF)), ex = Q6_Vh_vasl_VhR(Q6_Vh_vsub_VhVh(Q6_Vh_vsplat_R(30), lz), 10);
    const HVX_Vector sg = Q6_V_vand_VV(x, Q6_Vh_vsplat_R((short)0x8000)), v = Q6_V_vor_VV(sg, Q6_V_vor_VV(ex, man));
    return Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VhVh(a, Q6_V_vzero()), Q6_V_vzero(), v);
}
static inline HVX_Vector k69_h_from_hf(HVX_Vector b)
{
    const HVX_Vector e = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(b, 10), Q6_Vh_vsplat_R(31)), m = Q6_V_vor_VV(Q6_V_vand_VV(b, Q6_Vh_vsplat_R(0x3FF)), Q6_Vh_vsplat_R(0x400));
    const HVX_Vector k = Q6_Vh_vmax_VhVh(Q6_Vh_vsub_VhVh(e, Q6_Vh_vsplat_R(25)), Q6_Vh_vsplat_R(-15));             /* the shift left, -15..4 */
    const HVX_Vector rb = Q6_Vh_vasl_VhVh(Q6_Vh_vsplat_R(1), Q6_Vh_vsub_VhVh(Q6_Vh_vsub_VhVh(Q6_V_vzero(), k), Q6_Vh_vsplat_R(1)));   /* 1 << (-k - 1), 0 for k >= 0 */
    const HVX_Vector r0 = Q6_Vh_vasl_VhVh(Q6_Vh_vadd_VhVh(m, Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhVh(k, Q6_Vh_vsplat_R(-1)), Q6_V_vzero(), rb)), k);
    const HVX_Vector r = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VhVh(e, Q6_V_vzero()), Q6_V_vzero(), r0);                      /* zero and subnormals: 0 */
    return Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhVh(Q6_V_vzero(), b), Q6_Vh_vsub_VhVh(Q6_V_vzero(), r), r);                  /* the sign bit */
}
static inline HVX_Vector k69_sf_from_w(HVX_Vector x)
{
    const HVX_Vector a = Q6_Vw_vabs_Vw(x), lz = Q6_Vuw_vcl0_Vuw(a), m = Q6_Vw_vasl_VwVw(a, lz);     /* the top bit at 31 */
    const HVX_Vector man = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(m, 8), Q6_V_vsplat_R(0x7FFFFF)), ex = Q6_Vw_vasl_VwR(Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(158), lz), 23);
    const HVX_Vector sg = Q6_V_vand_VV(x, Q6_V_vsplat_R((int)0x80000000)), v = Q6_V_vor_VV(sg, Q6_V_vor_VV(ex, man));
    return Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(a, Q6_V_vzero()), Q6_V_vzero(), v);
}
static inline HVX_Vector k69_w_from_sf(HVX_Vector b)
{
    const HVX_Vector e = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(b, 23), Q6_V_vsplat_R(255)), m = Q6_V_vor_VV(Q6_V_vand_VV(b, Q6_V_vsplat_R(0x7FFFFF)), Q6_V_vsplat_R(0x800000));
    const HVX_Vector k = Q6_Vw_vmax_VwVw(Q6_Vw_vsub_VwVw(e, Q6_V_vsplat_R(150)), Q6_V_vsplat_R(-31));               /* the shift left, -31..7 */
    const HVX_Vector rb = Q6_Vw_vasl_VwVw(Q6_V_vsplat_R(1), Q6_Vw_vsub_VwVw(Q6_Vw_vsub_VwVw(Q6_V_vzero(), k), Q6_V_vsplat_R(1)));
    const HVX_Vector r0 = Q6_Vw_vasl_VwVw(Q6_Vw_vadd_VwVw(m, Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(k, Q6_V_vsplat_R(-1)), Q6_V_vzero(), rb)), k);
    const HVX_Vector r = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(e, Q6_V_vzero()), Q6_V_vzero(), r0);
    return Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), b), Q6_Vw_vsub_VwVw(Q6_V_vzero(), r), r);
}
#if __HVX_ARCH__ < 73   /* the replacements are always built, used in place of v73's only below it */
#define Q6_Vhf_equals_Vh(x) k69_hf_from_h(x)
#define Q6_Vh_equals_Vhf(x) k69_h_from_hf(x)
#define Q6_Vsf_equals_Vw(x) k69_sf_from_w(x)
#define Q6_Vw_equals_Vsf(x) k69_w_from_sf(x)
#endif
#if __HEXAGON_ARCH__ < 73
/* the scalar float <-> fp16 conversions the compiler calls below v73 (v73 on has instructions for them), absent from v69's DSP runtime
 * ("undefined symbol PLT __truncsfhf2"): IEEE, round to nearest even, as compiler-rt's */
__attribute__((visibility("default"))) uint16_t __truncsfhf2(float f)
{
    uint32_t x; __builtin_memcpy(&x, &f, 4); const uint32_t s = (x >> 16) & 0x8000; int e = (int)((x >> 23) & 0xFF) - 127 + 15; uint32_t m = x & 0x7FFFFF;
    if (((x >> 23) & 0xFF) == 0xFF) return (uint16_t)(s | 0x7C00 | (m ? 0x200 : 0));                        /* inf, nan */
    if (e >= 31) return (uint16_t)(s | 0x7C00);                                                                /* overflow */
    if (e <= 0)
    {   /* subnormal or zero */
        if (e < -10) return (uint16_t)s;
        m |= 0x800000; const int sh = 14 - e; uint32_t h = m >> sh; const uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (h & 1))) h++;
        return (uint16_t)(s | h);
    }
    uint32_t h = ((uint32_t)e << 10) | (m >> 13); const uint32_t rem = m & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;                                                      /* may carry into the exponent: fine */
    return (uint16_t)(s | h);
}
__attribute__((visibility("default"))) float __extendhfsf2(uint16_t h)
{
    const uint32_t s = (uint32_t)(h & 0x8000) << 16; uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF, x;
    if (e == 31) x = s | 0x7F800000 | (m << 13);
    else if (e) x = s | ((e + 112) << 23) | (m << 13);
    else if (!m) x = s;
    else { e = 113; while (!(m & 0x400)) { m <<= 1; e--; } x = s | (e << 23) | ((m & 0x3FF) << 13); }
    float f; __builtin_memcpy(&f, &x, 4); return f;
}
#endif
#endif
