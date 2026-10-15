// Kai: GGUF Q2_0 experts (--moe-expert-gguf), behind the MoeFormat interface of format.h.
#include <cstring>
#include <stdexcept>
#include <vector>

#include "../format.h"

namespace freetoken::cpu_moe {
namespace {

// -------------------------------- Q2_0 (W2A8) --------------------------------
// ggml Q2_0 (2026): per-64 block = fp16 scale d + 16 bytes; byte j holds elements 4j..4j+3,
// element 4j+k in bits 2k..2k+1, and w = (code - 1) * d (codes 0..3 -> -1, 0, +1, +2).
// W2A8 like the Q4_0 path: activations pre-quantized to Q8_0 (per-32 int8 + fp32 scale).
// Shifting and masking the 16 bytes yields the 2-bit PLANES: plane k = the 16 elements
// 4j+k, j = 0..15, a stride-4 subset of the block. So the activations of every 64-block are
// permuted ONCE per token / route into plane order (permute_q8_planes64: elem 4j+k -> slot
// 16k+j) and the kernels read them contiguously. The block's two Q8_0 scales (elements 0..31
// = j < 8 of every plane, 32..63 = j >= 8) split each plane in half: over the 32 bytes of
// planes (k, k+1) the 8 int32 lanes of the integer dot see the scales [s0 s0 s1 s1 s0 s0 s1 s1].
// The codes are fed to the u8 x s8 dot as they are (0..3); the "- 1" of every code is the
// block's activation sum, which is taken once per token / route as the same 8 int32 lanes
// (q2_lane_sums64: ``alanes``) and subtracted from the lane dots before the scales apply.
using q2dot_fn = float (*)(const uint8_t*, const int8_t*, const float*, const int32_t*, int);

// Quantize a bf16 activation row to Q8_0 (llama.cpp): per-32-block symmetric int8 in natural
// order + one fp32 scale (amax/127) per block. The same as the Q4_0 format's (kernels/q4_0.cpp).
void quant_q8_0(const bf16_t* x, int K, int8_t* aq, float* asb) {
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const bf16_t* xb = x + (size_t)b * 32;
    float xf[32], amax = 0.0f;
    for (int j = 0; j < 32; ++j) {
      xf[j] = bf16_to_f32(xb[j]);
      amax = std::max(amax, std::fabs(xf[j]));
    }
    const float d = amax > 0.0f ? amax / 127.0f : 1.0f;
    asb[b] = d;
    const float inv = amax > 0.0f ? 1.0f / d : 0.0f;
    int8_t* o = aq + (size_t)b * 32;
    for (int j = 0; j < 32; ++j)
      o[j] = (int8_t)std::max(-127, std::min(127, (int)std::lround(xf[j] * inv)));
  }
}

// In place: a Q8_0-quantized row in natural order (K % 64 == 0) -> plane order per 64-block.
inline void permute_q8_planes64(int8_t* aq, int K) {
  int8_t tmp[64];
  for (int b = 0; b < K / 64; ++b) {
    int8_t* blk = aq + (size_t)b * 64;
    for (int e = 0; e < 64; ++e) tmp[(e & 3) * 16 + (e >> 2)] = blk[e];
    std::memcpy(blk, tmp, 64);
  }
}

// Per 64-block of a plane-ordered row: the 8 int32 lane sums the u8 x s8 dot of planes (0,1)
// and (2,3) would see for an all-ones weight -- lane l = bytes 4l..4l+3 of both 32-byte halves.
inline void q2_lane_sums64(const int8_t* aq, int K, int32_t* out) {
  for (int b = 0; b < K / 64; ++b) {
    const int8_t* blk = aq + (size_t)b * 64;
    int32_t* o = out + (size_t)b * 8;
    for (int l = 0; l < 8; ++l) {
      int s = 0;
      for (int m = 0; m < 4; ++m) s += (int)blk[4 * l + m] + (int)blk[32 + 4 * l + m];
      o[l] = s;
    }
  }
}

float q2_0_dot_i8_scalar(const uint8_t* w, const int8_t* aq, const float* asb, const int32_t*, int K) {
  float acc = 0.0f;
  const int nb = K / 64;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* blk = w + (size_t)b * 18;
    uint16_t dh;
    std::memcpy(&dh, blk, sizeof(dh));
    const uint8_t* q = blk + 2;
    const int8_t* a = aq + (size_t)b * 64;  // plane order: a[16k + j] pairs with element 4j+k
    int s0 = 0, s1 = 0;                     // elements 0..31 (j < 8) / 32..63 (j >= 8)
    for (int k = 0; k < 4; ++k) {
      for (int j = 0; j < 16; ++j) {
        const int wq = (int)((q[j] >> (2 * k)) & 0x03) - 1;
        const int prod = wq * (int)a[16 * k + j];
        if (j < 8) s0 += prod; else s1 += prod;
      }
    }
    acc += fp16_to_f32(dh) * (asb[2 * b] * (float)s0 + asb[2 * b + 1] * (float)s1);
  }
  return acc;
}

#if CPU_MOE_X86
// fp16 block scale -> fp32 via HW F16C (single value in lane 0).
CPU_MOE_TARGET("f16c")
static inline float q2_scale(uint16_t h) {
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
}

// Planes (k, k+1) of one block as 32 UNSIGNED codes 0..3: bytes 0..15 = plane k, 16..31 =
// plane k+1 (the 16-bit shift lets bits cross byte lanes; the 0x03 mask drops them).
CPU_MOE_TARGET("avx2")
static inline void q2_planes(const uint8_t* blk, __m256i& w01, __m256i& w23) {
  const __m256i m3 = _mm256_set1_epi8(0x03);
  const __m128i qb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 2));
  w01 = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(qb, 2), qb), m3);
  w23 = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(qb, 6), _mm_srli_epi16(qb, 4)), m3);
}

// The per-lane scale vector of one block: d * [s0 s0 s1 s1 s0 s0 s1 s1] (see the plane note),
// built from the two adjacent Q8_0 scales with one 8-byte load and one lane permute.
CPU_MOE_TARGET("avx2,f16c")
static inline __m256 q2_lane_scale(uint16_t dh, const float* asb2) {
  const __m256i idx = _mm256_setr_epi32(0, 0, 1, 1, 0, 0, 1, 1);
  const __m128 s2 = _mm_castsi128_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(asb2)));
  const __m256 s = _mm256_permutevar8x32_ps(_mm256_castps128_ps256(s2), idx);
  return _mm256_mul_ps(_mm256_set1_ps(q2_scale(dh)), s);
}

// AVX2 W2A8: VPMADDUBSW (u8 codes x s8 activations, pair sums <= 762) + VPMADDWD, minus the
// precomputed activation lane sums, times the lane scales.
CPU_MOE_TARGET("avx2,fma,f16c")
float q2_0_dot_i8_avx2(const uint8_t* w, const int8_t* aq, const float* asb, const int32_t* alanes, int K) {
  const __m256i ones16 = _mm256_set1_epi16(1);
  __m256 accF = _mm256_setzero_ps();
  const int nb = K / 64;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* blk = w + (size_t)b * 18;
    _mm_prefetch(reinterpret_cast<const char*>(blk) + 512, _MM_HINT_T0);
    uint16_t dh;
    std::memcpy(&dh, blk, sizeof(dh));
    __m256i w01, w23;
    q2_planes(blk, w01, w23);
    const __m256i a01 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq + (size_t)b * 64));
    const __m256i a23 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq + (size_t)b * 64 + 32));
    const __m256i d01 = _mm256_madd_epi16(_mm256_maddubs_epi16(w01, a01), ones16);
    const __m256i d23 = _mm256_madd_epi16(_mm256_maddubs_epi16(w23, a23), ones16);
    const __m256i sums = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(alanes + (size_t)b * 8));
    const __m256i di = _mm256_sub_epi32(_mm256_add_epi32(d01, d23), sums);
    accF = _mm256_fmadd_ps(_mm256_cvtepi32_ps(di), q2_lane_scale(dh, asb + 2 * b), accF);
  }
  return hsum256(accF);
}

// AVX-VNNI W2A8: one VPDPBUSD per 32 weights, no sign juggling (the codes are unsigned).
CPU_MOE_TARGET("avx2,avxvnni,fma,f16c")
float q2_0_dot_i8_vnni(const uint8_t* w, const int8_t* aq, const float* asb, const int32_t* alanes, int K) {
  __m256 accF = _mm256_setzero_ps();
  const int nb = K / 64;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* blk = w + (size_t)b * 18;
    _mm_prefetch(reinterpret_cast<const char*>(blk) + 512, _MM_HINT_T0);
    uint16_t dh;
    std::memcpy(&dh, blk, sizeof(dh));
    __m256i w01, w23;
    q2_planes(blk, w01, w23);
    const __m256i a01 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq + (size_t)b * 64));
    const __m256i a23 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq + (size_t)b * 64 + 32));
    const __m256i sums = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(alanes + (size_t)b * 8));
    __m256i di = _mm256_dpbusd_avx_epi32(_mm256_sub_epi32(_mm256_setzero_si256(), sums), w01, a01);
    di = _mm256_dpbusd_avx_epi32(di, w23, a23);
    accF = _mm256_fmadd_ps(_mm256_cvtepi32_ps(di), q2_lane_scale(dh, asb + 2 * b), accF);
  }
  return hsum256(accF);
}
#endif  // CPU_MOE_X86

q2dot_fn select_q2dot() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
  if (cpu_has_avxvnni()) return q2_0_dot_i8_vnni;
  if (t >= ISA_AVX2) return q2_0_dot_i8_avx2;
#endif
  (void)t;
  return q2_0_dot_i8_scalar;
}

class Q2_0Format final : public MoeFormat {
 public:
  Q2_0Format(const MoeParams& p, const FormatArgs& a) : MoeFormat(p), dot_(select_q2dot()) {
    gate_up_ = a.bank("gate_up");
    down_ = a.bank("down");
    if (p.H % 64 != 0 || p.I % 64 != 0)
      throw std::runtime_error("Q2_0 CPU MoE requires H and I to be multiples of 64");
    const size_t tokens = p.max_tokens, routes = (size_t)p.max_tokens * p.top_k;
    xi8_.resize(tokens * p.H);
    xas_.resize(tokens * (p.H / 32));
    xal_.resize(tokens * (p.H / 64) * 8);
    gi8_.resize(routes * p.I);
    gas_.resize(routes * (p.I / 32));
    gal_.resize(routes * (p.I / 64) * 8);
    gu_row_bytes_ = (p.H / 64) * 18;  // K = H (gate_up rows)
    dn_row_bytes_ = (p.I / 64) * 18;  // K = I (down rows)
    // select_q2dot picks VPDPBUSD / VPMADDUBSW / scalar for the tier, so the tag reflects which one it resolved to.
    isa_ = std::string(tier_name(pick_isa())) + (cpu_has_avxvnni() ? "+vnni(q2_0-w2a8)" : "+q2_0-w2a8");
  }

  std::string isa_name() const override { return isa_; }

  void prep_input(int tok, const bf16_t* x) override {
    prep(x, p.H, xi8_.data() + (size_t)tok * p.H, xas_.data() + (size_t)tok * (p.H / 32),
         xal_.data() + (size_t)tok * (p.H / 64) * 8);
  }

  bool preps_inter() const override { return true; }

  void prep_inter(int64_t r, bf16_t* g) override {
    prep(g, p.I, gi8_.data() + (size_t)r * p.I, gas_.data() + (size_t)r * (p.I / 32),
         gal_.data() + (size_t)r * (p.I / 64) * 8);
  }

  void gate_up(int layer, int e, int tok, int i0, int i1, const bf16_t* /*x*/, float w_in,
               bf16_t* g) const override {
    const uint8_t* packed = static_cast<const uint8_t*>(tbl_at(gate_up_, layer));
    const int8_t* xi8 = xi8_.data() + (size_t)tok * p.H;
    const float* xas = xas_.data() + (size_t)tok * (p.H / 32);
    const int32_t* xal = xal_.data() + (size_t)tok * (p.H / 64) * 8;
    gated_rows(p, i0, i1, w_in, g, [&](int row) {
      return dot_(packed + ((size_t)e * (2 * p.I) + row) * (size_t)gu_row_bytes_, xi8, xas, xal, p.H);
    });
  }

  void down(int layer, int h0, int h1, const DownRoute* routes, int n, float* out) const override {
    const uint8_t* packed = static_cast<const uint8_t*>(tbl_at(down_, layer));
    for (int h = h0; h < h1; ++h) {
      for (int j = 0; j < n; ++j) {
        const size_t r = (size_t)routes[j].r;
        out[j * kRowBlock + h - h0] =
            dot_(packed + ((size_t)routes[j].e * p.H + h) * (size_t)dn_row_bytes_, gi8_.data() + r * p.I,
                 gas_.data() + r * (p.I / 32), gal_.data() + r * (p.I / 64) * 8, p.I);
      }
    }
  }

 private:
  // Q8_0-quantize a row, put every 64-block in plane order, and take its activation lane sums.
  static void prep(const bf16_t* v, int K, int8_t* aq, float* asb, int32_t* lanes) {
    quant_q8_0(v, K, aq, asb);
    permute_q8_planes64(aq, K);
    q2_lane_sums64(aq, K, lanes);
  }

  q2dot_fn dot_;
  // GGUF Q2_0 rows of 64-element blocks (fp16 scale + 16 code bytes): gate_up [E,2I,H/64*18], down [E,H,I/64*18].
  const uint64_t* gate_up_ = nullptr;
  const uint64_t* down_ = nullptr;
  int gu_row_bytes_ = 0, dn_row_bytes_ = 0;
  std::string isa_;
  // Q8_0 activations in plane order (per-32-block int8 + fp32 scale) and the 8 lane sums per 64-block,
  // for the input and the intermediate.
  AlignedVector<int8_t> xi8_, gi8_;
  AlignedVector<float> xas_, gas_;
  AlignedVector<int32_t> xal_, gal_;
};

}  // namespace

std::unique_ptr<MoeFormat> make_q2_0_format(const MoeParams& p, const FormatArgs& a) {
  return std::make_unique<Q2_0Format>(p, a);
}

}  // namespace freetoken::cpu_moe
