// Fused pair-bias attention for Uni-Mol (CPU, float32).
//
// Uni-Mol's attention layers carry the *pre-softmax logits* of every layer to the
// next one as an additive pair bias:
//
//     S_l = scale * Q K^T + S_{l-1}        (S_0 = Gaussian-kernel pair bias)
//     O_l = dropout(softmax(S_l)) V
//
// The eager implementation materialises Q K^T, adds the bias in place, clones the
// logits, runs softmax, and then a second bmm -- several full passes over
// B*H*L*L floats per layer.  With head_dim = 8 the bmm FLOPs are negligible and the
// whole op is memory bound, so we fuse it: each (batch, head) item is processed row
// by row with all intermediate state in L1/L2.
//
// Numerics: same maths as eager.  Differences are fp32 rounding (different FMA
// contraction, a polynomial expf with ~1 ulp error).  Dropout uses a stateless
// counter-based hash, so the backward pass regenerates the identical mask.
#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__) && !defined(UA_NO_CLONES)
#define UA_CLONES __attribute__((target_clones("arch=skylake-avx512", "arch=haswell", "default")))
#else
#define UA_CLONES
#endif
#define UA_INLINE static inline __attribute__((always_inline))

namespace {

UA_INLINE float fast_expf(float x) {
  // Cephes-style expf, valid for x in [-87.3, 88]; callers handle x <= -87.3 -> 0.
  x = x > 88.0f ? 88.0f : x;
  x = x < -87.3f ? -87.3f : x;
  const float n = std::floor(x * 1.44269504088896341f + 0.5f);
  float r = x - n * 0.693359375f;
  r = r - n * (-2.12194440e-4f);
  const float z = r * r;
  float y = 1.9875691500E-4f;
  y = y * r + 1.3981999507E-3f;
  y = y * r + 8.3334519073E-3f;
  y = y * r + 4.1665795894E-2f;
  y = y * r + 1.6666665459E-1f;
  y = y * r + 5.0000001201E-1f;
  y = y * z + r + 1.0f;
  const int32_t e = static_cast<int32_t>(n) + 127;
  const uint32_t bits = static_cast<uint32_t>(e) << 23;
  float p2;
  std::memcpy(&p2, &bits, sizeof(float));
  return y * p2;
}

// Stateless dropout hash -> uniform uint32 (murmur3 finaliser on a mixed counter).
UA_INLINE uint32_t hash32(uint64_t seed, uint64_t item, uint32_t i, uint32_t j) {
  uint32_t h = static_cast<uint32_t>(seed) ^ (static_cast<uint32_t>(seed >> 32) * 0x9E3779B1u);
  h ^= static_cast<uint32_t>(item) * 0x85EBCA6Bu;
  h ^= (static_cast<uint32_t>(item >> 32) + 0x7F4A7C15u) * 0xC2B2AE35u;
  h ^= i * 0x27D4EB2Fu;
  h += j * 0x165667B1u;
  h ^= h >> 16;
  h *= 0x85EBCA6Bu;
  h ^= h >> 13;
  h *= 0xC2B2AE35u;
  h ^= h >> 16;
  return h;
}

// softmax of one row of logits s[0..L) into p[0..L). Returns false if row is fully masked.
UA_INLINE void softmax_row(const float* __restrict__ s, float* __restrict__ p, int L) {
  float m = -std::numeric_limits<float>::infinity();
#pragma omp simd reduction(max : m)
  for (int j = 0; j < L; ++j) m = s[j] > m ? s[j] : m;
  if (m == -std::numeric_limits<float>::infinity()) {
    const float nan = std::numeric_limits<float>::quiet_NaN();  // matches torch.softmax on all -inf
    for (int j = 0; j < L; ++j) p[j] = nan;
    return;
  }
  float sum = 0.f;
#pragma omp simd reduction(+ : sum)
  for (int j = 0; j < L; ++j) {
    const float x = s[j] - m;
    const float e = x <= -87.3f ? 0.f : fast_expf(x);
    p[j] = e;
    sum += e;
  }
#pragma omp simd
  for (int j = 0; j < L; ++j) p[j] = p[j] / sum;
}

// One block of R query rows of one (batch, head) item.  kT/vT are [D][L] transposed copies,
// v is the original row-major [L][D].  s is [R][L] scratch.
template <int R, int DT>
UA_INLINE void fwd_block(const float* __restrict__ qi, const float* __restrict__ kT, const float* __restrict__ vi,
                         const float* __restrict__ bi, float* __restrict__ oi, float* __restrict__ so,
                         float* __restrict__ s, int i0, int L, int Drt, uint32_t thresh, float keep_scale,
                         uint64_t seed, uint64_t item) {
  const int D = DT ? DT : Drt;
  for (int r = 0; r < R; ++r) {
    float* sr = s + static_cast<size_t>(r) * L;
#pragma omp simd
    for (int j = 0; j < L; ++j) sr[j] = 0.f;
  }
  for (int d = 0; d < D; ++d) {
    const float* kd = kT + static_cast<size_t>(d) * L;
    float qv[R];
    for (int r = 0; r < R; ++r) qv[r] = qi[static_cast<size_t>(i0 + r) * D + d];
#pragma omp simd
    for (int j = 0; j < L; ++j) {
      const float kv = kd[j];
      for (int r = 0; r < R; ++r) s[static_cast<size_t>(r) * L + j] += qv[r] * kv;
    }
  }
  for (int r = 0; r < R; ++r) {
    float* sr = s + static_cast<size_t>(r) * L;
    const float* brow = bi + static_cast<size_t>(i0 + r) * L;
#pragma omp simd
    for (int j = 0; j < L; ++j) sr[j] += brow[j];
    if (so) {
      float* srow = so + static_cast<size_t>(i0 + r) * L;
#pragma omp simd
      for (int j = 0; j < L; ++j) srow[j] = sr[j];
    }
    softmax_row(sr, sr, L);  // in place: logits already copied to `so`
    if (thresh) {
      for (int j = 0; j < L; ++j) {
        const uint32_t h = hash32(seed, item, static_cast<uint32_t>(i0 + r), static_cast<uint32_t>(j));
        sr[j] = h >= thresh ? sr[j] * keep_scale : 0.f;
      }
    }
  }
  // O[r, :] = sum_j P[r, j] * V[j, :]  -- broadcast-FMA over the (contiguous) head dim, no horizontal sums
  float acc[R][DT ? DT : 1];
  if (DT) {
    for (int r = 0; r < R; ++r)
      for (int d = 0; d < DT; ++d) acc[r][d] = 0.f;
    for (int j = 0; j < L; ++j) {
      const float* vj = vi + static_cast<size_t>(j) * DT;
      for (int r = 0; r < R; ++r) {
        const float pj = s[static_cast<size_t>(r) * L + j];
#pragma omp simd
        for (int d = 0; d < DT; ++d) acc[r][d] += pj * vj[d];
      }
    }
    for (int r = 0; r < R; ++r) {
      float* orow = oi + static_cast<size_t>(i0 + r) * DT;
#pragma omp simd
      for (int d = 0; d < DT; ++d) orow[d] = acc[r][d];
    }
  } else {
    for (int r = 0; r < R; ++r) {
      float* orow = oi + static_cast<size_t>(i0 + r) * D;
      for (int d = 0; d < D; ++d) orow[d] = 0.f;
      for (int j = 0; j < L; ++j) {
        const float pj = s[static_cast<size_t>(r) * L + j];
        const float* vj = vi + static_cast<size_t>(j) * D;
#pragma omp simd
        for (int d = 0; d < D; ++d) orow[d] += pj * vj[d];
      }
    }
  }
}

template <int DT>
UA_INLINE void fwd_items_t(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                           const float* __restrict__ bias, float* __restrict__ o, float* __restrict__ sout,
                           int64_t b0, int64_t b1, int L, int Drt, float p_drop, uint64_t seed) {
  constexpr int RB = 4;
  const int D = DT ? DT : Drt;
  std::vector<float> scratch(static_cast<size_t>(D + RB) * L);
  float* kT = scratch.data();
  float* s = kT + static_cast<size_t>(D) * L;
  const uint32_t thresh = p_drop > 0.f ? static_cast<uint32_t>(static_cast<double>(p_drop) * 4294967296.0) : 0u;
  const float keep_scale = p_drop > 0.f ? 1.f / (1.f - p_drop) : 1.f;
  for (int64_t it = b0; it < b1; ++it) {
    const float* qi = q + it * L * D;
    const float* ki = k + it * L * D;
    const float* vi = v + it * L * D;
    const float* bi = bias + it * L * L;
    float* oi = o + it * L * D;
    float* so = sout ? sout + it * L * L : nullptr;
    for (int j = 0; j < L; ++j)
      for (int d = 0; d < D; ++d) kT[d * L + j] = ki[j * D + d];
    int i = 0;
    for (; i + RB <= L; i += RB)
      fwd_block<RB, DT>(qi, kT, vi, bi, oi, so, s, i, L, D, thresh, keep_scale, seed, static_cast<uint64_t>(it));
    for (; i < L; ++i)
      fwd_block<1, DT>(qi, kT, vi, bi, oi, so, s, i, L, D, thresh, keep_scale, seed, static_cast<uint64_t>(it));
  }
}

UA_CLONES
void fwd_items(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
               const float* __restrict__ bias, float* __restrict__ o, float* __restrict__ sout,
               int64_t b0, int64_t b1, int L, int D, float p_drop, uint64_t seed) {
  switch (D) {
    case 8: fwd_items_t<8>(q, k, v, bias, o, sout, b0, b1, L, D, p_drop, seed); break;
    case 16: fwd_items_t<16>(q, k, v, bias, o, sout, b0, b1, L, D, p_drop, seed); break;
    case 32: fwd_items_t<32>(q, k, v, bias, o, sout, b0, b1, L, D, p_drop, seed); break;
    default: fwd_items_t<0>(q, k, v, bias, o, sout, b0, b1, L, D, p_drop, seed); break;
  }
}

UA_CLONES
void bwd_items(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
               const float* __restrict__ sfwd, const float* __restrict__ dout, const float* __restrict__ dsout,
               float* __restrict__ dq, float* __restrict__ dk, float* __restrict__ dv, float* __restrict__ dbias,
               int64_t b0, int64_t b1, int L, int D, float p_drop, uint64_t seed) {
  // layout: kT, vT, dkT, dvT : D*L each; rows: pr, f, dp, ds : L each
  std::vector<float> scratch(static_cast<size_t>(4 * D + 4) * L);
  float* kT = scratch.data();
  float* vT = kT + static_cast<size_t>(D) * L;
  float* dkT = vT + static_cast<size_t>(D) * L;
  float* dvT = dkT + static_cast<size_t>(D) * L;
  float* pr = dvT + static_cast<size_t>(D) * L;
  float* fr = pr + L;
  float* dp = fr + L;
  float* ds = dp + L;
  const uint32_t thresh = p_drop > 0.f ? static_cast<uint32_t>(static_cast<double>(p_drop) * 4294967296.0) : 0u;
  const float keep_scale = p_drop > 0.f ? 1.f / (1.f - p_drop) : 1.f;
  for (int64_t it = b0; it < b1; ++it) {
    const float* qi = q + it * L * D;
    const float* ki = k + it * L * D;
    const float* vi = v + it * L * D;
    const float* si = sfwd + it * L * L;
    const float* doi = dout + it * L * D;
    const float* dsi = dsout ? dsout + it * L * L : nullptr;
    float* dqi = dq + it * L * D;
    float* dki = dk + it * L * D;
    float* dvi = dv + it * L * D;
    float* dbi = dbias + it * L * L;
    for (int j = 0; j < L; ++j)
      for (int d = 0; d < D; ++d) {
        kT[d * L + j] = ki[j * D + d];
        vT[d * L + j] = vi[j * D + d];
      }
    std::memset(dkT, 0, sizeof(float) * 2 * static_cast<size_t>(D) * L);  // dkT, dvT contiguous
    for (int i = 0; i < L; ++i) {
      softmax_row(si + static_cast<size_t>(i) * L, pr, L);
      if (thresh) {
        for (int j = 0; j < L; ++j) {
          const uint32_t h = hash32(seed, static_cast<uint64_t>(it), static_cast<uint32_t>(i), static_cast<uint32_t>(j));
          fr[j] = h >= thresh ? keep_scale : 0.f;
        }
      } else {
#pragma omp simd
        for (int j = 0; j < L; ++j) fr[j] = 1.f;
      }
#pragma omp simd
      for (int j = 0; j < L; ++j) dp[j] = 0.f;
      for (int d = 0; d < D; ++d) {
        const float dd = doi[i * D + d];
        const float* vd = vT + static_cast<size_t>(d) * L;
#pragma omp simd
        for (int j = 0; j < L; ++j) dp[j] += dd * vd[j];
      }
      float rs = 0.f;
#pragma omp simd reduction(+ : rs)
      for (int j = 0; j < L; ++j) {
        dp[j] = dp[j] * fr[j];  // grad wrt pre-dropout probabilities
        // masked (-inf) logits give pr == 0 (or NaN rows, propagated like eager)
        rs += pr[j] * dp[j];
      }
      float* dbrow = dbi + static_cast<size_t>(i) * L;
      if (dsi) {
        const float* dsrow = dsi + static_cast<size_t>(i) * L;
#pragma omp simd
        for (int j = 0; j < L; ++j) ds[j] = pr[j] * (dp[j] - rs) + dsrow[j];
      } else {
#pragma omp simd
        for (int j = 0; j < L; ++j) ds[j] = pr[j] * (dp[j] - rs);
      }
#pragma omp simd
      for (int j = 0; j < L; ++j) dbrow[j] = ds[j];
      for (int d = 0; d < D; ++d) {
        const float dd = doi[i * D + d];
        const float qd = qi[i * D + d];
        const float* kd = kT + static_cast<size_t>(d) * L;
        float* dkd = dkT + static_cast<size_t>(d) * L;
        float* dvd = dvT + static_cast<size_t>(d) * L;
        float accq = 0.f;
#pragma omp simd reduction(+ : accq)
        for (int j = 0; j < L; ++j) {
          accq += ds[j] * kd[j];
          dkd[j] += ds[j] * qd;
          dvd[j] += pr[j] * fr[j] * dd;
        }
        dqi[i * D + d] = accq;
      }
    }
    for (int j = 0; j < L; ++j)
      for (int d = 0; d < D; ++d) {
        dki[j * D + d] = dkT[d * L + j];
        dvi[j * D + d] = dvT[d * L + j];
      }
  }
}

void check_f32_contig(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}  // namespace

// q (already scaled), k, v: [BH, L, D];  bias: [BH, L, L].  Returns (out [BH,L,D], logits [BH,L,L]).
std::tuple<at::Tensor, at::Tensor> pair_attention_fwd(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                                      const at::Tensor& bias, double p_drop, int64_t seed) {
  check_f32_contig(q, "q");
  check_f32_contig(k, "k");
  check_f32_contig(v, "v");
  check_f32_contig(bias, "bias");
  TORCH_CHECK(q.dim() == 3 && bias.dim() == 3, "expected q [BH,L,D] and bias [BH,L,L]");
  const int64_t BH = q.size(0);
  const int L = static_cast<int>(q.size(1));
  const int D = static_cast<int>(q.size(2));
  TORCH_CHECK(bias.size(0) == BH && bias.size(1) == L && bias.size(2) == L, "bias shape mismatch");
  auto out = at::empty_like(q);
  auto logits = at::empty_like(bias);
  const float* qp = q.data_ptr<float>();
  const float* kp = k.data_ptr<float>();
  const float* vp = v.data_ptr<float>();
  const float* bp = bias.data_ptr<float>();
  float* op = out.data_ptr<float>();
  float* lp = logits.data_ptr<float>();
  at::parallel_for(0, BH, 1, [&](int64_t b0, int64_t b1) {
    fwd_items(qp, kp, vp, bp, op, lp, b0, b1, L, D, static_cast<float>(p_drop), static_cast<uint64_t>(seed));
  });
  return {out, logits};
}

// Returns (dq, dk, dv, dbias).  dlogits may be undefined (no gradient through the returned logits).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> pair_attention_bwd(
    const at::Tensor& dout, const c10::optional<at::Tensor>& dlogits, const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, const at::Tensor& logits, double p_drop, int64_t seed) {
  check_f32_contig(q, "q");
  check_f32_contig(k, "k");
  check_f32_contig(v, "v");
  check_f32_contig(logits, "logits");
  auto dout_c = dout.contiguous();
  at::Tensor dl;
  const float* dlp = nullptr;
  if (dlogits.has_value() && dlogits->defined()) {
    dl = dlogits->contiguous();
    dlp = dl.data_ptr<float>();
  }
  const int64_t BH = q.size(0);
  const int L = static_cast<int>(q.size(1));
  const int D = static_cast<int>(q.size(2));
  auto dq = at::empty_like(q);
  auto dk = at::empty_like(k);
  auto dv = at::empty_like(v);
  auto db = at::empty_like(logits);
  at::parallel_for(0, BH, 1, [&](int64_t b0, int64_t b1) {
    bwd_items(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), logits.data_ptr<float>(),
              dout_c.data_ptr<float>(), dlp, dq.data_ptr<float>(), dk.data_ptr<float>(), dv.data_ptr<float>(),
              db.data_ptr<float>(), b0, b1, L, D, static_cast<float>(p_drop), static_cast<uint64_t>(seed));
  });
  return {dq, dk, dv, db};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pair_attention_fwd", &pair_attention_fwd, "fused pair-bias attention forward (CPU fp32)");
  m.def("pair_attention_bwd", &pair_attention_bwd, "fused pair-bias attention backward (CPU fp32)");
}
