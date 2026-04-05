#include "devicecode.h"
#include <optix.h>

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

extern "C" {
__constant__ LaunchParams params;
}

// ---- RNG (PCG) ----

struct RNG {
  unsigned int state;
  __device__ RNG(unsigned int pixel, unsigned int sample, unsigned int depth) {
    state = (pixel * 17u + sample * 101u + depth * 1999u) * 747796405u +
            2891336453u;
  }
  __device__ float next() {
    unsigned int old = state;
    state = old * 747796405u + 2891336453u;
    unsigned int word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    unsigned int result = (word >> 22u) ^ word;
    // Use upper 23 bits to avoid float precision loss for large u32 values
    return (result >> 9) *
           (1.0f / 8388608.0f); // result >> 9 fits in 23-bit mantissa
  }
};

// ---- Math helpers ----

static __forceinline__ __device__ float3 make_f3(const float *p) {
  return make_float3(p[0], p[1], p[2]);
}

static __forceinline__ __device__ float dot3(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __forceinline__ __device__ float3 normalize3(float3 v) {
  float len = sqrtf(dot3(v, v));
  return make_float3(v.x / len, v.y / len, v.z / len);
}

static __forceinline__ __device__ float3 cross3(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

static __forceinline__ __device__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __device__ float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static __forceinline__ __device__ float3 operator*(float3 a, float s) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}

static __forceinline__ __device__ float3 operator*(float s, float3 a) {
  return a * s;
}

static __forceinline__ __device__ float3 operator*(float3 a, float3 b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

static __forceinline__ __device__ float3 reflect3(float3 I, float3 N) {
  return I - 2.0f * dot3(I, N) * N;
}

// Build tangent frame from normal, using geometry tangent if available
static __forceinline__ __device__ void
build_tangent_frame(float3 N, float3 &tangent, float3 &bitangent) {
  if (fabsf(N.x) > 0.9f)
    tangent = normalize3(cross3(make_float3(0, 1, 0), N));
  else
    tangent = normalize3(cross3(make_float3(1, 0, 0), N));
  bitangent = cross3(N, tangent);
}

// Build tangent frame from normal and geometry dpdu tangent
// Falls back to arbitrary frame if geom_tangent is zero
static __forceinline__ __device__ void
build_tangent_frame_geom(float3 N, float3 geom_tangent, float3 &tangent,
                         float3 &bitangent) {
  float len2 = dot3(geom_tangent, geom_tangent);
  if (len2 > 1e-6f) {
    // Orthogonalize tangent against normal (Gram-Schmidt)
    float3 t = geom_tangent - N * dot3(geom_tangent, N);
    float tlen2 = dot3(t, t);
    if (tlen2 > 1e-6f) {
      tangent = t * (1.0f / sqrtf(tlen2));
      bitangent = cross3(N, tangent);
      return;
    }
  }
  // Fallback to arbitrary frame
  build_tangent_frame(N, tangent, bitangent);
}

// Anisotropic GGX NDF
static __forceinline__ __device__ float ggx_D_aniso(float3 H, float3 N,
                                                    float3 T, float3 B,
                                                    float alpha_u,
                                                    float alpha_v) {
  float HdotT = dot3(H, T);
  float HdotB = dot3(H, B);
  float HdotN = dot3(H, N);
  float x = (HdotT / alpha_u) * (HdotT / alpha_u) +
            (HdotB / alpha_v) * (HdotB / alpha_v) + HdotN * HdotN;
  return 1.0f / (M_PIf * alpha_u * alpha_v * x * x);
}

// Anisotropic Smith G1 for GGX
static __forceinline__ __device__ float ggx_G1_aniso(float3 V, float3 N,
                                                     float3 T, float3 B,
                                                     float alpha_u,
                                                     float alpha_v) {
  float VdotT = dot3(V, T);
  float VdotB = dot3(V, B);
  float VdotN = dot3(V, N);
  float a2 = (alpha_u * VdotT) * (alpha_u * VdotT) +
             (alpha_v * VdotB) * (alpha_v * VdotB);
  float denom = VdotN + sqrtf(a2 + VdotN * VdotN);
  if (denom < 1e-8f)
    return 0.0f;
  return 2.0f * VdotN / denom;
}

// Isotropic wrappers
static __forceinline__ __device__ float ggx_D(float NdotH, float alpha) {
  float a2 = alpha * alpha;
  float d = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
  return a2 / (M_PIf * d * d);
}

static __forceinline__ __device__ float ggx_G1(float NdotX, float alpha) {
  float a2 = alpha * alpha;
  return 2.0f * NdotX / (NdotX + sqrtf(a2 + (1.0f - a2) * NdotX * NdotX));
}

// Exact conductor Fresnel reflectance per channel
static __forceinline__ __device__ float
conductor_fresnel_channel(float cos_i, float eta, float k) {
  float cos2 = cos_i * cos_i;
  float sin2 = 1.0f - cos2;
  float eta2 = eta * eta;
  float k2 = k * k;

  float t0 = eta2 - k2 - sin2;
  float a2b2 = sqrtf(t0 * t0 + 4.0f * eta2 * k2);
  float a = sqrtf(fmaxf(0.0f, 0.5f * (a2b2 + t0)));

  float term1 = a2b2 + cos2;
  float term2 = 2.0f * a * cos_i;
  float Rs = (term1 - term2) / (term1 + term2);

  float term3 = a2b2 * cos2 + sin2 * sin2;
  float term4 = term2 * sin2;
  float Rp = Rs * (term3 - term4) / (term3 + term4);

  return 0.5f * (Rs + Rp);
}

// Conductor Fresnel reflectance (RGB)
static __forceinline__ __device__ float3 conductor_fresnel(float cos_i,
                                                           float3 eta,
                                                           float3 k) {
  return make_float3(conductor_fresnel_channel(cos_i, eta.x, k.x),
                     conductor_fresnel_channel(cos_i, eta.y, k.y),
                     conductor_fresnel_channel(cos_i, eta.z, k.z));
}

// Schlick Fresnel with F0 (fallback when no eta/k)
static __forceinline__ __device__ float3 schlick_fresnel_f0_rgb(float cos_i,
                                                                float3 f0) {
  float x = 1.0f - cos_i;
  float x5 = x * x * x * x * x;
  return make_float3(f0.x + (1.0f - f0.x) * x5, f0.y + (1.0f - f0.y) * x5,
                     f0.z + (1.0f - f0.z) * x5);
}

// ---- GGX Energy Compensation (Kulla-Conty) ----

#define GGX_LUT_SIZE 32

// Look up precomputed directional albedo E(cosθ, α)
static __forceinline__ __device__ float ggx_E(float cos_theta, float alpha) {
  float u = cos_theta * (GGX_LUT_SIZE - 1);
  float v = alpha * (GGX_LUT_SIZE - 1);
  int iu = min(max((int)u, 0), GGX_LUT_SIZE - 1);
  int iv = min(max((int)v, 0), GGX_LUT_SIZE - 1);
  return params.ggx_e_lut[iu * GGX_LUT_SIZE + iv];
}

// Look up precomputed average albedo E_avg(α)
static __forceinline__ __device__ float ggx_E_avg(float alpha) {
  float v = alpha * (GGX_LUT_SIZE - 1);
  int iv = min(max((int)v, 0), GGX_LUT_SIZE - 1);
  return params.ggx_e_avg_lut[iv];
}

// Compute Kulla-Conty multi-scatter compensation BRDF
// Returns f_ms * F_avg for the given Fresnel term
static __forceinline__ __device__ float3 ggx_multiscatter(float NdotV,
                                                          float NdotL,
                                                          float alpha,
                                                          float3 F_avg) {
  float E_o = ggx_E(NdotV, alpha);
  float E_i = ggx_E(NdotL, alpha);
  float E_a = ggx_E_avg(alpha);
  float denom = M_PIf * (1.0f - E_a);
  if (denom < 1e-8f)
    return make_float3(0, 0, 0);
  float f_ms = (1.0f - E_o) * (1.0f - E_i) / denom;
  return F_avg * f_ms;
}

// Average Fresnel reflectance for conductors (integral of F over cosθ)
// For Schlick: F_avg = F0 + (1 - F0) / 21
static __forceinline__ __device__ float3 conductor_F_avg(float3 albedo,
                                                         float3 eta, float3 k) {
  bool has_ior =
      (eta.x > 0 || eta.y > 0 || eta.z > 0 || k.x > 0 || k.y > 0 || k.z > 0);
  if (has_ior) {
    // Numerical approximation: sample a few angles
    float3 sum = make_float3(0, 0, 0);
    for (int i = 0; i < 8; i++) {
      float mu = (i + 0.5f) / 8.0f;
      sum = sum + conductor_fresnel(mu, eta, k) * (2.0f * mu / 8.0f);
    }
    return sum;
  } else {
    // Schlick: F_avg = F0 + (1 - F0) / 21
    return albedo + (make_float3(1, 1, 1) - albedo) * (1.0f / 21.0f);
  }
}

// Evaluate Cook-Torrance specular BRDF with Kulla-Conty energy compensation
// eta/k = (0,0,0) means use Schlick with albedo as F0
// alpha_u/alpha_v: anisotropic roughness (set equal for isotropic)
// geom_tangent: geometry dpdu for anisotropy alignment (zero = arbitrary)
static __forceinline__ __device__ float3 eval_conductor_brdf(
    float3 V, float3 L, float3 N, float3 albedo, float alpha_u, float alpha_v,
    float3 eta, float3 k, float3 geom_tangent = make_float3(0, 0, 0)) {
  float NdotV = fmaxf(dot3(N, V), 0.001f);
  float NdotL = fmaxf(dot3(N, L), 0.001f);
  float3 H = normalize3(V + L);
  float VdotH = fmaxf(dot3(V, H), 0.0f);

  float D, G;
  if (alpha_u == alpha_v) {
    float NdotH = fmaxf(dot3(N, H), 0.0f);
    D = ggx_D(NdotH, alpha_u);
    G = ggx_G1(NdotV, alpha_u) * ggx_G1(NdotL, alpha_u);
  } else {
    float3 T, B;
    build_tangent_frame_geom(N, geom_tangent, T, B);
    D = ggx_D_aniso(H, N, T, B, alpha_u, alpha_v);
    G = ggx_G1_aniso(V, N, T, B, alpha_u, alpha_v) *
        ggx_G1_aniso(L, N, T, B, alpha_u, alpha_v);
  }

  // Fresnel
  float3 F;
  bool has_ior =
      (eta.x > 0 || eta.y > 0 || eta.z > 0 || k.x > 0 || k.y > 0 || k.z > 0);
  if (has_ior) {
    F = conductor_fresnel(VdotH, eta, k);
  } else {
    F = schlick_fresnel_f0_rgb(VdotH, albedo);
  }

  float denom = 4.0f * NdotV * NdotL;
  if (denom < 0.0001f)
    return make_float3(0, 0, 0);

  // Single-scatter term
  float3 f_ss = F * (D * G / denom);

  // Multi-scatter energy compensation (Kulla-Conty)
  float alpha_avg = 0.5f * (alpha_u + alpha_v);
  float3 F_a = conductor_F_avg(albedo, eta, k);
  float3 f_ms = ggx_multiscatter(NdotV, NdotL, alpha_avg, F_a);

  return f_ss + f_ms;
}

// Cosine-weighted hemisphere sample
static __forceinline__ __device__ float3 cosine_sample_hemisphere(float u1,
                                                                  float u2,
                                                                  float3 N) {
  float phi = 2.0f * M_PIf * u1;
  float cos_theta = sqrtf(u2);
  float sin_theta = sqrtf(1.0f - u2);

  float3 tangent;
  if (fabsf(N.x) > 0.9f)
    tangent = normalize3(cross3(make_float3(0, 1, 0), N));
  else
    tangent = normalize3(cross3(make_float3(1, 0, 0), N));
  float3 bitangent = cross3(N, tangent);

  return normalize3(tangent * (cosf(phi) * sin_theta) +
                    bitangent * (sinf(phi) * sin_theta) + N * cos_theta);
}

// GGX Visible Normal Distribution sampling (Heitz 2018)
// Samples half-vectors proportional to D(H) * max(VdotH, 0), avoiding invisible
// microfacets. V_world: view direction (toward camera), N: surface normal
// geom_tangent: geometry dpdu for anisotropy alignment (zero = arbitrary frame)
// Returns half-vector in world space
static __forceinline__ __device__ float3 ggx_sample_vndf(
    float u1, float u2, float3 V_world, float alpha_u, float alpha_v, float3 N,
    float3 geom_tangent = make_float3(0, 0, 0)) {
  float3 T, B;
  build_tangent_frame_geom(N, geom_tangent, T, B);

  // Transform V to local frame (T=x, B=y, N=z)
  float3 Vl = make_float3(dot3(V_world, T), dot3(V_world, B), dot3(V_world, N));

  // Transform to hemisphere configuration (stretch)
  float3 Vh = normalize3(make_float3(alpha_u * Vl.x, alpha_v * Vl.y, Vl.z));

  // Orthonormal basis around Vh
  float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
  float3 T1 = lensq > 1e-7f
                  ? make_float3(-Vh.y, Vh.x, 0.0f) * (1.0f / sqrtf(lensq))
                  : make_float3(1.0f, 0.0f, 0.0f);
  float3 T2 = cross3(Vh, T1);

  // Sample point on disk and reproject
  float r = sqrtf(u1);
  float phi = 2.0f * M_PIf * u2;
  float t1 = r * cosf(phi);
  float t2 = r * sinf(phi);
  float s = 0.5f * (1.0f + Vh.z);
  t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1)) + s * t2;

  // Reproject onto hemisphere
  float3 Nh =
      T1 * t1 + T2 * t2 + Vh * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2));

  // Transform back to ellipsoid configuration
  float3 H_local = normalize3(
      make_float3(alpha_u * Nh.x, alpha_v * Nh.y, fmaxf(0.0f, Nh.z)));

  // Transform back to world space
  return normalize3(T * H_local.x + B * H_local.y + N * H_local.z);
}

// Isotropic convenience wrapper (still uses VNDF)
static __forceinline__ __device__ float3
ggx_sample(float u1, float u2, float alpha, float3 V_world, float3 N,
           float3 geom_tangent = make_float3(0, 0, 0)) {
  return ggx_sample_vndf(u1, u2, V_world, alpha, alpha, N, geom_tangent);
}

// Schlick Fresnel approximation (for coated surfaces, uses F0 = 0.04 for
// dielectric coat)
static __forceinline__ __device__ float fresnel_schlick_f0(float cos_i,
                                                           float f0) {
  float x = 1.0f - cos_i;
  return f0 + (1.0f - f0) * x * x * x * x * x;
}

// Schlick Fresnel approximation
static __forceinline__ __device__ float fresnel_schlick(float cos_i,
                                                        float eta) {
  float r0 = (1.0f - eta) / (1.0f + eta);
  r0 = r0 * r0;
  float x = 1.0f - cos_i;
  return r0 + (1.0f - r0) * x * x * x * x * x;
}

static __forceinline__ __device__ bool refract3(float3 I, float3 N, float eta,
                                                float3 &T) {
  float cos_i = dot3(I, N);
  float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
  if (sin2_t > 1.0f)
    return false;
  T = eta * I - (eta * cos_i + sqrtf(1.0f - sin2_t)) * N;
  return true;
}

// ---- Shadow ray helper ----

struct ShadowResult {
  bool hit;        // true if something was hit
  float3 emission; // emission of hit surface (zero if miss)
};

static __forceinline__ __device__ ShadowResult
trace_shadow(float3 origin, float3 dir, float tmax, unsigned int ray_flags) {
  unsigned int p0, p1, p2, p3, p4, p5, p6, p7, p8;
  unsigned int sp9 = 0xFFFFFFFF;
  unsigned int sp10 = 0, sp11 = 0, sp12 = 0, sp13 = 0;
  unsigned int sp14 = 0, sp15 = 0, sp16 = 0, sp17 = 0, sp18 = 0, sp19 = 0;
  unsigned int sp20 = 0, sp21 = 0, sp22 = 0, sp23 = 0, sp24 = 0, sp25 = 0,
               sp26 = 0, sp27 = 0, sp28 = 0, sp29 = 0;
  optixTrace(params.traversable, origin, dir, 0.001f, tmax, 0.0f,
             OptixVisibilityMask(255), ray_flags, 0, 1, 0, p0, p1, p2, p3, p4,
             p5, p6, p7, p8, sp9, sp10, sp11, sp12, sp13, sp14, sp15, sp16,
             sp17, sp18, sp19, sp20, sp21, sp22, sp23, sp24, sp25, sp26, sp27,
             sp28, sp29);
  ShadowResult res;
  res.hit = (sp9 != 0xFFFFFFFF);
  res.emission = make_float3(__uint_as_float(sp10), __uint_as_float(sp11),
                             __uint_as_float(sp12));
  return res;
}

// ---- Environment map sampling ----

static __forceinline__ __device__ float3 sample_envmap(float3 dir) {
  if (!params.envmap_data)
    return make_f3(params.ambient_light);

  float theta = acosf(fminf(fmaxf(dir.y, -1.0f), 1.0f));
  float phi = atan2f(dir.z, dir.x);
  float u = 0.5f + phi / (2.0f * M_PIf);
  float v = theta / M_PIf;

  int w = params.envmap_width;
  int h = params.envmap_height;
  float fx = u * (w - 1);
  float fy = v * (h - 1);
  int ix = (int)fx;
  int iy = (int)fy;
  float dx = fx - ix;
  float dy = fy - iy;
  ix = max(0, min(ix, w - 1));
  iy = max(0, min(iy, h - 1));
  int ix1 = min(ix + 1, w - 1);
  int iy1 = min(iy + 1, h - 1);

  const float *d = params.envmap_data;
  float3 c00 = make_float3(d[(iy * w + ix) * 3], d[(iy * w + ix) * 3 + 1],
                           d[(iy * w + ix) * 3 + 2]);
  float3 c10 = make_float3(d[(iy * w + ix1) * 3], d[(iy * w + ix1) * 3 + 1],
                           d[(iy * w + ix1) * 3 + 2]);
  float3 c01 = make_float3(d[(iy1 * w + ix) * 3], d[(iy1 * w + ix) * 3 + 1],
                           d[(iy1 * w + ix) * 3 + 2]);
  float3 c11 = make_float3(d[(iy1 * w + ix1) * 3], d[(iy1 * w + ix1) * 3 + 1],
                           d[(iy1 * w + ix1) * 3 + 2]);

  return c00 * (1 - dx) * (1 - dy) + c10 * dx * (1 - dy) + c01 * (1 - dx) * dy +
         c11 * dx * dy;
}

static __forceinline__ __device__ int cdf_search(const float *cdf, int n,
                                                 float u) {
  int lo = 0, hi = n - 1;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (cdf[mid + 1] <= u)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

static __forceinline__ __device__ void
sample_envmap_direction(float u1, float u2, float3 &dir, float3 &color,
                        float &pdf) {
  int w = params.envmap_width;
  int h = params.envmap_height;

  int y = cdf_search(params.envmap_marginal_cdf, h, u1);
  float cdf_y0 = params.envmap_marginal_cdf[y];
  float cdf_y1 = params.envmap_marginal_cdf[y + 1];
  float frac_v = (cdf_y1 > cdf_y0) ? (u1 - cdf_y0) / (cdf_y1 - cdf_y0) : 0.0f;
  float v = (y + frac_v) / (float)h;

  const float *row_cdf = params.envmap_conditional_cdf + y * (w + 1);
  int x = cdf_search(row_cdf, w, u2);
  float cdf_x0 = row_cdf[x];
  float cdf_x1 = row_cdf[x + 1];
  float frac_u = (cdf_x1 > cdf_x0) ? (u2 - cdf_x0) / (cdf_x1 - cdf_x0) : 0.0f;
  float u = (x + frac_u) / (float)w;

  float theta = v * M_PIf;
  float phi = (u - 0.5f) * 2.0f * M_PIf;
  float sin_theta = sinf(theta);
  dir = make_float3(cosf(phi) * sin_theta, cosf(theta), sinf(phi) * sin_theta);
  color = sample_envmap(dir);

  float marginal_pdf = (cdf_y1 - cdf_y0) * (float)h;
  float conditional_pdf = (cdf_x1 - cdf_x0) * (float)w;
  pdf = marginal_pdf * conditional_pdf /
        (2.0f * M_PIf * M_PIf * fmaxf(sin_theta, 1e-6f));
}

// ---- NEE (Next Event Estimation) ----

// Evaluate BRDF * NdotL for a given light direction.
// For diffuse: albedo * NdotL / PI
// For conductor: eval_conductor_brdf(...) * NdotL
static __forceinline__ __device__ float3
eval_brdf_cosine(float3 V, float3 L, float3 N, float3 albedo, float alpha_u,
                 float alpha_v, bool is_conductor, float3 eta, float3 k,
                 float3 geom_tangent = make_float3(0, 0, 0)) {
  float NdotL = dot3(N, L);
  if (NdotL <= 0.0f)
    return make_float3(0, 0, 0);
  if (is_conductor) {
    return eval_conductor_brdf(V, L, N, albedo, alpha_u, alpha_v, eta, k,
                               geom_tangent) *
           NdotL;
  } else {
    return albedo * (NdotL / M_PIf);
  }
}

static __forceinline__ __device__ float3
nee_distant_lights(float3 hit_pos, float3 hit_normal, float3 V, float3 albedo,
                   float alpha_u, float alpha_v, bool is_conductor, float3 eta,
                   float3 k, float3 geom_tangent = make_float3(0, 0, 0)) {
  float3 result = make_float3(0, 0, 0);
  for (int i = 0; i < params.num_distant_lights; i++) {
    float3 L = make_f3(params.distant_lights[i].direction);
    float3 bw = eval_brdf_cosine(V, L, hit_normal, albedo, alpha_u, alpha_v,
                                 is_conductor, eta, k, geom_tangent);
    if (bw.x <= 0 && bw.y <= 0 && bw.z <= 0)
      continue;
    ShadowResult sr =
        trace_shadow(hit_pos, L, 1e16f, OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT);
    if (!sr.hit) {
      result = result + bw * make_f3(params.distant_lights[i].emission);
    }
  }
  return result;
}

static __forceinline__ __device__ float3 nee_sphere_lights(
    float3 hit_pos, float3 hit_normal, float3 V, float3 albedo, float alpha_u,
    float alpha_v, bool is_conductor, float3 eta, float3 k, float3 geom_tangent,
    unsigned int pixel_idx, unsigned int sample_idx, unsigned int depth) {
  int n = params.num_sphere_lights;
  if (n == 0)
    return make_float3(0, 0, 0);

  const int max_samples = 8;
  int n_samples = min(n, max_samples);
  float weight = (float)n / (float)n_samples;

  RNG sphere_select_rng(pixel_idx * 31, sample_idx, depth + 100);
  float3 result = make_float3(0, 0, 0);

  for (int s = 0; s < n_samples; s++) {
    int i;
    if (n_samples == n) {
      i = s;
    } else {
      i = (int)(sphere_select_rng.next() * n);
      if (i >= n)
        i = n - 1;
    }

    RNG light_rng(pixel_idx * 31 + i, sample_idx, depth + 101);
    float3 light_center = make_f3(params.sphere_lights[i].center);
    float light_radius = params.sphere_lights[i].radius;
    float3 to_light = light_center - hit_pos;
    float dist_to_center = sqrtf(dot3(to_light, to_light));
    float3 light_dir_norm = to_light * (1.0f / dist_to_center);

    float sin_theta_max2 =
        (light_radius * light_radius) / (dist_to_center * dist_to_center);
    float cos_theta_max = sqrtf(fmaxf(0.0f, 1.0f - sin_theta_max2));

    float u1 = light_rng.next();
    float u2 = light_rng.next();
    float cos_theta = 1.0f - u1 + u1 * cos_theta_max;
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    float phi = 2.0f * M_PIf * u2;

    float3 tangent;
    if (fabsf(light_dir_norm.x) > 0.9f)
      tangent = normalize3(cross3(make_float3(0, 1, 0), light_dir_norm));
    else
      tangent = normalize3(cross3(make_float3(1, 0, 0), light_dir_norm));
    float3 bitangent = cross3(light_dir_norm, tangent);

    float3 L = normalize3(tangent * (cosf(phi) * sin_theta) +
                          bitangent * (sinf(phi) * sin_theta) +
                          light_dir_norm * cos_theta);

    float3 bw = eval_brdf_cosine(V, L, hit_normal, albedo, alpha_u, alpha_v,
                                 is_conductor, eta, k, geom_tangent);
    if (bw.x <= 0 && bw.y <= 0 && bw.z <= 0)
      continue;

    ShadowResult sr = trace_shadow(hit_pos, L, 1e16f, OPTIX_RAY_FLAG_NONE);
    if (sr.emission.x > 0 || sr.emission.y > 0 || sr.emission.z > 0) {
      float pdf = 1.0f / (2.0f * M_PIf * (1.0f - cos_theta_max));
      result = result + bw * sr.emission * (weight / pdf);
    }
  }
  return result;
}

static __forceinline__ __device__ float3 nee_triangle_lights(
    float3 hit_pos, float3 hit_normal, float3 V, float3 albedo, float alpha_u,
    float alpha_v, bool is_conductor, float3 eta, float3 k, float3 geom_tangent,
    unsigned int pixel_idx, unsigned int sample_idx, unsigned int depth) {
  int n_groups = params.num_triangle_light_groups;
  if (n_groups == 0)
    return make_float3(0, 0, 0);

  // Two-level sampling: pick light group (object), then triangle within it
  const int max_group_samples = 4;
  int n_group_samples = min(n_groups, max_group_samples);

  RNG light_select_rng(pixel_idx * 37, sample_idx, depth + 200);
  float3 result = make_float3(0, 0, 0);

  for (int gs = 0; gs < n_group_samples; gs++) {
    // Level 1: select a light group (object)
    int gi;
    float group_pdf;
    if (n_group_samples == n_groups) {
      gi = gs;
      group_pdf = 1.0f;
    } else {
      float xi = light_select_rng.next();
      gi = cdf_search(params.triangle_light_group_cdf, n_groups, xi);
      float cdf0 = params.triangle_light_group_cdf[gi];
      float cdf1 = params.triangle_light_group_cdf[gi + 1];
      group_pdf = (cdf1 - cdf0) * n_group_samples;
    }

    const TriangleLightGroup &grp = params.triangle_light_groups[gi];
    int tri_count = (int)grp.count;
    if (tri_count == 0)
      continue;

    // Level 2: sample a random triangle within the group (uniform by area)
    int tri_idx = (int)(light_select_rng.next() * tri_count);
    if (tri_idx >= tri_count)
      tri_idx = tri_count - 1;
    int i = (int)grp.start + tri_idx;

    RNG light_rng(pixel_idx * 37 + i, sample_idx, depth + 201);
    const float *verts = params.triangle_light_vertices;
    int vi0 = (grp.vertex_offset + params.triangle_lights[i].i0) * 3;
    int vi1 = (grp.vertex_offset + params.triangle_lights[i].i1) * 3;
    int vi2 = (grp.vertex_offset + params.triangle_lights[i].i2) * 3;
    float3 lv0 = make_float3(verts[vi0], verts[vi0 + 1], verts[vi0 + 2]);
    float3 lv1 = make_float3(verts[vi1], verts[vi1 + 1], verts[vi1 + 2]);
    float3 lv2 = make_float3(verts[vi2], verts[vi2 + 1], verts[vi2 + 2]);
    float3 light_em = make_f3(grp.emission);
    float3 cross_e = cross3(lv1 - lv0, lv2 - lv0);
    float cross_len = sqrtf(dot3(cross_e, cross_e));
    float light_area = cross_len * 0.5f;
    float3 light_normal = (cross_len > 0.0f) ? cross_e * (1.0f / cross_len)
                                             : make_float3(0, 1, 0);

    float u1 = light_rng.next();
    float u2 = light_rng.next();
    if (u1 + u2 > 1.0f) {
      u1 = 1.0f - u1;
      u2 = 1.0f - u2;
    }
    float3 light_pos = lv0 * (1.0f - u1 - u2) + lv1 * u1 + lv2 * u2;

    float3 to_light = light_pos - hit_pos;
    float dist2 = dot3(to_light, to_light);
    float dist = sqrtf(dist2);
    float3 L = to_light * (1.0f / dist);

    float lndotl = -dot3(light_normal, L);
    if (lndotl <= 0.0f)
      continue;

    float3 bw = eval_brdf_cosine(V, L, hit_normal, albedo, alpha_u, alpha_v,
                                 is_conductor, eta, k, geom_tangent);
    if (bw.x <= 0 && bw.y <= 0 && bw.z <= 0)
      continue;

    ShadowResult sr = trace_shadow(hit_pos, L, 1e16f, OPTIX_RAY_FLAG_NONE);
    bool unoccluded = !sr.hit || (sr.emission.x > 0 || sr.emission.y > 0 ||
                                  sr.emission.z > 0);
    if (unoccluded) {
      // Weight: area_geo / (group_pdf * (1/tri_count))
      float geo = light_area * lndotl / dist2;
      float tri_pdf = group_pdf / (float)tri_count;
      result = result + bw * light_em * (geo / fmaxf(tri_pdf, 1e-8f));
    }
  }
  return result;
}

// Portal-guided environment light NEE
static __forceinline__ __device__ float3 nee_portal(
    float3 hit_pos, float3 hit_normal, float3 V, float3 albedo, float alpha_u,
    float alpha_v, bool is_conductor, float3 eta, float3 k, float3 geom_tangent,
    unsigned int pixel_idx, unsigned int sample_idx, unsigned int depth) {
  if (!params.has_portal)
    return make_float3(0, 0, 0);

  // Portal quad vertices
  float3 p0 = make_float3(params.portal[0], params.portal[1], params.portal[2]);
  float3 p1 = make_float3(params.portal[3], params.portal[4], params.portal[5]);
  float3 p2 = make_float3(params.portal[6], params.portal[7], params.portal[8]);
  float3 p3 =
      make_float3(params.portal[9], params.portal[10], params.portal[11]);

  // Portal edges and normal
  float3 e1 = p1 - p0;
  float3 e2 = p3 - p0;
  float3 portal_normal = cross3(e1, e2);
  float portal_area = sqrtf(dot3(portal_normal, portal_normal));
  if (portal_area < 1e-8f)
    return make_float3(0, 0, 0);
  portal_normal = portal_normal * (1.0f / portal_area);

  RNG portal_rng(pixel_idx * 59, sample_idx, depth + 400);
  float u1 = portal_rng.next();
  float u2 = portal_rng.next();

  // Sample point on quad (bilinear interpolation)
  float3 portal_point = p0 + e1 * u1 + e2 * u2;
  float3 to_portal = portal_point - hit_pos;
  float dist2 = dot3(to_portal, to_portal);
  float dist = sqrtf(dist2);
  float3 L = to_portal * (1.0f / dist);

  // Geometry term: cos at portal surface
  float cos_portal = fabsf(dot3(portal_normal, L * (-1.0f)));
  if (cos_portal < 1e-6f)
    return make_float3(0, 0, 0);

  // PDF: 1/area * distance^2 / cos_portal = solid angle PDF
  float pdf = dist2 / (portal_area * cos_portal);
  if (pdf <= 0.0f)
    return make_float3(0, 0, 0);

  float3 bw = eval_brdf_cosine(V, L, hit_normal, albedo, alpha_u, alpha_v,
                               is_conductor, eta, k, geom_tangent);
  if (bw.x <= 0 && bw.y <= 0 && bw.z <= 0)
    return make_float3(0, 0, 0);

  // Shadow ray toward portal point
  ShadowResult sr = trace_shadow(hit_pos, L, dist - 0.01f, OPTIX_RAY_FLAG_NONE);
  if (!sr.hit) {
    // Sample environment light color in the portal direction
    float3 env_color = sample_envmap(L);
    return bw * env_color * (1.0f / pdf);
  }
  return make_float3(0, 0, 0);
}

static __forceinline__ __device__ float3 nee_envmap(
    float3 hit_pos, float3 hit_normal, float3 V, float3 albedo, float alpha_u,
    float alpha_v, bool is_conductor, float3 eta, float3 k, float3 geom_tangent,
    unsigned int pixel_idx, unsigned int sample_idx, unsigned int depth) {
  if (!params.envmap_data || params.envmap_integral <= 0.0f)
    return make_float3(0, 0, 0);

  RNG env_rng(pixel_idx * 41, sample_idx, depth + 300);
  float3 L, env_color;
  float env_pdf;
  sample_envmap_direction(env_rng.next(), env_rng.next(), L, env_color,
                          env_pdf);

  if (env_pdf <= 0.0f)
    return make_float3(0, 0, 0);

  float3 bw = eval_brdf_cosine(V, L, hit_normal, albedo, alpha_u, alpha_v,
                               is_conductor, eta, k, geom_tangent);
  if (bw.x <= 0 && bw.y <= 0 && bw.z <= 0)
    return make_float3(0, 0, 0);

  // Shadow ray — check if direction is unoccluded (misses all geometry)
  ShadowResult sr = trace_shadow(hit_pos, L, 1e16f, OPTIX_RAY_FLAG_NONE);
  if (!sr.hit) {
    return bw * env_color * (1.0f / env_pdf);
  }
  return make_float3(0, 0, 0);
}

static __forceinline__ __device__ float3 compute_nee(
    float3 hit_pos, float3 hit_normal, float3 V, float3 albedo, float alpha_u,
    float alpha_v, bool is_conductor, float3 eta, float3 k, float3 geom_tangent,
    unsigned int pixel_idx, unsigned int sample_idx, unsigned int depth) {
  return nee_distant_lights(hit_pos, hit_normal, V, albedo, alpha_u, alpha_v,
                            is_conductor, eta, k, geom_tangent) +
         nee_sphere_lights(hit_pos, hit_normal, V, albedo, alpha_u, alpha_v,
                           is_conductor, eta, k, geom_tangent, pixel_idx,
                           sample_idx, depth) +
         nee_triangle_lights(hit_pos, hit_normal, V, albedo, alpha_u, alpha_v,
                             is_conductor, eta, k, geom_tangent, pixel_idx,
                             sample_idx, depth) +
         nee_envmap(hit_pos, hit_normal, V, albedo, alpha_u, alpha_v,
                    is_conductor, eta, k, geom_tangent, pixel_idx, sample_idx,
                    depth) +
         nee_portal(hit_pos, hit_normal, V, albedo, alpha_u, alpha_v,
                    is_conductor, eta, k, geom_tangent, pixel_idx, sample_idx,
                    depth);
}

// ---- Shared material helpers for raygen ----

static __forceinline__ __device__ float coat_fresnel_f0(float eta) {
  return (eta - 1.0f) * (eta - 1.0f) / ((eta + 1.0f) * (eta + 1.0f));
}

static __forceinline__ __device__ bool has_complex_ior(float3 eta, float3 k) {
  return (eta.x > 0 || eta.y > 0 || eta.z > 0 || k.x > 0 || k.y > 0 || k.z > 0);
}

static __forceinline__ __device__ float3 compute_conductor_fresnel_rgb(
    float cos_i, float3 albedo, float3 eta, float3 k) {
  if (has_complex_ior(eta, k))
    return conductor_fresnel(cos_i, eta, k);
  else
    return schlick_fresnel_f0_rgb(cos_i, albedo);
}

// Shared conductor specular bounce with VNDF + G1(L) + energy compensation.
// Returns false if the bounce is invalid (below horizon).
static __forceinline__ __device__ bool
conductor_specular_bounce(float u1, float u2, float3 view_dir, float alpha_u,
                          float alpha_v, float3 hit_normal, float3 hit_tangent,
                          float3 hit_albedo, float3 cond_eta, float3 cond_k,
                          float3 &out_direction, float3 &throughput_factor) {
  float3 H = ggx_sample_vndf(u1, u2, view_dir, alpha_u, alpha_v, hit_normal,
                             hit_tangent);
  float VdotH = fmaxf(dot3(view_dir, H), 0.001f);
  float3 Fr =
      compute_conductor_fresnel_rgb(VdotH, hit_albedo, cond_eta, cond_k);

  out_direction = reflect3(view_dir * (-1.0f), H);
  float NdotL = dot3(out_direction, hit_normal);
  if (NdotL <= 0.0f)
    return false;

  // VNDF weight = F * G1(L)
  float G1_L;
  if (alpha_u == alpha_v) {
    G1_L = ggx_G1(NdotL, alpha_u);
  } else {
    float3 T, B;
    build_tangent_frame_geom(hit_normal, hit_tangent, T, B);
    G1_L = ggx_G1_aniso(out_direction, hit_normal, T, B, alpha_u, alpha_v);
  }

  // Kulla-Conty energy compensation
  float alpha_avg = 0.5f * (alpha_u + alpha_v);
  float NdotV = fmaxf(dot3(view_dir, hit_normal), 0.001f);
  float E_o = ggx_E(NdotV, alpha_avg);
  float3 F_a = conductor_F_avg(hit_albedo, cond_eta, cond_k);
  float ms_boost = (E_o > 0.001f) ? (1.0f - E_o) / E_o : 0.0f;
  throughput_factor = Fr * G1_L + F_a * fmaxf(ms_boost * G1_L, 0.0f);
  return true;
}

// Set material-independent payloads from closesthit data
static __forceinline__ __device__ void
set_common_payloads(const MaterialData *mat, float3 hit_pos, float3 normal,
                    float3 albedo, float roughness, float roughness_v,
                    float3 tangent) {
  if (mat->material_type == MAT_DIELECTRIC) {
    optixSetPayload_0(__float_as_uint(mat->dielectric.eta));
    optixSetPayload_1(__float_as_uint(mat->dielectric.tint[0]));
    optixSetPayload_2(__float_as_uint(mat->dielectric.tint[1]));
    optixSetPayload_13(__float_as_uint(mat->dielectric.tint[2]));
  } else {
    optixSetPayload_0(__float_as_uint(albedo.x));
    optixSetPayload_1(__float_as_uint(albedo.y));
    optixSetPayload_2(__float_as_uint(albedo.z));
    optixSetPayload_13(__float_as_uint(roughness));
  }

  optixSetPayload_3(__float_as_uint(hit_pos.x));
  optixSetPayload_4(__float_as_uint(hit_pos.y));
  optixSetPayload_5(__float_as_uint(hit_pos.z));
  optixSetPayload_6(__float_as_uint(normal.x));
  optixSetPayload_7(__float_as_uint(normal.y));
  optixSetPayload_8(__float_as_uint(normal.z));
  optixSetPayload_9((unsigned int)mat->material_type);
  optixSetPayload_10(__float_as_uint(mat->emission[0]));
  optixSetPayload_11(__float_as_uint(mat->emission[1]));
  optixSetPayload_12(__float_as_uint(mat->emission[2]));

  float rv = roughness_v;
  if (rv == 0.0f)
    rv = roughness;
  optixSetPayload_22(__float_as_uint(rv));

  if (mat->material_type == MAT_CONDUCTOR ||
      mat->material_type == MAT_COATED_CONDUCTOR) {
    optixSetPayload_14(__float_as_uint(mat->conductor.eta[0]));
    optixSetPayload_15(__float_as_uint(mat->conductor.eta[1]));
    optixSetPayload_16(__float_as_uint(mat->conductor.eta[2]));
    optixSetPayload_17(__float_as_uint(mat->conductor.k[0]));
    optixSetPayload_18(__float_as_uint(mat->conductor.k[1]));
    optixSetPayload_19(__float_as_uint(mat->conductor.k[2]));
  }
  if (mat->material_type == MAT_COATED_CONDUCTOR ||
      mat->material_type == MAT_COATED_DIFFUSE) {
    optixSetPayload_20(__float_as_uint(mat->coat_roughness));
    optixSetPayload_21(__float_as_uint(mat->coat_eta));
  }
  if (mat->material_type == MAT_COATED_CONDUCTOR) {
    optixSetPayload_26(__float_as_uint(mat->coat_thickness));
    optixSetPayload_27(__float_as_uint(mat->coat_albedo[0]));
    optixSetPayload_28(__float_as_uint(mat->coat_albedo[1]));
    optixSetPayload_29(__float_as_uint(mat->coat_albedo[2]));
  }

  optixSetPayload_23(__float_as_uint(tangent.x));
  optixSetPayload_24(__float_as_uint(tangent.y));
  optixSetPayload_25(__float_as_uint(tangent.z));
}

// ---- Pack/unpack color ----

// sRGB transfer function (linear to sRGB)
static __forceinline__ __device__ float linear_to_srgb(float x) {
  if (x <= 0.0f)
    return 0.0f;
  if (x >= 1.0f)
    return 1.0f;
  if (x <= 0.0031308f)
    return 12.92f * x;
  return 1.055f * powf(x, 1.0f / 2.4f) - 0.055f;
}

static __forceinline__ __device__ unsigned int packColor(float3 c) {
  unsigned int ir = (unsigned int)(linear_to_srgb(c.x) * 255.0f);
  unsigned int ig = (unsigned int)(linear_to_srgb(c.y) * 255.0f);
  unsigned int ib = (unsigned int)(linear_to_srgb(c.z) * 255.0f);
  return (255u << 24) | (ib << 16) | (ig << 8) | ir;
}

// ---- Programs ----

extern "C" __global__ void __raygen__rg() {
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDimensions();
  const unsigned int pixel_idx = idx.y * dim.x + idx.x;

  float3 accum = make_float3(0, 0, 0);
  const unsigned int spp = params.samples_per_pixel;

  for (unsigned int s = 0; s < spp; s++) {
    RNG rng(pixel_idx, s, 0);

    float u = ((float)idx.x + rng.next()) / (float)dim.x;
    float v = ((float)idx.y + rng.next()) / (float)dim.y;
    float2 d = make_float2(u * 2.0f - 1.0f, v * 2.0f - 1.0f);

    float3 origin = make_f3(params.cam_eye);
    float3 U = make_f3(params.cam_u);
    float3 V = make_f3(params.cam_v);
    float3 W = make_f3(params.cam_w);
    float3 direction = normalize3(d.x * U + d.y * V + W);

    float3 throughput = make_float3(1, 1, 1);
    float3 radiance = make_float3(0, 0, 0);
    bool specular_bounce = true; // camera ray counts as specular

    for (unsigned int depth = 0; depth < params.max_depth; depth++) {
      unsigned int p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13;
      unsigned int p14 = 0, p15 = 0, p16 = 0, p17 = 0, p18 = 0, p19 = 0;
      unsigned int p20 = 0, p21 = 0, p22 = 0, p23 = 0, p24 = 0, p25 = 0;
      unsigned int p26 = 0, p27 = 0, p28 = 0, p29 = 0;
      p9 = 0xFFFFFFFF; // miss sentinel
      p10 = p11 = p12 = p13 = 0;

      optixTrace(params.traversable, origin, direction, 0.001f, 1e16f, 0.0f,
                 OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, p0, p1,
                 p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
                 p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27,
                 p28, p29);

      if (p9 == 0xFFFFFFFF) {
        // Miss - sample environment map or use constant ambient
        // Only add envmap on camera rays or specular bounces to avoid
        // double-counting with envmap NEE on diffuse paths
        if (specular_bounce || !params.envmap_data) {
          float3 bg = sample_envmap(direction);
          radiance = radiance + throughput * bg;
        }
        break;
      }

      float3 hit_pos = make_float3(__uint_as_float(p3), __uint_as_float(p4),
                                   __uint_as_float(p5));
      float3 hit_normal = make_float3(__uint_as_float(p6), __uint_as_float(p7),
                                      __uint_as_float(p8));
      float3 hit_albedo = make_float3(__uint_as_float(p0), __uint_as_float(p1),
                                      __uint_as_float(p2));
      float3 hit_emission = make_float3(
          __uint_as_float(p10), __uint_as_float(p11), __uint_as_float(p12));
      int mat_type = (int)p9;
      float3 hit_tangent = make_float3(
          __uint_as_float(p23), __uint_as_float(p24), __uint_as_float(p25));

      // Add emission on camera ray or after specular bounces.
      // Diffuse NEE already accounts for lights, so skip after diffuse bounce.
      if (specular_bounce &&
          (hit_emission.x > 0 || hit_emission.y > 0 || hit_emission.z > 0)) {
        radiance = radiance + throughput * hit_emission;
      }

      float3 zero3 = make_float3(0, 0, 0);

      // Common setup
      float3 view_dir = direction * (-1.0f);
      RNG bounce_rng(pixel_idx, s, depth + 1);

      if (mat_type == MAT_DIFFUSE) {
        radiance = radiance +
                   throughput * compute_nee(hit_pos, hit_normal, view_dir,
                                            hit_albedo, 0, 0, false, zero3,
                                            zero3, zero3, pixel_idx, s, depth);
        direction = cosine_sample_hemisphere(bounce_rng.next(),
                                             bounce_rng.next(), hit_normal);
        origin = hit_pos;
        throughput = throughput * hit_albedo;
        specular_bounce = false;
      } else if (mat_type == MAT_COATED_DIFFUSE) {
        float alpha = fmaxf(__uint_as_float(p13), 0.001f);
        float ceta = __uint_as_float(p21);
        if (ceta == 0.0f)
          ceta = 1.5f;
        float cos_i = fmaxf(fabsf(dot3(view_dir, hit_normal)), 0.001f);
        float F = fresnel_schlick_f0(cos_i, coat_fresnel_f0(ceta));

        if (bounce_rng.next() < F) {
          float3 H = ggx_sample(bounce_rng.next(), bounce_rng.next(), alpha,
                                view_dir, hit_normal);
          direction = reflect3(direction, H);
          if (dot3(direction, hit_normal) <= 0.0f)
            break;
          origin = hit_pos;
          specular_bounce = true;
        } else {
          radiance =
              radiance + throughput * compute_nee(hit_pos, hit_normal, view_dir,
                                                  hit_albedo, 0, 0, false,
                                                  zero3, zero3, zero3,
                                                  pixel_idx, s, depth);
          direction = cosine_sample_hemisphere(bounce_rng.next(),
                                               bounce_rng.next(), hit_normal);
          origin = hit_pos;
          throughput = throughput * hit_albedo;
          specular_bounce = false;
        }
      } else if (mat_type == MAT_CONDUCTOR) {
        float alpha_u = fmaxf(__uint_as_float(p13), 0.001f);
        float alpha_v = fmaxf(__uint_as_float(p22), 0.001f);
        float3 cond_eta = make_float3(
            __uint_as_float(p14), __uint_as_float(p15), __uint_as_float(p16));
        float3 cond_k = make_float3(__uint_as_float(p17), __uint_as_float(p18),
                                    __uint_as_float(p19));

        radiance =
            radiance +
            throughput * compute_nee(hit_pos, hit_normal, view_dir, hit_albedo,
                                     alpha_u, alpha_v, true, cond_eta, cond_k,
                                     hit_tangent, pixel_idx, s, depth);

        float3 bounce_dir, bounce_weight;
        if (!conductor_specular_bounce(bounce_rng.next(), bounce_rng.next(),
                                       view_dir, alpha_u, alpha_v, hit_normal,
                                       hit_tangent, hit_albedo, cond_eta,
                                       cond_k, bounce_dir, bounce_weight))
          break;
        direction = bounce_dir;
        origin = hit_pos;
        throughput = throughput * bounce_weight;
        specular_bounce = true;
      } else if (mat_type == MAT_COATED_CONDUCTOR) {
        float cond_alpha_u = fmaxf(__uint_as_float(p13), 0.001f);
        float cond_alpha_v = fmaxf(__uint_as_float(p22), 0.001f);
        float coat_alpha = fmaxf(__uint_as_float(p20), 0.001f);
        float coat_eta_val = __uint_as_float(p21);
        float coat_thick = __uint_as_float(p26);
        float3 coat_alb = make_float3(
            __uint_as_float(p27), __uint_as_float(p28), __uint_as_float(p29));
        float3 cond_eta = make_float3(
            __uint_as_float(p14), __uint_as_float(p15), __uint_as_float(p16));
        float3 cond_k = make_float3(__uint_as_float(p17), __uint_as_float(p18),
                                    __uint_as_float(p19));

        float cos_i = fmaxf(fabsf(dot3(view_dir, hit_normal)), 0.001f);
        float coat_F = fresnel_schlick_f0(cos_i, coat_fresnel_f0(coat_eta_val));

        if (bounce_rng.next() < coat_F) {
          float3 H = ggx_sample(bounce_rng.next(), bounce_rng.next(),
                                coat_alpha, view_dir, hit_normal);
          direction = reflect3(direction, H);
          if (dot3(direction, hit_normal) <= 0.0f)
            break;
          origin = hit_pos;
          specular_bounce = true;
        } else {
          // Coating absorption: Beer's law attenuation through the coating
          // layer Path length through coating ≈ thickness / cos(theta)
          float coat_path = coat_thick / fmaxf(cos_i, 0.01f);
          float3 coat_absorption = make_float3(expf(-coat_alb.x * coat_path),
                                               expf(-coat_alb.y * coat_path),
                                               expf(-coat_alb.z * coat_path));

          // Multi-layer scattering: geometric series for inter-layer bounces
          float coat_F_inner = coat_F;
          float3 Fr_avg = conductor_F_avg(hit_albedo, cond_eta, cond_k);
          float3 coat_tp =
              make_float3((1.0f - coat_F_inner) /
                              fmaxf(1.0f - coat_F_inner * Fr_avg.x, 0.01f),
                          (1.0f - coat_F_inner) /
                              fmaxf(1.0f - coat_F_inner * Fr_avg.y, 0.01f),
                          (1.0f - coat_F_inner) /
                              fmaxf(1.0f - coat_F_inner * Fr_avg.z, 0.01f));
          coat_tp = coat_tp * coat_absorption;

          radiance = radiance +
                     throughput * coat_tp *
                         compute_nee(hit_pos, hit_normal, view_dir, hit_albedo,
                                     cond_alpha_u, cond_alpha_v, true, cond_eta,
                                     cond_k, hit_tangent, pixel_idx, s, depth);

          float3 bounce_dir, bounce_weight;
          if (!conductor_specular_bounce(
                  bounce_rng.next(), bounce_rng.next(), view_dir, cond_alpha_u,
                  cond_alpha_v, hit_normal, hit_tangent, hit_albedo, cond_eta,
                  cond_k, bounce_dir, bounce_weight))
            break;
          direction = bounce_dir;
          origin = hit_pos;
          throughput = throughput * coat_tp * bounce_weight;
          specular_bounce = true;
        }
      } else if (mat_type == MAT_DIELECTRIC) {
        float eta_val = __uint_as_float(p0);
        float3 tint = make_float3(__uint_as_float(p1), __uint_as_float(p2),
                                  __uint_as_float(p13));
        RNG bounce_rng(pixel_idx, s, depth + 1);

        bool front_face = dot3(direction, hit_normal) < 0.0f;
        float3 outward_normal = front_face ? hit_normal : hit_normal * (-1.0f);
        float ratio = front_face ? (1.0f / eta_val) : eta_val;

        float cos_i = fminf(fabsf(dot3(direction, outward_normal)), 1.0f);
        float reflect_prob = fresnel_schlick(cos_i, ratio);

        float3 refracted;
        bool can_refract =
            refract3(direction * (-1.0f), outward_normal, ratio, refracted);

        if (!can_refract || bounce_rng.next() < reflect_prob) {
          direction = reflect3(direction, outward_normal);
        } else {
          direction = normalize3(refracted);
          // Apply absorption tint when entering the medium
          if (front_face) {
            throughput = throughput * tint;
          }
        }
        origin = hit_pos;
        specular_bounce = true;
      }
    }

    accum = accum + radiance;
  }

  float3 color = accum * (1.0f / (float)spp);
  params.image[pixel_idx] = packColor(color);
}

extern "C" __global__ void __miss__ms() {
  // Signal miss via payload 9
  optixSetPayload_9(0xFFFFFFFF);
}

extern "C" __global__ void __closesthit__ch() {
  const HitGroupData *data =
      reinterpret_cast<const HitGroupData *>(optixGetSbtDataPointer());
  const MaterialData *mat = data->mat;
  const float2 bary = optixGetTriangleBarycentrics();
  const int prim_idx = optixGetPrimitiveIndex();

  // Compute hit position
  const float t = optixGetRayTmax();
  const float3 ray_orig = optixGetWorldRayOrigin();
  const float3 ray_dir = optixGetWorldRayDirection();
  float3 hit_pos = ray_orig + ray_dir * t;

  // Compute vertex indices (shared by position, normal, texcoord lookups)
  int idx0, idx1, idx2;
  if (data->indices) {
    idx0 = data->indices[prim_idx * 3 + 0];
    idx1 = data->indices[prim_idx * 3 + 1];
    idx2 = data->indices[prim_idx * 3 + 2];
  } else {
    idx0 = prim_idx * 3 + 0;
    idx1 = prim_idx * 3 + 1;
    idx2 = prim_idx * 3 + 2;
  }

  // Stochastic mix material selection (resolves nested mix tree)
  {
    unsigned int mix_hash = __float_as_uint(ray_orig.x) ^
                            (__float_as_uint(ray_orig.y) * 2654435761u) ^
                            (__float_as_uint(ray_dir.x) * 2246822519u) ^
                            ((unsigned int)prim_idx * 3266489917u);
    while (mat->mix_mat1 && mat->mix_mat2) {
      float amount = mat->mix_amount_value;
      if (mat->mix_amount_data && data->texcoords) {
        float wt_ = 1.0f - bary.x - bary.y;
        float mu = wt_ * data->texcoords[idx0 * 2] +
                   bary.x * data->texcoords[idx1 * 2] +
                   bary.y * data->texcoords[idx2 * 2];
        float mv = wt_ * data->texcoords[idx0 * 2 + 1] +
                   bary.x * data->texcoords[idx1 * 2 + 1] +
                   bary.y * data->texcoords[idx2 * 2 + 1];
        mu = mu - floorf(mu);
        mv = mv - floorf(mv);
        int mx = max(0, min((int)(mu * (mat->mix_amount_width - 1)),
                            mat->mix_amount_width - 1));
        int my = max(0, min((int)((1.0f - mv) * (mat->mix_amount_height - 1)),
                            mat->mix_amount_height - 1));
        amount = mat->mix_amount_data[my * mat->mix_amount_width + mx];
      }
      // Different random per nesting level
      mix_hash ^= mix_hash >> 16;
      mix_hash *= 0x45d9f3b;
      mix_hash ^= mix_hash >> 16;
      float rnd = (mix_hash >> 8) * (1.0f / 16777216.0f);
      mat = (rnd < amount) ? mat->mix_mat2 : mat->mix_mat1;
    }
  }

  // Load triangle vertices
  float3 v0, v1, v2;
  if (data->indices) {
    v0 = make_float3(data->vertices[idx0 * 3], data->vertices[idx0 * 3 + 1],
                     data->vertices[idx0 * 3 + 2]);
    v1 = make_float3(data->vertices[idx1 * 3], data->vertices[idx1 * 3 + 1],
                     data->vertices[idx1 * 3 + 2]);
    v2 = make_float3(data->vertices[idx2 * 3], data->vertices[idx2 * 3 + 1],
                     data->vertices[idx2 * 3 + 2]);
  } else {
    v0 = make_float3(data->vertices[prim_idx * 9 + 0],
                     data->vertices[prim_idx * 9 + 1],
                     data->vertices[prim_idx * 9 + 2]);
    v1 = make_float3(data->vertices[prim_idx * 9 + 3],
                     data->vertices[prim_idx * 9 + 4],
                     data->vertices[prim_idx * 9 + 5]);
    v2 = make_float3(data->vertices[prim_idx * 9 + 6],
                     data->vertices[prim_idx * 9 + 7],
                     data->vertices[prim_idx * 9 + 8]);
  }

  // Load per-vertex normals (if available)
  float3 n0, n1, n2;
  bool has_normals = (data->normals != 0);
  if (has_normals) {
    n0 = make_float3(data->normals[idx0 * 3], data->normals[idx0 * 3 + 1],
                     data->normals[idx0 * 3 + 2]);
    n1 = make_float3(data->normals[idx1 * 3], data->normals[idx1 * 3 + 1],
                     data->normals[idx1 * 3 + 2]);
    n2 = make_float3(data->normals[idx2 * 3], data->normals[idx2 * 3 + 1],
                     data->normals[idx2 * 3 + 2]);
  }

  float3 shading_normal;
  float wt = 1.0f - bary.x - bary.y;
  if (has_normals) {
    shading_normal = normalize3(n0 * wt + n1 * bary.x + n2 * bary.y);
  } else {
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    shading_normal = normalize3(cross3(edge1, edge2));
  }

  // Ensure normal faces the ray
  if (dot3(shading_normal, ray_dir) > 0.0f)
    shading_normal = shading_normal * (-1.0f);

  // Compute tangent (dpdu) from triangle UVs for anisotropic GGX
  float3 geom_tangent = make_float3(0, 0, 0);
  if (data->texcoords) {
    float2 uv0 =
        make_float2(data->texcoords[idx0 * 2], data->texcoords[idx0 * 2 + 1]);
    float2 uv1 =
        make_float2(data->texcoords[idx1 * 2], data->texcoords[idx1 * 2 + 1]);
    float2 uv2 =
        make_float2(data->texcoords[idx2 * 2], data->texcoords[idx2 * 2 + 1]);
    float3 dp1 = v1 - v0, dp2 = v2 - v0;
    float duv1_u = uv1.x - uv0.x, duv1_v = uv1.y - uv0.y;
    float duv2_u = uv2.x - uv0.x, duv2_v = uv2.y - uv0.y;
    float det = duv1_u * duv2_v - duv1_v * duv2_u;
    if (fabsf(det) > 1e-8f) {
      float inv_det = 1.0f / det;
      geom_tangent = normalize3((dp1 * duv2_v - dp2 * duv1_v) * inv_det);
    }
  }

  // Bump mapping (matches PBRT's BumpMap)
  if (mat->bump_data && data->texcoords) {
    // Interpolate UVs
    float2 uv0 =
        make_float2(data->texcoords[idx0 * 2], data->texcoords[idx0 * 2 + 1]);
    float2 uv1 =
        make_float2(data->texcoords[idx1 * 2], data->texcoords[idx1 * 2 + 1]);
    float2 uv2 =
        make_float2(data->texcoords[idx2 * 2], data->texcoords[idx2 * 2 + 1]);
    float u_coord = wt * uv0.x + bary.x * uv1.x + bary.y * uv2.x;
    float v_coord = wt * uv0.y + bary.x * uv1.y + bary.y * uv2.y;

    // Compute UV Jacobian (dpdu, dpdv, dndu, dndv)
    float3 dp1 = v1 - v0, dp2 = v2 - v0;
    float duv1_u = uv1.x - uv0.x, duv1_v = uv1.y - uv0.y;
    float duv2_u = uv2.x - uv0.x, duv2_v = uv2.y - uv0.y;
    float det = duv1_u * duv2_v - duv1_v * duv2_u;

    float3 dpdu, dpdv, dndu, dndv;
    bool degenerate_uv = (fabsf(det) < 1e-8f);
    if (!degenerate_uv) {
      float inv_det = 1.0f / det;
      dpdu = (dp1 * duv2_v - dp2 * duv1_v) * inv_det;
      dpdv = (dp2 * duv1_u - dp1 * duv2_u) * inv_det;
      // Compute dndu/dndv from vertex normals
      if (has_normals) {
        float3 dn1 = n1 - n0, dn2 = n2 - n0;
        dndu = (dn1 * duv2_v - dn2 * duv1_v) * inv_det;
        dndv = (dn2 * duv1_u - dn1 * duv2_u) * inv_det;
      } else {
        dndu = make_float3(0, 0, 0);
        dndv = make_float3(0, 0, 0);
      }
    } else {
      // Degenerate UVs: fall back to arbitrary tangent frame
      if (fabsf(shading_normal.x) > 0.9f)
        dpdu = normalize3(cross3(make_float3(0, 1, 0), shading_normal));
      else
        dpdu = normalize3(cross3(make_float3(1, 0, 0), shading_normal));
      dpdv = cross3(shading_normal, dpdu);
      dndu = make_float3(0, 0, 0);
      dndv = make_float3(0, 0, 0);
    }

    // Compute ray differentials for adaptive finite-difference step
    // For primary rays: dpdx/dpdy from camera pixel spacing
    // For bounced rays: this is an approximation (correct scale, wrong
    // direction)
    float3 cam_u = make_f3(params.cam_u);
    float3 cam_v = make_f3(params.cam_v);
    float3 dDdx = cam_u * (2.0f / (float)params.width);
    float3 dDdy = cam_v * (2.0f / (float)params.height);

    // Intersect offset rays with tangent plane at hit point to get dpdx/dpdy
    float3 N = shading_normal;
    float NdotD = dot3(N, ray_dir);
    float3 dpdx, dpdy;
    if (fabsf(NdotD) > 1e-8f) {
      // Offset ray: origin + t_x * (direction + dDdx) lands on tangent plane
      float t_x = t * dot3(N, ray_dir) / dot3(N, ray_dir + dDdx);
      float t_y = t * dot3(N, ray_dir) / dot3(N, ray_dir + dDdy);
      dpdx = (ray_dir + dDdx) * t_x - ray_dir * t;
      dpdy = (ray_dir + dDdy) * t_y - ray_dir * t;
    } else {
      dpdx = dDdx * t;
      dpdy = dDdy * t;
    }

    // Compute dudx/dudy/dvdx/dvdy by projecting dpdx/dpdy onto UV space
    // Pick 2D plane where normal has smallest component (best-conditioned)
    float du = 0.0005f, dv = 0.0005f;
    if (!degenerate_uv) {
      int dim0, dim1;
      float anx = fabsf(N.x), any = fabsf(N.y), anz = fabsf(N.z);
      if (anx > any && anx > anz) {
        dim0 = 1;
        dim1 = 2;
      } else if (any > anz) {
        dim0 = 0;
        dim1 = 2;
      } else {
        dim0 = 0;
        dim1 = 1;
      }

      // Access vector components by index
      float dpdu_d[3] = {dpdu.x, dpdu.y, dpdu.z};
      float dpdv_d[3] = {dpdv.x, dpdv.y, dpdv.z};
      float dpdx_d[3] = {dpdx.x, dpdx.y, dpdx.z};
      float dpdy_d[3] = {dpdy.x, dpdy.y, dpdy.z};

      // Solve 2x2 system: [dpdu dpdv] * [du; dv] = dp  (projected onto dim0,
      // dim1)
      float a00 = dpdu_d[dim0], a01 = dpdv_d[dim0];
      float a10 = dpdu_d[dim1], a11 = dpdv_d[dim1];
      float det2 = a00 * a11 - a01 * a10;

      if (fabsf(det2) > 1e-10f) {
        float inv2 = 1.0f / det2;
        float dudx = (a11 * dpdx_d[dim0] - a01 * dpdx_d[dim1]) * inv2;
        float dvdx = (-a10 * dpdx_d[dim0] + a00 * dpdx_d[dim1]) * inv2;
        float dudy = (a11 * dpdy_d[dim0] - a01 * dpdy_d[dim1]) * inv2;
        float dvdy = (-a10 * dpdy_d[dim0] + a00 * dpdy_d[dim1]) * inv2;

        du = 0.5f * (fabsf(dudx) + fabsf(dudy));
        dv = 0.5f * (fabsf(dvdx) + fabsf(dvdy));
        if (du == 0.0f)
          du = 0.0005f;
        if (dv == 0.0f)
          dv = 0.0005f;
      }
    }

    // Sample displacement texture
    int bw_ = mat->bump_width;
    int bh_ = mat->bump_height;
    auto sample_bump = [&](float su, float sv) -> float {
      su = su - floorf(su);
      sv = sv - floorf(sv);
      int ix = (int)(su * (bw_ - 1));
      int iy = (int)((1.0f - sv) * (bh_ - 1));
      ix = max(0, min(ix, bw_ - 1));
      iy = max(0, min(iy, bh_ - 1));
      return mat->bump_data[iy * bw_ + ix];
    };

    float displace = sample_bump(u_coord, v_coord);
    float u_displace = sample_bump(u_coord + du, v_coord);
    float v_displace = sample_bump(u_coord, v_coord + dv);

    // PBRT formula:
    // dpdu' = dpdu + (uDisplace - displace) / du * N + displace * dndu
    // dpdv' = dpdv + (vDisplace - displace) / dv * N + displace * dndv
    float3 dpdu_bumped =
        dpdu + N * ((u_displace - displace) / du) + dndu * displace;
    float3 dpdv_bumped =
        dpdv + N * ((v_displace - displace) / dv) + dndv * displace;

    shading_normal = normalize3(cross3(dpdu_bumped, dpdv_bumped));
    if (dot3(shading_normal, ray_dir) > 0.0f)
      shading_normal = shading_normal * (-1.0f);
  }

  float3 albedo = make_f3(mat->albedo);

  // Checkerboard procedural texture
  if (mat->material_type == MAT_DIFFUSE && mat->diffuse.has_checkerboard) {
    float u_coord = 0.0f, v_coord = 0.0f;
    if (data->texcoords) {
      u_coord = wt * data->texcoords[idx0 * 2] +
                bary.x * data->texcoords[idx1 * 2] +
                bary.y * data->texcoords[idx2 * 2];
      v_coord = wt * data->texcoords[idx0 * 2 + 1] +
                bary.x * data->texcoords[idx1 * 2 + 1] +
                bary.y * data->texcoords[idx2 * 2 + 1];
    }
    u_coord *= mat->diffuse.checker_scale_u;
    v_coord *= mat->diffuse.checker_scale_v;
    int cu = (int)floorf(u_coord);
    int cv = (int)floorf(v_coord);
    if (((cu ^ cv) & 1) == 0)
      albedo = make_f3(mat->diffuse.checker_color1);
    else
      albedo = make_f3(mat->diffuse.checker_color2);
  }

  // Image texture sampling
  if (mat->texture_data && data->texcoords) {
    float u_coord = wt * data->texcoords[idx0 * 2] +
                    bary.x * data->texcoords[idx1 * 2] +
                    bary.y * data->texcoords[idx2 * 2];
    float v_coord = wt * data->texcoords[idx0 * 2 + 1] +
                    bary.x * data->texcoords[idx1 * 2 + 1] +
                    bary.y * data->texcoords[idx2 * 2 + 1];

    // Wrap UVs
    u_coord = u_coord - floorf(u_coord);
    v_coord = v_coord - floorf(v_coord);

    // Bilinear sample
    float fx = u_coord * (mat->texture_width - 1);
    float fy = (1.0f - v_coord) * (mat->texture_height - 1); // flip V
    int ix = (int)fx;
    int iy = (int)fy;
    float dx = fx - ix;
    float dy = fy - iy;
    int ix1 = min(ix + 1, mat->texture_width - 1);
    int iy1 = min(iy + 1, mat->texture_height - 1);

    int tw = mat->texture_width;
    float3 c00 = make_float3(mat->texture_data[(iy * tw + ix) * 3],
                             mat->texture_data[(iy * tw + ix) * 3 + 1],
                             mat->texture_data[(iy * tw + ix) * 3 + 2]);
    float3 c10 = make_float3(mat->texture_data[(iy * tw + ix1) * 3],
                             mat->texture_data[(iy * tw + ix1) * 3 + 1],
                             mat->texture_data[(iy * tw + ix1) * 3 + 2]);
    float3 c01 = make_float3(mat->texture_data[(iy1 * tw + ix) * 3],
                             mat->texture_data[(iy1 * tw + ix) * 3 + 1],
                             mat->texture_data[(iy1 * tw + ix) * 3 + 2]);
    float3 c11 = make_float3(mat->texture_data[(iy1 * tw + ix1) * 3],
                             mat->texture_data[(iy1 * tw + ix1) * 3 + 1],
                             mat->texture_data[(iy1 * tw + ix1) * 3 + 2]);

    albedo = c00 * (1 - dx) * (1 - dy) + c10 * dx * (1 - dy) +
             c01 * (1 - dx) * dy + c11 * dx * dy;
  }

  // Roughness: sample from texture if available, otherwise use constant
  float roughness_val = mat->roughness;
  if (mat->roughness_data && data->texcoords) {
    float ru = wt * data->texcoords[idx0 * 2] +
               bary.x * data->texcoords[idx1 * 2] +
               bary.y * data->texcoords[idx2 * 2];
    float rv = wt * data->texcoords[idx0 * 2 + 1] +
               bary.x * data->texcoords[idx1 * 2 + 1] +
               bary.y * data->texcoords[idx2 * 2 + 1];
    ru = ru - floorf(ru);
    rv = rv - floorf(rv);
    int rx = max(0, min((int)(ru * (mat->roughness_width - 1)),
                        mat->roughness_width - 1));
    int ry = max(0, min((int)((1.0f - rv) * (mat->roughness_height - 1)),
                        mat->roughness_height - 1));
    roughness_val = mat->roughness_data[ry * mat->roughness_width + rx];
  }

  // Normal mapping: perturb shading normal using tangent-space normal map
  if (mat->normalmap_data && data->texcoords) {
    float nu = wt * data->texcoords[idx0 * 2] +
               bary.x * data->texcoords[idx1 * 2] +
               bary.y * data->texcoords[idx2 * 2];
    float nv = wt * data->texcoords[idx0 * 2 + 1] +
               bary.x * data->texcoords[idx1 * 2 + 1] +
               bary.y * data->texcoords[idx2 * 2 + 1];
    nu = nu - floorf(nu);
    nv = nv - floorf(nv);

    // Bilinear sample normal map
    int nw = mat->normalmap_width;
    int nh = mat->normalmap_height;
    float nfx = nu * (nw - 1);
    float nfy = (1.0f - nv) * (nh - 1);
    int nix = max(0, min((int)nfx, nw - 1));
    int niy = max(0, min((int)nfy, nh - 1));
    int nix1 = min(nix + 1, nw - 1);
    int niy1 = min(niy + 1, nh - 1);
    float ndx = nfx - nix;
    float ndy = nfy - niy;
    const float *nd = mat->normalmap_data;
    float3 nm00 =
        make_float3(nd[(niy * nw + nix) * 3], nd[(niy * nw + nix) * 3 + 1],
                    nd[(niy * nw + nix) * 3 + 2]);
    float3 nm10 =
        make_float3(nd[(niy * nw + nix1) * 3], nd[(niy * nw + nix1) * 3 + 1],
                    nd[(niy * nw + nix1) * 3 + 2]);
    float3 nm01 =
        make_float3(nd[(niy1 * nw + nix) * 3], nd[(niy1 * nw + nix) * 3 + 1],
                    nd[(niy1 * nw + nix) * 3 + 2]);
    float3 nm11 =
        make_float3(nd[(niy1 * nw + nix1) * 3], nd[(niy1 * nw + nix1) * 3 + 1],
                    nd[(niy1 * nw + nix1) * 3 + 2]);
    float3 nm_sample = nm00 * (1 - ndx) * (1 - ndy) + nm10 * ndx * (1 - ndy) +
                       nm01 * (1 - ndx) * ndy + nm11 * ndx * ndy;

    // Remap [0,1] → [-1,1]
    float3 ts_normal = normalize3(make_float3(nm_sample.x * 2.0f - 1.0f,
                                              nm_sample.y * 2.0f - 1.0f,
                                              nm_sample.z * 2.0f - 1.0f));

    // Build TBN from geom_tangent and shading_normal
    float3 T, B;
    build_tangent_frame_geom(shading_normal, geom_tangent, T, B);

    // Transform tangent-space normal to world space
    shading_normal = normalize3(T * ts_normal.x + B * ts_normal.y +
                                shading_normal * ts_normal.z);

    if (dot3(shading_normal, ray_dir) > 0.0f)
      shading_normal = shading_normal * (-1.0f);
  }

  set_common_payloads(mat, hit_pos, shading_normal, albedo, roughness_val,
                      mat->roughness_v, geom_tangent);
}

// Closest hit for sphere
extern "C" __global__ void __closesthit__sphere() {
  const HitGroupData *data =
      reinterpret_cast<const HitGroupData *>(optixGetSbtDataPointer());
  const MaterialData *mat = data->mat;
  const float t = optixGetRayTmax();
  const float3 ray_orig = optixGetWorldRayOrigin();
  const float3 ray_dir = optixGetWorldRayDirection();
  float3 hit_pos = ray_orig + ray_dir * t;

  float3 obj_hit = optixTransformPointFromWorldToObjectSpace(hit_pos);
  float3 obj_normal = normalize3(obj_hit);
  float3 world_normal =
      normalize3(optixTransformNormalFromObjectToWorldSpace(obj_normal));
  if (dot3(world_normal, ray_dir) > 0.0f)
    world_normal = world_normal * (-1.0f);

  set_common_payloads(mat, hit_pos, world_normal, make_f3(mat->albedo),
                      mat->roughness, mat->roughness_v, make_float3(0, 0, 0));
}

// Any-hit program for alpha cutout
extern "C" __global__ void __anyhit__alpha() {
  const HitGroupData *data =
      reinterpret_cast<const HitGroupData *>(optixGetSbtDataPointer());
  const MaterialData *mat = data->mat;
  if (!mat->alpha_data || !data->texcoords)
    return;

  const float2 bary = optixGetTriangleBarycentrics();
  const int prim_idx = optixGetPrimitiveIndex();

  int i0, i1, i2;
  if (data->indices) {
    i0 = data->indices[prim_idx * 3];
    i1 = data->indices[prim_idx * 3 + 1];
    i2 = data->indices[prim_idx * 3 + 2];
  } else {
    i0 = prim_idx * 3;
    i1 = i0 + 1;
    i2 = i0 + 2;
  }
  float w = 1.0f - bary.x - bary.y;
  float u = w * data->texcoords[i0 * 2] + bary.x * data->texcoords[i1 * 2] +
            bary.y * data->texcoords[i2 * 2];
  float v = w * data->texcoords[i0 * 2 + 1] +
            bary.x * data->texcoords[i1 * 2 + 1] +
            bary.y * data->texcoords[i2 * 2 + 1];
  u = u - floorf(u);
  v = v - floorf(v);
  int ix = max(0, min((int)(u * (mat->alpha_width - 1)), mat->alpha_width - 1));
  int iy = max(0, min((int)((1.0f - v) * (mat->alpha_height - 1)),
                      mat->alpha_height - 1));
  float alpha = mat->alpha_data[iy * mat->alpha_width + ix];
  if (alpha < 0.5f) {
    optixIgnoreIntersection();
  }
}

// Custom sphere intersection that handles rays from both inside and outside
extern "C" __global__ void __intersection__sphere() {
  const float3 ray_orig = optixGetObjectRayOrigin();
  const float3 ray_dir = optixGetObjectRayDirection();
  const float tmin = optixGetRayTmin();
  const float tmax = optixGetRayTmax();

  // Sphere at origin, radius from SBT data
  // We store radius in the first float of vertices ptr (repurposed)
  const HitGroupData *data =
      reinterpret_cast<const HitGroupData *>(optixGetSbtDataPointer());
  float radius = __int_as_float(data->num_vertices); // radius stored here

  // Ray-sphere intersection: |o + t*d|^2 = r^2
  float a = dot3(ray_dir, ray_dir);
  float b = 2.0f * dot3(ray_orig, ray_dir);
  float c = dot3(ray_orig, ray_orig) - radius * radius;
  float discriminant = b * b - 4.0f * a * c;

  if (discriminant < 0.0f)
    return;

  float sqrt_disc = sqrtf(discriminant);
  float t1 = (-b - sqrt_disc) / (2.0f * a);
  float t2 = (-b + sqrt_disc) / (2.0f * a);

  // Report the nearest valid intersection
  if (t1 >= tmin && t1 <= tmax) {
    optixReportIntersection(t1, 0);
  } else if (t2 >= tmin && t2 <= tmax) {
    optixReportIntersection(t2, 0);
  }
}
