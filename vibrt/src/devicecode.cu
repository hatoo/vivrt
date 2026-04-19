#include "devicecode.h"
#include <optix.h>

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif
#ifndef INV_PIf
#define INV_PIf 0.31830988618379067154f
#endif

extern "C" {
__constant__ LaunchParams params;
}

// ---------- RNG (PCG) ----------
struct RNG {
  unsigned int state;
  __device__ RNG(unsigned int pixel, unsigned int sample, unsigned int bounce) {
    state = (pixel * 17u + sample * 101u + bounce * 1999u) * 747796405u +
            2891336453u;
  }
  __device__ float next() {
    unsigned int old = state;
    state = old * 747796405u + 2891336453u;
    unsigned int w = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    unsigned int r = (w >> 22u) ^ w;
    return (r >> 9) * (1.0f / 8388608.0f);
  }
};

// ---------- Math ----------
static __forceinline__ __device__ float3 make_f3(const float *p) {
  return make_float3(p[0], p[1], p[2]);
}
static __forceinline__ __device__ float dot3(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
static __forceinline__ __device__ float3 normalize3(float3 v) {
  float len = sqrtf(dot3(v, v));
  return len > 1e-20f ? make_float3(v.x / len, v.y / len, v.z / len)
                      : make_float3(0, 0, 1);
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
static __forceinline__ __device__ float3 operator-(float3 a) {
  return make_float3(-a.x, -a.y, -a.z);
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
static __forceinline__ __device__ float3 operator/(float3 a, float s) {
  return a * (1.0f / s);
}
static __forceinline__ __device__ float luminance(float3 c) {
  return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

static __forceinline__ __device__ void build_frame(float3 n, float3 &t,
                                                   float3 &b) {
  float s = n.z >= 0.0f ? 1.0f : -1.0f;
  float a = -1.0f / (s + n.z);
  float bb = n.x * n.y * a;
  t = make_float3(1.0f + s * n.x * n.x * a, s * bb, -s * n.x);
  b = make_float3(bb, s + n.y * n.y * a, -n.y);
}

// ---------- Texture sampling (bilinear) ----------
static __device__ float3 sample_rgba(const float *data, int w, int h,
                                     int /*channels*/, float2 uv) {
  if (data == nullptr || w <= 0 || h <= 0)
    return make_float3(1, 1, 1);
  float u = uv.x - floorf(uv.x);
  float v = uv.y - floorf(uv.y);
  float fx = u * (float)w - 0.5f;
  float fy = (1.0f - v) * (float)h - 0.5f;
  int x0 = (int)floorf(fx);
  int y0 = (int)floorf(fy);
  float dx = fx - (float)x0;
  float dy = fy - (float)y0;
  auto wrap = [](int v, int m) {
    int r = v % m;
    return r < 0 ? r + m : r;
  };
  int x1 = wrap(x0 + 1, w);
  int y1 = wrap(y0 + 1, h);
  x0 = wrap(x0, w);
  y0 = wrap(y0, h);
  auto fetch = [&](int x, int y) {
    const float *p = &data[(y * w + x) * 4];
    return make_float3(p[0], p[1], p[2]);
  };
  float3 c00 = fetch(x0, y0);
  float3 c10 = fetch(x1, y0);
  float3 c01 = fetch(x0, y1);
  float3 c11 = fetch(x1, y1);
  float3 c0 = c00 * (1.0f - dx) + c10 * dx;
  float3 c1 = c01 * (1.0f - dx) + c11 * dx;
  return c0 * (1.0f - dy) + c1 * dy;
}

// ---------- Fresnel (Schlick) ----------
static __forceinline__ __device__ float schlick_scalar(float cos_theta,
                                                       float F0) {
  float x = 1.0f - fmaxf(cos_theta, 0.0f);
  float x2 = x * x;
  return F0 + (1.0f - F0) * x2 * x2 * x;
}
static __forceinline__ __device__ float3 schlick_rgb(float cos_theta,
                                                     float3 F0) {
  float x = 1.0f - fmaxf(cos_theta, 0.0f);
  float x2 = x * x;
  float x5 = x2 * x2 * x;
  return F0 + (make_float3(1, 1, 1) - F0) * x5;
}

// Dielectric Fresnel (exact unpolarised).
static __forceinline__ __device__ float fresnel_dielectric(float cos_i,
                                                           float eta) {
  cos_i = fminf(fmaxf(cos_i, -1.0f), 1.0f);
  if (cos_i < 0.0f) {
    eta = 1.0f / eta;
    cos_i = -cos_i;
  }
  float sin2_t = (1.0f - cos_i * cos_i) / (eta * eta);
  if (sin2_t >= 1.0f)
    return 1.0f;
  float cos_t = sqrtf(fmaxf(0.0f, 1.0f - sin2_t));
  float rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
  float rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
  return 0.5f * (rs * rs + rp * rp);
}

// ---------- GGX (isotropic) ----------
static __forceinline__ __device__ float ggx_D(float NoH, float alpha) {
  float a2 = alpha * alpha;
  float d = NoH * NoH * (a2 - 1.0f) + 1.0f;
  return a2 / fmaxf(M_PIf * d * d, 1e-12f);
}
static __forceinline__ __device__ float smith_G1(float NoX, float alpha) {
  float a2 = alpha * alpha;
  return 2.0f * NoX / (NoX + sqrtf(a2 + (1.0f - a2) * NoX * NoX));
}

// Sample GGX VNDF (Heitz 2018), V in local space with N=+Z.
static __device__ float3 sample_ggx_vndf(float3 V, float alpha, float u1,
                                         float u2) {
  float3 Vh = normalize3(make_float3(alpha * V.x, alpha * V.y, V.z));
  float len2 = Vh.x * Vh.x + Vh.y * Vh.y;
  float3 T1 = len2 > 0.0f
                  ? make_float3(-Vh.y, Vh.x, 0.0f) * (1.0f / sqrtf(len2))
                  : make_float3(1, 0, 0);
  float3 T2 = cross3(Vh, T1);
  float r = sqrtf(u1);
  float phi = 2.0f * M_PIf * u2;
  float t1 = r * cosf(phi);
  float t2 = r * sinf(phi);
  float s = 0.5f * (1.0f + Vh.z);
  t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1)) + s * t2;
  float3 Nh =
      T1 * t1 + T2 * t2 + Vh * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2));
  float3 H =
      normalize3(make_float3(alpha * Nh.x, alpha * Nh.y, fmaxf(0.0f, Nh.z)));
  return H;
}

// PDF of GGX VNDF sample for reflection: D(H) * G1(V) * |V.H| / |V.N| / (4
// |V.H|) = D(H) * G1(V) / (4 |V.N|)
static __forceinline__ __device__ float ggx_vndf_pdf(float NoV, float NoH,
                                                     float alpha) {
  return ggx_D(NoH, alpha) * smith_G1(NoV, alpha) / fmaxf(4.0f * NoV, 1e-12f);
}

// ---------- GGX (anisotropic) ----------
// All inputs in the local tangent frame (z = Ns, x = T, y = B).
static __forceinline__ __device__ float
ggx_D_aniso(float3 Hlocal, float ax, float ay) {
  float x = Hlocal.x / fmaxf(ax, 1e-4f);
  float y = Hlocal.y / fmaxf(ay, 1e-4f);
  float z = Hlocal.z;
  float d = x * x + y * y + z * z;
  return 1.0f / fmaxf(M_PIf * ax * ay * d * d, 1e-12f);
}

static __forceinline__ __device__ float
smith_G1_aniso(float3 Vlocal, float ax, float ay) {
  float axv = ax * Vlocal.x;
  float ayv = ay * Vlocal.y;
  float Vz = fmaxf(Vlocal.z, 1e-6f);
  return 2.0f * Vz / fmaxf(Vz + sqrtf(axv * axv + ayv * ayv + Vz * Vz), 1e-12f);
}

static __device__ float3 sample_ggx_vndf_aniso(float3 V, float ax, float ay,
                                               float u1, float u2) {
  float3 Vh = normalize3(make_float3(ax * V.x, ay * V.y, V.z));
  float len2 = Vh.x * Vh.x + Vh.y * Vh.y;
  float3 T1 = len2 > 0.0f
                  ? make_float3(-Vh.y, Vh.x, 0.0f) * (1.0f / sqrtf(len2))
                  : make_float3(1, 0, 0);
  float3 T2 = cross3(Vh, T1);
  float r = sqrtf(u1);
  float phi = 2.0f * M_PIf * u2;
  float t1 = r * cosf(phi);
  float t2 = r * sinf(phi);
  float s = 0.5f * (1.0f + Vh.z);
  t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1)) + s * t2;
  float3 Nh =
      T1 * t1 + T2 * t2 + Vh * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2));
  return normalize3(
      make_float3(ax * Nh.x, ay * Nh.y, fmaxf(0.0f, Nh.z)));
}

static __forceinline__ __device__ float
ggx_vndf_pdf_aniso(float3 Vlocal, float3 Hlocal, float ax, float ay) {
  if (Hlocal.z <= 0.0f || Vlocal.z <= 0.0f)
    return 0.0f;
  float D = ggx_D_aniso(Hlocal, ax, ay);
  float G1 = smith_G1_aniso(Vlocal, ax, ay);
  return D * G1 / fmaxf(4.0f * Vlocal.z, 1e-12f);
}

// ---------- GGX energy compensation (Kulla-Conty) ----------
// LUT layout matches pipeline::generate_ggx_energy_lut():
//   e_lut[ci * n + ai]  with  cos_theta = (ci+0.5)/n,  alpha = (ai+0.5)/n
//   e_avg[ai]           with  alpha     = (ai+0.5)/n
#define GGX_LUT_N 32

static __forceinline__ __device__ float ggx_lut_coord(float x) {
  // Map x ∈ [0,1] to fractional texel index for center-pixel-convention LUT.
  float t = x * (float)GGX_LUT_N - 0.5f;
  return fminf(fmaxf(t, 0.0f), (float)(GGX_LUT_N - 1));
}

static __device__ float ggx_e_lookup(float NoV, float alpha) {
  float fc = ggx_lut_coord(fminf(fmaxf(NoV, 0.0f), 1.0f));
  float fa = ggx_lut_coord(fminf(fmaxf(alpha, 0.0f), 1.0f));
  int c0 = (int)floorf(fc);
  int a0 = (int)floorf(fa);
  int c1 = min(c0 + 1, GGX_LUT_N - 1);
  int a1 = min(a0 + 1, GGX_LUT_N - 1);
  float dc = fc - (float)c0;
  float da = fa - (float)a0;
  const float *L = params.ggx_e_lut;
  float v00 = L[c0 * GGX_LUT_N + a0];
  float v10 = L[c1 * GGX_LUT_N + a0];
  float v01 = L[c0 * GGX_LUT_N + a1];
  float v11 = L[c1 * GGX_LUT_N + a1];
  float v0 = v00 * (1.0f - dc) + v10 * dc;
  float v1 = v01 * (1.0f - dc) + v11 * dc;
  return v0 * (1.0f - da) + v1 * da;
}

static __device__ float ggx_e_avg_lookup(float alpha) {
  float fa = ggx_lut_coord(fminf(fmaxf(alpha, 0.0f), 1.0f));
  int a0 = (int)floorf(fa);
  int a1 = min(a0 + 1, GGX_LUT_N - 1);
  float da = fa - (float)a0;
  const float *L = params.ggx_e_avg_lut;
  return L[a0] * (1.0f - da) + L[a1] * da;
}

// Hemispherical-average Fresnel under Schlick approximation:
//   F_avg = ∫ F_Schlick(μ) · 2μ dμ  =  (20·F0 + 1) / 21
static __forceinline__ __device__ float f_avg_schlick(float F0) {
  return F0 * (20.0f / 21.0f) + (1.0f / 21.0f);
}
static __forceinline__ __device__ float3 f_avg_schlick(float3 F0) {
  return F0 * (20.0f / 21.0f) +
         make_float3(1.0f / 21.0f, 1.0f / 21.0f, 1.0f / 21.0f);
}

// ---------- Path tracing payload ----------
struct PathVertex {
  float3 P;  // hit position
  float3 Ng; // geometric normal (world)
  float3 Ns; // shading normal (world)
  float3 T;  // tangent (world)
  float2 uv; // mesh UVs (0,0 if absent)
  float3 vc; // interpolated vertex colour (1,1,1 if mesh has none)
  PrincipledGpu *mat;
  int hit; // 1 if hit, 0 if miss
};

static __forceinline__ __device__ void *unpack_ptr(unsigned int hi,
                                                   unsigned int lo) {
  return (void *)(((unsigned long long)hi << 32) | (unsigned long long)lo);
}
static __forceinline__ __device__ void pack_ptr(void *p, unsigned int &hi,
                                                unsigned int &lo) {
  unsigned long long u = (unsigned long long)p;
  hi = (unsigned int)(u >> 32);
  lo = (unsigned int)(u & 0xFFFFFFFFu);
}

static __forceinline__ __device__ PathVertex *get_path_vertex() {
  return (PathVertex *)unpack_ptr(optixGetPayload_0(), optixGetPayload_1());
}

// ---------- Env map ----------
static __forceinline__ __device__ float3 rotate_y(float3 v, float c, float s) {
  return make_float3(v.x * c + v.z * s, v.y, -v.x * s + v.z * c);
}

static __device__ float3 world_background(float3 dir) {
  if (params.world_type == 0) {
    return make_f3(params.world_color) * params.world_strength;
  }
  // envmap
  float c = cosf(-params.envmap_rotation_z_rad);
  float s = sinf(-params.envmap_rotation_z_rad);
  float3 d = make_float3(dir.x * c - dir.y * s, dir.x * s + dir.y * c, dir.z);
  float theta = acosf(fminf(fmaxf(d.z, -1.0f), 1.0f));
  float phi = atan2f(d.y, d.x);
  float u = phi * (0.5f * INV_PIf);
  if (u < 0.0f)
    u += 1.0f;
  float v = theta * INV_PIf;
  int w = params.envmap_width;
  int h = params.envmap_height;
  int x = min((int)(u * w), w - 1);
  int y = min((int)(v * h), h - 1);
  const float *px = &params.envmap_data[(y * w + x) * 3];
  return make_float3(px[0], px[1], px[2]) * params.world_strength;
}

static __device__ int cdf_search(const float *cdf, int n, float r) {
  int lo = 0, hi = n - 1;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (cdf[mid + 1] < r)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

struct EnvSample {
  float3 dir;
  float3 L;
  float pdf; // solid-angle PDF
};

// Power heuristic (β = 2) for two-strategy MIS.
static __forceinline__ __device__ float power_heuristic(float pa, float pb) {
  float a = pa * pa;
  float b = pb * pb;
  return a / fmaxf(a + b, 1e-20f);
}

// Solid-angle PDF of the envmap importance sampler for a given world-space dir.
// Inverse of sample_envmap: maps (theta, phi) back to the texel and reads the
// luminance, normalised by envmap_integral.
static __device__ float envmap_pdf(float3 dir) {
  if (params.world_type != 1 || params.envmap_integral <= 0.0f)
    return 0.0f;
  float c = cosf(-params.envmap_rotation_z_rad);
  float s = sinf(-params.envmap_rotation_z_rad);
  float3 d = make_float3(dir.x * c - dir.y * s, dir.x * s + dir.y * c, dir.z);
  float theta = acosf(fminf(fmaxf(d.z, -1.0f), 1.0f));
  float phi = atan2f(d.y, d.x);
  float u = phi * (0.5f * INV_PIf);
  if (u < 0.0f)
    u += 1.0f;
  float v = theta * INV_PIf;
  int w = params.envmap_width;
  int h = params.envmap_height;
  int x = min((int)(u * w), w - 1);
  int y = min((int)(v * h), h - 1);
  const float *px = &params.envmap_data[(y * w + x) * 3];
  float3 L = make_float3(px[0], px[1], px[2]) * params.world_strength;
  float lum = luminance(L);
  float sin_t = sinf(theta);
  if (sin_t <= 0.0f)
    return 0.0f;
  return lum * (float)(w * h) /
         (2.0f * M_PIf * M_PIf * sin_t * params.envmap_integral);
}

static __device__ EnvSample sample_envmap(RNG &rng) {
  EnvSample s;
  s.dir = make_float3(0, 0, 1);
  s.L = make_float3(0, 0, 0);
  s.pdf = 0.0f;
  if (params.world_type != 1 || params.envmap_integral <= 0.0f)
    return s;
  int w = params.envmap_width;
  int h = params.envmap_height;
  float u1 = rng.next();
  float u2 = rng.next();
  int y = cdf_search(params.envmap_marginal_cdf, h, u1);
  const float *row = &params.envmap_conditional_cdf[y * (w + 1)];
  int x = cdf_search(row, w, u2);

  float v = ((float)y + 0.5f) / (float)h;
  float u = ((float)x + 0.5f) / (float)w;
  float theta = v * M_PIf;
  float phi = u * 2.0f * M_PIf;
  float sin_t = sinf(theta);
  float3 d = make_float3(sin_t * cosf(phi), sin_t * sinf(phi), cosf(theta));
  // Apply rotation (inverse of what we do in background lookup).
  float c = cosf(params.envmap_rotation_z_rad);
  float sr = sinf(params.envmap_rotation_z_rad);
  s.dir = make_float3(d.x * c - d.y * sr, d.x * sr + d.y * c, d.z);
  const float *px = &params.envmap_data[(y * w + x) * 3];
  s.L = make_float3(px[0], px[1], px[2]) * params.world_strength;

  float lum = luminance(s.L);
  if (sin_t > 0.0f && params.envmap_integral > 0.0f) {
    s.pdf = lum * (float)(w * h) /
            (2.0f * M_PIf * M_PIf * sin_t * params.envmap_integral);
  }
  return s;
}

// ---------- Visibility (shadow rays) ----------
// Two ray types: 0 = radiance, 1 = shadow. SBT stride = 2. Shadow hit group
// is a dummy (CH disabled, no AH) so only the miss program runs, which sets
// payload[0]=1 meaning "reached infinity = light visible".
static __device__ bool shadow_visible(float3 P, float3 dir, float tmax) {
  unsigned int miss_flag = 0;
  optixTrace(params.traversable, P, dir, 1e-4f, tmax - 1e-3f, 0.0f,
             OptixVisibilityMask(255),
             OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
                 OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
             1, 2, 1, miss_flag);
  return miss_flag != 0;
}

// ---------- Colour graph evaluator ----------
// Register-sized cap — classroom's paintedCeiling only needs 6 nodes today.
#define COLOR_GRAPH_MAX_NODES 32

static __forceinline__ __device__ float3 mix_blend_rgb(
    float3 a, float3 b, float fac, unsigned int blend, bool clamp_out) {
  float facm = 1.0f - fac;
  float3 out;
  switch (blend) {
    default:
    case 0: // MIX
      out = a * facm + b * fac;
      break;
    case 1: // MULTIPLY
      out = make_float3(a.x * (facm + b.x * fac),
                        a.y * (facm + b.y * fac),
                        a.z * (facm + b.z * fac));
      break;
    case 2: // ADD
      out = a + b * fac;
      break;
    case 3: // SUBTRACT
      out = a - b * fac;
      break;
    case 4: // SCREEN
      out = make_float3(1.0f, 1.0f, 1.0f) -
            (make_float3(facm, facm, facm) + (make_float3(1.0f, 1.0f, 1.0f) - b) * fac) *
                (make_float3(1.0f, 1.0f, 1.0f) - a);
      break;
    case 5: // DIVIDE — per-channel, guard against div-by-zero
      out.x = b.x == 0.0f ? a.x : a.x * facm + a.x / b.x * fac;
      out.y = b.y == 0.0f ? a.y : a.y * facm + a.y / b.y * fac;
      out.z = b.z == 0.0f ? a.z : a.z * facm + a.z / b.z * fac;
      break;
    case 6: { // DIFFERENCE
      float3 d = make_float3(fabsf(a.x - b.x), fabsf(a.y - b.y), fabsf(a.z - b.z));
      out = a * facm + d * fac;
      break;
    }
    case 7: // DARKEN
      out.x = a.x * facm + fminf(a.x, b.x) * fac;
      out.y = a.y * facm + fminf(a.y, b.y) * fac;
      out.z = a.z * facm + fminf(a.z, b.z) * fac;
      break;
    case 8: // LIGHTEN (Blender's asymmetric form)
      out.x = fmaxf(a.x, b.x * fac);
      out.y = fmaxf(a.y, b.y * fac);
      out.z = fmaxf(a.z, b.z * fac);
      break;
    case 9: { // OVERLAY per-channel
      auto ov = [&](float ax, float bx) {
        return ax < 0.5f ? ax * (facm + 2.0f * fac * bx)
                         : 1.0f - (facm + 2.0f * fac * (1.0f - bx)) * (1.0f - ax);
      };
      out.x = ov(a.x, b.x);
      out.y = ov(a.y, b.y);
      out.z = ov(a.z, b.z);
      break;
    }
    case 10: { // SOFT_LIGHT
      float3 scr = make_float3(1.0f - (1.0f - b.x) * (1.0f - a.x),
                               1.0f - (1.0f - b.y) * (1.0f - a.y),
                               1.0f - (1.0f - b.z) * (1.0f - a.z));
      float3 inner = make_float3(
          (1.0f - a.x) * b.x * a.x + a.x * scr.x,
          (1.0f - a.y) * b.y * a.y + a.y * scr.y,
          (1.0f - a.z) * b.z * a.z + a.z * scr.z);
      out = a * facm + inner * fac;
      break;
    }
    case 11: // LINEAR_LIGHT
      out = a + (b * 2.0f - make_float3(1.0f, 1.0f, 1.0f)) * fac;
      break;
  }
  if (clamp_out) {
    out.x = fmaxf(0.0f, fminf(1.0f, out.x));
    out.y = fmaxf(0.0f, fminf(1.0f, out.y));
    out.z = fmaxf(0.0f, fminf(1.0f, out.z));
  }
  return out;
}

static __forceinline__ __device__ float3 math_apply_rgb(
    float3 inp, unsigned int op, float b, float c) {
  float3 o;
  switch (op) {
    default:
    case 0: // ADD
      o = make_float3(inp.x + b, inp.y + b, inp.z + b);
      break;
    case 1: // SUBTRACT
      o = make_float3(inp.x - b, inp.y - b, inp.z - b);
      break;
    case 2: // MULTIPLY
      o = make_float3(inp.x * b, inp.y * b, inp.z * b);
      break;
    case 3: // DIVIDE
      o = make_float3(b == 0.0f ? 0.0f : inp.x / b,
                      b == 0.0f ? 0.0f : inp.y / b,
                      b == 0.0f ? 0.0f : inp.z / b);
      break;
    case 4: // POWER
      o = make_float3(powf(fmaxf(inp.x, 0.0f), b),
                      powf(fmaxf(inp.y, 0.0f), b),
                      powf(fmaxf(inp.z, 0.0f), b));
      break;
    case 5: // MULTIPLY_ADD
      o = make_float3(inp.x * b + c, inp.y * b + c, inp.z * b + c);
      break;
    case 6: // MINIMUM
      o = make_float3(fminf(inp.x, b), fminf(inp.y, b), fminf(inp.z, b));
      break;
    case 7: // MAXIMUM
      o = make_float3(fmaxf(inp.x, b), fmaxf(inp.y, b), fmaxf(inp.z, b));
      break;
  }
  return o;
}

// RGB ↔ HSV helpers in the shape Blender uses (HSV channels in [0,1]).
static __forceinline__ __device__ float3 rgb_to_hsv_bl(float3 c) {
  float cmax = fmaxf(c.x, fmaxf(c.y, c.z));
  float cmin = fminf(c.x, fminf(c.y, c.z));
  float v = cmax;
  float d = cmax - cmin;
  float s = cmax > 0.0f ? d / cmax : 0.0f;
  float h = 0.0f;
  if (d > 0.0f) {
    if (cmax == c.x) {
      h = (c.y - c.z) / d + (c.y < c.z ? 6.0f : 0.0f);
    } else if (cmax == c.y) {
      h = (c.z - c.x) / d + 2.0f;
    } else {
      h = (c.x - c.y) / d + 4.0f;
    }
    h *= (1.0f / 6.0f);
  }
  return make_float3(h, s, v);
}

static __forceinline__ __device__ float3 hsv_to_rgb_bl(float3 c) {
  float h = c.x - floorf(c.x); // wrap to [0,1)
  float s = fmaxf(0.0f, fminf(1.0f, c.y));
  float v = c.z;
  float i = floorf(h * 6.0f);
  float f = h * 6.0f - i;
  float p = v * (1.0f - s);
  float q = v * (1.0f - f * s);
  float t = v * (1.0f - (1.0f - f) * s);
  int ii = ((int)i) % 6;
  if (ii < 0) ii += 6;
  switch (ii) {
    case 0: return make_float3(v, t, p);
    case 1: return make_float3(q, v, p);
    case 2: return make_float3(p, v, t);
    case 3: return make_float3(p, q, v);
    case 4: return make_float3(t, p, v);
    default: return make_float3(v, p, q);
  }
}

static __device__ float3 eval_color_graph(
    ColorGraphNode *nodes, int n_nodes, int output, float2 base_uv) {
  float3 slots[COLOR_GRAPH_MAX_NODES];
  int n = n_nodes < COLOR_GRAPH_MAX_NODES ? n_nodes : COLOR_GRAPH_MAX_NODES;
  for (int i = 0; i < n; i++) {
    ColorGraphNode nd = nodes[i];
    unsigned int tag = nd.tag;
    const unsigned int *pp = nd.payload;
    if (tag == COLOR_NODE_CONST) {
      slots[i] = make_float3(__uint_as_float(pp[0]),
                             __uint_as_float(pp[1]),
                             __uint_as_float(pp[2]));
    } else if (tag == COLOR_NODE_IMAGE_TEX) {
      unsigned long long ptr =
          ((unsigned long long)pp[1] << 32) | (unsigned long long)pp[0];
      const float *tex = (const float *)ptr;
      int w = (int)pp[2];
      int h = (int)pp[3];
      int ch = (int)pp[4];
      float m0 = __uint_as_float(pp[5]);
      float m1 = __uint_as_float(pp[6]);
      float m2 = __uint_as_float(pp[7]);
      float m3 = __uint_as_float(pp[8]);
      float m4 = __uint_as_float(pp[9]);
      float m5 = __uint_as_float(pp[10]);
      float2 uv = make_float2(m0 * base_uv.x + m1 * base_uv.y + m2,
                              m3 * base_uv.x + m4 * base_uv.y + m5);
      slots[i] = sample_rgba(tex, w, h, ch, uv);
    } else if (tag == COLOR_NODE_MIX) {
      int ia = (int)pp[0];
      int ib = (int)pp[1];
      unsigned int blend = pp[2];
      unsigned int fac_src = pp[3];
      float fac;
      if (fac_src == 0u) {
        fac = __uint_as_float(pp[6]);
      } else {
        int ifac = (int)pp[4];
        float3 fv = slots[ifac];
        fac = 0.2126f * fv.x + 0.7152f * fv.y + 0.0722f * fv.z;
      }
      bool clamp_out = pp[5] != 0u;
      slots[i] = mix_blend_rgb(slots[ia], slots[ib], fac, blend, clamp_out);
    } else if (tag == COLOR_NODE_INVERT) {
      int iin = (int)pp[0];
      float fac = __uint_as_float(pp[1]);
      float3 cc = slots[iin];
      float facm = 1.0f - fac;
      slots[i] = make_float3(cc.x * facm + (1.0f - cc.x) * fac,
                             cc.y * facm + (1.0f - cc.y) * fac,
                             cc.z * facm + (1.0f - cc.z) * fac);
    } else if (tag == COLOR_NODE_MATH) {
      int iin = (int)pp[0];
      unsigned int op = pp[1];
      bool clamp_out = pp[2] != 0u;
      float b = __uint_as_float(pp[3]);
      float c = __uint_as_float(pp[4]);
      float3 o = math_apply_rgb(slots[iin], op, b, c);
      if (clamp_out) {
        o.x = fmaxf(0.0f, fminf(1.0f, o.x));
        o.y = fmaxf(0.0f, fminf(1.0f, o.y));
        o.z = fmaxf(0.0f, fminf(1.0f, o.z));
      }
      slots[i] = o;
    } else if (tag == COLOR_NODE_HUE_SAT) {
      int iin = (int)pp[0];
      float hue = __uint_as_float(pp[1]);
      float sat = __uint_as_float(pp[2]);
      float val = __uint_as_float(pp[3]);
      float fac = __uint_as_float(pp[4]);
      float3 src = slots[iin];
      float3 hsv = rgb_to_hsv_bl(src);
      // Blender's HueSaturation subtracts 0.5 from Hue before shifting so
      // the default (0.5) is identity.
      hsv.x = hsv.x + (hue - 0.5f);
      hsv.y = fmaxf(0.0f, fminf(1.0f, hsv.y * sat));
      hsv.z = hsv.z * val;
      float3 shifted = hsv_to_rgb_bl(hsv);
      float facm = 1.0f - fac;
      slots[i] = make_float3(src.x * facm + shifted.x * fac,
                             src.y * facm + shifted.y * fac,
                             src.z * facm + shifted.z * fac);
    } else {
      slots[i] = make_float3(1.0f, 1.0f, 1.0f);
    }
  }
  if (output < 0 || output >= n)
    return make_float3(1.0f, 1.0f, 1.0f);
  return slots[output];
}

// ---------- Principled BSDF ----------
struct MaterialEval {
  float3 base_color;
  float metallic;
  float roughness;
  float alpha;   // isotropic α = roughness², used for LUT + transmission
  float alpha_x; // anisotropic αx (along tangent T)
  float alpha_y; // anisotropic αy (along bitangent B)
  float ior;
  float transmission;
  float3 emission;
  float3 Ns; // shading normal
  float3 T;  // rotated by tangent_rotation for anisotropy
  float3 B;
  PrincipledGpu *mat; // raw material ptr for coat/sheen/sss params
};

static __device__ MaterialEval eval_material(const PathVertex &v) {
  MaterialEval e;
  PrincipledGpu *m = v.mat;
  e.mat = m;
  e.base_color = make_f3(m->base_color);
  e.metallic = m->metallic;
  e.roughness = m->roughness;
  e.ior = m->ior;
  e.transmission = m->transmission;
  e.emission = make_f3(m->emission);

  const float *M = m->uv_transform;
  float2 uv =
      make_float2(M[0] * v.uv.x + M[1] * v.uv.y + M[2],
                  M[3] * v.uv.x + M[4] * v.uv.y + M[5]);

  if (m->color_graph_nodes != nullptr && m->color_graph_len > 0) {
    // Graph evaluation reads the raw mesh UV; each ImageTex node carries
    // its own uv_transform so sources with different Mapping nodes don't
    // get jammed into one global scale.
    float3 g = eval_color_graph(m->color_graph_nodes, m->color_graph_len,
                                m->color_graph_output, v.uv);
    e.base_color = e.base_color * g;
  } else if (m->base_color_tex != nullptr) {
    float3 t =
        sample_rgba(m->base_color_tex, m->base_color_tex_w, m->base_color_tex_h,
                    m->base_color_tex_channels, uv);
    e.base_color = e.base_color * t;
  }
  if (m->use_vertex_color) {
    e.base_color = e.base_color * v.vc;
  }
  if (m->roughness_tex != nullptr) {
    float3 t = sample_rgba(m->roughness_tex, m->roughness_tex_w,
                           m->roughness_tex_h, m->roughness_tex_channels, uv);
    e.roughness = e.roughness * t.x;
  }
  if (m->metallic_tex != nullptr) {
    float3 t = sample_rgba(m->metallic_tex, m->metallic_tex_w,
                           m->metallic_tex_h, m->metallic_tex_channels, uv);
    e.metallic = e.metallic * t.x;
  }
  e.metallic = fminf(fmaxf(e.metallic, 0.0f), 1.0f);
  e.roughness = fmaxf(e.roughness, 0.02f);
  e.alpha = fmaxf(e.roughness * e.roughness, 1e-4f);

  float3 Ns = v.Ns;
  // Composite tangent-space perturbation: start from (0,0,1), apply bump
  // (central-difference over heightmap) first, then normal map (multiplied).
  float3 nm_t = make_float3(0.0f, 0.0f, 1.0f);
  bool any_perturb = false;
  if (m->bump_tex != nullptr && m->bump_strength != 0.0f) {
    int bw = m->bump_tex_w;
    int bh = m->bump_tex_h;
    if (bw > 0 && bh > 0) {
      float du = 1.0f / (float)bw;
      float dv = 1.0f / (float)bh;
      float h_px = sample_rgba(m->bump_tex, bw, bh, m->bump_tex_channels,
                               make_float2(uv.x + du, uv.y)).x;
      float h_mx = sample_rgba(m->bump_tex, bw, bh, m->bump_tex_channels,
                               make_float2(uv.x - du, uv.y)).x;
      float h_py = sample_rgba(m->bump_tex, bw, bh, m->bump_tex_channels,
                               make_float2(uv.x, uv.y + dv)).x;
      float h_my = sample_rgba(m->bump_tex, bw, bh, m->bump_tex_channels,
                               make_float2(uv.x, uv.y - dv)).x;
      // Central-difference slope × user strength. Empirical 0.25 keeps the
      // bump in a "subtle micro-relief" range at Strength=1.
      float sx = -(h_px - h_mx) * m->bump_strength * 0.25f;
      float sy = -(h_py - h_my) * m->bump_strength * 0.25f;
      nm_t = normalize3(make_float3(sx, sy, 1.0f));
      any_perturb = true;
    }
  }
  if (m->normal_tex != nullptr) {
    float3 n = sample_rgba(m->normal_tex, m->normal_tex_w, m->normal_tex_h,
                           m->normal_tex_channels, uv);
    float s = m->normal_strength;
    float3 nn = make_float3((n.x * 2.0f - 1.0f) * s,
                            (n.y * 2.0f - 1.0f) * s,
                            fmaxf(n.z * 2.0f - 1.0f, 0.01f));
    nn = normalize3(nn);
    // Compose: treat nm_t as frame, re-express nn in its tangent basis.
    if (any_perturb) {
      float3 nt, nb;
      build_frame(nm_t, nt, nb);
      nm_t = normalize3(nt * nn.x + nb * nn.y + nm_t * nn.z);
    } else {
      nm_t = nn;
    }
    any_perturb = true;
  }
  if (any_perturb) {
    float3 T, B;
    build_frame(Ns, T, B);
    Ns = normalize3(T * nm_t.x + B * nm_t.y + Ns * nm_t.z);
  }
  e.Ns = Ns;
  build_frame(e.Ns, e.T, e.B);

  // Anisotropic α. Disney remap: aspect = sqrt(1 - 0.9·|aniso|).
  float aniso = fminf(fmaxf(m->anisotropy, -1.0f), 1.0f);
  float aspect = sqrtf(fmaxf(1.0f - 0.9f * fabsf(aniso), 1e-4f));
  if (aniso >= 0.0f) {
    e.alpha_x = e.alpha / aspect;
    e.alpha_y = e.alpha * aspect;
  } else {
    e.alpha_x = e.alpha * aspect;
    e.alpha_y = e.alpha / aspect;
  }
  // Tangent rotation around Ns.
  float rot = m->tangent_rotation;
  if (rot != 0.0f) {
    float c = cosf(rot);
    float s = sinf(rot);
    float3 Trot = e.T * c + e.B * s;
    float3 Brot = e.B * c - e.T * s;
    e.T = normalize3(Trot);
    e.B = normalize3(Brot);
  }
  return e;
}

// Evaluate BSDF * |cos(wi, Ns)| for a given outgoing wo and incoming wi (both
// world space, pointing away from P). Returns f_r and the sampling PDF used
// by the combined sample routine (for MIS weighting).
struct BsdfEval {
  float3 f;
  float pdf;
};

static __device__ BsdfEval eval_bsdf(const MaterialEval &e, float3 wo,
                                     float3 wi) {
  BsdfEval r;
  r.f = make_float3(0, 0, 0);
  r.pdf = 0.0f;

  float NoV = dot3(e.Ns, wo);
  float NoL = dot3(e.Ns, wi);
  bool reflect = NoL > 0.0f;
  bool transmit = !reflect && e.transmission > 0.0f;
  // SSS can also contribute diffusely when the light is slightly behind the
  // surface (wraparound lobe).
  bool sss_backlit = !reflect && e.mat->sss_weight > 0.0f && NoL > -1.0f;
  if (NoV <= 0.0f)
    return r;

  float F0_d =
      ((e.ior - 1.0f) / (e.ior + 1.0f)) * ((e.ior - 1.0f) / (e.ior + 1.0f));

  // Weights for MIS between lobes
  float w_diffuse = (1.0f - e.metallic) * (1.0f - e.transmission);
  float w_spec = e.metallic + (1.0f - e.metallic) * (1.0f - e.transmission);
  float w_trans = (1.0f - e.metallic) * e.transmission;
  float w_sum = w_diffuse + w_spec + w_trans;
  if (w_sum <= 0.0f)
    return r;
  float p_diff = w_diffuse / w_sum;
  float p_spec = w_spec / w_sum;
  float p_trans = w_trans / w_sum;

  // Subsurface (approximate): wraparound Lambert + radius-weighted tint.
  // Not a true random-walk SSS but captures the characteristic "wax / skin /
  // paper" behaviour where light wraps past the terminator and shifts hue by
  // channel-dependent mean-free-path.
  float sss_w = e.mat->sss_weight;
  float3 sss_tint = make_float3(1, 1, 1);
  if (sss_w > 0.0f) {
    float3 rr = make_f3(e.mat->sss_radius);
    float rmax = fmaxf(fmaxf(rr.x, rr.y), fmaxf(rr.z, 1e-4f));
    sss_tint = make_float3(rr.x / rmax, rr.y / rmax, rr.z / rmax);
  }

  // Wraparound brightness for SSS: allows a bit of back-lit contribution.
  float NoL_std = fmaxf(NoL, 0.0f);
  float NoL_wrap = 0.5f * (1.0f + NoL);

  if (reflect) {
    // Diffuse (Lambert, possibly wrap-shifted by SSS)
    if (w_diffuse > 0.0f) {
      float effective = (1.0f - sss_w) * NoL_std + sss_w * NoL_wrap;
      float3 bc_sss = e.base_color * (make_float3(1, 1, 1) * (1.0f - sss_w) +
                                      sss_tint * sss_w);
      r.f = r.f + bc_sss * (INV_PIf * w_diffuse * effective);
      r.pdf += p_diff * fmaxf(effective, 0.0f) * INV_PIf;
    }
    // Specular (metallic + dielectric spec layer)
    float3 H = normalize3(wo + wi);
    float NoH = fmaxf(dot3(e.Ns, H), 0.0f);
    float VoH = fmaxf(dot3(wo, H), 0.0f);
    float3 Vloc = make_float3(dot3(wo, e.T), dot3(wo, e.B), dot3(wo, e.Ns));
    float3 Lloc = make_float3(dot3(wi, e.T), dot3(wi, e.B), dot3(wi, e.Ns));
    float3 Hloc = make_float3(dot3(H, e.T), dot3(H, e.B), dot3(H, e.Ns));
    float D = ggx_D_aniso(Hloc, e.alpha_x, e.alpha_y);
    float G1_v = smith_G1_aniso(Vloc, e.alpha_x, e.alpha_y);
    float G1_l = smith_G1_aniso(Lloc, e.alpha_x, e.alpha_y);
    float G = G1_v * G1_l;
    float3 F_metal = schlick_rgb(VoH, e.base_color);
    float F_dielec = schlick_scalar(VoH, F0_d);
    float3 f_metal = F_metal * (D * G / fmaxf(4.0f * NoV * NoL, 1e-8f));
    float3 f_dielec = make_float3(F_dielec, F_dielec, F_dielec) *
                      (D * G / fmaxf(4.0f * NoV * NoL, 1e-8f));
    float3 f_spec = e.metallic * f_metal +
                    (1.0f - e.metallic) * (1.0f - e.transmission) * f_dielec;

    // Kulla-Conty multi-scattering compensation (reflection only).
    float E_wo = ggx_e_lookup(NoV, e.alpha);
    float E_wi = ggx_e_lookup(NoL, e.alpha);
    float E_avg = ggx_e_avg_lookup(e.alpha);
    float one_minus_Eavg = fmaxf(1.0f - E_avg, 1e-4f);
    float f_ms_scalar =
        (1.0f - E_wo) * (1.0f - E_wi) / fmaxf(M_PIf * one_minus_Eavg, 1e-8f);

    float3 F_avg_m = f_avg_schlick(e.base_color);
    float3 one3 = make_float3(1.0f, 1.0f, 1.0f);
    float3 denom_m = one3 - F_avg_m * (1.0f - E_avg);
    float3 F_ms_m = make_float3(
        (F_avg_m.x * F_avg_m.x) * E_avg / fmaxf(denom_m.x, 1e-4f),
        (F_avg_m.y * F_avg_m.y) * E_avg / fmaxf(denom_m.y, 1e-4f),
        (F_avg_m.z * F_avg_m.z) * E_avg / fmaxf(denom_m.z, 1e-4f));
    float3 f_metal_ms = F_ms_m * f_ms_scalar;

    float F_avg_d = f_avg_schlick(F0_d);
    float denom_d = fmaxf(1.0f - F_avg_d * (1.0f - E_avg), 1e-4f);
    float F_ms_d = (F_avg_d * F_avg_d) * E_avg / denom_d;
    float f_dielec_ms = F_ms_d * f_ms_scalar;

    f_spec = f_spec + e.metallic * f_metal_ms +
             (1.0f - e.metallic) * (1.0f - e.transmission) *
                 make_float3(f_dielec_ms, f_dielec_ms, f_dielec_ms);

    r.f = r.f + f_spec * NoL;
    float pdf_spec = ggx_vndf_pdf_aniso(Vloc, Hloc, e.alpha_x, e.alpha_y);
    r.pdf += p_spec * pdf_spec;

    // --- Coat (clearcoat): additive isotropic GGX dielectric above base. ---
    PrincipledGpu *mc = e.mat;
    float coat_w = mc->coat_weight;
    if (coat_w > 0.0f) {
      float coat_alpha = fmaxf(mc->coat_roughness * mc->coat_roughness, 1e-4f);
      float cF0 =
          ((mc->coat_ior - 1.0f) / (mc->coat_ior + 1.0f)) *
          ((mc->coat_ior - 1.0f) / (mc->coat_ior + 1.0f));
      float Fc = schlick_scalar(VoH, cF0);
      float cD = ggx_D(NoH, coat_alpha);
      float cG = smith_G1(NoV, coat_alpha) * smith_G1(NoL, coat_alpha);
      float coat_f = coat_w * Fc * cD * cG / fmaxf(4.0f * NoV * NoL, 1e-8f);
      // Attenuate existing lobes by the coat "loss" at view+light directions.
      float loss = coat_w * 0.5f * (schlick_scalar(NoV, cF0) +
                                    schlick_scalar(NoL, cF0));
      r.f = r.f * (1.0f - loss);
      r.f = r.f + make_float3(coat_f, coat_f, coat_f) * NoL;
      float pdf_coat = ggx_vndf_pdf(NoV, NoH, coat_alpha);
      r.pdf += coat_w * pdf_coat; // add (unnormalised) coat sampling weight
    }

    // --- Sheen: soft angular cosine lobe on top of diffuse. ---
    float sheen_w = mc->sheen_weight;
    if (sheen_w > 0.0f) {
      float sr = fmaxf(mc->sheen_roughness, 1e-3f);
      float sf = powf(fmaxf(1.0f - NoV, 0.0f), 1.0f / sr) * sheen_w;
      float3 tint = make_f3(mc->sheen_tint);
      r.f = r.f + tint * sf * NoL;
    }
  } else if (sss_backlit) {
    // Back-lit diffuse via SSS wrap lobe only.
    float w_wrap = fmaxf(NoL_wrap, 0.0f);
    if (w_wrap > 0.0f && w_diffuse > 0.0f) {
      float3 bc_sss = e.base_color * (make_float3(1, 1, 1) * (1.0f - sss_w) +
                                      sss_tint * sss_w);
      r.f = r.f + bc_sss * (INV_PIf * w_diffuse * sss_w * w_wrap);
      r.pdf += p_diff * w_wrap * INV_PIf * sss_w;
    }
  } else if (transmit) {
    // Rough dielectric transmission: evaluate using Walter et al. BTDF.
    float eta = e.ior;
    float3 n_oriented = NoV > 0.0f ? e.Ns : -e.Ns;
    if (NoV < 0.0f)
      eta = 1.0f / eta;
    float3 H = -normalize3(wo * eta + wi);
    if (dot3(H, n_oriented) < 0.0f)
      H = -H;
    float VoH = dot3(wo, H);
    float LoH = dot3(wi, H);
    float NoH = dot3(e.Ns, H);
    float F = fresnel_dielectric(VoH, eta);
    float D = ggx_D(fmaxf(NoH, 0.0f), e.alpha);
    float G = smith_G1(fabsf(NoV), e.alpha) * smith_G1(fabsf(NoL), e.alpha);
    float denom =
        (eta * VoH + LoH) * (eta * VoH + LoH) * fabsf(NoV) * fabsf(NoL);
    float btdf =
        (fabsf(VoH) * fabsf(LoH) * (1.0f - F) * D * G) / fmaxf(denom, 1e-8f);
    float3 tint = e.base_color;
    r.f = tint * btdf * fabsf(NoL) * w_trans;
    // Approximate pdf: Jacobian of half-vector to wi
    float jacobian =
        fabsf(LoH) / fmaxf((eta * VoH + LoH) * (eta * VoH + LoH), 1e-8f);
    float pdf_h = ggx_D(fmaxf(NoH, 0.0f), e.alpha) * fmaxf(NoH, 0.0f);
    r.pdf += p_trans * pdf_h * jacobian;
  }
  return r;
}

// Sample BSDF, returning new direction wi, evaluated f*cosθ, and PDF.
struct BsdfSample {
  float3 wi;
  float3 f;
  float pdf;
  bool specular; // true for rough-specular that shouldn't NEE-double-count
};

static __device__ BsdfSample sample_bsdf(const MaterialEval &e, float3 wo,
                                         RNG &rng) {
  BsdfSample s;
  s.wi = make_float3(0, 0, 1);
  s.f = make_float3(0, 0, 0);
  s.pdf = 0.0f;
  s.specular = false;

  float NoV = dot3(e.Ns, wo);
  if (NoV <= 0.0f)
    return s;

  float w_diffuse = (1.0f - e.metallic) * (1.0f - e.transmission);
  float w_spec = e.metallic + (1.0f - e.metallic) * (1.0f - e.transmission);
  float w_trans = (1.0f - e.metallic) * e.transmission;
  float total = w_diffuse + w_spec + w_trans;
  if (total <= 0.0f)
    return s;
  float p_diff = w_diffuse / total;
  float p_spec = w_spec / total;

  float u = rng.next();
  float u1 = rng.next();
  float u2 = rng.next();

  if (u < p_diff) {
    // Cosine-weighted hemisphere sample
    float r = sqrtf(u1);
    float phi = 2.0f * M_PIf * u2;
    float3 local = make_float3(r * cosf(phi), r * sinf(phi),
                               sqrtf(fmaxf(0.0f, 1.0f - u1)));
    s.wi = normalize3(e.T * local.x + e.B * local.y + e.Ns * local.z);
  } else if (u < p_diff + p_spec) {
    // GGX VNDF specular (anisotropic)
    float3 Vlocal = make_float3(dot3(wo, e.T), dot3(wo, e.B), dot3(wo, e.Ns));
    float3 Hlocal =
        sample_ggx_vndf_aniso(Vlocal, e.alpha_x, e.alpha_y, u1, u2);
    float3 H = normalize3(e.T * Hlocal.x + e.B * Hlocal.y + e.Ns * Hlocal.z);
    float VoH = dot3(wo, H);
    float3 wi = normalize3(H * (2.0f * VoH) - wo);
    if (dot3(e.Ns, wi) <= 0.0f)
      return s;
    s.wi = wi;
    if (e.alpha < 0.02f * 0.02f)
      s.specular = true;
  } else {
    // Rough-dielectric transmission
    float eta = e.ior;
    float3 Vlocal = make_float3(dot3(wo, e.T), dot3(wo, e.B), dot3(wo, e.Ns));
    float3 Hlocal = sample_ggx_vndf(Vlocal, e.alpha, u1, u2);
    float3 H = normalize3(e.T * Hlocal.x + e.B * Hlocal.y + e.Ns * Hlocal.z);
    float VoH = dot3(wo, H);
    float F = fresnel_dielectric(VoH, eta);
    // Randomly pick reflect vs transmit by Fresnel.
    if (rng.next() < F) {
      float3 wi = normalize3(H * (2.0f * VoH) - wo);
      if (dot3(e.Ns, wi) <= 0.0f)
        return s;
      s.wi = wi;
    } else {
      float cosi = VoH;
      float eta_t = 1.0f / eta;
      float k = 1.0f - eta_t * eta_t * (1.0f - cosi * cosi);
      if (k < 0.0f)
        return s;
      float3 wt = -wo * eta_t + H * (eta_t * cosi - sqrtf(k));
      s.wi = normalize3(wt);
      if (dot3(e.Ns, s.wi) >= 0.0f)
        return s;
    }
    if (e.alpha < 0.02f * 0.02f)
      s.specular = true;
  }

  BsdfEval ev = eval_bsdf(e, wo, s.wi);
  s.f = ev.f;
  s.pdf = ev.pdf;
  return s;
}

// ---------- Direct lighting (NEE) ----------
static __device__ float3 direct_light(const MaterialEval &e, float3 P,
                                      float3 wo, RNG &rng) {
  float3 L = make_float3(0, 0, 0);

  // Point lights: sample centre (simple)
  for (int i = 0; i < params.num_point_lights; i++) {
    PointLight &pl = params.point_lights[i];
    float3 ld = make_f3(pl.position) - P;
    float d = sqrtf(dot3(ld, ld));
    if (d < 1e-4f)
      continue;
    float3 wi = ld / d;
    if (!shadow_visible(P, wi, d))
      continue;
    BsdfEval b = eval_bsdf(e, wo, wi);
    L = L + b.f * make_f3(pl.emission) / fmaxf(d * d, 1e-6f);
  }

  // Sun lights: sample within cone
  for (int i = 0; i < params.num_sun_lights; i++) {
    SunLight &sl = params.sun_lights[i];
    float3 dir = make_f3(sl.direction);
    // Small cone sample around the sun direction.
    float cos_a = sl.cos_angle;
    float u1 = rng.next();
    float u2 = rng.next();
    float ct = 1.0f - u1 * (1.0f - cos_a);
    float st = sqrtf(fmaxf(0.0f, 1.0f - ct * ct));
    float phi = 2.0f * M_PIf * u2;
    float3 T, B;
    build_frame(dir, T, B);
    float3 wi =
        normalize3(T * (st * cosf(phi)) + B * (st * sinf(phi)) + dir * ct);
    if (!shadow_visible(P, wi, 1e20f))
      continue;
    BsdfEval b = eval_bsdf(e, wo, wi);
    L = L + b.f * make_f3(sl.emission);
  }

  // Spot lights
  for (int i = 0; i < params.num_spot_lights; i++) {
    SpotLight &sp = params.spot_lights[i];
    float3 ld = make_f3(sp.position) - P;
    float d = sqrtf(dot3(ld, ld));
    if (d < 1e-4f)
      continue;
    float3 wi = ld / d;
    // Direction from light to point = -wi, cos against spot axis
    float cos_a = dot3(make_f3(sp.direction), -wi);
    if (cos_a <= sp.cos_outer)
      continue;
    float falloff = 1.0f;
    if (cos_a < sp.cos_inner) {
      float t =
          (cos_a - sp.cos_outer) / fmaxf(sp.cos_inner - sp.cos_outer, 1e-4f);
      falloff = t * t * (3.0f - 2.0f * t);
    }
    if (!shadow_visible(P, wi, d))
      continue;
    BsdfEval b = eval_bsdf(e, wo, wi);
    L = L + b.f * make_f3(sp.emission) * falloff / fmaxf(d * d, 1e-6f);
  }

  // Area rect lights: sample one proportional to power
  if (params.num_rect_lights > 0) {
    float u = rng.next();
    int idx = cdf_search(params.rect_light_cdf, params.num_rect_lights, u);
    AreaRectLight &ar = params.rect_lights[idx];
    float pmf_light =
        (params.rect_light_cdf[idx + 1] - params.rect_light_cdf[idx]);
    if (pmf_light > 0.0f) {
      float su = rng.next();
      float sv = rng.next();
      float3 Pl = make_f3(ar.corner) + make_f3(ar.u_axis) * (su * ar.size_u) +
                  make_f3(ar.v_axis) * (sv * ar.size_v);
      float3 ld = Pl - P;
      float d2 = dot3(ld, ld);
      float d = sqrtf(d2);
      if (d > 1e-4f) {
        float3 wi = ld / d;
        float cos_light = -dot3(make_f3(ar.normal), wi);
        if (cos_light > 0.0f && shadow_visible(P, wi, d)) {
          BsdfEval b = eval_bsdf(e, wo, wi);
          float area = ar.size_u * ar.size_v;
          float pdf_area = 1.0f / fmaxf(area, 1e-8f);
          float pdf_solid = pdf_area * d2 / cos_light;
          float pdf = pdf_solid * pmf_light;
          if (pdf > 0.0f) {
            L = L + b.f * make_f3(ar.emission) / pdf;
          }
        }
      }
    }
  }

  // Envmap: one importance sample, MIS-weighted against BSDF sampling.
  if (params.world_type == 1 && params.envmap_integral > 0.0f) {
    EnvSample es = sample_envmap(rng);
    if (es.pdf > 0.0f && shadow_visible(P, es.dir, 1e20f)) {
      BsdfEval b = eval_bsdf(e, wo, es.dir);
      if (b.pdf > 0.0f) {
        float w = power_heuristic(es.pdf, b.pdf);
        L = L + b.f * es.L * (w / es.pdf);
      }
    }
  }

  return L;
}

// ---------- Closesthit / Miss / Raygen ----------

extern "C" __global__ void __closesthit__ch() {
  PathVertex *v = get_path_vertex();
  HitGroupData *hg = (HitGroupData *)optixGetSbtDataPointer();

  unsigned int prim = optixGetPrimitiveIndex();
  int i0 = hg->indices[prim * 3 + 0];
  int i1 = hg->indices[prim * 3 + 1];
  int i2 = hg->indices[prim * 3 + 2];
  float3 p0 = make_f3(&hg->vertices[i0 * 3]);
  float3 p1 = make_f3(&hg->vertices[i1 * 3]);
  float3 p2 = make_f3(&hg->vertices[i2 * 3]);

  float2 bary = optixGetTriangleBarycentrics();
  float b0 = 1.0f - bary.x - bary.y;
  float3 P_local = p0 * b0 + p1 * bary.x + p2 * bary.y;

  float3 P = optixTransformPointFromObjectToWorldSpace(P_local);

  // Geometric normal in object space
  float3 e1 = p1 - p0;
  float3 e2 = p2 - p0;
  float3 Ng_local = normalize3(cross3(e1, e2));
  float3 Ng = normalize3(optixTransformNormalFromObjectToWorldSpace(Ng_local));

  float3 Ns = Ng;
  if (hg->normals != nullptr) {
    float3 n0 = make_f3(&hg->normals[i0 * 3]);
    float3 n1 = make_f3(&hg->normals[i1 * 3]);
    float3 n2 = make_f3(&hg->normals[i2 * 3]);
    float3 Ns_local = normalize3(n0 * b0 + n1 * bary.x + n2 * bary.y);
    Ns = normalize3(optixTransformNormalFromObjectToWorldSpace(Ns_local));
  }

  float2 uv = make_float2(0.0f, 0.0f);
  if (hg->uvs != nullptr) {
    float2 uv0 = make_float2(hg->uvs[i0 * 2 + 0], hg->uvs[i0 * 2 + 1]);
    float2 uv1 = make_float2(hg->uvs[i1 * 2 + 0], hg->uvs[i1 * 2 + 1]);
    float2 uv2 = make_float2(hg->uvs[i2 * 2 + 0], hg->uvs[i2 * 2 + 1]);
    uv.x = b0 * uv0.x + bary.x * uv1.x + bary.y * uv2.x;
    uv.y = b0 * uv0.y + bary.x * uv1.y + bary.y * uv2.y;
  }

  // Tangent from dP/du
  float3 T = make_float3(1, 0, 0);
  float3 B = cross3(Ns, T);
  if (dot3(B, B) < 1e-6f) {
    T = make_float3(0, 1, 0);
  }

  float3 vc = make_float3(1.0f, 1.0f, 1.0f);
  if (hg->vertex_colors != nullptr) {
    float3 c0 = make_f3(&hg->vertex_colors[i0 * 3]);
    float3 c1 = make_f3(&hg->vertex_colors[i1 * 3]);
    float3 c2 = make_f3(&hg->vertex_colors[i2 * 3]);
    vc = c0 * b0 + c1 * bary.x + c2 * bary.y;
  }

  v->P = P;
  v->Ng = Ng;
  v->Ns = Ns;
  v->T = T;
  v->uv = uv;
  v->vc = vc;
  v->mat = hg->mat;
  if (hg->material_indices != nullptr && hg->num_materials > 0) {
    unsigned int mi = hg->material_indices[prim];
    if ((int)mi < hg->num_materials)
      v->mat = hg->materials[mi];
  }
  v->hit = 1;
}

extern "C" __global__ void __closesthit__shadow() {
  // Not used when TERMINATE_ON_FIRST_HIT is set — kept to satisfy SBT.
}

// Shared by radiance and shadow ray types: alpha-mask cutout, plus
// transparent-shadow pass-through for transmissive materials on shadow rays.
extern "C" __global__ void __anyhit__ah() {
  HitGroupData *hg = (HitGroupData *)optixGetSbtDataPointer();
  unsigned int prim = optixGetPrimitiveIndex();
  PrincipledGpu *m = hg->mat;
  if (hg->material_indices != nullptr && hg->num_materials > 0) {
    unsigned int mi = hg->material_indices[prim];
    if ((int)mi < hg->num_materials)
      m = hg->materials[mi];
  }
  // Shadow rays pass through transmissive surfaces (approximation of
  // Cycles' transparent shadows — no color attenuation for now).
  bool is_shadow_ray =
      (optixGetRayFlags() & OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT) != 0;
  if (is_shadow_ray && m->transmission > 0.5f) {
    optixIgnoreIntersection();
  }
  if (m->alpha_threshold <= 0.0f || m->base_color_tex == nullptr)
    return;
  // Recompute UV from barycentrics.
  if (hg->uvs == nullptr)
    return;
  int i0 = hg->indices[prim * 3 + 0];
  int i1 = hg->indices[prim * 3 + 1];
  int i2 = hg->indices[prim * 3 + 2];
  float2 bary = optixGetTriangleBarycentrics();
  float b0 = 1.0f - bary.x - bary.y;
  float2 uv0 = make_float2(hg->uvs[i0 * 2 + 0], hg->uvs[i0 * 2 + 1]);
  float2 uv1 = make_float2(hg->uvs[i1 * 2 + 0], hg->uvs[i1 * 2 + 1]);
  float2 uv2 = make_float2(hg->uvs[i2 * 2 + 0], hg->uvs[i2 * 2 + 1]);
  float2 uv =
      make_float2(b0 * uv0.x + bary.x * uv1.x + bary.y * uv2.x,
                  b0 * uv0.y + bary.x * uv1.y + bary.y * uv2.y);
  const float *M = m->uv_transform;
  uv = make_float2(M[0] * uv.x + M[1] * uv.y + M[2],
                   M[3] * uv.x + M[4] * uv.y + M[5]);
  // Bilinear fetch of the alpha channel, wrapped.
  int w = m->base_color_tex_w;
  int h = m->base_color_tex_h;
  if (w <= 0 || h <= 0)
    return;
  float u = uv.x - floorf(uv.x);
  float v = uv.y - floorf(uv.y);
  float fx = u * (float)w - 0.5f;
  float fy = (1.0f - v) * (float)h - 0.5f;
  int x0 = (int)floorf(fx);
  int y0 = (int)floorf(fy);
  float dx = fx - (float)x0;
  float dy = fy - (float)y0;
  auto wrap = [](int v, int m) {
    int r = v % m;
    return r < 0 ? r + m : r;
  };
  int x1 = wrap(x0 + 1, w);
  int y1 = wrap(y0 + 1, h);
  x0 = wrap(x0, w);
  y0 = wrap(y0, h);
  const float *data = m->base_color_tex;
  float a00 = data[(y0 * w + x0) * 4 + 3];
  float a10 = data[(y0 * w + x1) * 4 + 3];
  float a01 = data[(y1 * w + x0) * 4 + 3];
  float a11 = data[(y1 * w + x1) * 4 + 3];
  float a0 = a00 * (1.0f - dx) + a10 * dx;
  float a1 = a01 * (1.0f - dx) + a11 * dx;
  float alpha = a0 * (1.0f - dy) + a1 * dy;
  if (alpha < m->alpha_threshold)
    optixIgnoreIntersection();
}

extern "C" __global__ void __miss__ms() {
  PathVertex *v = get_path_vertex();
  v->hit = 0;
}

extern "C" __global__ void __miss__shadow() { optixSetPayload_0(1u); }

static __forceinline__ __device__ float3 clamp_indirect(float3 c,
                                                         float max_lum) {
  if (max_lum <= 0.0f)
    return c;
  float l = luminance(c);
  return (l > max_lum) ? c * (max_lum / l) : c;
}

// Ray/rect-light intersection: returns closest one-sided front-face hit inside
// [t_min, t_max). Rect-light planes are invisible from behind (Cycles default:
// emission on one side, backside doesn't render). We match NEE's sign
// convention where `ar.normal` points along emission direction.
static __device__ int intersect_rect_lights(float3 origin, float3 dir,
                                            float t_min, float t_max,
                                            float &t_out) {
  int best = -1;
  float t_best = t_max;
  for (int i = 0; i < params.num_rect_lights; i++) {
    AreaRectLight &ar = params.rect_lights[i];
    float3 normal = make_f3(ar.normal);
    float3 corner = make_f3(ar.corner);
    float3 u_axis = make_f3(ar.u_axis);
    float3 v_axis = make_f3(ar.v_axis);
    float denom = dot3(normal, dir);
    // Hit the emissive face when the ray is heading against the normal.
    if (denom >= -1e-6f)
      continue;
    float t = dot3(corner - origin, normal) / denom;
    if (t <= t_min || t >= t_best)
      continue;
    float3 Ph = origin + dir * t;
    float3 d = Ph - corner;
    float su = dot3(d, u_axis);
    float sv = dot3(d, v_axis);
    if (su < 0.0f || su > ar.size_u)
      continue;
    if (sv < 0.0f || sv > ar.size_v)
      continue;
    t_best = t;
    best = i;
  }
  t_out = t_best;
  return best;
}

static __device__ float3 trace_path(float3 origin, float3 dir, RNG &rng) {
  float3 throughput = make_float3(1, 1, 1);
  float3 L = make_float3(0, 0, 0);
  bool last_specular = true;
  float prev_bsdf_pdf = 0.0f;

  for (unsigned int bounce = 0; bounce < params.max_depth; bounce++) {
    PathVertex v;
    v.hit = 0;
    unsigned int hi, lo;
    pack_ptr(&v, hi, lo);
    optixTrace(params.traversable, origin, dir, 1e-4f, 1e20f, 0.0f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
               0, // SBT offset (radiance ray type)
               2, // SBT stride (2 ray types: radiance + shadow)
               0, // miss index (radiance miss)
               hi, lo);

    // Geometry hit distance (∞ on miss). Needed so a rect light only counts
    // when it occludes geometry rather than being behind a wall.
    float t_geom = 1e20f;
    if (v.hit != 0) {
      float3 delta = v.P - origin;
      t_geom = sqrtf(dot3(delta, delta));
    }
    float t_rect;
    int rect_idx = intersect_rect_lights(origin, dir, 1e-4f, t_geom, t_rect);
    if (rect_idx >= 0) {
      // Hit an area light in front of any geometry. Add emission on primary
      // rays and after specular bounces; NEE covers it for diffuse paths.
      if (bounce == 0 || last_specular) {
        float3 em =
            throughput * make_f3(params.rect_lights[rect_idx].emission);
        if (bounce > 0)
          em = clamp_indirect(em, params.clamp_indirect);
        L = L + em;
      }
      break;
    }

    if (v.hit == 0) {
      float3 bg = world_background(dir);
      float w = 1.0f;
      if (bounce > 0 && !last_specular && params.world_type == 1) {
        float p_env = envmap_pdf(dir);
        w = power_heuristic(prev_bsdf_pdf, p_env);
      }
      float3 bg_contrib = throughput * bg * w;
      if (bounce > 0)
        bg_contrib = clamp_indirect(bg_contrib, params.clamp_indirect);
      L = L + bg_contrib;
      break;
    }

    MaterialEval e = eval_material(v);
    // Flip normal if ray hit backside (for dielectric transmission).
    if (dot3(v.Ns, -dir) < 0.0f) {
      e.Ns = -e.Ns;
      build_frame(e.Ns, e.T, e.B);
    }

    // Emission (add on primary or after specular bounce)
    if (bounce == 0 || last_specular) {
      float3 e_contrib = throughput * e.emission;
      if (bounce > 0)
        e_contrib = clamp_indirect(e_contrib, params.clamp_indirect);
      L = L + e_contrib;
    }

    // NEE
    float3 wo = -dir;
    float3 nee = throughput * direct_light(e, v.P, wo, rng);
    if (bounce > 0)
      nee = clamp_indirect(nee, params.clamp_indirect);
    L = L + nee;

    // Sample BSDF for next bounce
    BsdfSample bs = sample_bsdf(e, wo, rng);
    if (bs.pdf <= 0.0f)
      break;
    float3 contrib = bs.f / bs.pdf;
    throughput = throughput * contrib;
    last_specular = bs.specular;
    prev_bsdf_pdf = bs.pdf;

    // Russian roulette after a few bounces
    if (bounce >= 3) {
      float q = fminf(fmaxf(luminance(throughput), 0.05f), 0.95f);
      if (rng.next() > q)
        break;
      throughput = throughput / q;
    }

    // Offset along geometric normal to reduce self-intersection
    float3 offset_n = dot3(bs.wi, v.Ng) > 0.0f ? v.Ng : -v.Ng;
    origin = v.P + offset_n * 1e-4f;
    dir = bs.wi;
  }
  return L;
}

extern "C" __global__ void __raygen__rg() {
  uint3 idx = optixGetLaunchIndex();
  uint3 dim = optixGetLaunchDimensions();
  unsigned int pixel = idx.y * dim.x + idx.x;

  float3 eye = make_f3(params.cam_eye);
  float3 U = make_f3(params.cam_u);
  float3 V = make_f3(params.cam_v);
  float3 W = make_f3(params.cam_w);

  float3 accum = make_float3(0, 0, 0);
  for (unsigned int s = 0; s < params.samples_per_pixel; s++) {
    RNG rng(pixel, s, 0u);
    float jx = rng.next();
    float jy = rng.next();
    float px = (2.0f * ((float)idx.x + jx) / (float)dim.x) - 1.0f;
    float py = (2.0f * ((float)idx.y + jy) / (float)dim.y) - 1.0f;
    float3 dir = normalize3(U * px + V * py + W);
    float3 origin = eye;

    if (params.cam_lens_radius > 0.0f) {
      float u1 = rng.next();
      float u2 = rng.next();
      float r = sqrtf(u1);
      float phi = 2.0f * M_PIf * u2;
      float lx = r * cosf(phi) * params.cam_lens_radius;
      float ly = r * sinf(phi) * params.cam_lens_radius;
      float3 Uunit = normalize3(U);
      float3 Vunit = normalize3(V);
      float3 focus_point = origin + dir * params.cam_focal_distance;
      origin = origin + Uunit * lx + Vunit * ly;
      dir = normalize3(focus_point - origin);
    }

    accum = accum + trace_path(origin, dir, rng);
  }
  float inv = 1.0f / (float)params.samples_per_pixel;
  accum = accum * inv;

  // y is 0 at top of image; we'll flip on CPU side for EXR.
  float *out = &params.image[pixel * 4];
  out[0] = accum.x;
  out[1] = accum.y;
  out[2] = accum.z;
  out[3] = 1.0f;
}
