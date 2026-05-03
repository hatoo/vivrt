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
  // Blender's image.pixels (and what the exporter writes to the bin) is
  // bottom-up: buffer row 0 is the bottom of the image. Blender UV v=0 is
  // also the bottom, so no flip on the y axis.
  float fy = v * (float)h - 0.5f;
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

// ---------- Volumes (homogeneous, Henyey-Greenstein phase) ----------

// Maximum nesting depth for mesh-bounded volumes within a single ray. Four is
// generous: real scenes top out at "two overlapping volumes" (e.g. a fog cube
// inside a smoke cube). World volume sits implicitly below the stack.
#define VOL_STACK_MAX 4

struct VolumeStack {
  const Volume *entries[VOL_STACK_MAX];
  int depth;
};

static __forceinline__ __device__ const Volume *
volume_stack_top(const VolumeStack &s) {
  if (s.depth > 0)
    return s.entries[s.depth - 1];
  return params.world_volume; // may be nullptr (vacuum)
}

static __forceinline__ __device__ void
volume_stack_push(VolumeStack &s, const Volume *v) {
  if (v == nullptr)
    return;
  if (s.depth < VOL_STACK_MAX)
    s.entries[s.depth++] = v;
  // Overflow silently — losing the deepest nesting is preferable to corrupt
  // memory; the scene would already be pathological at this nesting depth.
}

static __forceinline__ __device__ void
volume_stack_pop(VolumeStack &s, const Volume *v) {
  // Pop by reference equality scanned from top. BVH order isn't strict LIFO
  // (a back-face exit can come before the matching front-face entry was
  // recorded if rays miss tessellation), so a strict pop would desync.
  for (int i = s.depth - 1; i >= 0; i--) {
    if (s.entries[i] == v) {
      for (int j = i; j < s.depth - 1; j++)
        s.entries[j] = s.entries[j + 1];
      s.depth--;
      return;
    }
  }
}

// Average per-channel σ_t — used as the scalar pdf for distance sampling.
// We then track per-channel beam transmittance separately so chromatic
// volumes don't lose colour information through the sampler.
static __forceinline__ __device__ float vol_sigma_t_avg(const Volume *v) {
  return (v->sigma_t[0] + v->sigma_t[1] + v->sigma_t[2]) * (1.0f / 3.0f);
}

static __forceinline__ __device__ float3 vol_beam_transmittance(const Volume *v,
                                                                float t) {
  return make_float3(expf(-v->sigma_t[0] * t), expf(-v->sigma_t[1] * t),
                     expf(-v->sigma_t[2] * t));
}

// Closed-form path emission integral over [0, t] for a homogeneous volume:
//   ∫_0^t exp(-σ_t s) σ_e ds = σ_e/σ_t · (1 - exp(-σ_t t))   (per channel)
// Computed deterministically — no MC noise from emissive media. Falls back
// to σ_e · t in the σ_t→0 limit (Beer-Lambert is identity, integrand = σ_e).
static __device__ float3 vol_path_emission(const Volume *v, float t) {
  float3 e = make_float3(v->emission[0], v->emission[1], v->emission[2]);
  if (e.x <= 0.0f && e.y <= 0.0f && e.z <= 0.0f)
    return make_float3(0, 0, 0);
  auto component = [t](float sig_t, float em) {
    if (em <= 0.0f)
      return 0.0f;
    if (sig_t < 1e-8f)
      return em * t;
    return em * (1.0f - expf(-sig_t * t)) / sig_t;
  };
  return make_float3(component(v->sigma_t[0], e.x),
                     component(v->sigma_t[1], e.y),
                     component(v->sigma_t[2], e.z));
}

// Henyey-Greenstein phase function p(cos θ) and importance sampling.
// p(μ) = (1 - g²) / (4π · (1 + g² - 2g·μ)^(3/2))
static __forceinline__ __device__ float phase_hg_eval(float g, float cos_theta) {
  float g2 = g * g;
  float denom = 1.0f + g2 - 2.0f * g * cos_theta;
  return (1.0f - g2) /
         (4.0f * M_PIf * sqrtf(fmaxf(denom, 1e-12f)) * fmaxf(denom, 1e-12f));
}

// Sample a scatter direction from HG phase, returned in the local frame
// {x = T_perp1, y = T_perp2, z = forward}. Caller transforms to world.
static __device__ float3 phase_hg_sample(float g, float u1, float u2,
                                         float &pdf) {
  float cos_theta;
  if (fabsf(g) < 1e-3f) {
    cos_theta = 1.0f - 2.0f * u1; // isotropic
  } else {
    float k = (1.0f - g * g) / (1.0f - g + 2.0f * g * u1);
    cos_theta = (1.0f + g * g - k * k) / (2.0f * g);
  }
  cos_theta = fmaxf(-1.0f, fminf(1.0f, cos_theta));
  float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
  float phi = 2.0f * M_PIf * u2;
  pdf = phase_hg_eval(g, cos_theta);
  return make_float3(sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta);
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
  // Per-instance uniform random in [0..1). Derived from
  // `optixGetInstanceId()` via a hash so each particle / instance gets a
  // stable but-different value — feeds `COLOR_NODE_OBJECT_RANDOM` so the
  // material graph can lerp ColorRamps and Mix nodes per instance.
  float object_random;
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

// Bilinear-filter an equirect texture (3-channel, packed) at a world-space
// direction `dir`. Wraps u (longitude is periodic) and clamps v (poles are
// not). `rotation` is a 3×3 row-major matrix applied to `dir` before the
// equirect lookup so per-layer Mapping rotations don't have to bake into
// pixels.
static __device__ float3 sample_equirect_layer(const float *data, int w, int h,
                                                const float *rotation,
                                                float3 dir) {
  float3 d = make_float3(rotation[0] * dir.x + rotation[1] * dir.y + rotation[2] * dir.z,
                         rotation[3] * dir.x + rotation[4] * dir.y + rotation[5] * dir.z,
                         rotation[6] * dir.x + rotation[7] * dir.y + rotation[8] * dir.z);
  float theta = acosf(fminf(fmaxf(d.z, -1.0f), 1.0f));
  float phi = atan2f(d.y, d.x);
  float u = phi * (0.5f * INV_PIf);
  if (u < 0.0f)
    u += 1.0f;
  float v = theta * INV_PIf;
  float fx = u * (float)w - 0.5f;
  float fy = v * (float)h - 0.5f;
  int x0 = (int)floorf(fx);
  int y0 = (int)floorf(fy);
  float dx = fx - (float)x0;
  float dy = fy - (float)y0;
  auto wrap = [](int v, int m) {
    int r = v % m;
    return r < 0 ? r + m : r;
  };
  int x1 = wrap(x0 + 1, w);
  x0 = wrap(x0, w);
  y0 = max(0, min(h - 1, y0));
  int y1 = max(0, min(h - 1, y0 + 1));
  auto fetch = [&](int x, int y) {
    const float *p = &data[(y * w + x) * 3];
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

// Build a 3×3 rotation matrix from `envmap_rotation_z_rad` for the legacy
// single-envmap path so it can share `sample_equirect_layer`.
static __forceinline__ __device__ void make_rotation_z(float angle,
                                                        float (&m)[9]) {
  float c = cosf(angle), s = sinf(angle);
  m[0] = c; m[1] = -s; m[2] = 0;
  m[3] = s; m[4] = c;  m[5] = 0;
  m[6] = 0; m[7] = 0;  m[8] = 1;
}

static __device__ float3 world_background(float3 dir, bool is_camera_ray) {
  if (params.world_type == 0) {
    return make_f3(params.world_color) * params.world_strength;
  }
  // Light-Path camera-ray split: camera-visible misses read layer b
  // directly (the artist's backplate), while all other rays fall through
  // to envmap_data (= the lighting envmap = layer a). The CDF / NEE
  // sampler stays on envmap_data so MIS pdfs match the radiance lighting
  // rays observe — camera rays don't go through MIS, so the asymmetry
  // is unbiased.
  if (params.world_type == 2 &&
      params.envmap_split_by_camera_ray != 0 &&
      is_camera_ray &&
      params.envmap_data_b != nullptr) {
    float3 col = sample_equirect_layer(params.envmap_data_b,
                                       params.envmap_width_b,
                                       params.envmap_height_b,
                                       params.envmap_rotation_b, dir);
    return col * params.envmap_strength_b;
  }
  // Both single-layer and mixed worlds read from `envmap_data` so that
  // MIS-combined NEE and BSDF-sampled-miss strategies see the same
  // radiance at any given direction (essential for unbiased estimation).
  // For mixed worlds, `envmap_data` is the host-rasterised mix at CDF
  // resolution; the per-layer high-res `envmap_data_a` / `_b` are
  // currently kept around for future high-res sampling once the CDF
  // construction also moves to the high-res grid.
  float rot[9];
  make_rotation_z(-params.envmap_rotation_z_rad, rot);
  float3 col = sample_equirect_layer(params.envmap_data,
                                     params.envmap_width,
                                     params.envmap_height, rot, dir);
  return col * params.world_strength;
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
//
// Sampling distribution: each pixel (x, y) is picked with discrete probability
// P(x, y) = lum_raw × sin(θ_y) / Z, where Z = Σ lum_raw × sin(θ_y) =
// envmap_integral. The pixel covers solid-angle Ω = (2π² sin θ) / (W·H), so the
// solid-angle pdf is P / Ω = lum_raw × W·H / (2π² × Z) — no sin θ in the
// denominator, and `lum` must come from the raw texel (no world_strength) so
// the strength factor stays on the radiance side and direct-light contribution
// scales correctly with `world_strength`.
static __device__ float envmap_pdf(float3 dir) {
  // Active for both single-layer (world_type==1) and mixed (==2) envmaps.
  // The CDF lives on `envmap_data` either way (mixed worlds rasterise the
  // blend on the host so this PDF lookup stays consistent with the
  // direction sampler).
  if (params.world_type == 0 || params.envmap_integral <= 0.0f)
    return 0.0f;
  // For mixed worlds `envmap_rotation_z_rad` is 0 and this collapses to
  // identity — consistent with `world_background`'s shared lookup path.
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
  float lum_raw =
      0.2126f * px[0] + 0.7152f * px[1] + 0.0722f * px[2];
  return lum_raw * (float)(w * h) /
         (2.0f * M_PIf * M_PIf * params.envmap_integral);
}

static __device__ EnvSample sample_envmap(RNG &rng) {
  EnvSample s;
  s.dir = make_float3(0, 0, 1);
  s.L = make_float3(0, 0, 0);
  s.pdf = 0.0f;
  // Active for both single-layer and mixed envmaps; the CDF lives on
  // `envmap_data` for both cases (host rasterises the mix for type==2).
  if (params.world_type == 0 || params.envmap_integral <= 0.0f)
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
  // Apply rotation (inverse of what `world_background` does). For mixed
  // worlds `envmap_rotation_z_rad` is left at 0 (the rasterised grid is
  // already in world space) so this collapses to the identity, which
  // is the correct "undo" of `world_background`'s identity lookup.
  float c = cosf(params.envmap_rotation_z_rad);
  float sr = sinf(params.envmap_rotation_z_rad);
  s.dir = make_float3(d.x * c - d.y * sr, d.x * sr + d.y * c, d.z);
  // Read s.L from the same rasterised grid the CDF / pdf were built
  // from. Returning a different (higher-res) radiance value for the
  // same direction would make `f / pdf` inconsistent and over-count
  // bright bands of the mixed envmap. The slight loss in HDRI peak
  // brightness on NEE samples is compensated by BSDF-sampled paths
  // (which DO read high-res via `world_background`) — MIS combines
  // both strategies into an unbiased estimator only when each
  // strategy's `f / pdf` is internally consistent.
  const float *px = &params.envmap_data[(y * w + x) * 3];
  s.L = make_float3(px[0], px[1], px[2]) * params.world_strength;

  // pdf is over raw texel luminance only — see envmap_pdf for the derivation.
  // Folding world_strength into `lum` here would cancel out of the
  // L/pdf ratio in NEE and make contribution invariant to `world_strength`.
  float lum_raw =
      0.2126f * px[0] + 0.7152f * px[1] + 0.0722f * px[2];
  if (params.envmap_integral > 0.0f) {
    s.pdf = lum_raw * (float)(w * h) /
            (2.0f * M_PIf * M_PIf * params.envmap_integral);
  }
  return s;
}

// ---------- Visibility (shadow rays) ----------
// Two ray types: 0 = radiance, 1 = shadow. SBT stride = 2. Shadow hit group
// is a dummy (CH disabled, no AH) so only the miss program runs, which sets
// payload[0]=1 meaning "reached infinity = light visible".
static __device__ bool shadow_visible(float3 P, float3 dir, float tmax) {
  unsigned int miss_flag = 0;
  // Mask bit 1 = "blocks shadow rays"; instances with cast_shadow=false clear
  // it (e.g. classroom's paper-lantern drum) so NEE sees through them.
  optixTrace(params.traversable, P, dir, 1e-4f, tmax - 1e-3f, 0.0f,
             OptixVisibilityMask(0x02),
             OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
                 OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
             1, 2, 1, miss_flag);
  return miss_flag != 0;
}

// Same visibility query, but returns RGB beam transmittance through any
// volumes the shadow ray traverses on its way to the light. Volume-only
// boundaries are walked iteratively (CH fires on the radiance hit group,
// updates the local stack, and we re-trace from the next epsilon).
//
// Returns (0, 0, 0) on opaque occlusion. Capped at `VOL_STACK_MAX * 4`
// trace iterations to bound the worst case (alternating boundary
// crossings); past that we conservatively assume the light is unreachable
// — pathological scenes only.
static __device__ float3 shadow_transmittance(float3 P, float3 dir, float tmax,
                                              VolumeStack stack) {
  // Fast path: no volumes anywhere. Standard binary visibility — the
  // any-hit handler already lets transmissive surfaces pass through
  // (`m->transmission > 0.5f -> optixIgnoreIntersection`), so direct
  // light reaches the pool floor through clear water without paying
  // for the iterative loop on every shadow ray.
  if (params.world_volume == nullptr && stack.depth == 0) {
    return shadow_visible(P, dir, tmax) ? make_float3(1, 1, 1)
                                        : make_float3(0, 0, 0);
  }

  float3 tr = make_float3(1, 1, 1);
  float3 cur_origin = P;
  float remaining = tmax;
  const int max_iters = VOL_STACK_MAX * 4 + 4;

  for (int iter = 0; iter < max_iters; iter++) {
    if (remaining <= 1e-4f)
      return tr;

    // Trace using the radiance ray type so CH fires and we get hit info.
    PathVertex v;
    v.hit = 0;
    unsigned int hi, lo;
    pack_ptr(&v, hi, lo);
    optixTrace(params.traversable, cur_origin, dir, 1e-4f, remaining - 1e-3f,
               0.0f, OptixVisibilityMask(0x02), OPTIX_RAY_FLAG_NONE, 0, 2, 0,
               hi, lo);

    const Volume *cur_vol = volume_stack_top(stack);
    float t_seg;
    if (v.hit == 0) {
      t_seg = remaining; // ran out before any geometry
    } else {
      float3 d = v.P - cur_origin;
      t_seg = sqrtf(dot3(d, d));
    }

    // Beam transmittance through the current volume on this segment.
    if (cur_vol != nullptr) {
      tr = tr * vol_beam_transmittance(cur_vol, t_seg);
      if (luminance(tr) < 1e-6f)
        return make_float3(0, 0, 0);
    }

    if (v.hit == 0)
      return tr; // reached light

    // Surface in the way. Three cases:
    //   1. Pure volume container — walk past, push/pop the volume stack.
    //   2. Transmissive surface (Glass / Refraction / `transmission > 0`) —
    //      attenuate by (1−F)·base_color and continue past, so the sun
    //      reaches the pool floor through the water plane. Without this
    //      the water acts as an opaque occluder for NEE and the only
    //      light hitting underwater surfaces is the path-traced indirect
    //      bounce, which is dim and noisy.
    //   3. Anything else — opaque, return zero.
    if (v.mat != nullptr && v.mat->volume != nullptr &&
        v.mat->volume_only != 0) {
      bool entering = dot3(v.Ng, dir) < 0.0f;
      if (entering)
        volume_stack_push(stack, v.mat->volume);
      else
        volume_stack_pop(stack, v.mat->volume);
      // Advance just past the surface. Use ray dir (not Ng) — we want to
      // stay on the same side of the boundary we just crossed.
      float p_scale =
          fmaxf(fmaxf(fabsf(v.P.x), fabsf(v.P.y)), fabsf(v.P.z));
      float eps = fmaxf(1e-4f, p_scale * 1e-5f);
      cur_origin = v.P + dir * eps;
      remaining -= t_seg + eps;
      continue;
    }
    if (v.mat != nullptr && v.mat->transmission > 0.0f) {
      // Transparent shadow: keep the ray going through the surface,
      // attenuated by `(1 − F) · base_color · transmission`. We can't
      // call `eval_material` here because it's defined later in the
      // translation unit and forward-declaring it just to share the
      // (textured) base colour would pull in a lot of unrelated state;
      // the constant `base_color` is correct for water / clear-glass
      // (the common case) and only loses tint texture variation, which
      // is rarely present on transmissive shaders.
      float3 base = make_f3(v.mat->base_color);
      float cos_i = fabsf(dot3(v.Ns, dir));
      float F0_d = ((v.mat->ior - 1.0f) / (v.mat->ior + 1.0f)) *
                   ((v.mat->ior - 1.0f) / (v.mat->ior + 1.0f));
      float F = schlick_scalar(cos_i, F0_d);
      float3 t_factor = base * (1.0f - F) * v.mat->transmission;
      // Surfaces with `transmission < 1` (e.g. Principled at 0.5) still
      // partially absorb — fold the missing fraction into 0 so the ray
      // does NOT pretend to also transmit the diffuse / specular share.
      tr = tr * t_factor;
      if (luminance(tr) < 1e-6f)
        return make_float3(0, 0, 0);
      float p_scale =
          fmaxf(fmaxf(fabsf(v.P.x), fabsf(v.P.y)), fabsf(v.P.z));
      float eps = fmaxf(1e-4f, p_scale * 1e-5f);
      cur_origin = v.P + dir * eps;
      remaining -= t_seg + eps;
      continue;
    }
    return make_float3(0, 0, 0); // opaque occlusion
  }
  // Exceeded the iteration cap — conservatively block.
  return make_float3(0, 0, 0);
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
    float3 inp, unsigned int op, float b, float c, bool swap) {
  // `swap` flips operand order for non-commutative ops so the node can act as
  // `b OP input` instead of `input OP b` — mirrors Cycles ShaderNodeMath with
  // the texture chain on input[1]. Commutative ops ignore swap.
  float3 o;
  switch (op) {
    default:
    case 0: // ADD
      o = make_float3(inp.x + b, inp.y + b, inp.z + b);
      break;
    case 1: // SUBTRACT
      if (swap) {
        o = make_float3(b - inp.x, b - inp.y, b - inp.z);
      } else {
        o = make_float3(inp.x - b, inp.y - b, inp.z - b);
      }
      break;
    case 2: // MULTIPLY
      o = make_float3(inp.x * b, inp.y * b, inp.z * b);
      break;
    case 3: // DIVIDE
      if (swap) {
        // b / inp, per channel. Cycles returns 0 when the denominator is 0.
        o = make_float3(inp.x == 0.0f ? 0.0f : b / inp.x,
                        inp.y == 0.0f ? 0.0f : b / inp.y,
                        inp.z == 0.0f ? 0.0f : b / inp.z);
      } else {
        o = make_float3(b == 0.0f ? 0.0f : inp.x / b,
                        b == 0.0f ? 0.0f : inp.y / b,
                        b == 0.0f ? 0.0f : inp.z / b);
      }
      break;
    case 4: // POWER
      if (swap) {
        float base = fmaxf(b, 0.0f);
        o = make_float3(powf(base, inp.x), powf(base, inp.y), powf(base, inp.z));
      } else {
        o = make_float3(powf(fmaxf(inp.x, 0.0f), b),
                        powf(fmaxf(inp.y, 0.0f), b),
                        powf(fmaxf(inp.z, 0.0f), b));
      }
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
    ColorGraphNode *nodes, int n_nodes, int output, float2 base_uv, float3 vc,
    float object_random) {
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
      bool swap = pp[5] != 0u;
      float3 o = math_apply_rgb(slots[iin], op, b, c, swap);
      if (clamp_out) {
        o.x = fmaxf(0.0f, fminf(1.0f, o.x));
        o.y = fmaxf(0.0f, fminf(1.0f, o.y));
        o.z = fmaxf(0.0f, fminf(1.0f, o.z));
      }
      slots[i] = o;
    } else if (tag == COLOR_NODE_RGB_CURVE) {
      int iin = (int)pp[0];
      unsigned long long ptr =
          ((unsigned long long)pp[2] << 32) | (unsigned long long)pp[1];
      const float *lut = (const float *)ptr;
      float3 src = slots[iin];
      auto fetch = [&](int channel, float x) {
        float xc = fmaxf(0.0f, fminf(1.0f, x));
        float fx = xc * 255.0f;
        int i0 = (int)floorf(fx);
        int i1 = i0 < 255 ? i0 + 1 : 255;
        float t = fx - (float)i0;
        int base = channel * 256;
        return lut[base + i0] * (1.0f - t) + lut[base + i1] * t;
      };
      slots[i] = make_float3(fetch(0, src.x), fetch(1, src.y), fetch(2, src.z));
    } else if (tag == COLOR_NODE_BRIGHT_CONTRAST) {
      int iin = (int)pp[0];
      float bright = __uint_as_float(pp[1]);
      float contrast = __uint_as_float(pp[2]);
      // Cycles: a = 1 + contrast; b = bright - contrast/2.
      float a = 1.0f + contrast;
      float b = bright - 0.5f * contrast;
      float3 s = slots[iin];
      slots[i] = make_float3(fmaxf(0.0f, a * s.x + b),
                             fmaxf(0.0f, a * s.y + b),
                             fmaxf(0.0f, a * s.z + b));
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
    } else if (tag == COLOR_NODE_VERTEX_COLOR) {
      slots[i] = vc;
    } else if (tag == COLOR_NODE_OBJECT_RANDOM) {
      slots[i] = make_float3(object_random, object_random, object_random);
    } else if (tag == COLOR_NODE_COLOR_RAMP) {
      // Payload: [in_slot, lut_ptr_lo, lut_ptr_hi, lut_len]. The host bakes
      // the ramp to `lut_len` linear-RGB stops; we look up by Fac.x with
      // bilinear interpolation across the table.
      int iin = (int)pp[0];
      unsigned long long ptr =
          ((unsigned long long)pp[2] << 32) | (unsigned long long)pp[1];
      const float *lut = (const float *)ptr;
      int len = (int)pp[3];
      float3 src = slots[iin];
      // ColorRamp's Fac socket reads as either a scalar or the R channel of
      // an incoming colour — match `_socket_constant_rgb`'s convention.
      float fac = src.x;
      fac = fmaxf(0.0f, fminf(1.0f, fac));
      float fx = fac * (float)(len - 1);
      int i0 = (int)floorf(fx);
      int i1 = i0 < (len - 1) ? i0 + 1 : (len - 1);
      float t = fx - (float)i0;
      float3 c0 = make_float3(lut[i0 * 3 + 0], lut[i0 * 3 + 1], lut[i0 * 3 + 2]);
      float3 c1 = make_float3(lut[i1 * 3 + 0], lut[i1 * 3 + 1], lut[i1 * 3 + 2]);
      slots[i] = make_float3(c0.x * (1.0f - t) + c1.x * t,
                             c0.y * (1.0f - t) + c1.y * t,
                             c0.z * (1.0f - t) + c1.z * t);
    } else {
      // Unknown tag means the host built a graph with a node type the device
      // doesn't recognize — a real bug in the exporter / scene_loader. Print
      // once per launch (gated by launch-idx==0) so the log isn't flooded.
      uint3 _li = optixGetLaunchIndex();
      if (_li.x == 0 && _li.y == 0) {
        printf("[vibrt] warn: eval_color_graph: unknown node tag %u at slot %d -- returning white\n",
               tag, i);
      }
      slots[i] = make_float3(1.0f, 1.0f, 1.0f);
    }
  }
  if (output < 0 || output >= n) {
    uint3 _li = optixGetLaunchIndex();
    if (_li.x == 0 && _li.y == 0) {
      printf("[vibrt] warn: eval_color_graph: output index %d out of range [0, %d) -- returning white\n",
             output, n);
    }
    return make_float3(1.0f, 1.0f, 1.0f);
  }
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
  // Filter-glossy-inflated coat α. Pre-computed in eval_material so the
  // clearcoat lobe in eval_bsdf / sample_bsdf doesn't have to re-derive
  // it from the material's coat_roughness on every call (and so it
  // picks up the path's min_alpha state along with the main lobe).
  float coat_alpha;
  float ior;
  float transmission;
  float3 emission;
  float3 Ns; // shading normal
  float3 T;  // rotated by tangent_rotation for anisotropy
  float3 B;
  PrincipledGpu *mat; // raw material ptr for coat/sheen/sss params
};

// `min_alpha` is the path's "filter glossy" floor — Cycles' filter-glossy
// settings inflate the BSDF roughness on indirect glossy bounces so that
// near-mirror bounces can't resample a tiny solid angle and produce
// fireflies. We approximate it by raising `alpha`, `alpha_x`, `alpha_y`
// to at least this value. 0 = primary ray (no inflation).
static __device__ MaterialEval eval_material(const PathVertex &v,
                                              float min_alpha = 0.0f) {
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
                                m->color_graph_output, v.uv, v.vc,
                                v.object_random);
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
  if (m->transmission_tex != nullptr) {
    float3 t =
        sample_rgba(m->transmission_tex, m->transmission_tex_w,
                    m->transmission_tex_h, m->transmission_tex_channels, uv);
    e.transmission = e.transmission * t.x;
  }
  if (m->emission_tex != nullptr) {
    float3 t =
        sample_rgba(m->emission_tex, m->emission_tex_w,
                    m->emission_tex_h, m->emission_tex_channels, uv);
    e.emission = e.emission * t;
  }
  e.metallic = fminf(fmaxf(e.metallic, 0.0f), 1.0f);
  e.transmission = fminf(fmaxf(e.transmission, 0.0f), 1.0f);
  // No separate roughness floor — the alpha clamp below is the sampler's
  // numerical safeguard; forcing a minimum roughness on top of it would
  // silently turn artist-authored mirrors into slightly-rough dielectrics.
  e.alpha = fmaxf(e.roughness * e.roughness, 1e-4f);
  // Filter-glossy: the path tracer raises `min_alpha` once it's already
  // taken at least one glossy / specular bounce so that subsequent
  // near-mirror evals can't refocus the radiance into a tiny solid
  // angle. Approximates Cycles' Light Paths > Filter Glossy. min_alpha=0
  // (primary ray) leaves alpha untouched.
  if (min_alpha > e.alpha)
    e.alpha = min_alpha;

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
      // Central-difference slope × user strength. Linear in bump_strength
      // so Strength=1 reproduces Blender's default amplitude and the user
      // doesn't need to pre-scale for an empirical "subtle" factor.
      float sx = -(h_px - h_mx) * m->bump_strength;
      float sy = -(h_py - h_my) * m->bump_strength;
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
                             n.z * 2.0f - 1.0f);
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
  // If closesthit interpolated an authored tangent (hair strand axis), use
  // it as the anisotropy / hair tangent after re-orthogonalising against the
  // possibly-perturbed shading normal. Otherwise fall back to a synthetic
  // tangent — the previous behaviour for every non-hair surface.
  if (dot3(v.T, v.T) > 0.5f) {
    float3 Tperp = v.T - e.Ns * dot3(v.T, e.Ns);
    float Tn2 = dot3(Tperp, Tperp);
    if (Tn2 > 1e-8f) {
      e.T = Tperp * (1.0f / sqrtf(Tn2));
      e.B = normalize3(cross3(e.Ns, e.T));
    } else {
      build_frame(e.Ns, e.T, e.B);
    }
  } else {
    build_frame(e.Ns, e.T, e.B);
  }

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
  // Coat α also gets the filter-glossy floor applied. Pebble materials
  // in pabellon use a Fresnel-driven MixShader → my exporter routes the
  // glossy side onto the coat lobe with a low coat_roughness (~0.45),
  // which without inflation can refocus indirect bounces through the
  // pool's surface into a sharper cone than the path has already
  // sampled. Same min_alpha as the main lobe — Cycles' filter_glossy
  // is a single state shared across all glossy lobes.
  e.coat_alpha = fmaxf(m->coat_roughness * m->coat_roughness, 1e-4f);
  if (min_alpha > e.coat_alpha)
    e.coat_alpha = min_alpha;
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

// Kajiya-Kay hair: simplified tangent-aligned specular + "sin(T,L)" diffuse.
// Not energy-conservative by construction — it's a classic approximation. The
// specular uses the tilt-shifted H direction so the highlight cones forward
// along the strand (Offset slider on Cycles' Hair BSDF). Sampling is plain
// cosine-weighted hemisphere; MIS picks up the slack.
static __device__ BsdfEval eval_hair(const MaterialEval &e, float3 wo,
                                     float3 wi) {
  BsdfEval r;
  r.f = make_float3(0, 0, 0);
  r.pdf = 0.0f;
  float NoL = dot3(e.Ns, wi);
  if (NoL <= 0.0f)
    return r;

  // Diffuse: peaks when light is perpendicular to the strand.
  float TdotL = dot3(wi, e.T);
  float sin_TL = sqrtf(fmaxf(1.0f - TdotL * TdotL, 0.0f));

  // Specular: tangent-aligned cone. sin(offset) tilts the cone along T so the
  // highlight shifts toward the tip of the strand (matches Cycles' "Offset").
  float3 H = wo + wi;
  float Hlen = sqrtf(fmaxf(dot3(H, H), 1e-12f));
  H = H * (1.0f / Hlen);
  float TdotH = dot3(H, e.T);
  float tilt = sinf(e.mat->hair_offset);
  float TdotH_eff = TdotH - tilt;
  float cone = sqrtf(fmaxf(1.0f - TdotH_eff * TdotH_eff, 0.0f));
  float exponent = 2.0f / fmaxf(e.mat->hair_roughness_u *
                                    e.mat->hair_roughness_u,
                                1e-4f);
  float spec = powf(fmaxf(cone, 0.0f), exponent);

  r.f = (e.base_color * sin_TL + make_float3(spec, spec, spec))
        * (INV_PIf * NoL);
  r.pdf = NoL * INV_PIf;
  return r;
}

static __device__ BsdfEval eval_bsdf(const MaterialEval &e, float3 wo,
                                     float3 wi) {
  BsdfEval r;
  r.f = make_float3(0, 0, 0);
  r.pdf = 0.0f;

  float NoV = dot3(e.Ns, wo);
  if (NoV <= 0.0f)
    return r;
  if (e.mat->hair_weight > 0.5f)
    return eval_hair(e, wo, wi);

  float NoL = dot3(e.Ns, wi);
  bool reflect = NoL > 0.0f;
  bool transmit = !reflect && e.transmission > 0.0f;
  // SSS can also contribute diffusely when the light is slightly behind the
  // surface (wraparound lobe).
  bool sss_backlit = !reflect && e.mat->sss_weight > 0.0f && NoL > -1.0f;
  // Translucent BSDF: Lambertian on the back hemisphere. Active when wi is
  // on the side opposite wo (NoL < 0) — light passing through a thin sheet.
  bool translucent_active =
      !reflect && e.mat->translucent_weight > 0.0f && e.transmission <= 0.0f;

  float F0_d =
      ((e.ior - 1.0f) / (e.ior + 1.0f)) * ((e.ior - 1.0f) / (e.ior + 1.0f));

  // Coat Fresnel parameters (used both for MIS weighting and the coat BRDF).
  PrincipledGpu *mc = e.mat;
  float coat_w_mat = mc->coat_weight;
  float coat_alpha = e.coat_alpha;  // already filter-glossy-clamped
  float cF0 = ((mc->coat_ior - 1.0f) / (mc->coat_ior + 1.0f)) *
              ((mc->coat_ior - 1.0f) / (mc->coat_ior + 1.0f));

  // Weights for MIS between lobes. Coat gets `coat_weight·F_coat(V)` so its
  // sampling budget follows the actual lobe intensity at this view angle —
  // at grazing the coat dominates, at normal it contributes weakly.
  // translucent_weight steals from the forward diffuse budget: t_w=1 → no
  // forward Lambert, all diffuse energy on the back hemisphere; t_w=0.5 →
  // even split (Mix Shader of Diffuse + Translucent).
  float t_w = mc->translucent_weight;
  float w_base_diff = (1.0f - e.metallic) * (1.0f - e.transmission);
  float w_diffuse = w_base_diff * (1.0f - t_w);
  float w_translucent = w_base_diff * t_w;
  float w_spec = e.metallic + (1.0f - e.metallic) * (1.0f - e.transmission);
  float w_trans = (1.0f - e.metallic) * e.transmission;
  float w_coat = coat_w_mat * schlick_scalar(NoV, cF0);
  float w_sum = w_diffuse + w_translucent + w_spec + w_trans + w_coat;
  if (w_sum <= 0.0f)
    return r;
  float p_diff = w_diffuse / w_sum;
  float p_translucent = w_translucent / w_sum;
  float p_spec = w_spec / w_sum;
  float p_trans = w_trans / w_sum;
  float p_coat = w_coat / w_sum;

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
    // Diffuse (Lambert, possibly wrap-shifted by SSS). The wrap term only
    // modifies f — the sampler is plain cosine hemisphere, so the MIS pdf
    // has to stay at NoL/π to match what was actually sampled.
    if (w_diffuse > 0.0f) {
      float effective = (1.0f - sss_w) * NoL_std + sss_w * NoL_wrap;
      float3 bc_sss = e.base_color * (make_float3(1, 1, 1) * (1.0f - sss_w) +
                                      sss_tint * sss_w);
      r.f = r.f + bc_sss * (INV_PIf * w_diffuse * effective);
      r.pdf += p_diff * NoL_std * INV_PIf;
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
    // Glass / transmissive materials sample the reflect-vs-refract split via
    // `fresnel_dielectric` (exact), so the eval has to use the same Fresnel
    // function — Schlick over-predicts F by ~2× at low IOR (1.0–1.2), which
    // for water (IOR=1.1) at the camera's grazing pool angle made
    // glass-reflection paths carry too much radiance and transmission paths
    // too little. The two formulas match within ~1% for IOR≥1.4 (the typical
    // dielectric range), so non-glass materials don't notice.
    float F_dielec = (e.transmission > 0.0f)
        ? fresnel_dielectric(VoH, e.ior)
        : schlick_scalar(VoH, F0_d);
    float3 f_metal = F_metal * (D * G / fmaxf(4.0f * NoV * NoL, 1e-8f));
    float3 f_dielec = make_float3(F_dielec, F_dielec, F_dielec) *
                      (D * G / fmaxf(4.0f * NoV * NoL, 1e-8f));
    // Dielectric surface reflection. For a matte material this is the
    // standard "spec layer on top of diffuse"; for a glass material
    // (transmission=1) it's the Fresnel-reflective component the
    // transmission lobe's sampler picked when `rng.next() < F`. Without
    // this term, a Glass BSDF surface evaluated on a reflected wi
    // returns f=0 and pdf=0, the path tracer drops the reflection
    // entirely, and at grazing angles where glass should mirror-reflect
    // the rays just transmit straight through to the sky — pabellon's
    // water plane appeared as the bright sunset sky shining through,
    // not as a reflective pond.
    float3 f_spec = e.metallic * f_metal +
                    (1.0f - e.metallic) * f_dielec;

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
             (1.0f - e.metallic) *
                 make_float3(f_dielec_ms, f_dielec_ms, f_dielec_ms);

    r.f = r.f + f_spec * NoL;
    float pdf_spec = ggx_vndf_pdf_aniso(Vloc, Hloc, e.alpha_x, e.alpha_y);
    r.pdf += p_spec * pdf_spec;
    // Glass reflection samples this same Fresnel lobe via the
    // transmission branch's `rng.next() < F` decision. Account for that
    // contribution so MIS-weighted estimators get the right pdf when
    // we land on a reflected wi for a transmissive surface.
    if (e.transmission > 0.0f && p_trans > 0.0f) {
        r.pdf += p_trans * F_dielec * pdf_spec;
    }

    // --- Coat (clearcoat): additive isotropic GGX dielectric above base. ---
    if (coat_w_mat > 0.0f) {
      float Fc = schlick_scalar(VoH, cF0);
      float cD = ggx_D(NoH, coat_alpha);
      float cG = smith_G1(NoV, coat_alpha) * smith_G1(NoL, coat_alpha);
      float coat_f = coat_w_mat * Fc * cD * cG / fmaxf(4.0f * NoV * NoL, 1e-8f);
      // Round-trip coat transmission attenuates base lobes: (1-F) on entry
      // and on exit.
      float T_view = 1.0f - coat_w_mat * schlick_scalar(NoV, cF0);
      float T_light = 1.0f - coat_w_mat * schlick_scalar(NoL, cF0);
      r.f = r.f * (T_view * T_light);
      r.f = r.f + make_float3(coat_f, coat_f, coat_f) * NoL;
      // Isotropic GGX VNDF density for the coat-sampler. Narrow car-paint
      // clearcoats (coat_roughness ≈ 0.1) are hard for the broad base-spec
      // sampler to hit, so without this branch coat contributions arrive
      // almost entirely through NEE and highlights from environment /
      // indirect bounces fall several-× short of Cycles.
      float pdf_coat = cD * smith_G1(NoV, coat_alpha) /
                       fmaxf(4.0f * NoV, 1e-12f);
      r.pdf += p_coat * pdf_coat;
    }

    // --- Sheen: soft grazing-angle lobe on top of diffuse. Symmetric in
    // NoV and NoL so the BRDF is reciprocal, and normalised by 1/π so the
    // integral stays bounded as sheen_roughness→0. Still an ad-hoc shape
    // (not Charlie), but energy-sane and reciprocal.
    float sheen_w = mc->sheen_weight;
    if (sheen_w > 0.0f) {
      float sr = fmaxf(mc->sheen_roughness, 1e-3f);
      float inv_sr = 1.0f / sr;
      float sf = powf(fmaxf(1.0f - NoV, 0.0f), inv_sr) *
                 powf(fmaxf(1.0f - NoL, 0.0f), inv_sr) * sheen_w * INV_PIf;
      float3 tint = make_f3(mc->sheen_tint);
      r.f = r.f + tint * sf * NoL;
    }
  } else if (transmit) {
    // Rough dielectric transmission (Walter et al.). Half-vector
    // h ∝ η_i·ωo + η_t·ωi → divide by η_i, so with eta = η_t/η_i the form is
    // `wo + wi * eta`. This matches the H the VNDF sampler implicitly used
    // in sample_bsdf, so eval's pdf can be inverted back to sampler's pdf.
    float eta = e.ior;
    float3 n_oriented = NoV > 0.0f ? e.Ns : -e.Ns;
    if (NoV < 0.0f)
      eta = 1.0f / eta;
    float3 H = -normalize3(wo + wi * eta);
    if (dot3(H, n_oriented) < 0.0f)
      H = -H;
    float VoH = dot3(wo, H);
    float LoH = dot3(wi, H);
    float abs_VoH = fabsf(VoH);
    float abs_LoH = fabsf(LoH);
    float abs_NoH = fabsf(dot3(e.Ns, H));
    float abs_NoV = fabsf(NoV);
    float abs_NoL = fabsf(NoL);
    float F = fresnel_dielectric(VoH, eta);
    float D = ggx_D(abs_NoH, e.alpha);
    float G1_v = smith_G1(abs_NoV, e.alpha);
    float G = G1_v * smith_G1(abs_NoL, e.alpha);
    float sqrt_den = VoH + eta * LoH;
    float den2 = fmaxf(sqrt_den * sqrt_den, 1e-8f);
    float btdf = (1.0f - F) * D * G * eta * eta * abs_VoH * abs_LoH /
                 (abs_NoV * abs_NoL * den2);
    r.f = e.base_color * btdf * abs_NoL * w_trans;
    // pdf_wi = p_trans · P(refract branch) · VNDF(H) · |dH/dwi|.
    float pdf_h = D * G1_v * abs_VoH / fmaxf(abs_NoV, 1e-12f);
    float jacobian = eta * eta * abs_LoH / den2;
    r.pdf += p_trans * (1.0f - F) * pdf_h * jacobian;
  } else if (translucent_active || sss_backlit) {
    // Back hemisphere (NoL < 0). Translucent (proper back-Lambert) and the
    // SSS wrap lobe both contribute additively here. Refraction is mutually
    // exclusive (handled above) — Glass + Translucent isn't a Cycles combo.
    if (translucent_active && w_translucent > 0.0f) {
      float abs_NoL = -NoL;
      r.f = r.f + e.base_color * (INV_PIf * w_translucent * abs_NoL);
      // sample_bsdf uses a cosine-weighted *back* hemisphere, so the pdf
      // matches the forward-Lambert form with |NoL|.
      r.pdf += p_translucent * abs_NoL * INV_PIf;
    }
    if (sss_backlit) {
      // Wrap-Lambert SSS hack — the cosine-hemisphere sampler can't reach
      // below-surface directions on its own, so r.pdf gets no SSS term;
      // NEE carries these paths.
      float w_wrap = fmaxf(NoL_wrap, 0.0f);
      if (w_wrap > 0.0f && w_diffuse > 0.0f) {
        float3 bc_sss = e.base_color * (make_float3(1, 1, 1) * (1.0f - sss_w) +
                                        sss_tint * sss_w);
        r.f = r.f + bc_sss * (INV_PIf * w_diffuse * sss_w * w_wrap);
      }
    }
  }
  return r;
}

// Sample BSDF, returning new direction wi, evaluated f*cosθ, and PDF.
// Path-tracer per-type bounce counters key off this lobe id. Cycles caps
// `diffuse_bounces` / `glossy_bounces` / `transmission_bounces` separately
// from the total `max_bounces`, so the kernel needs to know which lobe
// each sampled direction came from.
#define LOBE_DIFFUSE 0
#define LOBE_GLOSSY 1
#define LOBE_TRANSMISSION 2

struct BsdfSample {
  float3 wi;
  float3 f;
  float pdf;
  bool specular; // true for rough-specular that shouldn't NEE-double-count
  int lobe;      // LOBE_DIFFUSE / LOBE_GLOSSY / LOBE_TRANSMISSION
};

static __device__ BsdfSample sample_bsdf(const MaterialEval &e, float3 wo,
                                         RNG &rng) {
  BsdfSample s;
  s.wi = make_float3(0, 0, 1);
  s.f = make_float3(0, 0, 0);
  s.pdf = 0.0f;
  s.specular = false;
  s.lobe = LOBE_DIFFUSE;

  float NoV = dot3(e.Ns, wo);
  if (NoV <= 0.0f)
    return s;

  // Hair: cosine-weighted sample around Ns. Not optimal for the tangent
  // cone lobe, but MIS against NEE covers for it and stratification is cheap.
  if (e.mat->hair_weight > 0.5f) {
    float u1 = rng.next();
    float u2 = rng.next();
    float r = sqrtf(u1);
    float phi = 2.0f * M_PIf * u2;
    float3 local = make_float3(r * cosf(phi), r * sinf(phi),
                               sqrtf(fmaxf(0.0f, 1.0f - u1)));
    s.wi = normalize3(e.T * local.x + e.B * local.y + e.Ns * local.z);
    BsdfEval ev = eval_hair(e, wo, s.wi);
    s.f = ev.f;
    s.pdf = ev.pdf;
    // Hair-Kajiya is mostly a forward-cone lobe + a diffuse base; bucket
    // it as glossy because the dominant contribution shouldn't deplete
    // the diffuse-bounces budget that thinks of brick / paint walls.
    s.lobe = LOBE_GLOSSY;
    return s;
  }

  // Coat weight mirrors eval_bsdf: scale by F_coat(V) so sampling budget
  // follows the actual coat intensity at this view angle (grazing >> normal).
  float coat_w_mat = e.mat->coat_weight;
  float coat_alpha = e.coat_alpha;  // already filter-glossy-clamped
  float cF0 = ((e.mat->coat_ior - 1.0f) / (e.mat->coat_ior + 1.0f)) *
              ((e.mat->coat_ior - 1.0f) / (e.mat->coat_ior + 1.0f));
  float t_w = e.mat->translucent_weight;
  float w_base_diff = (1.0f - e.metallic) * (1.0f - e.transmission);
  float w_diffuse = w_base_diff * (1.0f - t_w);
  float w_translucent = w_base_diff * t_w;
  float w_spec = e.metallic + (1.0f - e.metallic) * (1.0f - e.transmission);
  float w_trans = (1.0f - e.metallic) * e.transmission;
  float w_coat = coat_w_mat * schlick_scalar(NoV, cF0);
  float total = w_diffuse + w_translucent + w_spec + w_trans + w_coat;
  if (total <= 0.0f)
    return s;
  float p_diff = w_diffuse / total;
  float p_translucent = w_translucent / total;
  float p_spec = w_spec / total;
  float p_trans = w_trans / total;

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
    s.lobe = LOBE_DIFFUSE;
  } else if (u < p_diff + p_translucent) {
    // Translucent: cosine-weighted *back* hemisphere. Negate the local Z so
    // wi exits the surface on the side opposite Ns. Bucketed as
    // LOBE_DIFFUSE so it shares Cycles' diffuse-bounces budget — Cycles
    // categorises Translucent BSDF as a diffuse path-type as well.
    float r = sqrtf(u1);
    float phi = 2.0f * M_PIf * u2;
    float3 local = make_float3(r * cosf(phi), r * sinf(phi),
                               -sqrtf(fmaxf(0.0f, 1.0f - u1)));
    s.wi = normalize3(e.T * local.x + e.B * local.y + e.Ns * local.z);
    s.lobe = LOBE_DIFFUSE;
  } else if (u < p_diff + p_translucent + p_spec) {
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
    // Fires when roughness has been clamped to the minimum (author asked for
    // a mirror); NEE can't find such a narrow lobe, so flag as specular and
    // let trace_path add BSDF-sampled emission directly.
    if (e.alpha <= 0.02f * 0.02f)
      s.specular = true;
    s.lobe = LOBE_GLOSSY;
  } else if (u < p_diff + p_translucent + p_spec + p_trans) {
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
    if (e.alpha <= 0.02f * 0.02f)
      s.specular = true;
    s.lobe = LOBE_TRANSMISSION;
  } else {
    // Coat: isotropic GGX VNDF on coat_alpha. Always reflects, so no
    // Fresnel reflect/transmit branch here.
    float3 Vlocal = make_float3(dot3(wo, e.T), dot3(wo, e.B), dot3(wo, e.Ns));
    float3 Hlocal = sample_ggx_vndf(Vlocal, coat_alpha, u1, u2);
    float3 H = normalize3(e.T * Hlocal.x + e.B * Hlocal.y + e.Ns * Hlocal.z);
    float VoH = dot3(wo, H);
    float3 wi = normalize3(H * (2.0f * VoH) - wo);
    if (dot3(e.Ns, wi) <= 0.0f)
      return s;
    s.wi = wi;
    if (coat_alpha <= 0.02f * 0.02f)
      s.specular = true;
    s.lobe = LOBE_GLOSSY;
  }

  BsdfEval ev = eval_bsdf(e, wo, s.wi);
  s.f = ev.f;
  s.pdf = ev.pdf;
  return s;
}

// ---------- IES profile lookup ----------
//
// Returns the IES table's normalised intensity multiplier in [0, 1] for
// the world-space sample direction `dir_world` (light → surface). When
// `ies_data == nullptr` returns 1.0 (no IES, isotropic).
//
// Layout of `ies_data` matches `upload_ies_buffers` in render.rs:
//   [0           .. n_v          ): vertical angles in degrees (theta)
//   [n_v         .. n_v + n_h    ): horizontal angles in degrees (phi)
//   [n_v + n_h   .. + n_v * n_h  ): normalised candelas, row-major over
//                                    phi × theta — `[h * n_v + v]`
//
// Convention: `light_rotation` is the light's object→world 3×3 (row-
// major). We invert (transpose for pure rotations) to convert
// `dir_world` into the light's local frame, then derive theta from the
// local -Z axis (the IES "down" convention) and phi from atan2 around
// local +Z.
static __device__ float ies_lookup(const float *ies_data,
                                   unsigned int n_v, unsigned int n_h,
                                   const float *light_rotation,
                                   float3 dir_world) {
  if (ies_data == nullptr || n_v == 0 || n_h == 0)
    return 1.0f;
  // light_rotation row-major: rows are world-space basis vectors of the
  // light's local axes. world→local = transpose(rotation), so the local
  // dir is (rotation[col 0] · world, rotation[col 1] · world, ...)
  // expressed as: local.x = R[0]·d + R[3]·d.y + R[6]·d.z (column 0).
  float3 local;
  local.x = light_rotation[0] * dir_world.x + light_rotation[3] * dir_world.y +
            light_rotation[6] * dir_world.z;
  local.y = light_rotation[1] * dir_world.x + light_rotation[4] * dir_world.y +
            light_rotation[7] * dir_world.z;
  local.z = light_rotation[2] * dir_world.x + light_rotation[5] * dir_world.y +
            light_rotation[8] * dir_world.z;
  float len = sqrtf(dot3(local, local));
  if (len < 1e-12f)
    return 1.0f;
  // theta from -Z (IES "down" convention). cos(theta) = dot(local, -Z) =
  // -local.z / len.
  float cos_t = fminf(1.0f, fmaxf(-1.0f, -local.z / len));
  float theta_deg = acosf(cos_t) * (180.0f / M_PIf);
  // phi: angle around the IES axis. atan2(local.y, local.x) in radians,
  // then to degrees in [0, 360).
  float phi_rad = atan2f(local.y, local.x);
  if (phi_rad < 0.0f)
    phi_rad += 2.0f * M_PIf;
  float phi_deg = phi_rad * (180.0f / M_PIf);

  // Bilinear lookup in the table.
  const float *thetas = ies_data;
  const float *phis = ies_data + n_v;
  const float *cands = ies_data + n_v + n_h;

  // Vertical-angle bracket. Clamp to table range; IES profiles often
  // have a sparse range like [0, 90] (downlights) so anything below
  // theta[0] takes theta[0] and anything above theta[n_v-1] takes the
  // last entry (typically 0 candela for properly authored downlights).
  unsigned int v_lo = 0, v_hi = 0;
  float tv = 0.0f;
  if (theta_deg <= thetas[0]) {
    v_lo = 0; v_hi = 0;
  } else if (theta_deg >= thetas[n_v - 1]) {
    v_lo = n_v - 1; v_hi = n_v - 1;
  } else {
    for (unsigned int i = 0; i + 1 < n_v; i++) {
      if (thetas[i] <= theta_deg && theta_deg < thetas[i + 1]) {
        v_lo = i;
        v_hi = i + 1;
        float span = thetas[v_hi] - thetas[v_lo];
        tv = (span > 0.0f) ? (theta_deg - thetas[v_lo]) / span : 0.0f;
        break;
      }
    }
  }

  // Horizontal-angle bracket. n_h == 1 means radially symmetric — phi
  // is ignored. Otherwise fold phi according to the table's coverage:
  // phi_max ≤ 90 → quadrant symmetric, ≤ 180 → bilateral, else full.
  unsigned int h_lo = 0, h_hi = 0;
  float th = 0.0f;
  if (n_h > 1) {
    float phi_max = phis[n_h - 1];
    float p = phi_deg;
    if (phi_max <= 90.0f) {
      p = fmodf(p, 180.0f);
      if (p > 90.0f) p = 180.0f - p;
      if (p > 90.0f) p = 90.0f;
    } else if (phi_max <= 180.0f) {
      if (p > 180.0f) p = 360.0f - p;
    }
    if (p <= phis[0]) {
      h_lo = 0; h_hi = 0;
    } else if (p >= phis[n_h - 1]) {
      h_lo = n_h - 1; h_hi = n_h - 1;
    } else {
      for (unsigned int i = 0; i + 1 < n_h; i++) {
        if (phis[i] <= p && p < phis[i + 1]) {
          h_lo = i;
          h_hi = i + 1;
          float span = phis[h_hi] - phis[h_lo];
          th = (span > 0.0f) ? (p - phis[h_lo]) / span : 0.0f;
          break;
        }
      }
    }
  }

  float a = cands[h_lo * n_v + v_lo];
  float b = cands[h_lo * n_v + v_hi];
  float c = cands[h_hi * n_v + v_lo];
  float d = cands[h_hi * n_v + v_hi];
  float ab = a * (1.0f - tv) + b * tv;
  float cd = c * (1.0f - tv) + d * tv;
  return ab * (1.0f - th) + cd * th;
}

// ---------- Direct lighting (NEE) ----------
//
// `vstack` is the shadow ray's starting volume stack. For surface NEE this is
// the same stack `trace_path` is currently maintaining; for in-scatter volume
// NEE the caller does the same thing (the scatter point is *inside* the
// current volume, not on a boundary, so the stack starts unchanged).
//
// `shadow_transmittance` returns RGB so each light contribution is attenuated
// by the per-channel volume transmittance. On scenes without volumes it
// short-circuits to the legacy binary `shadow_visible` path.
static __device__ float3 direct_light(const MaterialEval &e, float3 P,
                                      float3 wo, RNG &rng,
                                      const VolumeStack &vstack) {
  float3 L = make_float3(0, 0, 0);

  // Point lights: sample centre (simple)
  for (int i = 0; i < params.num_point_lights; i++) {
    PointLight &pl = params.point_lights[i];
    float3 ld = make_f3(pl.position) - P;
    float d = sqrtf(dot3(ld, ld));
    if (d < 1e-4f)
      continue;
    float3 wi = ld / d;
    float3 vis = shadow_transmittance(P, wi, d, vstack);
    if (luminance(vis) <= 0.0f)
      continue;
    BsdfEval b = eval_bsdf(e, wo, wi);
    // IES: directional intensity multiplier in [0, 1] sampled in the
    // light's local frame. `wi` points from surface to light, so the
    // direction "from light toward surface" is `-wi`. ies_lookup
    // returns 1.0 when no IES is attached.
    float ies = ies_lookup(pl.ies_data, pl.ies_n_v, pl.ies_n_h,
                           pl.light_rotation, -wi);
    L = L + b.f * vis * make_f3(pl.emission) * ies / fmaxf(d * d, 1e-6f);
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
    float3 vis = shadow_transmittance(P, wi, 1e20f, vstack);
    if (luminance(vis) <= 0.0f)
      continue;
    BsdfEval b = eval_bsdf(e, wo, wi);
    L = L + b.f * vis * make_f3(sl.emission);
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
    float3 vis = shadow_transmittance(P, wi, d, vstack);
    if (luminance(vis) <= 0.0f)
      continue;
    BsdfEval b = eval_bsdf(e, wo, wi);
    float ies = ies_lookup(sp.ies_data, sp.ies_n_v, sp.ies_n_h,
                           sp.light_rotation, -wi);
    L = L + b.f * vis * make_f3(sp.emission) * falloff * ies /
                fmaxf(d * d, 1e-6f);
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
        // Two-sided rects emit from both faces — take |cos_light| so
        // surface points on either side of the panel see it.
        if (ar.two_sided != 0u)
          cos_light = fabsf(cos_light);
        if (cos_light > 0.0f) {
          float3 vis = shadow_transmittance(P, wi, d, vstack);
          if (luminance(vis) > 0.0f) {
            BsdfEval b = eval_bsdf(e, wo, wi);
            float area = ar.size_u * ar.size_v;
            float pdf_area = 1.0f / fmaxf(area, 1e-8f);
            float pdf_solid = pdf_area * d2 / cos_light;
            float pdf = pdf_solid * pmf_light;
            if (pdf > 0.0f) {
              float ies = ies_lookup(ar.ies_data, ar.ies_n_v, ar.ies_n_h,
                                     ar.light_rotation, -wi);
              L = L + b.f * vis * make_f3(ar.emission) * ies / pdf;
            }
          }
        }
      }
    }
  }

  // Envmap: one importance sample, MIS-weighted against BSDF sampling.
  // Activated for both single-layer (`world_type==1`) and mixed
  // (`world_type==2`) envmaps. Without this gate covering both, mixed
  // worlds would skip envmap NEE entirely — BSDF-sampled paths into the
  // sky would still pick up envmap radiance but with no NEE strategy
  // competing on MIS weights, doubling the effective contribution
  // (previously masquerading as 2× brightness on diffuse surfaces in
  // pabellon's pool basin).
  if (params.world_type != 0 && params.envmap_integral > 0.0f) {
    EnvSample es = sample_envmap(rng);
    if (es.pdf > 0.0f) {
      float3 vis = shadow_transmittance(P, es.dir, 1e20f, vstack);
      if (luminance(vis) > 0.0f) {
        BsdfEval b = eval_bsdf(e, wo, es.dir);
        if (b.pdf > 0.0f) {
          float w = power_heuristic(es.pdf, b.pdf);
          L = L + b.f * vis * es.L * (w / es.pdf);
        }
      }
    }
  }

  return L;
}

// In-scatter NEE for volume scatter events. Same lights as `direct_light`
// but the BSDF role is played by the HG phase function and there's no
// `MaterialEval` — the scatter point is in free space inside `vol`.
//
// `wi_world` is the incoming ray direction (so wo_world = -wi_world is the
// "outgoing" direction back toward the scatter event's predecessor); the
// phase function depends on the angle between wi_world and the chosen
// shadow direction. PDFs aren't MIS-weighted against phase sampling here —
// volume MIS only matters at scatter events that are followed by another
// MIS-able interaction, and one-sample importance sampling of the lights
// is already a substantial improvement over BSDF-only.
static __device__ float3 direct_light_volume(float3 P, float3 wi_world,
                                              const Volume *vol,
                                              const VolumeStack &vstack,
                                              RNG &rng) {
  float3 L = make_float3(0, 0, 0);
  float g = vol->anisotropy;

  auto add = [&](float3 wi, float3 emission, float3 vis) {
    float cos_t = dot3(wi_world, wi);
    float ph = phase_hg_eval(g, cos_t);
    L = L + vis * emission * ph;
  };

  for (int i = 0; i < params.num_point_lights; i++) {
    PointLight &pl = params.point_lights[i];
    float3 ld = make_f3(pl.position) - P;
    float d = sqrtf(dot3(ld, ld));
    if (d < 1e-4f)
      continue;
    float3 wi = ld / d;
    float3 vis = shadow_transmittance(P, wi, d, vstack);
    if (luminance(vis) <= 0.0f)
      continue;
    add(wi, make_f3(pl.emission) / fmaxf(d * d, 1e-6f), vis);
  }
  for (int i = 0; i < params.num_sun_lights; i++) {
    SunLight &sl = params.sun_lights[i];
    float3 dir = make_f3(sl.direction);
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
    float3 vis = shadow_transmittance(P, wi, 1e20f, vstack);
    if (luminance(vis) <= 0.0f)
      continue;
    add(wi, make_f3(sl.emission), vis);
  }
  for (int i = 0; i < params.num_spot_lights; i++) {
    SpotLight &sp = params.spot_lights[i];
    float3 ld = make_f3(sp.position) - P;
    float d = sqrtf(dot3(ld, ld));
    if (d < 1e-4f)
      continue;
    float3 wi = ld / d;
    float cos_a = dot3(make_f3(sp.direction), -wi);
    if (cos_a <= sp.cos_outer)
      continue;
    float falloff = 1.0f;
    if (cos_a < sp.cos_inner) {
      float t =
          (cos_a - sp.cos_outer) / fmaxf(sp.cos_inner - sp.cos_outer, 1e-4f);
      falloff = t * t * (3.0f - 2.0f * t);
    }
    float3 vis = shadow_transmittance(P, wi, d, vstack);
    if (luminance(vis) <= 0.0f)
      continue;
    add(wi, make_f3(sp.emission) * falloff / fmaxf(d * d, 1e-6f), vis);
  }
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
        if (ar.two_sided != 0u)
          cos_light = fabsf(cos_light);
        if (cos_light > 0.0f) {
          float3 vis = shadow_transmittance(P, wi, d, vstack);
          if (luminance(vis) > 0.0f) {
            float area = ar.size_u * ar.size_v;
            float pdf_area = 1.0f / fmaxf(area, 1e-8f);
            float pdf_solid = pdf_area * d2 / cos_light;
            float pdf = pdf_solid * pmf_light;
            if (pdf > 0.0f) {
              float ies = ies_lookup(ar.ies_data, ar.ies_n_v, ar.ies_n_h,
                                     ar.light_rotation, -wi);
              add(wi, make_f3(ar.emission) * ies / pdf, vis);
            }
          }
        }
      }
    }
  }
  if (params.world_type != 0 && params.envmap_integral > 0.0f) {
    EnvSample es = sample_envmap(rng);
    if (es.pdf > 0.0f) {
      float3 vis = shadow_transmittance(P, es.dir, 1e20f, vstack);
      if (luminance(vis) > 0.0f) {
        // MIS-weight against phase sampling. After a phase-sampled scatter
        // a missed ray hitting the envmap will be MIS-weighted in the
        // miss-path block (uses prev_bsdf_pdf = phase pdf). To make the
        // estimator unbiased we apply the matching light-side weight here.
        float cos_t = dot3(wi_world, es.dir);
        float ph_pdf = phase_hg_eval(g, cos_t);
        float w = power_heuristic(es.pdf, ph_pdf);
        // `add` multiplies by phase_eval, which equals ph_pdf, so the
        // contribution is throughput * phase * vis * L * w / pdf_light.
        add(es.dir, es.L * (w / es.pdf), vis);
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

  // Authored per-vertex tangent (currently only emitted for hair ribbons).
  // Read in object space so the strand axis sits in the same frame as the
  // ribbon vertices, then transform alongside the normal. Zero means "no
  // authored tangent" — eval_material falls back to build_frame() in that
  // case, preserving the shader's existing isotropic behaviour.
  float3 T = make_float3(0.0f, 0.0f, 0.0f);
  if (hg->tangents != nullptr) {
    float3 t0 = make_f3(&hg->tangents[i0 * 3]);
    float3 t1 = make_f3(&hg->tangents[i1 * 3]);
    float3 t2 = make_f3(&hg->tangents[i2 * 3]);
    float3 T_local = t0 * b0 + t1 * bary.x + t2 * bary.y;
    if (dot3(T_local, T_local) > 1e-12f) {
      float3 Tw = optixTransformVectorFromObjectToWorldSpace(T_local);
      float Tlen2 = dot3(Tw, Tw);
      if (Tlen2 > 1e-12f) {
        T = Tw * (1.0f / sqrtf(Tlen2));
      }
    }
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
  // Hash the OptiX instance id into a uniform [0,1) for ObjectInfo.Random.
  // PCG-style multiplicative hash: cheap, decorrelated, deterministic per
  // instance — what Cycles uses for the same node.
  unsigned int iid = optixGetInstanceId();
  unsigned int h = iid * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  h = (h >> 22u) ^ h;
  v->object_random = (float)h * (1.0f / 4294967296.0f);
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
  // Translucent thin sheets (Add(Diffuse, Translucent), sss_weight>0)
  // are NOT auto-passed: Cycles only lets ~30-50% through them, and
  // letting them be fully transparent makes the foreground pool floor
  // ~50% brighter than Cycles' reference. Cycles-style soft shadows
  // through translucent BSDFs would need a payload-accumulating
  // shadow-ray loop; for now respecting the alpha-cutout silhouette
  // (which is what's checked further below) gets us closer to
  // reference than full pass-through.
  bool is_shadow_ray =
      (optixGetRayFlags() & OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT) != 0;
  if (is_shadow_ray && m->transmission > 0.5f) {
    optixIgnoreIntersection();
  }
  // Pure volume-container surfaces are invisible to *binary* shadow rays
  // (the legacy `shadow_visible` path used by surface NEE on volumeless
  // scenes — TERMINATE_ON_FIRST_HIT is set there). Without this check the
  // smoke's mesh would shadow lights behind it as if it were opaque.
  //
  // We do NOT skip volume_only surfaces on radiance rays — `trace_path`
  // depends on closest-hit firing at the boundary so it can update the
  // volume stack. `shadow_transmittance` similarly uses the radiance ray
  // type (no terminate-on-first-hit) so CH runs there too.
  if (is_shadow_ray && m->volume != nullptr && m->volume_only != 0) {
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
  // Blender's image.pixels (and what the exporter writes to the bin) is
  // bottom-up: buffer row 0 is the bottom of the image. Blender UV v=0 is
  // also the bottom, so no flip on the y axis.
  float fy = v * (float)h - 0.5f;
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
    // Hidden emitters don't occlude camera/specular rays — they only
    // contribute via NEE. Skipping them here also prevents a hidden rect
    // in front of a visible one from shadowing the latter.
    if (ar.camera_visible == 0u)
      continue;
    float3 normal = make_f3(ar.normal);
    float3 corner = make_f3(ar.corner);
    float3 u_axis = make_f3(ar.u_axis);
    float3 v_axis = make_f3(ar.v_axis);
    float denom = dot3(normal, dir);
    // Hit the emissive face when the ray is heading against the normal.
    // Two-sided rects (emissive meshes) emit from both faces, so the
    // sign of `denom` doesn't matter beyond being non-zero.
    if (ar.two_sided == 0u) {
      if (denom >= -1e-6f)
        continue;
    } else {
      if (fabsf(denom) < 1e-6f)
        continue;
    }
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

static __device__ float3 trace_path(float3 origin, float3 dir, RNG &rng,
                                    float first_ray_tmin) {
  float3 throughput = make_float3(1, 1, 1);
  float3 L = make_float3(0, 0, 0);
  bool last_specular = true;
  float prev_bsdf_pdf = 0.0f;
  // Cycles caps diffuse / glossy / transmission bounces independently of
  // the total path-length cap. Track per-lobe counters; once one runs out
  // the path terminates (diffuse-saturated scenes — lone_monk's brick
  // courtyard with diffuse_bounces=2 — used to receive 4× too much
  // brick-on-brick indirect light because the kernel only honoured the
  // total `max_depth=8`).
  unsigned int diffuse_bounces = 0;
  unsigned int glossy_bounces = 0;
  unsigned int transmission_bounces = 0;
  // Filter-glossy state machine (mirrors Cycles' Light Paths > Filter
  // Glossy). Track the smallest BSDF pdf the path has sampled at a
  // glossy / transmission bounce. The next surface's BSDF α is clamped
  // up by an amount that keeps its lobe at least as wide as
  // 1/(filter_glossy · path_min_pdf) — concretely
  // `min_alpha = sqrt(filter_glossy / path_min_pdf)`. Camera rays start
  // with min_pdf=∞ → min_alpha=0 (no inflation), so primary visibility
  // sees authored materials unmodified; only paths that already passed
  // through a sharp lobe get blurred. Cycles' filter_glossy=0 short-
  // circuits the inflation entirely.
  float path_min_ray_pdf = 1e30f;
  float min_alpha = 0.0f;
  VolumeStack vstack;
  vstack.depth = 0;

  int camera_skip = 0;
  for (unsigned int bounce = 0; bounce < params.max_depth; bounce++) {
    PathVertex v;
    v.hit = 0;
    unsigned int hi, lo;
    pack_ptr(&v, hi, lo);
    // Bounce 0 is the camera ray and must honour Cycles' Camera "Clip
    // Start": ignore intersections closer than `first_ray_tmin` from the
    // eye. We do this via OptiX's t_min instead of pushing `origin`
    // forward, because a forward push can land the new origin inside
    // geometry (e.g. flat_archiviz puts the camera ~0.6m from the east
    // wall; pushing 2m forward placed the origin INSIDE that wall and
    // primary rays lost the entire left half of the frame). Subsequent
    // bounces use a small epsilon so reflections / refractions still
    // self-clear.
    //
    // Once we're inside the bounce-0 loop, also advance through back-face
    // hits. Cycles' clip_start is meant to hide a thick wall the camera
    // sits next to (flat_archiviz: 0.35m wall front=1.7m, back=2.05m); a
    // strict t_min skips the front face but we still hit the back face
    // just past it and shade it as a dark interior. The back-face advance
    // mimics Cycles' "skip the whole hidden object" intent.
    float ray_tmin = (bounce == 0 && camera_skip == 0) ? first_ray_tmin : 1e-4f;
    optixTrace(params.traversable, origin, dir, ray_tmin, 1e20f, 0.0f,
               OptixVisibilityMask(0x01), OPTIX_RAY_FLAG_NONE,
               0, // SBT offset (radiance ray type)
               2, // SBT stride (2 ray types: radiance + shadow)
               0, // miss index (radiance miss)
               hi, lo);
    if (bounce == 0 && v.hit != 0 && dot3(v.Ng, dir) > 0.0f) {
      // Only skip back-faces that are near `clip_start` — the workaround's
      // intent is to hide a wall the camera sits inside (flat_archiviz:
      // 0.35m wall, front at 1.7m=clip_start, back at 2.05m). Distant
      // back-faces (e.g. ies_light's room walls authored with normals
      // facing outward, hit at ~5.9m through clear air) are NOT what we
      // want to skip — doing so makes the camera ray miss the entire
      // scene and renders pitch black. Use a 1m thickness budget past
      // `clip_start`: enough for a thick interior wall, tight enough to
      // not eat any genuine back-facing geometry past the camera region.
      float t_hit = sqrtf(dot3(v.P - origin, v.P - origin));
      bool near_clip = t_hit < first_ray_tmin + 1.0f;
      if (near_clip && camera_skip < 4) {
        // Back-face hit on the camera ray: step past it and retry without
        // counting this as a bounce. Cap the retry depth so a pathological
        // all-back-faces direction can't loop forever.
        float p_scale =
            fmaxf(fmaxf(fabsf(v.P.x), fabsf(v.P.y)), fabsf(v.P.z));
        float eps = fmaxf(1e-3f, p_scale * 1e-5f);
        origin = v.P + dir * eps;
        camera_skip++;
        bounce--;  // wraps to UINT_MAX, the for-loop's bounce++ brings it back to 0
        continue;
      }
      // Cap exhausted (or back-face is far from clip_start so we treat it
      // as a regular interior hit) — flag it so the host can warn after the
      // launch, then fall through and shade this back-face. One increment
      // per sample-per-pixel where the cap is the reason we fall through;
      // distant back-faces are silent because the eval_material path below
      // flips Ns and shades them like a normal interior surface.
      if (near_clip && params.primary_back_face_skip_exhausted != nullptr) {
        atomicAdd(params.primary_back_face_skip_exhausted, 1u);
      }
    }

    // Geometry hit distance (∞ on miss). Needed so a rect light only counts
    // when it occludes geometry rather than being behind a wall.
    float t_geom = 1e20f;
    if (v.hit != 0) {
      float3 delta = v.P - origin;
      t_geom = sqrtf(dot3(delta, delta));
    }
    float t_rect;
    int rect_idx = intersect_rect_lights(origin, dir, ray_tmin, t_geom, t_rect);
    float t_segment = (rect_idx >= 0) ? t_rect : t_geom;

    // Volume distance sampling: if we're currently inside a volume, decide
    // whether the ray scatters before reaching the next surface/light. Path
    // emission is integrated in closed form over the entire segment so
    // emissive media don't add MC noise.
    const Volume *cur_vol = volume_stack_top(vstack);
    if (cur_vol != nullptr) {
      float sigma_t_avg = vol_sigma_t_avg(cur_vol);
      if (sigma_t_avg > 1e-12f) {
        float u = rng.next();
        float t_scatter =
            -logf(fmaxf(1.0f - u, 1e-30f)) / sigma_t_avg;
        if (t_scatter < t_segment) {
          // --- Volume scatter event ---
          // Path emission [0, t_scatter] (deterministic, noise-free).
          L = L + throughput * vol_path_emission(cur_vol, t_scatter);
          float3 P_scatter = origin + dir * t_scatter;
          float3 tr = vol_beam_transmittance(cur_vol, t_scatter);
          float pdf_t = sigma_t_avg * expf(-sigma_t_avg * t_scatter);
          float3 sigma_s = make_float3(cur_vol->sigma_s[0],
                                       cur_vol->sigma_s[1],
                                       cur_vol->sigma_s[2]);
          throughput = throughput * tr * sigma_s / fmaxf(pdf_t, 1e-30f);

          // In-scatter NEE: phase-weighted contribution from each light.
          float3 nee = throughput * direct_light_volume(P_scatter, dir,
                                                         cur_vol, vstack, rng);
          if (bounce > 0)
            nee = clamp_indirect(nee, params.clamp_indirect);
          L = L + nee;

          // Sample new direction via HG phase function. Frame is built
          // around the previous direction so wi_local.z corresponds to
          // forward scatter.
          float u1 = rng.next();
          float u2 = rng.next();
          float pdf_phase;
          float3 wi_local =
              phase_hg_sample(cur_vol->anisotropy, u1, u2, pdf_phase);
          float3 T_frame, B_frame;
          build_frame(dir, T_frame, B_frame);
          float3 wi = T_frame * wi_local.x + B_frame * wi_local.y +
                      dir * wi_local.z;
          // Phase sampling is exact (sampled from p / pdf == 1), so no
          // additional throughput multiplier is needed.
          last_specular = false;
          prev_bsdf_pdf = pdf_phase;

          if (bounce >= 3) {
            float q = fminf(fmaxf(luminance(throughput), 0.05f), 0.95f);
            if (rng.next() > q)
              break;
            throughput = throughput / q;
          }
          origin = P_scatter;
          dir = wi;
          continue;
        }
        // No scatter — path emission across the full segment, then apply
        // beam transmittance to the endpoint and divide by survival prob so
        // the no-scatter branch stays an unbiased estimator.
        L = L + throughput * vol_path_emission(cur_vol, t_segment);
        float3 tr = vol_beam_transmittance(cur_vol, t_segment);
        float pdf_surv = expf(-sigma_t_avg * t_segment);
        throughput = throughput * tr / fmaxf(pdf_surv, 1e-30f);
      } else {
        // Pure emitter (σ_t = 0): only path emission, no transmittance work.
        L = L + throughput * vol_path_emission(cur_vol, t_segment);
      }
    }

    if (rect_idx >= 0) {
      // Hit a camera-visible area light in front of any geometry. Hidden
      // rects are pre-filtered by intersect_rect_lights and never reach
      // here, so they don't occlude primary/specular rays. Add emission
      // on primary rays and after specular bounces; NEE covers diffuse.
      if (bounce == 0 || last_specular) {
        AreaRectLight &ar = params.rect_lights[rect_idx];
        // IES applies to direct camera-ray hits too — Cycles' IES Texture
        // multiplies emission for every visibility class, not just NEE.
        // `dir` is camera→light here; the IES table is sampled in the
        // light's local frame.
        float ies = ies_lookup(ar.ies_data, ar.ies_n_v, ar.ies_n_h,
                               ar.light_rotation, dir);
        float3 em = throughput * make_f3(ar.emission) * ies;
        if (bounce > 0)
          em = clamp_indirect(em, params.clamp_indirect);
        L = L + em;
      }
      break;
    }

    if (v.hit == 0) {
      // is_camera_ray for the world-bg split: bounce==0 is the primary
      // ray, and a chain of specular bounces (mirror / glass) preserves
      // the camera-ray semantic in Cycles' Light Path classification.
      bool is_camera_ray = (bounce == 0) || last_specular;
      float3 bg = world_background(dir, is_camera_ray);
      float w = 1.0f;
      if (bounce > 0 && !last_specular && params.world_type != 0) {
        float p_env = envmap_pdf(dir);
        w = power_heuristic(prev_bsdf_pdf, p_env);
      }
      float3 bg_contrib = throughput * bg * w;
      if (bounce > 0)
        bg_contrib = clamp_indirect(bg_contrib, params.clamp_indirect);
      L = L + bg_contrib;
      break;
    }

    // Volume-only boundary surface: invisible to shading. Update the volume
    // stack and pass straight through. Costs a bounce only if the boundary
    // surface ends up nested inside another bounded volume; the common
    // single-volume case still resolves in the next iteration.
    if (v.mat != nullptr && v.mat->volume != nullptr &&
        v.mat->volume_only != 0) {
      bool entering = dot3(v.Ng, dir) < 0.0f;
      if (entering)
        volume_stack_push(vstack, v.mat->volume);
      else
        volume_stack_pop(vstack, v.mat->volume);
      float p_scale = fmaxf(fmaxf(fabsf(v.P.x), fabsf(v.P.y)), fabsf(v.P.z));
      float eps = fmaxf(1e-4f, p_scale * 1e-5f);
      origin = v.P + dir * eps;
      // Don't bump bounce count — the boundary is invisible, treating it as
      // a bounce would shorten paths gratuitously inside dense volumes.
      // The for-loop still advances `bounce`, but in practice volume
      // boundaries are rare enough that the few-extra-iterations cost is
      // negligible and the semantics stay simple.
      continue;
    }

    MaterialEval e = eval_material(v, min_alpha);
    // Flip normal if ray hit backside (for dielectric transmission). Mirror
    // the bitangent instead of rebuilding the frame so the authored tangent
    // rotation (anisotropy axis) survives on back-faces; negating one axis
    // keeps (T, B, Ns) right-handed. Also invert e.ior: sample_bsdf and
    // eval_bsdf read it as η_t/η_i for the current incident side, so on a
    // back-face hit (ray exiting the medium) we need 1/ior to get correct
    // Fresnel and TIR.
    if (dot3(v.Ns, -dir) < 0.0f) {
      e.Ns = -e.Ns;
      e.B = -e.B;
      e.ior = 1.0f / e.ior;
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
    float3 nee = throughput * direct_light(e, v.P, wo, rng, vstack);
    // bounce==0 → camera-ray NEE, clamp via Cycles' sample_clamp_direct.
    // bounce>0 → indirect-bounce NEE, clamp via sample_clamp_indirect.
    // Cycles applies clamping at exactly the same path-vertex layering.
    if (bounce == 0)
      nee = clamp_indirect(nee, params.clamp_direct);
    else
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

    // Cycles-style per-lobe bounce caps. Cycles' `diffuse_bounces=N`
    // means up to N diffuse reflections may appear in the path, which
    // makes NEE fire at the surface that the Nth bounce LANDS on.
    // Translating that to our sample-then-trace loop: we allow the
    // counter to reach N (NEE at the next iteration is the Nth-bounce
    // indirect contribution), and only terminate when the *next* bounce
    // of the same lobe would exceed N — i.e. on count > limit. Using
    // `>=` here (the obvious-looking choice) under-samples by one bounce.
    if (bs.lobe == LOBE_DIFFUSE) {
      diffuse_bounces++;
      if (diffuse_bounces > params.max_diffuse_bounces)
        break;
    } else if (bs.lobe == LOBE_GLOSSY) {
      glossy_bounces++;
      if (glossy_bounces > params.max_glossy_bounces)
        break;
    } else if (bs.lobe == LOBE_TRANSMISSION) {
      transmission_bounces++;
      if (transmission_bounces > params.max_transmission_bounces)
        break;
    }
    // Update filter-glossy state. After every glossy / transmission
    // bounce, narrow the path's pdf floor to the just-sampled value so
    // the next surface's α is widened proportionally. Diffuse bounces
    // don't update min_pdf because their pdf is already broad
    // (cosine-weighted hemisphere integrates to 1).
    if (params.filter_glossy > 0.0f
        && (bs.lobe == LOBE_GLOSSY || bs.lobe == LOBE_TRANSMISSION)) {
      path_min_ray_pdf = fminf(path_min_ray_pdf, bs.pdf);
      // α_min = sqrt(filter_glossy / min_pdf). Cycles' filter_glossy of
      // 1.0 with a typical glossy bounce pdf of 100 → α_min = 0.10
      // (roughness ≈ 0.32). filter_glossy=5.0 (pabellon) gives α_min
      // ≈ 0.22 (roughness ≈ 0.47) — substantial widening of any
      // near-mirror lobe encountered downstream.
      float new_min_alpha =
          sqrtf(params.filter_glossy /
                fmaxf(path_min_ray_pdf, 1e-6f));
      // Cap at α=1 (roughness=1, fully diffuse-like) so high
      // filter_glossy values on a very sharp bounce don't produce
      // numerically silly alphas.
      new_min_alpha = fminf(new_min_alpha, 1.0f);
      min_alpha = fmaxf(min_alpha, new_min_alpha);
    }

    // Russian roulette after a few bounces
    if (bounce >= 3) {
      float q = fminf(fmaxf(luminance(throughput), 0.05f), 0.95f);
      if (rng.next() > q)
        break;
      throughput = throughput / q;
    }

    // If the surface has a volume container (Surface + Volume both
    // authored — e.g. coloured glass with subsurface fog), update the
    // volume stack on transmission. Reflection keeps us on the same side.
    if (v.mat != nullptr && v.mat->volume != nullptr) {
      bool was_outside_to_inside = dot3(v.Ng, dir) < 0.0f;
      bool now_outside_to_inside = dot3(v.Ng, bs.wi) < 0.0f;
      if (was_outside_to_inside != now_outside_to_inside) {
        if (now_outside_to_inside)
          volume_stack_push(vstack, v.mat->volume);
        else
          volume_stack_pop(vstack, v.mat->volume);
      }
    }

    // Offset along geometric normal to reduce self-intersection. Scale with
    // |P| so large-coordinate scenes (e.g. archviz authored in cm) don't
    // fall below float-precision noise while the 1e-4 floor still covers
    // unit-scale scenes.
    float3 offset_n = dot3(bs.wi, v.Ng) > 0.0f ? v.Ng : -v.Ng;
    float p_scale = fmaxf(fmaxf(fabsf(v.P.x), fabsf(v.P.y)), fabsf(v.P.z));
    float eps = fmaxf(1e-4f, p_scale * 1e-5f);
    origin = v.P + offset_n * eps;
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

  // Denoiser guide AOVs + Mist/Z support: one un-jittered primary ray into
  // the first surface. Keeps guides crisp (no sub-pixel averaging) and
  // cheap (one trace/pixel). The depth_aov drives the addon's Mist pass so
  // Cycles-authored compositors (mist haze, depth-based masks) work on top
  // of vibrt's output; misses store `cam_clip_end` so Mist saturates to 1.
  if (params.albedo_aov != nullptr || params.normal_aov != nullptr ||
      params.depth_aov != nullptr) {
    float px0 = (2.0f * ((float)idx.x + 0.5f) / (float)dim.x) - 1.0f
                + 2.0f * params.cam_shift_x;
    float py0 = (2.0f * ((float)idx.y + 0.5f) / (float)dim.y) - 1.0f
                + 2.0f * params.cam_shift_y;
    float3 dir0 = normalize3(U * px0 + V * py0 + W);
    PathVertex v;
    v.hit = 0;
    // Skip volume-only boundaries on the AOV ray — the denoiser guides
    // should describe the *visible* surface (the wall behind a smoke
    // cube), not the invisible volume container. Capped to a few
    // iterations so a pathological scene can't loop forever. The first
    // iteration honours `cam_clip_start` via t_min so the AOV agrees
    // with what the SPP loop below sees; later iterations use a small
    // epsilon to clear the previous boundary surface.
    float3 origin = eye;
    float ray_tmin = params.cam_clip_start;
    for (int i = 0; i < VOL_STACK_MAX + 2; i++) {
      v.hit = 0;
      unsigned int hi, lo;
      pack_ptr(&v, hi, lo);
      optixTrace(params.traversable, origin, dir0, ray_tmin, 1e20f, 0.0f,
                 OptixVisibilityMask(0x01), OPTIX_RAY_FLAG_NONE, 0, 2, 0, hi, lo);
      if (v.hit == 0)
        break;
      if (v.mat == nullptr || v.mat->volume == nullptr ||
          v.mat->volume_only == 0)
        break;
      float p_scale =
          fmaxf(fmaxf(fabsf(v.P.x), fabsf(v.P.y)), fabsf(v.P.z));
      float eps = fmaxf(1e-4f, p_scale * 1e-5f);
      origin = v.P + dir0 * eps;
      ray_tmin = 1e-4f;
    }
    float3 alb = make_float3(0, 0, 0);
    float3 nrm = make_float3(0, 0, 0);
    if (v.hit != 0) {
      MaterialEval e = eval_material(v);
      if (dot3(e.Ns, -dir0) < 0.0f)
        e.Ns = -e.Ns;
      alb = e.base_color;
      float3 Uu = normalize3(U);
      float3 Vu = normalize3(V);
      // W is already unit (camera forward).
      nrm = make_float3(dot3(e.Ns, Uu), dot3(e.Ns, Vu), dot3(e.Ns, W));
    }
    if (params.albedo_aov != nullptr) {
      float *a = &params.albedo_aov[pixel * 3];
      a[0] = alb.x;
      a[1] = alb.y;
      a[2] = alb.z;
    }
    if (params.normal_aov != nullptr) {
      float *n = &params.normal_aov[pixel * 3];
      n[0] = nrm.x;
      n[1] = nrm.y;
      n[2] = nrm.z;
    }
    if (params.depth_aov != nullptr) {
      float d;
      if (v.hit != 0) {
        float3 d3 = v.P - eye;
        d = sqrtf(dot3(d3, d3));
      } else {
        d = params.cam_clip_end;
      }
      params.depth_aov[pixel] = d;
    }
  }

  float3 accum = make_float3(0, 0, 0);
  for (unsigned int s = 0; s < params.samples_per_pixel; s++) {
    RNG rng(pixel, s, 0u);
    float jx = rng.next();
    float jy = rng.next();
    float px = (2.0f * ((float)idx.x + jx) / (float)dim.x) - 1.0f
               + 2.0f * params.cam_shift_x;
    float py = (2.0f * ((float)idx.y + jy) / (float)dim.y) - 1.0f
               + 2.0f * params.cam_shift_y;
    float3 dir = normalize3(U * px + V * py + W);
    // Honour Cycles' Camera "Clip Start" via t_min on the first ray
    // inside trace_path. We deliberately do NOT push `origin` forward:
    // when the camera sits close to a wall on one side (flat_archiviz
    // ~0.6m east wall, lens=50mm, clip_start=2m), shifting the origin
    // by `dir * clip_start` lands inside the wall for rays going that
    // way, and OptiX returns no hit → a hard black corner on that side
    // of the frame.
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
      // Cycles' focus_distance is measured from the camera (eye),
      // not from the near-clip plane. Place the focus point at
      // eye + dir*focal_distance so the bokeh is sharp where Cycles
      // says it should be.
      float3 focus_point = eye + dir * params.cam_focal_distance;
      origin = eye + Uunit * lx + Vunit * ly;
      dir = normalize3(focus_point - origin);
    }

    accum = accum + trace_path(origin, dir, rng, params.cam_clip_start);
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
