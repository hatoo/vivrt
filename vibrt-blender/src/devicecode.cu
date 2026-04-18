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

static __forceinline__ __device__ void
build_frame(float3 n, float3 &t, float3 &b) {
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
  float3 Nh = T1 * t1 + T2 * t2 +
              Vh * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2));
  float3 H =
      normalize3(make_float3(alpha * Nh.x, alpha * Nh.y, fmaxf(0.0f, Nh.z)));
  return H;
}

// PDF of GGX VNDF sample for reflection: D(H) * G1(V) * |V.H| / |V.N| / (4 |V.H|)
// = D(H) * G1(V) / (4 |V.N|)
static __forceinline__ __device__ float ggx_vndf_pdf(float NoV, float NoH,
                                                     float alpha) {
  return ggx_D(NoH, alpha) * smith_G1(NoV, alpha) / fmaxf(4.0f * NoV, 1e-12f);
}

// ---------- Path tracing payload ----------
struct PathVertex {
  float3 P;          // hit position
  float3 Ng;         // geometric normal (world)
  float3 Ns;         // shading normal (world)
  float3 T;          // tangent (world)
  float2 uv;         // mesh UVs (0,0 if absent)
  PrincipledGpu *mat;
  int hit;           // 1 if hit, 0 if miss
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

// ---------- Principled BSDF ----------
struct MaterialEval {
  float3 base_color;
  float metallic;
  float roughness;
  float alpha;   // roughness^2, clamped
  float ior;
  float transmission;
  float3 emission;
  float3 Ns;     // shading normal
  float3 T;
  float3 B;
};

static __device__ MaterialEval eval_material(const PathVertex &v) {
  MaterialEval e;
  PrincipledGpu *m = v.mat;
  e.base_color = make_f3(m->base_color);
  e.metallic = m->metallic;
  e.roughness = m->roughness;
  e.ior = m->ior;
  e.transmission = m->transmission;
  e.emission = make_f3(m->emission);
  if (m->base_color_tex != nullptr) {
    float3 t = sample_rgba(m->base_color_tex, m->base_color_tex_w,
                           m->base_color_tex_h, m->base_color_tex_channels,
                           v.uv);
    e.base_color = e.base_color * t;
  }
  if (m->roughness_tex != nullptr) {
    float3 t =
        sample_rgba(m->roughness_tex, m->roughness_tex_w, m->roughness_tex_h,
                    m->roughness_tex_channels, v.uv);
    e.roughness = e.roughness * t.x;
  }
  if (m->metallic_tex != nullptr) {
    float3 t =
        sample_rgba(m->metallic_tex, m->metallic_tex_w, m->metallic_tex_h,
                    m->metallic_tex_channels, v.uv);
    e.metallic = e.metallic * t.x;
  }
  e.metallic = fminf(fmaxf(e.metallic, 0.0f), 1.0f);
  e.roughness = fmaxf(e.roughness, 0.02f);
  e.alpha = fmaxf(e.roughness * e.roughness, 1e-4f);

  float3 Ns = v.Ns;
  if (m->normal_tex != nullptr) {
    float3 n = sample_rgba(m->normal_tex, m->normal_tex_w, m->normal_tex_h,
                           m->normal_tex_channels, v.uv);
    float3 nm = make_float3(n.x * 2.0f - 1.0f, n.y * 2.0f - 1.0f,
                            fmaxf(n.z * 2.0f - 1.0f, 0.01f));
    float3 T, B;
    build_frame(Ns, T, B);
    Ns = normalize3(T * nm.x + B * nm.y + Ns * nm.z);
  }
  e.Ns = Ns;
  build_frame(e.Ns, e.T, e.B);
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

  if (reflect) {
    // Diffuse (Lambert)
    if (w_diffuse > 0.0f) {
      r.f = r.f + e.base_color * (INV_PIf * w_diffuse * NoL);
      r.pdf += p_diff * NoL * INV_PIf;
    }
    // Specular (metallic + dielectric spec layer)
    float3 H = normalize3(wo + wi);
    float NoH = fmaxf(dot3(e.Ns, H), 0.0f);
    float VoH = fmaxf(dot3(wo, H), 0.0f);
    float D = ggx_D(NoH, e.alpha);
    float G1_v = smith_G1(NoV, e.alpha);
    float G1_l = smith_G1(NoL, e.alpha);
    float G = G1_v * G1_l;
    float3 F_metal = schlick_rgb(VoH, e.base_color);
    float F_dielec = schlick_scalar(VoH, F0_d);
    float3 f_metal = F_metal * (D * G / fmaxf(4.0f * NoV * NoL, 1e-8f));
    float3 f_dielec =
        make_float3(F_dielec, F_dielec, F_dielec) *
        (D * G / fmaxf(4.0f * NoV * NoL, 1e-8f));
    float3 f_spec =
        e.metallic * f_metal + (1.0f - e.metallic) * (1.0f - e.transmission) *
                                   f_dielec;
    r.f = r.f + f_spec * NoL;
    float pdf_spec = ggx_vndf_pdf(NoV, NoH, e.alpha);
    r.pdf += p_spec * pdf_spec;
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
    float G = smith_G1(fabsf(NoV), e.alpha) *
              smith_G1(fabsf(NoL), e.alpha);
    float denom =
        (eta * VoH + LoH) * (eta * VoH + LoH) * fabsf(NoV) * fabsf(NoL);
    float btdf = (fabsf(VoH) * fabsf(LoH) * (1.0f - F) * D * G) /
                 fmaxf(denom, 1e-8f);
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
    // GGX VNDF specular
    float3 Vlocal = make_float3(dot3(wo, e.T), dot3(wo, e.B), dot3(wo, e.Ns));
    float3 Hlocal = sample_ggx_vndf(Vlocal, e.alpha, u1, u2);
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
    float3 wi = normalize3(T * (st * cosf(phi)) + B * (st * sinf(phi)) +
                           dir * ct);
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
      float t = (cos_a - sp.cos_outer) /
                fmaxf(sp.cos_inner - sp.cos_outer, 1e-4f);
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
    float pmf_light = (params.rect_light_cdf[idx + 1] -
                       params.rect_light_cdf[idx]);
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

  // Envmap: one importance sample
  if (params.world_type == 1 && params.envmap_integral > 0.0f) {
    EnvSample es = sample_envmap(rng);
    if (es.pdf > 0.0f && shadow_visible(P, es.dir, 1e20f)) {
      BsdfEval b = eval_bsdf(e, wo, es.dir);
      if (b.pdf > 0.0f) {
        L = L + b.f * es.L / es.pdf;
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
  float3 Ng =
      normalize3(optixTransformNormalFromObjectToWorldSpace(Ng_local));

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

  v->P = P;
  v->Ng = Ng;
  v->Ns = Ns;
  v->T = T;
  v->uv = uv;
  v->mat = hg->mat;
  v->hit = 1;
}

extern "C" __global__ void __closesthit__shadow() {
  // Not used when TERMINATE_ON_FIRST_HIT is set — kept to satisfy SBT.
}

extern "C" __global__ void __miss__ms() {
  PathVertex *v = get_path_vertex();
  v->hit = 0;
}

extern "C" __global__ void __miss__shadow() {
  optixSetPayload_0(1u);
}

static __device__ float3 trace_path(float3 origin, float3 dir, RNG &rng) {
  float3 throughput = make_float3(1, 1, 1);
  float3 L = make_float3(0, 0, 0);
  bool last_specular = true;

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

    if (v.hit == 0) {
      L = L + throughput * world_background(dir);
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
      L = L + throughput * e.emission;
    }

    // NEE
    float3 wo = -dir;
    L = L + throughput * direct_light(e, v.P, wo, rng);

    // Sample BSDF for next bounce
    BsdfSample bs = sample_bsdf(e, wo, rng);
    if (bs.pdf <= 0.0f)
      break;
    float3 contrib = bs.f / bs.pdf;
    throughput = throughput * contrib;
    last_specular = bs.specular;

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
