#include <optix.h>
#include "devicecode.h"

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

extern "C" {
__constant__ LaunchParams params;
}

// ---- RNG (PCG hash) ----

static __forceinline__ __device__ unsigned int pcg_hash(unsigned int input)
{
    unsigned int state = input * 747796405u + 2891336453u;
    unsigned int word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

struct RNG {
    unsigned int state;
    __device__ RNG(unsigned int pixel, unsigned int sample, unsigned int depth) {
        state = pcg_hash(pixel * 17 + sample * 101 + depth * 1999);
    }
    __device__ float next() {
        state = pcg_hash(state);
        return state / 4294967296.0f;
    }
};

// ---- Math helpers ----

static __forceinline__ __device__ float3 make_f3(const float* p) {
    return make_float3(p[0], p[1], p[2]);
}

static __forceinline__ __device__ float dot3(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

static __forceinline__ __device__ float3 normalize3(float3 v) {
    float len = sqrtf(dot3(v, v));
    return make_float3(v.x/len, v.y/len, v.z/len);
}

static __forceinline__ __device__ float3 cross3(float3 a, float3 b) {
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

static __forceinline__ __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

static __forceinline__ __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

static __forceinline__ __device__ float3 operator*(float3 a, float s) {
    return make_float3(a.x*s, a.y*s, a.z*s);
}

static __forceinline__ __device__ float3 operator*(float s, float3 a) {
    return a * s;
}

static __forceinline__ __device__ float3 operator*(float3 a, float3 b) {
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}

static __forceinline__ __device__ float3 reflect3(float3 I, float3 N) {
    return I - 2.0f * dot3(I, N) * N;
}

// Cosine-weighted hemisphere sample
static __forceinline__ __device__ float3 cosine_sample_hemisphere(float u1, float u2, float3 N) {
    float phi = 2.0f * M_PIf * u1;
    float cos_theta = sqrtf(u2);
    float sin_theta = sqrtf(1.0f - u2);

    float3 tangent;
    if (fabsf(N.x) > 0.9f)
        tangent = normalize3(cross3(make_float3(0,1,0), N));
    else
        tangent = normalize3(cross3(make_float3(1,0,0), N));
    float3 bitangent = cross3(N, tangent);

    return normalize3(
        tangent * (cosf(phi) * sin_theta) +
        bitangent * (sinf(phi) * sin_theta) +
        N * cos_theta
    );
}

// GGX microfacet sampling (for glossy materials)
static __forceinline__ __device__ float3 ggx_sample(float u1, float u2, float alpha, float3 N) {
    // Sample GGX distribution of normals
    float phi = 2.0f * M_PIf * u1;
    float cos_theta = sqrtf((1.0f - u2) / (1.0f + (alpha * alpha - 1.0f) * u2));
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));

    // Build local frame
    float3 tangent;
    if (fabsf(N.x) > 0.9f)
        tangent = normalize3(cross3(make_float3(0,1,0), N));
    else
        tangent = normalize3(cross3(make_float3(1,0,0), N));
    float3 bitangent = cross3(N, tangent);

    // Half vector in world space
    return normalize3(
        tangent * (cosf(phi) * sin_theta) +
        bitangent * (sinf(phi) * sin_theta) +
        N * cos_theta
    );
}

// Schlick Fresnel approximation (for coated surfaces, uses F0 = 0.04 for dielectric coat)
static __forceinline__ __device__ float fresnel_schlick_f0(float cos_i, float f0) {
    float x = 1.0f - cos_i;
    return f0 + (1.0f - f0) * x * x * x * x * x;
}

// Schlick Fresnel approximation
static __forceinline__ __device__ float fresnel_schlick(float cos_i, float eta) {
    float r0 = (1.0f - eta) / (1.0f + eta);
    r0 = r0 * r0;
    float x = 1.0f - cos_i;
    return r0 + (1.0f - r0) * x * x * x * x * x;
}

static __forceinline__ __device__ bool refract3(float3 I, float3 N, float eta, float3& T) {
    float cos_i = dot3(I, N);
    float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
    if (sin2_t > 1.0f) return false;
    T = eta * I - (eta * cos_i + sqrtf(1.0f - sin2_t)) * N;
    return true;
}

// ---- Payload ----

static __forceinline__ __device__ void setPayload(float3 color) {
    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}

static __forceinline__ __device__ void setHitInfo(float3 pos, float3 normal, int mat_idx) {
    // payloads 3-8: hit position, normal, material index
    optixSetPayload_3(__float_as_uint(pos.x));
    optixSetPayload_4(__float_as_uint(pos.y));
    optixSetPayload_5(__float_as_uint(pos.z));
    optixSetPayload_6(__float_as_uint(normal.x));
    optixSetPayload_7(__float_as_uint(normal.y));
    optixSetPayload_8(__float_as_uint(normal.z));
    optixSetPayload_9(mat_idx);
}

// ---- Pack/unpack color ----

static __forceinline__ __device__ unsigned int packColor(float3 c) {
    auto clamp01 = [](float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); };
    // Apply simple gamma (sRGB approximation)
    float r = sqrtf(clamp01(c.x));
    float g = sqrtf(clamp01(c.y));
    float b = sqrtf(clamp01(c.z));
    unsigned int ir = (unsigned int)(r * 255.0f);
    unsigned int ig = (unsigned int)(g * 255.0f);
    unsigned int ib = (unsigned int)(b * 255.0f);
    return (255u << 24) | (ib << 16) | (ig << 8) | ir;
}

// ---- Programs ----

extern "C" __global__ void __raygen__rg()
{
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

        for (unsigned int depth = 0; depth < params.max_depth; depth++) {
            unsigned int p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13;
            p9 = 0xFFFFFFFF; // miss sentinel
            p10 = p11 = p12 = p13 = 0;

            optixTrace(
                params.traversable,
                origin, direction,
                0.001f, 1e16f, 0.0f,
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE,
                0, 1, 0,
                p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13
            );

            if (p9 == 0xFFFFFFFF) {
                // Miss - add environment light
                float3 bg = make_f3(params.ambient_light);
                radiance = radiance + throughput * bg;
                break;
            }

            float3 hit_pos = make_float3(__uint_as_float(p3), __uint_as_float(p4), __uint_as_float(p5));
            float3 hit_normal = make_float3(__uint_as_float(p6), __uint_as_float(p7), __uint_as_float(p8));
            float3 hit_albedo = make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
            float3 hit_emission = make_float3(__uint_as_float(p10), __uint_as_float(p11), __uint_as_float(p12));
            int mat_type = (int)p9;

            // Add emission only on direct camera ray (depth 0).
            // Indirect illumination from emissive surfaces is handled by NEE.
            if (depth == 0 && (hit_emission.x > 0 || hit_emission.y > 0 || hit_emission.z > 0)) {
                radiance = radiance + throughput * hit_emission;
            }

            if (mat_type == MAT_DIFFUSE) {
                // Direct lighting from distant lights
                for (int i = 0; i < params.num_distant_lights; i++) {
                    float3 light_dir = make_f3(params.distant_lights[i].direction);
                    float3 light_em = make_f3(params.distant_lights[i].emission);
                    float ndotl = dot3(hit_normal, light_dir);
                    if (ndotl > 0.0f) {
                        // Shadow ray
                        unsigned int shadow_p9 = 0xFFFFFFFF;
                        unsigned int sp10 = 0, sp11 = 0, sp12 = 0, sp13 = 0;
                        optixTrace(
                            params.traversable,
                            hit_pos, light_dir,
                            0.001f, 1e16f, 0.0f,
                            OptixVisibilityMask(255),
                            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                            0, 1, 0,
                            p0, p1, p2, p3, p4, p5, p6, p7, p8, shadow_p9, sp10, sp11, sp12, sp13
                        );
                        if (shadow_p9 == 0xFFFFFFFF) {
                            radiance = radiance + throughput * hit_albedo * light_em * ndotl * (1.0f / M_PIf);
                        }
                    }
                }

                // Direct lighting from sphere area lights (NEE)
                for (int i = 0; i < params.num_sphere_lights; i++) {
                    RNG light_rng(pixel_idx * 31 + i, s, depth + 100);
                    float3 light_center = make_f3(params.sphere_lights[i].center);
                    float light_radius = params.sphere_lights[i].radius;
                    float3 light_em = make_f3(params.sphere_lights[i].emission);

                    // Sample a point on the sphere
                    float3 to_light = light_center - hit_pos;
                    float dist_to_center = sqrtf(dot3(to_light, to_light));
                    float3 light_dir_norm = to_light * (1.0f / dist_to_center);

                    // Sample uniformly on sphere surface visible from hit point
                    float sin_theta_max2 = (light_radius * light_radius) / (dist_to_center * dist_to_center);
                    float cos_theta_max = sqrtf(fmaxf(0.0f, 1.0f - sin_theta_max2));

                    // Sample within the cone subtended by the sphere
                    float u1 = light_rng.next();
                    float u2 = light_rng.next();
                    float cos_theta = 1.0f - u1 + u1 * cos_theta_max;
                    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
                    float phi = 2.0f * M_PIf * u2;

                    // Build local frame around light direction
                    float3 tangent;
                    if (fabsf(light_dir_norm.x) > 0.9f)
                        tangent = normalize3(cross3(make_float3(0,1,0), light_dir_norm));
                    else
                        tangent = normalize3(cross3(make_float3(1,0,0), light_dir_norm));
                    float3 bitangent = cross3(light_dir_norm, tangent);

                    float3 sample_dir = normalize3(
                        tangent * (cosf(phi) * sin_theta) +
                        bitangent * (sinf(phi) * sin_theta) +
                        light_dir_norm * cos_theta
                    );

                    float ndotl = dot3(hit_normal, sample_dir);
                    if (ndotl > 0.0f) {
                        // Distance to sphere surface along sample direction
                        float t_max = dist_to_center + light_radius;

                        // Shadow ray
                        unsigned int shadow_p9 = 0xFFFFFFFF;
                        unsigned int sp10 = 0, sp11 = 0, sp12 = 0, sp13 = 0;
                        optixTrace(
                            params.traversable,
                            hit_pos, sample_dir,
                            0.001f, t_max, 0.0f,
                            OptixVisibilityMask(255),
                            OPTIX_RAY_FLAG_NONE,
                            0, 1, 0,
                            p0, p1, p2, p3, p4, p5, p6, p7, p8, shadow_p9, sp10, sp11, sp12, sp13
                        );

                        // Check if we hit the light (emission > 0)
                        float3 shadow_emission = make_float3(
                            __uint_as_float(sp10), __uint_as_float(sp11), __uint_as_float(sp12));
                        if (shadow_emission.x > 0 || shadow_emission.y > 0 || shadow_emission.z > 0) {
                            // Solid angle PDF of cone sampling
                            float pdf = 1.0f / (2.0f * M_PIf * (1.0f - cos_theta_max));
                            radiance = radiance + throughput * hit_albedo * shadow_emission * ndotl
                                * (1.0f / (M_PIf * pdf));
                        }
                    }
                }

                // Direct lighting from triangle area lights (NEE)
                for (int i = 0; i < params.num_triangle_lights; i++) {
                    RNG light_rng(pixel_idx * 37 + i, s, depth + 200);
                    float3 lv0 = make_f3(params.triangle_lights[i].v0);
                    float3 lv1 = make_f3(params.triangle_lights[i].v1);
                    float3 lv2 = make_f3(params.triangle_lights[i].v2);
                    float3 light_em = make_f3(params.triangle_lights[i].emission);
                    float3 light_normal = make_f3(params.triangle_lights[i].normal);
                    float light_area = params.triangle_lights[i].area;

                    // Sample random point on triangle
                    float u1 = light_rng.next();
                    float u2 = light_rng.next();
                    if (u1 + u2 > 1.0f) { u1 = 1.0f - u1; u2 = 1.0f - u2; }
                    float3 light_pos = lv0 * (1.0f - u1 - u2) + lv1 * u1 + lv2 * u2;

                    float3 to_light = light_pos - hit_pos;
                    float dist2 = dot3(to_light, to_light);
                    float dist = sqrtf(dist2);
                    float3 light_dir = to_light * (1.0f / dist);

                    float ndotl = dot3(hit_normal, light_dir);
                    float lndotl = -dot3(light_normal, light_dir); // light faces toward hit
                    if (ndotl > 0.0f && lndotl > 0.0f) {
                        unsigned int shadow_p9 = 0xFFFFFFFF;
                        unsigned int sp10=0,sp11=0,sp12=0,sp13=0;
                        optixTrace(params.traversable, hit_pos, light_dir,
                            0.001f, dist - 0.001f, 0.0f,
                            OptixVisibilityMask(255), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                            0, 1, 0,
                            p0,p1,p2,p3,p4,p5,p6,p7,p8,shadow_p9,sp10,sp11,sp12,sp13);
                        if (shadow_p9 == 0xFFFFFFFF) {
                            // PDF = 1/area, geometry factor = cos_light * cos_hit / dist^2
                            float weight = light_area * lndotl * ndotl / (dist2 * M_PIf);
                            radiance = radiance + throughput * hit_albedo * light_em * weight;
                        }
                    }
                }

                // Indirect: cosine-weighted bounce
                RNG bounce_rng(pixel_idx, s, depth + 1);
                direction = cosine_sample_hemisphere(bounce_rng.next(), bounce_rng.next(), hit_normal);
                origin = hit_pos;
                throughput = throughput * hit_albedo;
            }
            else if (mat_type == MAT_COATED_DIFFUSE) {
                float hit_roughness = __uint_as_float(p13);
                float alpha = fmaxf(hit_roughness * hit_roughness, 0.001f);
                RNG bounce_rng(pixel_idx, s, depth + 1);

                // Fresnel determines specular vs diffuse probability
                float cos_i = fmaxf(fabsf(dot3(direction * (-1.0f), hit_normal)), 0.001f);
                float F = fresnel_schlick_f0(cos_i, 0.04f); // dielectric coat F0 ≈ 0.04

                if (bounce_rng.next() < F) {
                    // Specular reflection: sample GGX
                    float3 H = ggx_sample(bounce_rng.next(), bounce_rng.next(), alpha, hit_normal);
                    direction = reflect3(direction, H);
                    if (dot3(direction, hit_normal) <= 0.0f) break; // below surface
                    origin = hit_pos;
                    // Specular doesn't tint by albedo (dielectric coat)
                } else {
                    // Diffuse: same as MAT_DIFFUSE with NEE
                    for (int i = 0; i < params.num_distant_lights; i++) {
                        float3 light_dir = make_f3(params.distant_lights[i].direction);
                        float3 light_em = make_f3(params.distant_lights[i].emission);
                        float ndotl = dot3(hit_normal, light_dir);
                        if (ndotl > 0.0f) {
                            unsigned int shadow_p9 = 0xFFFFFFFF;
                            unsigned int sp10 = 0, sp11 = 0, sp12 = 0, sp13 = 0;
                            optixTrace(params.traversable, hit_pos, light_dir,
                                0.001f, 1e16f, 0.0f, OptixVisibilityMask(255),
                                OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0, 1, 0,
                                p0, p1, p2, p3, p4, p5, p6, p7, p8, shadow_p9, sp10, sp11, sp12, sp13);
                            if (shadow_p9 == 0xFFFFFFFF)
                                radiance = radiance + throughput * hit_albedo * light_em * ndotl * (1.0f / M_PIf);
                        }
                    }
                    // NEE for sphere lights
                    for (int i = 0; i < params.num_sphere_lights; i++) {
                        RNG light_rng(pixel_idx * 31 + i, s, depth + 100);
                        float3 light_center = make_f3(params.sphere_lights[i].center);
                        float light_radius = params.sphere_lights[i].radius;
                        float3 light_em = make_f3(params.sphere_lights[i].emission);
                        float3 to_light = light_center - hit_pos;
                        float dist_to_center = sqrtf(dot3(to_light, to_light));
                        float3 light_dir_norm = to_light * (1.0f / dist_to_center);
                        float sin_theta_max2 = (light_radius * light_radius) / (dist_to_center * dist_to_center);
                        float cos_theta_max = sqrtf(fmaxf(0.0f, 1.0f - sin_theta_max2));
                        float u1l = light_rng.next(), u2l = light_rng.next();
                        float cos_theta_l = 1.0f - u1l + u1l * cos_theta_max;
                        float sin_theta_l = sqrtf(fmaxf(0.0f, 1.0f - cos_theta_l * cos_theta_l));
                        float phi_l = 2.0f * M_PIf * u2l;
                        float3 tgt;
                        if (fabsf(light_dir_norm.x) > 0.9f) tgt = normalize3(cross3(make_float3(0,1,0), light_dir_norm));
                        else tgt = normalize3(cross3(make_float3(1,0,0), light_dir_norm));
                        float3 btgt = cross3(light_dir_norm, tgt);
                        float3 sample_dir = normalize3(tgt*(cosf(phi_l)*sin_theta_l) + btgt*(sinf(phi_l)*sin_theta_l) + light_dir_norm*cos_theta_l);
                        float ndotl = dot3(hit_normal, sample_dir);
                        if (ndotl > 0.0f) {
                            unsigned int shadow_p9 = 0xFFFFFFFF;
                            unsigned int sp10=0,sp11=0,sp12=0,sp13=0;
                            optixTrace(params.traversable, hit_pos, sample_dir, 0.001f, dist_to_center+light_radius, 0.0f,
                                OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0,
                                p0,p1,p2,p3,p4,p5,p6,p7,p8,shadow_p9,sp10,sp11,sp12,sp13);
                            float3 shadow_em = make_float3(__uint_as_float(sp10),__uint_as_float(sp11),__uint_as_float(sp12));
                            if (shadow_em.x > 0 || shadow_em.y > 0 || shadow_em.z > 0) {
                                float pdf = 1.0f / (2.0f * M_PIf * (1.0f - cos_theta_max));
                                radiance = radiance + throughput * hit_albedo * shadow_em * ndotl * (1.0f / (M_PIf * pdf));
                            }
                        }
                    }
                    direction = cosine_sample_hemisphere(bounce_rng.next(), bounce_rng.next(), hit_normal);
                    origin = hit_pos;
                    throughput = throughput * hit_albedo;
                }
            }
            else if (mat_type == MAT_DIELECTRIC) {
                float eta_val = __uint_as_float(p0); // eta stored in p0 for dielectric
                RNG bounce_rng(pixel_idx, s, depth + 1);

                bool front_face = dot3(direction, hit_normal) < 0.0f;
                float3 outward_normal = front_face ? hit_normal : hit_normal * (-1.0f);
                float ratio = front_face ? (1.0f / eta_val) : eta_val;

                float cos_i = fminf(fabsf(dot3(direction, outward_normal)), 1.0f);
                float reflect_prob = fresnel_schlick(cos_i, ratio);

                float3 refracted;
                bool can_refract = refract3(direction * (-1.0f), outward_normal, ratio, refracted);

                if (!can_refract || bounce_rng.next() < reflect_prob) {
                    direction = reflect3(direction, outward_normal);
                } else {
                    direction = normalize3(refracted);
                }
                origin = hit_pos;
                // Glass doesn't absorb (throughput unchanged)
            }
        }

        accum = accum + radiance;
    }

    float3 color = accum * (1.0f / (float)spp);
    params.image[pixel_idx] = packColor(color);
}

extern "C" __global__ void __miss__ms()
{
    // Signal miss via payload 9
    optixSetPayload_9(0xFFFFFFFF);
}

extern "C" __global__ void __closesthit__ch()
{
    const HitGroupData* data = reinterpret_cast<const HitGroupData*>(optixGetSbtDataPointer());
    const float2 bary = optixGetTriangleBarycentrics();
    const int prim_idx = optixGetPrimitiveIndex();

    // Compute hit position
    const float t = optixGetRayTmax();
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    float3 hit_pos = ray_orig + ray_dir * t;

    // Compute face normal from triangle vertices
    float3 v0, v1, v2;
    if (data->indices) {
        int i0 = data->indices[prim_idx * 3 + 0];
        int i1 = data->indices[prim_idx * 3 + 1];
        int i2 = data->indices[prim_idx * 3 + 2];
        v0 = make_float3(data->vertices[i0*3], data->vertices[i0*3+1], data->vertices[i0*3+2]);
        v1 = make_float3(data->vertices[i1*3], data->vertices[i1*3+1], data->vertices[i1*3+2]);
        v2 = make_float3(data->vertices[i2*3], data->vertices[i2*3+1], data->vertices[i2*3+2]);
    } else {
        v0 = make_float3(data->vertices[prim_idx*9+0], data->vertices[prim_idx*9+1], data->vertices[prim_idx*9+2]);
        v1 = make_float3(data->vertices[prim_idx*9+3], data->vertices[prim_idx*9+4], data->vertices[prim_idx*9+5]);
        v2 = make_float3(data->vertices[prim_idx*9+6], data->vertices[prim_idx*9+7], data->vertices[prim_idx*9+8]);
    }

    float3 shading_normal;
    if (data->normals) {
        // Smooth shading: interpolate per-vertex normals
        int i0, i1, i2;
        if (data->indices) {
            i0 = data->indices[prim_idx * 3 + 0];
            i1 = data->indices[prim_idx * 3 + 1];
            i2 = data->indices[prim_idx * 3 + 2];
        } else {
            i0 = prim_idx * 3 + 0;
            i1 = prim_idx * 3 + 1;
            i2 = prim_idx * 3 + 2;
        }
        float3 n0 = make_float3(data->normals[i0*3], data->normals[i0*3+1], data->normals[i0*3+2]);
        float3 n1 = make_float3(data->normals[i1*3], data->normals[i1*3+1], data->normals[i1*3+2]);
        float3 n2 = make_float3(data->normals[i2*3], data->normals[i2*3+1], data->normals[i2*3+2]);
        float w = 1.0f - bary.x - bary.y;
        shading_normal = normalize3(n0 * w + n1 * bary.x + n2 * bary.y);
    } else {
        // Flat shading: face normal from triangle edges
        float3 edge1 = v1 - v0;
        float3 edge2 = v2 - v0;
        shading_normal = normalize3(cross3(edge1, edge2));
    }

    // Ensure normal faces the ray
    if (dot3(shading_normal, ray_dir) > 0.0f)
        shading_normal = shading_normal * (-1.0f);

    float3 albedo = make_f3(data->albedo);

    // Checkerboard procedural texture
    if (data->has_checkerboard) {
        float u_coord = 0.0f, v_coord = 0.0f;
        if (data->texcoords && data->indices) {
            int i0 = data->indices[prim_idx * 3 + 0];
            int i1 = data->indices[prim_idx * 3 + 1];
            int i2 = data->indices[prim_idx * 3 + 2];
            float w = 1.0f - bary.x - bary.y;
            u_coord = w * data->texcoords[i0*2] + bary.x * data->texcoords[i1*2] + bary.y * data->texcoords[i2*2];
            v_coord = w * data->texcoords[i0*2+1] + bary.x * data->texcoords[i1*2+1] + bary.y * data->texcoords[i2*2+1];
        } else if (data->texcoords) {
            int base = prim_idx * 3;
            float w = 1.0f - bary.x - bary.y;
            u_coord = w * data->texcoords[base*2] + bary.x * data->texcoords[(base+1)*2] + bary.y * data->texcoords[(base+2)*2];
            v_coord = w * data->texcoords[base*2+1] + bary.x * data->texcoords[(base+1)*2+1] + bary.y * data->texcoords[(base+2)*2+1];
        }
        u_coord *= data->checker_scale_u;
        v_coord *= data->checker_scale_v;
        int cu = (int)floorf(u_coord);
        int cv = (int)floorf(v_coord);
        if (((cu ^ cv) & 1) == 0)
            albedo = make_f3(data->checker_color1);
        else
            albedo = make_f3(data->checker_color2);
    }

    if (data->material_type == MAT_DIELECTRIC) {
        // Store eta in p0, signal dielectric in p9
        optixSetPayload_0(__float_as_uint(data->eta));
        optixSetPayload_1(0);
        optixSetPayload_2(0);
    } else {
        optixSetPayload_0(__float_as_uint(albedo.x));
        optixSetPayload_1(__float_as_uint(albedo.y));
        optixSetPayload_2(__float_as_uint(albedo.z));
    }

    optixSetPayload_3(__float_as_uint(hit_pos.x));
    optixSetPayload_4(__float_as_uint(hit_pos.y));
    optixSetPayload_5(__float_as_uint(hit_pos.z));
    optixSetPayload_6(__float_as_uint(shading_normal.x));
    optixSetPayload_7(__float_as_uint(shading_normal.y));
    optixSetPayload_8(__float_as_uint(shading_normal.z));
    optixSetPayload_9((unsigned int)data->material_type);
    optixSetPayload_10(__float_as_uint(data->emission[0]));
    optixSetPayload_11(__float_as_uint(data->emission[1]));
    optixSetPayload_12(__float_as_uint(data->emission[2]));
    optixSetPayload_13(__float_as_uint(data->roughness));
}

// Closest hit for sphere (built-in intersection)
extern "C" __global__ void __closesthit__sphere()
{
    const HitGroupData* data = reinterpret_cast<const HitGroupData*>(optixGetSbtDataPointer());

    const float t = optixGetRayTmax();
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    float3 hit_pos = ray_orig + ray_dir * t;

    // For a unit sphere at the object origin, the normal is the hit point in object space
    // Use world-space hit position since sphere is at world origin in this simple case
    // We get the object-space hit point from the transform
    float3 obj_hit = optixTransformPointFromWorldToObjectSpace(hit_pos);
    float3 obj_normal = normalize3(obj_hit); // sphere normal = normalized position
    float3 world_normal = normalize3(optixTransformNormalFromObjectToWorldSpace(obj_normal));

    if (dot3(world_normal, ray_dir) > 0.0f)
        world_normal = world_normal * (-1.0f);

    if (data->material_type == MAT_DIELECTRIC) {
        optixSetPayload_0(__float_as_uint(data->eta));
        optixSetPayload_1(0);
        optixSetPayload_2(0);
    } else {
        float3 albedo = make_f3(data->albedo);
        optixSetPayload_0(__float_as_uint(albedo.x));
        optixSetPayload_1(__float_as_uint(albedo.y));
        optixSetPayload_2(__float_as_uint(albedo.z));
    }

    optixSetPayload_3(__float_as_uint(hit_pos.x));
    optixSetPayload_4(__float_as_uint(hit_pos.y));
    optixSetPayload_5(__float_as_uint(hit_pos.z));
    optixSetPayload_6(__float_as_uint(world_normal.x));
    optixSetPayload_7(__float_as_uint(world_normal.y));
    optixSetPayload_8(__float_as_uint(world_normal.z));
    optixSetPayload_9((unsigned int)data->material_type);
    optixSetPayload_10(__float_as_uint(data->emission[0]));
    optixSetPayload_11(__float_as_uint(data->emission[1]));
    optixSetPayload_12(__float_as_uint(data->emission[2]));
    optixSetPayload_13(__float_as_uint(data->roughness));
}
