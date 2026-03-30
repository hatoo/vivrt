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
            unsigned int p0, p1, p2, p3, p4, p5, p6, p7, p8, p9;
            p9 = 0xFFFFFFFF; // miss sentinel

            optixTrace(
                params.traversable,
                origin, direction,
                0.001f, 1e16f, 0.0f,
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE,
                0, 1, 0,
                p0, p1, p2, p3, p4, p5, p6, p7, p8, p9
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
            int mat_type = (int)p9;

            if (mat_type == MAT_DIFFUSE) {
                // Direct lighting from distant lights
                for (int i = 0; i < params.num_distant_lights; i++) {
                    float3 light_dir = make_f3(params.distant_lights[i].direction);
                    float3 light_em = make_f3(params.distant_lights[i].emission);
                    float ndotl = dot3(hit_normal, light_dir);
                    if (ndotl > 0.0f) {
                        // Shadow ray
                        unsigned int shadow_p9 = 0xFFFFFFFF;
                        optixTrace(
                            params.traversable,
                            hit_pos, light_dir,
                            0.001f, 1e16f, 0.0f,
                            OptixVisibilityMask(255),
                            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                            0, 1, 0,
                            p0, p1, p2, p3, p4, p5, p6, p7, p8, shadow_p9
                        );
                        if (shadow_p9 == 0xFFFFFFFF) {
                            radiance = radiance + throughput * hit_albedo * light_em * ndotl * (1.0f / M_PIf);
                        }
                    }
                }

                // Indirect: cosine-weighted bounce
                RNG bounce_rng(pixel_idx, s, depth + 1);
                direction = cosine_sample_hemisphere(bounce_rng.next(), bounce_rng.next(), hit_normal);
                origin = hit_pos;
                throughput = throughput * hit_albedo;
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
        if ((cu + cv) % 2 == 0)
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
}
