#pragma once

#include <glm/glm.hpp>
#include <curand_kernel.h>
#include <cstdint>

struct HitRecord;

#include "Ray.h"
#include "Hittable.h"
#include "Texture.h"
#include "Constants.h"

#ifndef TWO_PI_F
#define TWO_PI_F 6.28318530717958647693f
#endif

// Closed-form uniform-volume sample inside the unit ball. Replaces the
// rejection-sampling loop, which produced heavy warp divergence (the loop
// runs until the LAST lane in the warp accepts; ~5-6 iterations expected
// vs. ~1.91 per-lane mean). The cbrtf radial scale preserves the volume-
// uniform distribution of the original rejection sampler.
__device__ inline glm::vec3 random_in_unit_sphere(RandState* local_rand_state) {
	float z   = 1.0f - 2.0f * curand_uniform(local_rand_state);
	float r   = sqrtf(fmaxf(0.0f, 1.0f - z * z));
	float phi = TWO_PI_F * curand_uniform(local_rand_state);
	float sp, cp;
	__sincosf(phi, &sp, &cp);
	float scale = cbrtf(curand_uniform(local_rand_state));
	return scale * glm::vec3(r * cp, r * sp, z);
}

__device__ inline glm::vec3 random_in_hemisphere(RandState* local_rand_state, const glm::vec3 normal) {
	glm::vec3 in_unit_sphere = random_in_unit_sphere(local_rand_state);

	if (glm::dot(in_unit_sphere, normal) > 0.0f) {
		return in_unit_sphere;
	}
	return -in_unit_sphere;
}

// Branchless orthonormal basis (Pixar's revised Frisvad): produces a
// well-conditioned (t, b, n) frame without the divergent fabsf(n.x) > 0.9
// branch. See "Building an Orthonormal Basis, Revisited" (Duff et al. 2017).
__device__ inline void build_onb(const glm::vec3& n, glm::vec3& t, glm::vec3& b) {
	float sign = copysignf(1.0f, n.z);
	float a    = -1.0f / (sign + n.z);
	float bxy  = n.x * n.y * a;
	t = glm::vec3(1.0f + sign * n.x * n.x * a, sign * bxy,           -sign * n.x);
	b = glm::vec3(bxy,                          sign + n.y * n.y * a, -n.y);
}

// Cosine-weighted hemisphere sampling: PDF proportional to cos(theta)/pi
__device__ inline glm::vec3 cosine_weighted_hemisphere(RandState* local_rand_state, const glm::vec3& normal) {
	float r1 = curand_uniform(local_rand_state);
	float r2 = curand_uniform(local_rand_state);

	float phi = TWO_PI_F * r1;
	float sp, cp;
	__sincosf(phi, &sp, &cp);
	float sr2 = sqrtf(r2);
	float x = cp * sr2;
	float y = sp * sr2;
	float z = sqrtf(1.0f - r2);

	// Transform from local frame (normal = Z) to world space
	glm::vec3 t, b;
	build_onb(normal, t, b);
	return t * x + b * y + normal * z;
}

__device__ inline glm::vec3 reflect(const glm::vec3& v, const glm::vec3& n) {
	return v - 2.0f * glm::dot(v, n) * n;
}

__device__ inline bool near_zero(const glm::vec3& v) {
	float theta = 1e-8f;
	return (fabsf(v[0]) < theta) && (fabsf(v[1]) < theta) && (fabsf(v[2]) < theta);
}

__device__ inline glm::vec3 refract(const glm::vec3& uv, const glm::vec3& n, float etai_over_etat) {
	float cos_theta = fminf(glm::dot(-uv, n), 1.0f);
	glm::vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
	glm::vec3 r_out_parallel = -sqrtf(fabsf(1.0f - glm::dot(r_out_perp, r_out_perp))) * n;
	return r_out_perp + r_out_parallel;
}

// Schlick's approximation for the reflectance of a dielectric surface.
__device__ inline float schlick_reflectance(float cosine, float ref_idx) {
	float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
	r0 = r0 * r0;
	return r0 + (1.0f - r0) * powf((1.0f - cosine), 5);
}

// Tag for tagged-union Material dispatch. Replaces the previous vtable-based
// virtual call to scatter()/emitted(): per-bounce dispatch was hot in
// trace_color, and lanes hitting different material subclasses
// serialized through a vtable indirection. A switch on this tag lets PTXAS
// inline every case body and removes the vtable load.
enum class MaterialKind : uint8_t {
	LAMBERTIAN,
	METAL,
	DIELECTRIC,
	EMISSIVE,
	SUBSURFACE_SCATTER,
};

class Material {
public:
	MaterialKind kind;
	bool is_specular_flag;

	// Per-kind payload. All members are trivially-constructible (pointers,
	// floats, glm::vec3) so the union needs no custom ctor/dtor; the
	// subclass constructors below initialize the active member.
	union Payload {
		struct {
			Texture* albedo;
			Texture* normal_map;
			Texture* specular_map;
		} lambertian;
		struct {
			glm::vec3 albedo;
			float fuzz;
		} metal;
		struct {
			float ir;
		} dielectric;
		struct {
			Texture* emit;
			float intensity;
		} emissive;
		struct {
			glm::vec3 albedo;
			float scattering_distance;
			float ior;
			glm::vec3 extinction_coeff;
		} sss;

		__device__ Payload() {}
	} u;

protected:
	__device__ Material(MaterialKind k, bool spec) : kind(k), is_specular_flag(spec) {}

public:
	__device__ bool is_specular() const { return is_specular_flag; }

	__device__ bool scatter(const Ray& r_in, const HitRecord& rec,
	                        glm::vec3& attenuation, Ray& scattered,
	                        RandState* local_rand_state) const;

	__device__ glm::vec3 emitted(float u, float v, const glm::vec3& p) const;

	// Non-virtual destructor: switches on kind to free owned Textures.
	// All Material subclasses are stored via Material* and freed via
	// `delete mat_ptr`, so the active-kind branch handles cleanup directly.
	__device__ ~Material();
};

// Thin constructor shims preserving the existing `new Lambertian(...)`,
// `new Metal(...)`, etc. call sites in kernel.cu. Subclasses add no
// members and no virtual functions, so sizeof(Subclass) == sizeof(Material)
// and `delete (Material*)p` invokes Material's destructor directly.

class Lambertian final : public Material {
public:
	__device__ Lambertian(const glm::vec3& a) : Material(MaterialKind::LAMBERTIAN, false) {
		u.lambertian.albedo       = new SolidColor(a);
		u.lambertian.normal_map   = nullptr;
		u.lambertian.specular_map = nullptr;
	}
	__device__ Lambertian(Texture* a, Texture* n = nullptr, Texture* s = nullptr)
		: Material(MaterialKind::LAMBERTIAN, false) {
		u.lambertian.albedo       = a;
		u.lambertian.normal_map   = n;
		u.lambertian.specular_map = s;
	}
};

class Metal final : public Material {
public:
	__device__ Metal(const glm::vec3& a, float f) : Material(MaterialKind::METAL, true) {
		u.metal.albedo = a;
		u.metal.fuzz   = (f < 1.0f) ? f : 1.0f;
	}
};

class Dielectric final : public Material {
public:
	__device__ Dielectric(float index_of_refraction)
		: Material(MaterialKind::DIELECTRIC, true) {
		u.dielectric.ir = index_of_refraction;
	}
};

class Emissive final : public Material {
public:
	__device__ Emissive(Texture* a, float i = 1.0f)
		: Material(MaterialKind::EMISSIVE, false) {
		u.emissive.emit      = a;
		u.emissive.intensity = i;
	}
	__device__ Emissive(glm::vec3 c, float i = 1.0f)
		: Material(MaterialKind::EMISSIVE, false) {
		u.emissive.emit      = new SolidColor(c);
		u.emissive.intensity = i;
	}
};

class SubsurfaceScatter final : public Material {
public:
	__device__ SubsurfaceScatter(const glm::vec3& a, float sd, float ir, const glm::vec3& ext)
		: Material(MaterialKind::SUBSURFACE_SCATTER, false) {
		u.sss.albedo              = a;
		u.sss.scattering_distance = sd;
		u.sss.ior                 = ir;
		u.sss.extinction_coeff    = ext;
	}
};

// ---- Dispatch ----------------------------------------------------------
//
// Each scatter case below mirrors the body of the original Lambertian /
// Metal / Dielectric / Emissive / SubsurfaceScatter::scatter overrides.
// The switch lets the compiler inline all cases at the trace_color
// call site and CSE shared loads (rec.normal, rec.p, r_in.direction).
__device__ inline bool Material::scatter(const Ray& r_in, const HitRecord& rec,
                                          glm::vec3& attenuation, Ray& scattered,
                                          RandState* local_rand_state) const {
	switch (kind) {
		case MaterialKind::LAMBERTIAN: {
			Texture* albedo       = u.lambertian.albedo;
			Texture* normal_map   = u.lambertian.normal_map;
			Texture* specular_map = u.lambertian.specular_map;

			glm::vec3 n = rec.normal;

			if (normal_map) {
				glm::vec3 nmap = normal_map->value(rec.u, rec.v, rec.p) * 2.0f - 1.0f;
				n = glm::normalize(nmap.x * rec.tangent + nmap.y * rec.bitangent + nmap.z * rec.normal);
			}

			if (specular_map) {
				glm::vec3 spec_color = specular_map->value(rec.u, rec.v, rec.p);
				float spec_intensity = (spec_color.r + spec_color.g + spec_color.b) / 3.0f;
				if (curand_uniform(local_rand_state) < spec_intensity) {
					glm::vec3 reflected = reflect(glm::normalize(r_in.direction), n);
					scattered = Ray(rec.p, reflected);
					attenuation = albedo->value(rec.u, rec.v, rec.p);
					return (glm::dot(scattered.direction, n) > 0.0f);
				}
			}

			glm::vec3 scatter_direction = cosine_weighted_hemisphere(local_rand_state, n);
			if (near_zero(scatter_direction)) {
				scatter_direction = n;
			}

			scattered = Ray(rec.p, glm::normalize(scatter_direction));
			attenuation = albedo->value(rec.u, rec.v, rec.p);
			return true;
		}

		case MaterialKind::METAL: {
			glm::vec3 reflected = reflect(glm::normalize(r_in.direction), rec.normal);
			scattered = Ray(rec.p, reflected + u.metal.fuzz * random_in_unit_sphere(local_rand_state));
			attenuation = u.metal.albedo;
			return (glm::dot(scattered.direction, rec.normal) > 0.0f);
		}

		case MaterialKind::DIELECTRIC: {
			float ir = u.dielectric.ir;
			attenuation = glm::vec3(1.0f, 1.0f, 1.0f);
			float refraction_ratio = rec.front_face ? (1.0f / ir) : ir;
			glm::vec3 unit_direction = glm::normalize(r_in.direction);

			float cos_theta = fminf(glm::dot(-unit_direction, rec.normal), 1.0f);
			float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

			bool cannot_refract = refraction_ratio * sin_theta > 1.0f;

			glm::vec3 direction;
			if (cannot_refract || schlick_reflectance(cos_theta, refraction_ratio) > curand_uniform(local_rand_state)) {
				direction = reflect(unit_direction, rec.normal);
			} else {
				direction = refract(unit_direction, rec.normal, refraction_ratio);
			}
			scattered = Ray(rec.p, direction);
			return true;
		}

		case MaterialKind::EMISSIVE:
			return false;

		case MaterialKind::SUBSURFACE_SCATTER: {
			glm::vec3 unit_dir = glm::normalize(r_in.direction);
			float cos_theta = fminf(glm::dot(-unit_dir, rec.normal), 1.0f);

			float r0 = (1.0f - u.sss.ior) / (1.0f + u.sss.ior);
			r0 = r0 * r0;
			float fresnel = r0 + (1.0f - r0) * powf(1.0f - cos_theta, 5.0f);

			if (curand_uniform(local_rand_state) < fresnel) {
				glm::vec3 reflected = reflect(unit_dir, rec.normal);
				scattered = Ray(rec.p, reflected);
				attenuation = glm::vec3(1.0f);
				return true;
			}

			float xi = fmaxf(curand_uniform(local_rand_state), 1e-6f);
			float scatter_dist = -logf(xi) * u.sss.scattering_distance;

			glm::vec3 scatter_offset = scatter_dist * random_in_unit_sphere(local_rand_state);
			glm::vec3 exit_point = rec.p + scatter_offset;

			glm::vec3 exit_dir = cosine_weighted_hemisphere(local_rand_state, rec.normal);
			exit_point += T_SELF_INTERSECT * exit_dir;
			scattered = Ray(exit_point, exit_dir);

			glm::vec3 beer = glm::vec3(
				expf(-u.sss.extinction_coeff.x * scatter_dist),
				expf(-u.sss.extinction_coeff.y * scatter_dist),
				expf(-u.sss.extinction_coeff.z * scatter_dist));
			attenuation = u.sss.albedo * beer;
			return true;
		}
	}
	return false;
}

__device__ inline glm::vec3 Material::emitted(float u_coord, float v_coord, const glm::vec3& p) const {
	if (kind == MaterialKind::EMISSIVE) {
		return u.emissive.intensity * u.emissive.emit->value(u_coord, v_coord, p);
	}
	return glm::vec3(0.0f);
}

__device__ inline Material::~Material() {
	switch (kind) {
		case MaterialKind::LAMBERTIAN:
			delete u.lambertian.albedo;
			if (u.lambertian.normal_map)   delete u.lambertian.normal_map;
			if (u.lambertian.specular_map) delete u.lambertian.specular_map;
			break;
		case MaterialKind::EMISSIVE:
			delete u.emissive.emit;
			break;
		case MaterialKind::METAL:
		case MaterialKind::DIELECTRIC:
		case MaterialKind::SUBSURFACE_SCATTER:
			break;
	}
}
