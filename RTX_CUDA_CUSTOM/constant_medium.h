#ifndef CONSTANT_MEDIUM_H
#define CONSTANT_MEDIUM_H

#include "hitable.h"
#include "material.h"
#include "texture.h"
#include <curand_kernel.h>

class constant_medium : public hitable {
public:
    __device__ constant_medium(hitable* b, float d, tex* a, curandState* p_rand_state)
        : boundary(b),
        neg_inv_density(-1.0f / d),
        phase_function(new isotropic(a)),
        rand_state(p_rand_state)
    {}

    __device__ constant_medium(hitable* b, float d, vec3 c, curandState* p_rand_state)
        : boundary(b),
        neg_inv_density(-1.0f / d),
        phase_function(new isotropic(c)),
        rand_state(p_rand_state)
    {}

    __device__ virtual bool hit(
        const ray& r, float t_min, float t_max, hit_record& rec) const override;

    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
        return boundary->bounding_box(time0, time1, output_box);
    }

public:
    curandState* rand_state;
    hitable* boundary;
    material* phase_function;
    float neg_inv_density;
};

__device__ bool constant_medium::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    hit_record rec1, rec2;

    if (!boundary->hit(r, -FLT_MAX, FLT_MAX, rec1))
        return false;

    if (!boundary->hit(r, rec1.t + 0.0001f, FLT_MAX, rec2)) 
        return false;

    if (rec1.t < t_min) rec1.t = t_min;
    if (rec2.t > t_max) rec2.t = t_max;

    if (rec1.t >= rec2.t)
        return false;

    if (rec1.t < 0.0f)
        rec1.t = 0.0f;

    const float ray_length = r.direction().length();
    const float distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
    const float hit_distance = neg_inv_density * logf(curand_uniform(rand_state));

    if (hit_distance > distance_inside_boundary)
        return false;

    rec.t = rec1.t + hit_distance / ray_length;
    rec.p = r.point_at_parameter(rec.t);

    rec.normal = vec3(1.0f, 0.0f, 0.0f);  // arbitrary
    rec.front_face = true;     // also arbitrary
    rec.mat_ptr = phase_function;

    return true;
}
#endif
