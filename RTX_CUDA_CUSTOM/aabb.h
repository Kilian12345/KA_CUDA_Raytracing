#ifndef AABB_H
#define AABB_H

#include "vec3.h"
#include "ray.h"

class aabb {
public:
    __device__ aabb() {}
    __device__ aabb(const vec3& a, const vec3& b) { minimum = a; maximum = b; }

    __device__ vec3 min() const { return minimum; }
    __device__ vec3 max() const { return maximum; }

    __device__ inline bool aabb::hit(const ray& r, double t_min, double t_max) const {
        for (int a = 0; a < 3; a++) {
            auto invD = 1.0f / r.direction()[a];
            auto t0 = (min()[a] - r.origin()[a]) * invD;
            auto t1 = (max()[a] - r.origin()[a]) * invD;
            if (invD < 0.0f) {
                auto temp0 = t0;
                t0 = t1;
                t1 = temp0;
            }
            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;
            if (t_max <= t_min)
                return false;
        }
        return true;
    }

    vec3 minimum;
    vec3 maximum;
};

__device__ aabb surrounding_box(aabb box0, aabb box1) {
    vec3 small(fminf(box0.min().x(), box1.min().x()),
        fminf(box0.min().y(), box1.min().y()),
        fminf(box0.min().z(), box1.min().z()));

    vec3 big(fmaxf(box0.max().x(), box1.max().x()),
        fmaxf(box0.max().y(), box1.max().y()),
        fmaxf(box0.max().z(), box1.max().z()));

    return aabb(small, big);
}


#endif