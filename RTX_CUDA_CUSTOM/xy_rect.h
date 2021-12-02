#ifndef AARECT_H
#define AARECT_H

#include "rtweekend.h"
#include "hitable.h"

class xy_rect : public hitable {
public:
    __device__ xy_rect() {}

    __device__ xy_rect(double _x0, double _x1, double _y0, double _y1, double _k, material* mat)
        : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mp(mat) {
        trackerValue = 1;
    };

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec)const override;

    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
        // The bounding box must have non-zero width in each dimension, so pad the Z
        // dimension a small amount.
        output_box = aabb(vec3(x0, y0, k - 0.0001f), vec3(x1, y1, k + 0.0001f));
        return true;
    }

public:
    material* mp;
    float x0, x1, y0, y1, k;
};

class xz_rect : public hitable {
public:
    __device__ xz_rect() {}

    __device__ xz_rect(float _x0, float _x1, float _z0, float _z1, float _k, material* mat)
        : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mp(mat) {
        trackerValue = 1;
    };

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
        // The bounding box must have non-zero width in each dimension, so pad the Y
        // dimension a small amount.
        output_box = aabb(vec3(x0, k - 0.0001f, z0), vec3(x1, k + 0.0001f, z1));
        return true;
    }

public:
    material* mp;
    float x0, x1, z0, z1, k;
};

class yz_rect : public hitable {
public:
    __device__ yz_rect() {}

    __device__ yz_rect(float _y0, float _y1, float _z0, float _z1, float _k, material* mat)
        : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mp(mat) {
        trackerValue = 1;
    };

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
        // The bounding box must have non-zero width in each dimension, so pad the X
        // dimension a small amount.
        output_box = aabb(vec3(k - 0.0001, y0, z0), vec3(k + 0.0001, y1, z1));
        return true;
    }

public:
    material* mp;
    float y0, y1, z0, z1, k;
};


/// 
/// /////////////////////////////////// FUNCTIONS
/// 


__device__ bool xy_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    float t = (k - r.origin().z()) / r.direction().z();
    if (t < t_min || t > t_max)
        return false;
    float x = r.origin().x() + t * r.direction().x();
    float y = r.origin().y() + t * r.direction().y();
    if (x < x0 || x > x1 || y < y0 || y > y1)
        return false;
    rec.u = (x - x0) / (x1 - x0);
    rec.v = (y - y0) / (y1 - y0);
    rec.t = t;
    vec3 outward_normal = vec3(0, 0, 1);
    rec.set_face_normal(r,outward_normal);
    rec.mat_ptr = mp;
    rec.p = r.point_at_parameter(t);
    return true;
}

__device__ bool xz_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    float t = (k - r.origin().y()) / r.direction().y();
    if (t < t_min || t > t_max)
        return false;
    float x = r.origin().x() + t * r.direction().x();
    float z = r.origin().z() + t * r.direction().z();
    if (x < x0 || x > x1 || z < z0 || z > z1)
        return false;
    rec.u = (x - x0) / (x1 - x0);
    rec.v = (z - z0) / (z1 - z0);
    rec.t = t;
    vec3 outward_normal = vec3(0, 1, 0);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mp;
    rec.p = r.point_at_parameter(t);
    return true;
}

 __device__ bool yz_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    float t = (k - r.origin().x()) / r.direction().x();
    if (t < t_min || t > t_max)
        return false;
    float y = r.origin().y() + t * r.direction().y();
    float z = r.origin().z() + t * r.direction().z();
    if (y < y0 || y > y1 || z < z0 || z > z1)
        return false;
    rec.u = (y - y0) / (y1 - y0);
    rec.v = (z - z0) / (z1 - z0);
    rec.t = t;
    vec3 outward_normal = vec3(1, 0, 0);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mp;
    rec.p = r.point_at_parameter(t);
    return true;
}
#endif