#ifndef PERLIN_H
#define PERLIN_H

#include "vec3.h"
#include "rtweekend.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

class perlin {
public:
    __device__ perlin(curandState* local_rand_state) {
        ranvec = new vec3[point_count];
        for (int i = 0; i < point_count; ++i) {
            ranvec[i] = unit_vector(
                vec3(
                    random_range(local_rand_state, -1.0f, 1.0f),
                    random_range(local_rand_state, -1.0f, 1.0f),
                    random_range(local_rand_state, -1.0f, 1.0f)
                    ));
        }

        perm_x = perlin_generate_perm(local_rand_state);
        perm_y = perlin_generate_perm(local_rand_state);
        perm_z = perlin_generate_perm(local_rand_state);
    }

    __host__ __device__ ~perlin() {
        delete[] ranvec;
        delete[] perm_x;
        delete[] perm_y;
        delete[] perm_z;
    }

    __device__ float turb(const vec3& p, int depth = 7) const {
        float accum = 0.0f;
        vec3 temp_p = p;
        float weight = 1.0f;

        for (int i = 0; i < depth; i++) {
            accum += weight * noise(temp_p);
            weight *= 0.5f;
            temp_p *= 2;
        }

        return fabsf(accum);
    }

    __device__ float noise(const vec3& p) const {
        float u = p.x() - floorf(p.x());
        float v = p.y() - floorf(p.y());
        float w = p.z() - floorf(p.z());

        int i = static_cast<int>(floorf(p.x()));
        int j = static_cast<int>(floorf(p.y()));
        int k = static_cast<int>(floorf(p.z()));
        vec3 c[2][2][2];

        for (int di = 0; di < 2; di++)
            for (int dj = 0; dj < 2; dj++)
                for (int dk = 0; dk < 2; dk++)
                    c[di][dj][dk] = ranvec[
                        perm_x[(i + di) & 255] ^
                            perm_y[(j + dj) & 255] ^
                            perm_z[(k + dk) & 255]
                    ];

        return perlin_interp(c,u,v,w);
    }


private:
    static const int point_count = 256;
    vec3* ranvec;
    int* perm_x;
    int* perm_y;
    int* perm_z;

    __device__ static int* perlin_generate_perm(curandState* local_rand_state) {
        int* p = new int[point_count];

        for (int i = 0; i < perlin::point_count; i++)
            p[i] = i;

        permute(local_rand_state, p, point_count);

        return p;
    }

    __device__ static void permute(curandState* local_rand_state, int* p, int n) {
        for (int i = n - 1; i > 0; i--) {
            int target = random_int(local_rand_state, 0, i);
            int tmp = p[i];
            p[i] = p[target];
            p[target] = tmp;
        }
    }

    __device__ static float trilinear_interp(float c[2][2][2], float u, float v, float w) {
        auto accum = 0.0;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++)
                    accum += (i * u + (1 - i) * (1 - u)) *
                    (j * v + (1 - j) * (1 - v)) *
                    (k * w + (1 - k) * (1 - w)) * c[i][j][k];

        return accum;
    }

    __forceinline__ __device__ static float perlin_interp(vec3 c[2][2][2], float u, float v, float w) {
        float uu = u * u * (3.0f - 2.0f * u);
        float vv = v * v * (3.0f - 2.0f * v);
        float ww = w * w * (3.0f - 2.0f * w);
        float accum = 0.0f;

       for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    vec3 weight_v(u - i, v - j, w - k);
                    accum += (i * uu + (1.0f - i) * (1.0f - uu))
                        * (j * vv + (1.0f - j) * (1.0f - vv))
                        * (k * ww + (1.0f - k) * (1.0f - ww))
                        * dot(c[i][j][k], weight_v);
                }
            }
        }


        return accum;
    }
};

#endif