#ifndef TEX_H
#define TEX_H

#include <iostream>
#include "vec3.h"
#include "perlin.h"
#include "cuda_runtime.h"

class tex {
public:
    __device__ virtual vec3 value(float u, float v, const vec3& p) const = 0;
};

class solid_color : public tex {
public:
    __device__ solid_color() {}
    __device__ solid_color(vec3 c) : color_value(c) {}

    __device__ solid_color(float red, float green, float blue)
        : solid_color(vec3(red, green, blue)) {}

    __device__ virtual vec3 value(float u, float v, const vec3& p) const override {
        return color_value;
    }

private:
    vec3 color_value;
};


class checker_texture : public tex {
public:
    __device__ checker_texture() {}

    __device__ checker_texture(tex* _even, tex* _odd)
        : even(_even), odd(_odd) {}

    __device__ checker_texture(vec3 c1, vec3 c2)
        : even(new solid_color(c1)), odd(new solid_color(c2)) {}

    __device__ virtual vec3 value(float u, float v, const vec3& p) const override {
        auto sines = sinf(10 * p.x()) * sinf(10 * p.y()) * sinf(10 * p.z());
        if (sines < 0)
            return odd->value(u, v, p);
        else
            return even->value(u, v, p);
    }

public:
    tex* odd;
    tex* even;
};

class noise_texture : public tex{
public:
    __device__ noise_texture(curandState* local_rand_state)
        : perlin_noise(new perlin(local_rand_state)) {}

    __device__ noise_texture(curandState* local_rand_state, float sc = 1.0f)
        : perlin_noise(new perlin(local_rand_state)), scale(sc){}

    __device__ virtual vec3 value(float u, float v, const vec3& p) const override {
        return vec3(1, 1, 1) * 0.5f * (1.0f + sinf(scale*p.z() + 10 * perlin_noise->turb(p)));
    }

public:
    perlin* perlin_noise;
    float scale;
};

struct image_texture_infos {
    __host__ image_texture_infos()
        : bytes_per_pixel(0), filename(nullptr), data(nullptr), width(0), height(0), pitch(0) {}

    int bytes_per_pixel;
    const char* filename;
    unsigned char* data;
    int width, height;
    size_t pitch;
};

class image_texture : public tex {
public:

    __device__ image_texture()
        : tex_infos(nullptr), bytes_per_scanline(0), width(0), height(0) {}

    __device__ image_texture(image_texture_infos* _tex_infos, unsigned char* _tex_data)
        : tex_infos(_tex_infos),
        data(_tex_data),
        bytes_per_pixel(_tex_infos->bytes_per_pixel),
        width(_tex_infos->width), height(_tex_infos->height),
        pitch(_tex_infos->pitch){

        bytes_per_scanline = bytes_per_pixel * width;
    }

    __device__ virtual vec3 value(float u, float v, const vec3& p) const override {
        // If we have no texture data, then return solid cyan as a debugging aid.
        if (data == nullptr)
            return vec3(0, 1, 1);

        // Cuda Clamp input texture coordinates to [0,1] x [1,0]
        u = __saturatef(u);
        v = 1.0 - __saturatef(v);  // Flip V to image coordinates

        int i = u * width;
        int j = v * height;

        // Clamp integer mapping, since actual coordinates should be less than 1.0
        if (i >= width)  i = width - 1;
        if (j >= height) j = height - 1;

        const float color_scale = 1.0f / 255.0f;
        unsigned char* pixel = data + j * bytes_per_scanline + i * bytes_per_pixel;

        return vec3(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
    }

private:
    image_texture_infos* tex_infos;
    unsigned char* data;
    int bytes_per_scanline;
    int width, height, bytes_per_pixel;
    size_t pitch;
};


#endif