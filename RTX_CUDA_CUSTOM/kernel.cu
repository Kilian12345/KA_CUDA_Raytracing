#include <iostream>
#include <time.h>
#include <float.h>
#include <stdio.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"
#include "moving_sphere.h"
#include "stb_image.h"
#include "tex_loader.h"
#include "cuda_debug.h"
#include "xy_rect.h"
#include "box.h"
#include "constant_medium.h"
#include "bvh.h"

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, const vec3& background, hitable** world, curandState* local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0f, 1.0f, 1.0f);
    hit_record rec;
    vec3 emitted(0, 0, 0);

    for (int i = 0; i < 50; i++) {

        if (!(*world)->hit(cur_ray, 0.001f, FLT_MAX, rec))
            return cur_attenuation *= background;

        ray scattered;
        vec3 attenuation;
        emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);

        if (!rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state))
            return emitted * cur_attenuation;


        cur_attenuation *= attenuation;
        cur_ray = scattered;

    }
    return cur_attenuation; // exceeded recursion
}

__global__ void rand_init(curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3* fb, int max_x, int max_y, int ns, vec3 background, camera** cam, hitable** world, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, background, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);   
    //printf("col staruration = %f/%f/%f\n", col[0], col[1], col[2]);
    //if (col[0] > 6.0f || col[1] > 6.0f || col[2] > 6.0f) printf("col staruration = %f/%f/%f\n", col[0], col[1], col[2]);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);

    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

#pragma region WORLD




__global__ void create_world(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
            new lambertian(new checker_texture(vec3(0.2, 0.3, 0.1), vec3(0.9, 0.9, 0.9))));
        int i = 1;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a + RND, 0.2, b + RND);
                if (choose_mat < 0.8f) {
                    vec3 center2 = center + vec3(0.0f, random_range(rand_state, 0.0f, 0.5f), 0.0f);
                    d_list[i++] = new moving_sphere(center, center2, 0.0f, 1.0f, 0.2,
                        new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
                }
                else if (choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                        new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, 22 * 22 + 1 + 3);

        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0; (lookfrom - lookat).length();
        float aperture = 0.1;
        *d_camera = new camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            30.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus,
            0.0f,
            1.0f);
    }
}

__global__ void create_big_sphere(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        curandState local_rand_state = *rand_state;
        checker_texture* checker = new checker_texture(vec3(0.2, 0.3, 0.1), vec3(0.9, 0.9, 0.9));
        lambertian* lambert = new lambertian(checker);

        d_list[0] = new sphere(vec3(0, -10.0, 0), 10, lambert);
        d_list[1] = new sphere(vec3(0, 10.0, 0), 10, lambert);

        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, 2);

        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0; (lookfrom - lookat).length();
        float aperture = 0.0;
        *d_camera = new camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            20.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus,
            0.0f,
            1.0f);
    }
}

__global__ void create_perlin_sphere(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        curandState local_rand_state = *rand_state;
        noise_texture* perlin = new noise_texture(rand_state, 4);
        lambertian* lambert = new lambertian(perlin);

        d_list[0] = new sphere(vec3(0, -1000.0, 0), 1000, lambert);
        d_list[1] = new sphere(vec3(0, 2, 0), 2, lambert);

        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, 2);

        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0; (lookfrom - lookat).length();
        float aperture = 0.0;
        *d_camera = new camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            20.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus,
            0.0f,
            1.0f);
    }
}

__global__ void create_earth(hitable** d_list,
    hitable** d_world,
    camera** d_camera,
    int nx, int ny,
    curandState* rand_state,
    image_texture_infos* textures_infos,
    unsigned char* texture_data)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        curandState local_rand_state = *rand_state;
        image_texture* texture = new image_texture(textures_infos, texture_data);
        lambertian* lambert = new lambertian(texture);

        d_list[0] = new sphere(vec3(0, 0, 0), 2, lambert);
        d_list[1] = new sphere(vec3(0, -3, 3), 0.5, lambert);
        d_list[2] = new sphere(vec3(0, 3, -3), 0.25, lambert);
        d_list[3] = new sphere(vec3(-3, 0, 3), 1, lambert);
        d_list[4] = new sphere(vec3(3, 0, -3), 0.75, lambert);

        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, 5);

        vec3 lookfrom(20, 2, 3);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0; (lookfrom - lookat).length();
        float aperture = 0.0;
        *d_camera = new camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            20.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus,
            0.0f,
            1.0f);
    }
}

__global__ void pyra(hitable** d_list,
    hitable** d_world,
    camera** d_camera,
    int nx, int ny,
    curandState* rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        curandState local_rand_state = *rand_state;
        noise_texture* perlin = new noise_texture(rand_state, 4);
        lambertian* lambert = new lambertian(perlin);

        metal* met;

        int pyramid_length = 5;
        float sphere_radius = 0.6f;
        float padding = 0.0f;

        int sphere_num = 0;

        for (int i = 1; i <= pyramid_length; i++)
        {
            sphere_num += i * i;
        }

        int id_tracker = 0;

        for (int i = 0; i < pyramid_length; i++)
        {
            for (int y = i; y < pyramid_length; y++)
            {
                for (int w = i; w < pyramid_length; w++)
                {

                    met = new metal(vec3(0.1f * w, 0.1f * y, 0.1f * i), 0.1f);

                    float m = i * (sphere_radius + padding);

                    d_list[id_tracker] =
                        new sphere(
                            vec3(
                                (sphere_radius * 2.0f + padding) * w - m,
                                (sphere_radius * 2.0f + padding) * i + 0.6f - (i* 0.1f),
                                (sphere_radius * 2.0f + padding) * y - m
                            ),
                            sphere_radius,
                            met);
                    id_tracker++;
                }
            }
        }

        int total = sphere_num;

        d_list[total + 0] = new sphere(vec3(0, -5000, 0), 5000, lambert);
        //d_list[1] = new sphere(vec3(0, 2, 0), 2, lambert);

        diffuse_light* diff_light = new diffuse_light(vec3(1, 0, 0));
        //d_list[total + 1] = new xy_rect(-5, 0, -10, 10, -5, diff_light);

        diff_light = new diffuse_light(vec3(1, 1, 1));
        d_list[total + 1] = new xz_rect(-10, 10, -10, 10, 10, diff_light);
        
        total = total + 2;

        solid_color* solid;

        for (int i = 0; i < 1000; i++)
        {
            solid = new solid_color(vec3(RND, RND, RND));
            lambert = new lambertian(solid);

            float neg = RND >= 0.5f ? 1 : -1;
            float neg2 = RND >= 0.5f ? 1 : -1;

            d_list[total + i] = new sphere(vec3((RND * 50.0f * neg), RND * 0.2f + 0.3f, (RND * 50.0f * neg2)), RND * 0.3f + 0.1f, lambert);
        }




        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, total + 1000);

        vec3 lookfrom(25, 7, 10);
        vec3 lookat(-20, -2, -(pyramid_length * sphere_radius) * 2.0f);
        float dist_to_focus = 23;  (lookfrom - lookat).length();
        float aperture = 0.5f;
        *d_camera = new camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            20.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus,
            0.0f,
            1.0f);
    }
}

__global__ void sss(hitable** d_list,
    hitable** d_world,
    camera** d_camera,
    int nx, int ny,
    curandState* rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        curandState local_rand_state = *rand_state;

        lambertian* red = new lambertian(vec3(.65, .05, .05));
        lambertian* white = new lambertian(vec3(.73, .73, .73));
        lambertian* dark = new lambertian(vec3(.1, .1, .1));
        lambertian* green = new lambertian(vec3(.12, .45, .15));
        vec3 light_color = vec3(1.0f, 0.1f, 0.1f);
        light_color = unit_vector(light_color);
        diffuse_light* light = new diffuse_light(vec3(1.0f, 1.0f, 1.0f));
        diffuse_light* light2 = new diffuse_light(light_color * 1);

        diffuse_light* light_p;

        int width_tile = 8;
        int length_tile = 12;
        float width_div = 555 / width_tile;
        float length_div = 3000 / length_tile;

        for (int i = 0; i < length_tile; i++)
        {
            for (int y = 0; y < width_tile; y++)
            {
                light_p = new diffuse_light(vec3(RND, RND, RND));

                d_list[(i * width_tile) + y] = new xz_rect(width_div * y + 20, width_div * (y + 1), length_div * i + 20, length_div * (i + 1), 2, light_p);
            }
        }

        int total = length_tile * width_tile;

        d_list[total + 0] = new yz_rect(0, 555, 0, 3000, 555, dark);
        d_list[total + 1] = new yz_rect(0, 555, 0, 3000, 0, dark);
        d_list[total + 2] = new xz_rect(0, 555, 0, 3000, 555, dark);
        d_list[total + 3] = new xy_rect(0, 555, 0, 555, 3000, dark);

        sphere* boundary = new sphere(vec3(200, 150, 700), 100.0f, new dielectric(1.5f));
        sphere* boundary2 = new sphere(vec3(200, 250, 1500), 150.0f, new dielectric(1.5f));

        hitable* box1 = new box(vec3(0, 0, 0), vec3(165, 165, 165), white);
        box1 = new transform(box1, -18, vec3(350, 0, 400));

        hitable* box2 = new box(vec3(0, 0, 0), vec3(165, 360, 800), white);
        box2 = new transform(box2, 8, vec3(100, 0, 1600));

        d_list[total + 4] = boundary;
        d_list[total + 5] = boundary2;
        d_list[total + 6] = new constant_medium(boundary2, 5, vec3(0.2f, 0.8f, 0.3f), rand_state);
        d_list[total + 7] = box1;


        boundary = new sphere(vec3(0.0f, 0.0f, 0.0f), 5000.0f, new dielectric(1.5f));       
        d_list[total + 8] = new constant_medium(boundary, 0.0001f, vec3(1.0f, 1.0f, 1.0f), rand_state);


        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, total + 9);

        vec3 lookfrom(450, 500, -300);
        vec3 lookat(0, 0, 1500);
        float dist_to_focus = 10.0; (lookfrom - lookat).length();
        float aperture = 0.0;
        *d_camera = new camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            40.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus,
            0.0f,
            1.0f);
    }
}

__global__ void corridor(hitable** d_list,
    hitable** d_world,
    camera** d_camera,
    int nx, int ny,
    curandState* rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        curandState local_rand_state = *rand_state;

        lambertian* red = new lambertian(vec3(.65, .05, .05));
        lambertian* white = new lambertian(vec3(.73, .73, .73));
        lambertian* green = new lambertian(vec3(.12, .45, .15));
        vec3 light_color = vec3(1.0f, 0.1f, 0.1f);
        light_color = unit_vector(light_color);
        diffuse_light* light = new diffuse_light(vec3(1.0f, 1.0f,1.0f));
        diffuse_light* light2 = new diffuse_light(light_color * 1);

        d_list[0] = new yz_rect(0, 555, 0, 3000, 555, green);
        d_list[1] = new yz_rect(0, 555, 0, 3000, 0, red);
        d_list[2] = new yz_rect(50, 500, 2000, 2800, 1, light);
        d_list[3] = new yz_rect(50, 500, 500, 1000, 554, light2);
        d_list[4] = new xz_rect(0, 555, 0, 3000, 0, white);
        d_list[5] = new xz_rect(0, 555, 0, 3000, 555, white);
        d_list[6] = new xy_rect(0, 555, 0, 555, 3000, white);
        d_list[7] = new yz_rect(50, 500, 200, 1000, 1, new metal(vec3(0.8f, 0.8f, 0.9f), 0.0f));


        sphere* boundary = new sphere(vec3(200, 150, 700), 100.0f, new dielectric(1.5f));
        sphere* boundary2 = new sphere(vec3(200, 250, 1200), 150.0f, new metal(vec3(0.8f, 0.8f, 0.9f), 1.0f));

        hitable* box1 = new box(vec3(0, 0, 0), vec3(165, 165, 165), white);
        //box2 = new rotate_y(box2, -18);
        //box2 = new translate(box2, vec3(130, 0, 65));
        box1 = new transform(box1, -18, vec3(350, 0, 400));

        hitable* box2 = new box(vec3(0, 0, 0), vec3(165, 360, 800), white);
        //box2 = new rotate_y(box2, -18);
        //box2 = new translate(box2, vec3(130, 0, 65));
        box2 = new transform(box2, 8, vec3(100, 0, 1600));

        d_list[8] = boundary;
        d_list[9] = box1;
        d_list[10] = box2;

        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, 11);

        vec3 lookfrom(450, 500, -300);
        vec3 lookat(0, 0, 1500);
        float dist_to_focus = 10.0; (lookfrom - lookat).length();
        float aperture = 0.0;
        *d_camera = new camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            40.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus,
            0.0f,
            1.0f);
    }
}

__global__ void full_scene(hitable** d_list,
    hitable** d_world,
    camera** d_camera,
    int nx, int ny,
    image_texture_infos* textures_infos,
    unsigned char* texture_data,
    curandState* rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        curandState local_rand_state = *rand_state;

        const int boxes_per_side = 20;
        lambertian* ground = new lambertian(vec3(0.48f, 0.83f, 0.53f));

        for (int i = 0; i < boxes_per_side; i++) {
            for (int j = 0; j < boxes_per_side; j++) {
                float w = 100.0f;
                float x0 = -1000.0f + i * w;
                float z0 = -1000.0f + j * w;
                float y0 = 0.0f;
                float x1 = x0 + w;
                float y1 = random_range(rand_state, 1.0f, 90.0f);
                float z1 = z0 + w;

                d_list[i * boxes_per_side + j] = new box(vec3(x0, y0, z0), vec3(x1, y1, z1), ground);
            }
        }

        int base_index = boxes_per_side * boxes_per_side;
        base_index -= 1;

        diffuse_light* light = new diffuse_light(vec3(3.0f, 3.0f, 3.0f));
        d_list[base_index + 1] = new xz_rect(123.0f, 423.0f, 147.0f, 412.0f, 554.0f, light);

        vec3 center1 = vec3(400.0f, 400.0f, 200.0f);
        vec3 center2 = center1 + vec3(30.0f, 0.0f, 0.0f);
        lambertian* moving_sphere_material = new lambertian(vec3(0.7f, 0.3f, 0.1f));
        d_list[base_index + 2] = new moving_sphere(center1, center2, 0.0f, 1.0f, 50.0f, moving_sphere_material);

        d_list[base_index + 3] = new sphere(vec3(260.0f, 150.0f, 45.0f), 50.0f, new dielectric(1.5f));
        d_list[base_index + 4] = new sphere(vec3(0.0f, 150.0f, 145.0f), 50.0f, new metal(vec3(0.8f, 0.8f, 0.9f), 1.0f));

        sphere* boundary = new sphere(vec3(360.0f, 150.0f, 145.0f), 70.0f, new dielectric(1.5f));
        d_list[base_index + 5] = boundary;
        d_list[base_index + 6] = new constant_medium(boundary, 0.2f, vec3(0.2f, 0.4f, 0.9f), rand_state);
        boundary = new sphere(vec3(0.0f, 0.0f, 0.0f), 5000.0f, new dielectric(1.5f));
        d_list[base_index + 7] = new constant_medium(boundary, .0001f, vec3(1.0f, 1.0f, 1.0f), rand_state);

        lambertian* emat = new lambertian(new image_texture(textures_infos, texture_data));
        d_list[base_index + 8] = new sphere(vec3(400.0f, 200.0f, 400.0f), 100.0f, emat);
        noise_texture* pertext = new noise_texture(rand_state, 0.1f);
        d_list[base_index + 9] = new sphere(vec3(220.0f, 280.0f, 300.0f), 80.0f, new lambertian(pertext));

        int base_index_2 = base_index + 9;
        
        int ns = 1000;
        lambertian* white = new lambertian(vec3(.73f, .73f, .73f));
        for (int j = 1; j <= ns; j++) {
            vec3 center = vec3(
                random_range(rand_state, 0.0f, 165.0f),
                random_range(rand_state, 0.0f, 165.0f),
                random_range(rand_state, 0.0f, 165.0f)
            );

            hitable* l_sphere = new sphere(center + vec3(-100.0f, 270.0f, 395.0f), 10.0f, white);
            d_list[base_index_2 + j] = l_sphere;
        }

        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, base_index_2 + 1000);

        vec3 lookfrom(478, 278, -600);
        vec3 lookat(278, 278, 0);
        float dist_to_focus = 10.0; (lookfrom - lookat).length();
        float aperture = 0.0;
        *d_camera = new camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            40.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus,
            0.0f,
            1.0f);
    }
}
#pragma endregion

__global__ void free_world(float number_of_objects, hitable** d_list,
    hitable** d_world,
    camera** d_camera,
    image_texture_infos** textures_infos,
    unsigned char** d_texture_data
) {
    for (int i = 0; i < number_of_objects; i++) {
        delete ((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete* d_world;
    delete* d_camera;
    delete* textures_infos;
    delete* d_texture_data;

}

int main() {
    int nx = 400;
    float aspect_ratio = 16.0f/9.0f;
    int ny = nx / aspect_ratio;
    int ns = 100;
    int tx = 8; // sqrt(1024) = 32
    int ty = 8;


    int actual_scene = 4; // 0 normal || 1 big_sphere || 2 perlin_sphere || 3 earth || 4 pyra || 5 sss || 6 corridor || 7 full_scene

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    // allocate FB
    vec3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // allocate random state on the device
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
    curandState* d_rand_state2;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init << <1, 1 >> > (d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hitables & the camera
    hitable** d_list;
    int num_hitables = 20 * 20 + 9 + 1000;
    checkCudaErrors(cudaMalloc((void**)&d_list, num_hitables * sizeof(hitable*)));

    hitable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));

    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));

    // Texture
    image_texture_infos** textures_infos = new image_texture_infos * ();
    textures_infos[0] = new image_texture_infos(); // Create a texture
    unsigned char** d_texture_data = new unsigned char*;
    size_t* pitch = new size_t();

    // Background
    vec3 background(0, 0, 0);

    switch (actual_scene) {
    case 0:
        create_world << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2);
        background = vec3(0.70, 0.80, 1.00);
        break;
    case 1:
        create_big_sphere << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2);
        background = vec3(0.70, 0.80, 1.00);
        break;
    case 2:
        create_perlin_sphere << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2);
        background = vec3(0.70, 0.80, 1.00);
        break;
    case 3:
        checkCudaErrors(cudaMallocManaged((void**)&textures_infos[0], sizeof(image_texture_infos)));
        texture_loader(0, textures_infos, d_texture_data, pitch, "D:\\PROJECT\\GRAPH_PROG\\RTX_WKND\\RayTracerNextWeek_CUDA\\pic\\earthmap.jpg");
        create_earth << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2, textures_infos[0], d_texture_data[0]);
        background = vec3(0.70, 0.80, 1.00);
        break;
    case 4:
        pyra << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2);
        background = vec3(0.0, 0.0, 0.0);
        break;
    case 5:
        sss << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2);
        background = vec3(0.0, 0.0, 0.0);
        break;
    case 6:
        corridor<< <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2);
        background = vec3(0.0, 0.0, 0.0);
        break;
    default:
    case 7:
        checkCudaErrors(cudaMallocManaged((void**)&textures_infos[0], sizeof(image_texture_infos)));
        texture_loader(0, textures_infos, d_texture_data, pitch, "D:\\PROJECT\\GRAPH_PROG\\RTX_WKND\\RayTracerNextWeek_CUDA\\pic\\earthmap.jpg");
        full_scene << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, textures_infos[0], d_texture_data[0], d_rand_state2);
        background = vec3(0.0, 0.0, 0.0);
        break;
    }

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init << <blocks, threads >> > (nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render << <blocks, threads >> > (fb, nx, ny, ns, background, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            int ir = int(255.99 * fb[pixel_index].r());
            int ig = int(255.99 * fb[pixel_index].g());
            int ib = int(255.99 * fb[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_camera));

    free_world << <1, 1 >> > (20 * 20 + 9 + 1000, d_list, d_world, d_camera, textures_infos, d_texture_data);

    /*switch (actual_scene) {
    case 0:
        free_world << <1, 1 >> > (22 * 22 + 1 + 3, d_list, d_world, d_camera, textures_infos, d_texture_data);
        break;
    case 1||2:
        free_world << <1, 1 >> > (2, d_list, d_world, d_camera, textures_infos, d_texture_data);
        break;
    case 3:
        free_world << <1, 1 >> > (5, d_list, d_world, d_camera, textures_infos, d_texture_data);
        break;
    default:
    case 4:
        free_world << <1, 1 >> > (4, d_list, d_world, d_camera, textures_infos, d_texture_data);
        break;

    }*/

    //checkCudaErrors(cudaDeviceSynchronize()); //IF ACTIVATED, THIS WILL RETURN cudaError_t 700
    checkCudaErrors(cudaGetLastError());

    cudaDeviceReset();
}