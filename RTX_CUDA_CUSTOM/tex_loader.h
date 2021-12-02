#ifndef TEX_LOADER_H
#define TEX_LOADER_H

#include <stdio.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "texture.h"
#include "rtw_stb_image.h"
#include "cuda_debug.h"

class tex_loader {
public:
    __host__ void texture_loader(int tex_index, image_texture_infos** _tex_infos, unsigned char** device_data, size_t* pitch, const char* filename);

};

__host__ void texture_loader(int tex_index, image_texture_infos** _tex_infos, unsigned char** device_data, size_t* pitch, const char* filename) {

    _tex_infos[tex_index]->filename = filename;

    int byte_per_pixel_local = _tex_infos[tex_index]->bytes_per_pixel;

    _tex_infos[tex_index]->data = stbi_load(
        _tex_infos[tex_index]->filename,
        &_tex_infos[tex_index]->width,
        &_tex_infos[tex_index]->height,
        &_tex_infos[tex_index]->bytes_per_pixel,
        byte_per_pixel_local);

    // Real width because we have 3 byte of infos per pixel (RGB)
    int real_width = _tex_infos[tex_index]->width * sizeof(unsigned char);
    real_width *= _tex_infos[tex_index]->bytes_per_pixel;

    // Allocate in Device
    checkCudaErrors(cudaMallocPitch(
        &(device_data[tex_index]),
        &(pitch[tex_index]),
        real_width,
        _tex_infos[tex_index]->height)
    );

    // Copy Host data in Device
    checkCudaErrors(cudaMemcpy2D(
        device_data[tex_index],
        pitch[tex_index],
        _tex_infos[tex_index]->data,
        real_width,
        real_width,
        _tex_infos[tex_index]->height,
        cudaMemcpyHostToDevice)
    );

    _tex_infos[tex_index]->pitch = pitch[tex_index];


    std::cerr << "_tex_infos->filename = " << _tex_infos[tex_index]->filename << " \n"
        << "_tex_infos->width = " << _tex_infos[tex_index]->width << " \n"
        << "_tex_infos->height = " << _tex_infos[tex_index]->height << " \n"
        << "_tex_infos->bytes_per_pixel = " << _tex_infos[tex_index]->bytes_per_pixel << " \n";


    if (!_tex_infos[tex_index]->data) {
        std::cerr << "ERROR: Could not load texture image file '" << _tex_infos[tex_index]->filename << "'.\n";
        _tex_infos[tex_index]->width = _tex_infos[tex_index]->height = 0;
    }
}

#endif