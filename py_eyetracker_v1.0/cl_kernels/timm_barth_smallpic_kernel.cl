
#define YCOORD(id) ((id) / image_height)
#define XCOORD(id) ((id) % image_height)
#define PIXELID(x, y) ((y) + ((x) * image_width))


__kernel void timm_and_barth(
    int image_width,
    int image_height,
    int local_radius,
    __global float* x_derivatives,
    __global float* y_derivatives,
    __global float* inverse_image,
    __global float* output_image)
{
    size_t gid = get_global_id(0);
    size_t numpixels = image_width * image_height;

    int center_x = XCOORD(gid);
    int center_y = YCOORD(gid);

    float accumul = .0f;
    
    for (size_t pixelid = 0; pixelid < numpixels; pixelid++) {
        int x = XCOORD(pixelid);
        int y = YCOORD(pixelid);
        float displac_x = (float)(x - center_x);
        float displac_y = (float)(y - center_y);
        float displac_norm = sqrt(displac_x * displac_x + displac_y * displac_y) + 0.00001;

        displac_x = displac_x / displac_norm;
        displac_y = displac_y / displac_norm;

        float dotprod = displac_x * x_derivatives[pixelid] + displac_y * y_derivatives[pixelid];
        if (dotprod > 0) {
            accumul += (dotprod * dotprod) * inverse_image[gid];
        }
    }

    output_image[gid] = accumul / (float)numpixels;
}
