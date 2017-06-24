
#define XCOORD(id) ((id) % image_width)
#define YCOORD(id) ((id) / image_width)
#define PIXELID(x, y) ((x) + ((y) * image_width))


__kernel void timm_and_barth(
    unsigned int image_width,
    unsigned int image_height,
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

    for (size_t point_x = (center_x - local_radius > 0 ? center_x - local_radius : 0);
         point_x < image_width && point_x < center_x + local_radius;
         point_x++
        ) {
            for (size_t point_y = (center_y - local_radius > 0 ? center_y - local_radius : 0);
                 point_y < image_height && point_y < center_y + local_radius;
                 point_y++
                ) {
                    size_t pixelid = PIXELID(point_x, point_y);
                    float displac_x = (float)(point_x - center_x);
                    float displac_y = (float)(point_y - center_y);
                    float displac_norm = sqrt(displac_x * displac_x + displac_y * displac_y) + 0.00001;
                    if (displac_norm > local_radius) {continue;}

                    displac_x = displac_x / displac_norm;
                    displac_y = displac_y / displac_norm;

                    float dotprod = displac_x * x_derivatives[pixelid] + displac_y * y_derivatives[pixelid];
                    if (dotprod > 0) {
                        accumul += (dotprod * dotprod) * inverse_image[gid];
                    }
                }
        }

    output_image[gid] = accumul / (float)numpixels;
}
