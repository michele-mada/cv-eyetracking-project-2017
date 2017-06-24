
#define COLOR_MAX 255

#define XCOORD_IN(id) ((id) % image_width)
#define YCOORD_IN(id) ((id) / image_width)
#define PIXELID_IN(x, y) ((x) + ((y) * image_width))
#define PIXELID_OUT(x, y, bin) ((bin) + ((x) * nbins) + ((y) * image_width * nbins))


// Use one work item per pixel
 __kernel void IIF(
    unsigned int image_width,
    unsigned int image_height,
    uchar nbins,
    float k,
    __global float* image,
    __global float* hist,
    __global float* transform)
{
    size_t gid = get_global_id(0);
    float intensity = image[gid];
    
    int bin_number = floor(intensity * nbins);
    
    unsigned int x = XCOORD_IN(gid);
    unsigned int y = YCOORD_IN(gid);
    
    float intensity_adjusted = k * intensity * 100;
    if (intensity_adjusted < k) intensity_adjusted = k;
    float intensity_adjusted_2 = intensity_adjusted * intensity_adjusted;
    
    float accum = 0.0f;
    
    for (int b = 0; b < nbins; b++) {
        accum += exp( -(                                \
                ((b - bin_number)*(b - bin_number))     \
                                /                       \
                (2 * intensity_adjusted_2)              \
            ) )                                         \
        * hist[PIXELID_OUT(x, y, b)];
    }
    
    transform[PIXELID_IN(x, y)] = accum;
    
}
