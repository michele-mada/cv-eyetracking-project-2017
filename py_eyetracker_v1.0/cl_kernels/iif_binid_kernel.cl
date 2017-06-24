
#define COLOR_MAX 255

#define XCOORD_IN(id) ((id) % image_width)
#define YCOORD_IN(id) ((id) / image_width)
#define PIXELID_IN(x, y) ((x) + ((y) * image_width))
#define PIXELID_OUT(x, y, bin) ((bin) + ((x) * nbins) + ((y) * image_height * nbins))


// Use one work item per pixel
 __kernel void iif_binid(
    unsigned int image_width,
    unsigned int image_height,
    uchar nbins,
    __global float* image,
    __global float* output_binid)
{
    size_t gid = get_global_id(0);
    float intensity = image[gid];
    
    uchar bin_number = floor(intensity * nbins);
    
    for (size_t b = 0; b < nbins; b++) {
        output_binid[PIXELID_OUT(XCOORD_IN(gid), 
                                 YCOORD_IN(gid), 
                                 b)
            ] = (float)bin_number;
    }
    
}
