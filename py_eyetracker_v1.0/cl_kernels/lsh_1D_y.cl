
#define COLOR_MAX 255

#define PIXELID(x, y, bin) ((bin) + ((x) * nbins) + ((y) * image_width * nbins))


// Use one work item per column
 __kernel void lsh_1D_y(
    unsigned int image_width,
    unsigned int image_height,
    uchar nbins,
    float alpha,
    __global float* histogram_left,
    __global float* histogram_right,
    __global float* normalization_left,
    __global float* normalization_right)
{
    size_t colid = get_global_id(0);
    
    for (uchar bin = 0; bin < nbins; bin++) {
        
        for (int y = 1; y < image_height; y++) {
            size_t pixelid = PIXELID(colid, y, bin);
            size_t prev_pixelid = PIXELID(colid, y-1, bin);
            
            histogram_left[pixelid] = histogram_left[pixelid] + alpha * histogram_left[prev_pixelid];
            normalization_left[pixelid] = normalization_left[pixelid] + alpha * normalization_left[prev_pixelid];
        }
        
        for (int y2 = 0; y2 < image_height-1; y2++) {
            int y = (image_height-2) - y2;
            size_t pixelid = PIXELID(colid, y, bin);
            size_t next_pixelid = PIXELID(colid, y+1, bin);
            
            histogram_right[pixelid] = histogram_right[pixelid] + alpha * histogram_right[next_pixelid];
            normalization_right[pixelid] = normalization_right[pixelid] + alpha * normalization_right[next_pixelid];
        }
        
    }
    
}
