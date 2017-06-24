
#define COLOR_MAX 255

#define PIXELID(x, y, bin) ((bin) + ((x) * nbins) + ((y) * image_width * nbins))


// Use one work item per row
 __kernel void lsh_1D_x(
    unsigned int image_width,
    unsigned int image_height,
    uchar nbins,
    float alpha,
    __global float* histogram_left,
    __global float* histogram_right,
    __global float* normalization_left,
    __global float* normalization_right)
{
    size_t rowid = get_global_id(0);
    
    for (uchar bin = 0; bin < nbins; bin++) {
        
        for (int x = 1; x < image_width; x++) {
            size_t pixelid = PIXELID(x, rowid, bin);
            size_t prev_pixelid = PIXELID(x-1, rowid, bin);
            
            histogram_left[pixelid] = histogram_left[pixelid] + alpha * histogram_left[prev_pixelid];
            normalization_left[pixelid] = normalization_left[pixelid] + alpha * normalization_left[prev_pixelid];
        }
        
        for (int x2 = 0; x2 < image_width-1; x2++) {
            int x = (image_width-2) - x2;
            size_t pixelid = PIXELID(x, rowid, bin);
            size_t next_pixelid = PIXELID(x+1, rowid, bin);
            
            histogram_right[pixelid] = histogram_right[pixelid] + alpha * histogram_right[next_pixelid];
            normalization_right[pixelid] = normalization_right[pixelid] + alpha * normalization_right[next_pixelid];
        }
        
    }
    
}
