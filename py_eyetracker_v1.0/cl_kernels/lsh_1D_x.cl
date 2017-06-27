
#define COLOR_MAX 255
#define LEFT 0
#define RIGHT 1

#define PIXELID(x, y, bin) ((bin) + ((x) * nbins) + ((y) * image_width * nbins))

#define BUFFERID_X(x, isright) ((x) + (image_width * (isright)) + (2 * image_width * worker_id))


// Use one work item per row
 __kernel void lsh_1D_x(
    int image_width,
    int image_height,
    int nbins,
    float alpha,
    __global float* Q,
    __global float* F,
    __local float* histogram_buffer,
    __local float* normalization_buffer,
    __global float* histogram_out, 
    __global float* normalization_out)
{
    size_t rowid = get_global_id(0);
    size_t worker_id = get_local_id(0);
    
    for (int bin = 0; bin < nbins; bin++) {
        
        histogram_buffer[BUFFERID_X(0, LEFT)] = Q[PIXELID(0, rowid, bin)];
        normalization_buffer[BUFFERID_X(0, LEFT)] = F[PIXELID(0, rowid, bin)];
        
        for (int x = 1; x < image_width; x++) {
            size_t pixelid = PIXELID(x, rowid, bin);
            
            histogram_buffer[BUFFERID_X(x, LEFT)] = Q[pixelid] + alpha * histogram_buffer[BUFFERID_X(x-1, LEFT)];
            normalization_buffer[BUFFERID_X(x, LEFT)] = F[pixelid] + alpha * normalization_buffer[BUFFERID_X(x-1, LEFT)];
        }
        
        histogram_buffer[BUFFERID_X(image_width-1, RIGHT)] = Q[PIXELID(image_width-1, rowid, bin)];
        normalization_buffer[BUFFERID_X(image_width-1, RIGHT)] = F[PIXELID(image_width-1, rowid, bin)];
        
        for (int x2 = 0; x2 < image_width-1; x2++) {
            int x = (image_width-2) - x2;
            size_t pixelid = PIXELID(x, rowid, bin);
            
            histogram_buffer[BUFFERID_X(x, RIGHT)] = Q[pixelid] + alpha * histogram_buffer[BUFFERID_X(x+1, RIGHT)];
            normalization_buffer[BUFFERID_X(x, RIGHT)] = F[pixelid] + alpha * normalization_buffer[BUFFERID_X(x+1, RIGHT)];
        }
        
        for (int x = 0; x < image_width; x++) {
            size_t pixelid = PIXELID(x, rowid, bin);
            
            histogram_out[pixelid] = histogram_buffer[BUFFERID_X(x, LEFT)] + histogram_buffer[BUFFERID_X(x, RIGHT)] - Q[pixelid];
            normalization_out[pixelid] = normalization_buffer[BUFFERID_X(x, LEFT)] + normalization_buffer[BUFFERID_X(x, RIGHT)] - 1.0;
        }
        
    }
    
}
