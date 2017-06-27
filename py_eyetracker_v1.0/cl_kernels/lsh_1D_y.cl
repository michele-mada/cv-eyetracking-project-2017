
#define COLOR_MAX 255
#define LEFT 0
#define RIGHT 1

#define PIXELID(x, y, bin) ((bin) + ((x) * nbins) + ((y) * image_width * nbins))

#define BUFFERID_Y(y, isright) ((y) + (image_height * (isright)) + (2 * image_height * worker_id))


// Use one work item per column
 __kernel void lsh_1D_y(
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
    size_t colid = get_global_id(0);
    size_t worker_id = get_local_id(0);
    
    for (int bin = 0; bin < nbins; bin++) {
        
        histogram_buffer[BUFFERID_Y(0, LEFT)] = Q[PIXELID(colid, 0, bin)];
        normalization_buffer[BUFFERID_Y(0, LEFT)] = F[PIXELID(colid, 0, bin)];
        
        for (int y = 1; y < image_height; y++) {
            size_t pixelid = PIXELID(colid, y, bin);
            
            histogram_buffer[BUFFERID_Y(y, LEFT)] = Q[pixelid] + alpha * histogram_buffer[BUFFERID_Y(y-1, LEFT)];
            normalization_buffer[BUFFERID_Y(y, LEFT)] = F[pixelid] + alpha * normalization_buffer[BUFFERID_Y(y-1, LEFT)];
        }
        
        histogram_buffer[BUFFERID_Y(image_height-1, RIGHT)] = Q[PIXELID(colid, image_height-1, bin)];
        normalization_buffer[BUFFERID_Y(image_height-1, RIGHT)] = F[PIXELID(colid, image_height-1, bin)];
        
        for (int y2 = 0; y2 < image_height-1; y2++) {
            int y = (image_height-2) - y2;
            size_t pixelid = PIXELID(colid, y, bin);
            
            histogram_buffer[BUFFERID_Y(y, RIGHT)] = Q[pixelid] + alpha * histogram_buffer[BUFFERID_Y(y+1, RIGHT)];
            normalization_buffer[BUFFERID_Y(y, RIGHT)] = F[pixelid] + alpha * normalization_buffer[BUFFERID_Y(y+1, RIGHT)];
        }
        
        for (int y = 0; y < image_height; y++) {
            size_t pixelid = PIXELID(colid, y, bin);
            
            histogram_out[pixelid] = histogram_buffer[BUFFERID_Y(y, LEFT)] + histogram_buffer[BUFFERID_Y(y, RIGHT)] - Q[pixelid];
            normalization_out[pixelid] = normalization_buffer[BUFFERID_Y(y, LEFT)] + normalization_buffer[BUFFERID_Y(y, RIGHT)] - 1.0;
        }
    }
    
}
