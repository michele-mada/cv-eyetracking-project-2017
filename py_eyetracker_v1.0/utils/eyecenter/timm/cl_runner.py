import pyopencl as cl
import numpy as np

from skimage.filters import scharr_h, scharr_v, gaussian


class CLTimmBarth:

    def __init__(self, precomputation=lambda i: i):
        self.precomputation = precomputation
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)

    def load_program(self, program_path="timm_barth_kernel.cl"):
        with open(program_path, "r") as fp:
            self.program = cl.Program(self.context, fp.read()).build()

    def host_side_compute(self, floatimage):
        return self.precomputation(floatimage)

    def compute(self, floatimage, locality):
        width, height = np.shape(floatimage)
        numpixels = width * height
        x_gradient, y_gradient, inverse = self.host_side_compute(floatimage)
        x_gradient = np.reshape(x_gradient, (numpixels,)).astype(np.float32)
        y_gradient = np.reshape(y_gradient, (numpixels,)).astype(np.float32)
        inverse = np.reshape(inverse, (numpixels,)).astype(np.float32)

        mf = cl.mem_flags
        self.buf_x_grad = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_gradient)
        self.buf_y_grad = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_gradient)
        self.buf_inverse = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inverse)
        self.output_buf = cl.Buffer(self.context, mf.READ_WRITE, inverse.nbytes)

        kernel = self.program.timm_and_barth
        kernel.set_scalar_arg_dtypes([np.uintc, np.uintc, np.intc] + [None] * 4)
        kernel.set_arg(0, np.uintc(width))
        kernel.set_arg(1, np.uintc(height))
        kernel.set_arg(2, np.intc(locality))
        kernel.set_arg(3, self.buf_x_grad)
        kernel.set_arg(4, self.buf_y_grad)
        kernel.set_arg(5, self.buf_inverse)
        kernel.set_arg(6, self.output_buf)


        cl.enqueue_nd_range_kernel(self.queue, kernel, inverse.shape, None).wait()

        result = np.empty_like(inverse)
        cl.enqueue_read_buffer(self.queue, self.output_buf, result).wait()
        return np.reshape(result, (width, height)).astype(np.float)




