import pyopencl as cl
import numpy as np
import math
from skimage import img_as_ubyte


class CL_Q:
    """
    Just a wrapper around all the opencl boilerplate code
    """

    def __init__(self):
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)

    def load_program(self, program_path="cl_kernels/lsh_Q_kernel.cl"):
        with open(program_path, "r") as fp:
            self.program = cl.Program(self.context, fp.read()).build()

    def compute(self, image, num_bins):
        width, height = np.shape(image)
        numpixels = width * height

        image = np.reshape(image, (numpixels,)).astype(np.float32)
        result = np.zeros((numpixels * num_bins, ), dtype=np.float32)

        mf = cl.mem_flags
        self.buf_image = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image)
        self.output_buf = cl.Buffer(self.context, mf.READ_WRITE, result.nbytes)

        kernel = self.program.lsh_Q
        kernel.set_scalar_arg_dtypes([np.uintc, np.uintc, np.ubyte] + [None] * 2)
        kernel.set_arg(0, np.uintc(width))
        kernel.set_arg(1, np.uintc(height))
        kernel.set_arg(2, np.ubyte(num_bins))
        kernel.set_arg(3, self.buf_image)
        kernel.set_arg(4, self.output_buf)

        cl.enqueue_nd_range_kernel(self.queue, kernel, image.shape, None).wait()

        cl.enqueue_read_buffer(self.queue, self.output_buf, result).wait()
        return np.reshape(result, (width, height, num_bins)).astype(np.float32)




