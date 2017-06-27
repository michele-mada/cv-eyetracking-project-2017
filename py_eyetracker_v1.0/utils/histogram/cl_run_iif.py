import pyopencl as cl
import numpy as np
import math
from skimage import img_as_ubyte


class CL_IIF_BINID:
    """
    Just a wrapper around all the opencl boilerplate code
    """

    def __init__(self):
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)

    def load_program(self, program_path="cl_kernels/iif_binid_kernel.cl"):
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

        kernel = self.program.iif_binid
        kernel.set_scalar_arg_dtypes([np.uintc, np.uintc, np.ubyte] + [None] * 2)
        kernel.set_arg(0, np.uintc(width))
        kernel.set_arg(1, np.uintc(height))
        kernel.set_arg(2, np.ubyte(num_bins))
        kernel.set_arg(3, self.buf_image)
        kernel.set_arg(4, self.output_buf)

        cl.enqueue_nd_range_kernel(self.queue, kernel, image.shape, None).wait()

        cl.enqueue_read_buffer(self.queue, self.output_buf, result).wait()
        return np.reshape(result, (width, height, num_bins)).astype(np.float32)


class CL_IIF:
    """
    Just a wrapper around all the opencl boilerplate code
    """

    def __init__(self, num_bins=32):
        self.num_bins = num_bins
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)

    def load_program(self, program_path="cl_kernels/iif_kernel.cl"):
        with open(program_path, "r") as fp:
            self.source = "#define NBINS %d\n" % self.num_bins + fp.read()
            self.program = cl.Program(self.context, self.source).build()

    def compute(self, floatimage, histogram, k):
        width, height, nbins = np.shape(histogram)
        numpixels = width * height

        image_linear = np.reshape(floatimage, (numpixels,)).astype(np.float32)
        histogram_linear = np.reshape(histogram, (np.size(histogram),)).astype(np.float32)
        transform = np.zeros_like(image_linear).astype(np.float32)

        mf = cl.mem_flags
        self.buf_image = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image_linear)
        self.buf_histogram = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=histogram_linear)
        self.output_buf = cl.Buffer(self.context, mf.READ_WRITE, transform.nbytes)

        kernel = self.program.IIF
        kernel.set_scalar_arg_dtypes([np.uintc, np.uintc, np.float32] + [None] * 3)
        kernel.set_arg(0, np.uintc(width))
        kernel.set_arg(1, np.uintc(height))
        kernel.set_arg(2, np.float32(k))
        kernel.set_arg(3, self.buf_image)
        kernel.set_arg(4, self.buf_histogram)
        kernel.set_arg(5, self.output_buf)

        cl.enqueue_nd_range_kernel(self.queue, kernel, image_linear.shape, None).wait()

        cl.enqueue_read_buffer(self.queue, self.output_buf, transform).wait()
        return np.reshape(transform, (width, height)).astype(np.float)




