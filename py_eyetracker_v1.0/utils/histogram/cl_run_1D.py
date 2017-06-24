import pyopencl as cl
import numpy as np
import math
from skimage import img_as_ubyte


class CL_hist_1D:
    """
    Just a wrapper around all the opencl boilerplate code
    """

    def __init__(self, direction="x"):
        self.direction = direction
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)

    def load_program(self, program_path="cl_kernels/lsh_1D_%s.cl"):
        with open(program_path % self.direction, "r") as fp:
            self.program = cl.Program(self.context, fp.read()).build()

    def compute(self, alpha, init_hist, init_norm=None):
        width, height, num_bins = np.shape(init_hist)

        left_histogram = np.copy(init_hist).reshape((np.size(init_hist),)).astype(np.float32)
        right_histogram = np.copy(init_hist).reshape((np.size(init_hist),)).astype(np.float32)
        if init_norm is None:
            left_normalization = np.ones_like(left_histogram).astype(np.float32)
            right_normalization = np.ones_like(right_histogram).astype(np.float32)
        else:
            left_normalization = np.copy(init_norm).reshape(np.shape(left_histogram)).astype(np.float32)
            right_normalization = np.copy(init_norm).reshape(np.shape(right_histogram)).astype(np.float32)

        mf = cl.mem_flags
        self.left_hist_buf = cl.Buffer(self.context, mf.READ_WRITE, size=left_histogram.nbytes)
        self.right_hist_buf = cl.Buffer(self.context, mf.READ_WRITE, size=right_histogram.nbytes)
        self.left_norm_buf = cl.Buffer(self.context, mf.READ_WRITE, size=left_normalization.nbytes)
        self.right_norm_buf = cl.Buffer(self.context, mf.READ_WRITE, size=right_normalization.nbytes)

        kernel = getattr(self.program, "lsh_1D_%s" % self.direction)
        kernel.set_scalar_arg_dtypes([np.uintc, np.uintc, np.ubyte, np.float32] + [None] * 4)
        kernel.set_arg(0, np.uintc(width))
        kernel.set_arg(1, np.uintc(height))
        kernel.set_arg(2, np.ubyte(num_bins))
        kernel.set_arg(3, np.float32(alpha))
        kernel.set_arg(4, self.left_hist_buf)
        kernel.set_arg(5, self.right_hist_buf)
        kernel.set_arg(6, self.left_norm_buf)
        kernel.set_arg(7, self.right_norm_buf)

        work_items = (height,)
        if self.direction == "y":
            work_items = (width,)

        cl.enqueue_write_buffer(self.queue, self.left_hist_buf, left_histogram).wait()
        cl.enqueue_write_buffer(self.queue, self.right_hist_buf, right_histogram).wait()
        cl.enqueue_write_buffer(self.queue, self.left_norm_buf, left_normalization).wait()
        cl.enqueue_write_buffer(self.queue, self.right_norm_buf, right_normalization).wait()

        cl.enqueue_nd_range_kernel(self.queue, kernel, work_items, None).wait()

        cl.enqueue_read_buffer(self.queue, self.left_hist_buf, left_histogram).wait()
        cl.enqueue_read_buffer(self.queue, self.right_hist_buf, right_histogram).wait()
        cl.enqueue_read_buffer(self.queue, self.left_norm_buf, left_normalization).wait()
        cl.enqueue_read_buffer(self.queue, self.right_norm_buf, right_normalization).wait()

        return (
            np.reshape(left_histogram, np.shape(init_hist)),
            np.reshape(right_histogram, np.shape(init_hist)),
            np.reshape(left_normalization, np.shape(init_hist)),
            np.reshape(right_normalization, np.shape(init_hist)),
        )




