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

    def get_max_local_mem(self):
        devices = self.context.get_info(cl.context_info.DEVICES)
        device = devices[0]
        return device.local_mem_size

    def compute(self, linearized_Q, linearized_F, alpha, shapetuple):
        height, width, num_bins = shapetuple

        mf = cl.mem_flags
        self.buffer_Q = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=linearized_Q)
        self.buffer_F = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=linearized_F)
        self.hist_out_buf = cl.Buffer(self.context, mf.READ_WRITE, size=linearized_Q.nbytes)
        self.norm_out_buf = cl.Buffer(self.context, mf.READ_WRITE, size=linearized_Q.nbytes)

        work_items = height
        if self.direction == "y":
            work_items = width

        direction_size = width
        if self.direction == "y":
            direction_size = height

        half_required_local_mem = 4 * 2 * direction_size
        n_workers = self.get_max_local_mem() // (half_required_local_mem * 2)

        while work_items % n_workers != 0:
            n_workers -= 1

        #print("n workers: ", n_workers)

        self.local_mem_hist = cl.LocalMemory(half_required_local_mem * n_workers)
        self.local_mem_norm = cl.LocalMemory(half_required_local_mem * n_workers)

        kernel = getattr(self.program, "lsh_1D_%s" % self.direction)
        kernel.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, np.float32] + [None] * 6)
        kernel.set_arg(0, np.int32(width))
        kernel.set_arg(1, np.int32(height))
        kernel.set_arg(2, np.int32(num_bins))
        kernel.set_arg(3, np.float32(alpha))
        kernel.set_arg(4, self.buffer_Q)
        kernel.set_arg(5, self.buffer_F)
        kernel.set_arg(6, self.local_mem_hist)
        kernel.set_arg(7, self.local_mem_norm)
        kernel.set_arg(8, self.hist_out_buf)
        kernel.set_arg(9, self.norm_out_buf)

        cl.enqueue_nd_range_kernel(self.queue, kernel, (work_items,), (n_workers,)).wait()

        hist_result = np.empty_like(linearized_Q, dtype=np.float32)
        norm_result = np.empty_like(linearized_Q, dtype=np.float32)

        cl.enqueue_read_buffer(self.queue, self.hist_out_buf, hist_result).wait()
        cl.enqueue_read_buffer(self.queue, self.norm_out_buf, norm_result).wait()

        return (
            hist_result,
            norm_result
        )




