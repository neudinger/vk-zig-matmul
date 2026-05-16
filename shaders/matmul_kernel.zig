const max_elements = 1024 * 1024;

const Buffer = extern struct {
    data: [max_elements]f32,
};

const PushConstants = extern struct {
    m: u32,
    n: u32,
    k: u32,
    a_stride: u32,
    b_stride: u32,
    c_stride: u32,
};

extern var a_buffer: Buffer addrspace(.storage_buffer);
extern var b_buffer: Buffer addrspace(.storage_buffer);
extern var c_buffer: Buffer addrspace(.storage_buffer);
extern var push_constants: PushConstants addrspace(.push_constant);
extern const global_invocation_id: @Vector(3, u32) addrspace(.input);

export fn main() callconv(.spirv_kernel) void {
    const col = global_invocation_id[0];
    const row = global_invocation_id[1];
    const pc = push_constants;
    if (row >= pc.m or col >= pc.n) return;

    var sum: f32 = 0;
    var i: u32 = 0;
    while (i < pc.k) : (i += 1) {
        sum += a_buffer.data[row * pc.a_stride + i] * b_buffer.data[i * pc.b_stride + col];
    }
    c_buffer.data[row * pc.c_stride + col] = sum;
}
