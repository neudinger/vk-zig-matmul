using namespace linalg;

struct PushConstants
{
    uint m;
    uint n;
    uint k;
    uint a_stride;
    uint b_stride;
    uint c_stride;
};

[[vk::binding(0, 0)]] StructuredBuffer<half> a_buffer;
[[vk::binding(1, 0)]] StructuredBuffer<half> b_buffer;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> c_buffer;
[[vk::push_constant]] PushConstants push_constants;

typealias AccMat = CoopMat<float, MemoryScope.Workgroup, 128, 128, CoopMatMatrixUse.MatrixAccumulator>;
typealias AMat = CoopMat<half, MemoryScope.Workgroup, 128, 16, CoopMatMatrixUse.MatrixA>;
typealias BMat = CoopMat<half, MemoryScope.Workgroup, 16, 128, CoopMatMatrixUse.MatrixB>;

TensorLayout<2, CoopMatClampMode.Undefined> tensorLayout(uint dim0, uint dim1, uint stride0, uint stride1)
{
    TensorLayout<2, CoopMatClampMode.Undefined> layout;
    return layout.Dimension(dim0, dim1).Stride(stride0, stride1);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 workgroup_id : SV_GroupID)
{
    uint tile_row = workgroup_id.y * 128;
    uint tile_col = workgroup_id.x * 128;

    AccMat result;
    result.fill(0.0f);

    let layout_a = tensorLayout(push_constants.m, push_constants.k, push_constants.a_stride, 1);
    let layout_b = tensorLayout(push_constants.k, push_constants.n, push_constants.b_stride, 1);
    let layout_c = tensorLayout(push_constants.m, push_constants.n, push_constants.c_stride, 1);

    for (uint kk = 0; kk < push_constants.k; kk += 16)
    {
        let a_slice = layout_a.Slice(tile_row, 128, kk, 16);
        let b_slice = layout_b.Slice(kk, 16, tile_col, 128);
        let a = AMat::Load(a_buffer, 0, a_slice);
        let b = BMat::Load(b_buffer, 0, b_slice);
        result = coopMatMulAdd<float, false>(a, b, result);
    }

    let c_slice = layout_c.Slice(tile_row, 128, tile_col, 128);
    result.Store(c_buffer, 0, c_slice);
}
