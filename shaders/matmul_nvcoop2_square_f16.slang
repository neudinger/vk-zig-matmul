using namespace linalg;

#ifndef K_UNROLL
#define K_UNROLL 1
#endif

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

AMat loadA(TensorLayout<2, CoopMatClampMode.Undefined> layout_a, uint tile_row, uint kk)
{
    let a_slice = layout_a.Slice(tile_row, 128, kk, 16);
    return AMat::Load(a_buffer, 0, a_slice);
}

BMat loadB(TensorLayout<2, CoopMatClampMode.Undefined> layout_b, uint tile_col, uint kk)
{
    let b_slice = layout_b.Slice(kk, 16, tile_col, 128);
    return BMat::Load(b_buffer, 0, b_slice);
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

    uint kk = 0;

#if K_UNROLL >= 8
    for (; kk + 128 <= push_constants.k; kk += 128)
    {
        let a0 = loadA(layout_a, tile_row, kk + 0);
        let b0 = loadB(layout_b, tile_col, kk + 0);
        let a1 = loadA(layout_a, tile_row, kk + 16);
        let b1 = loadB(layout_b, tile_col, kk + 16);
        let a2 = loadA(layout_a, tile_row, kk + 32);
        let b2 = loadB(layout_b, tile_col, kk + 32);
        let a3 = loadA(layout_a, tile_row, kk + 48);
        let b3 = loadB(layout_b, tile_col, kk + 48);
        let a4 = loadA(layout_a, tile_row, kk + 64);
        let b4 = loadB(layout_b, tile_col, kk + 64);
        let a5 = loadA(layout_a, tile_row, kk + 80);
        let b5 = loadB(layout_b, tile_col, kk + 80);
        let a6 = loadA(layout_a, tile_row, kk + 96);
        let b6 = loadB(layout_b, tile_col, kk + 96);
        let a7 = loadA(layout_a, tile_row, kk + 112);
        let b7 = loadB(layout_b, tile_col, kk + 112);
        result = coopMatMulAdd<float, false>(a0, b0, result);
        result = coopMatMulAdd<float, false>(a1, b1, result);
        result = coopMatMulAdd<float, false>(a2, b2, result);
        result = coopMatMulAdd<float, false>(a3, b3, result);
        result = coopMatMulAdd<float, false>(a4, b4, result);
        result = coopMatMulAdd<float, false>(a5, b5, result);
        result = coopMatMulAdd<float, false>(a6, b6, result);
        result = coopMatMulAdd<float, false>(a7, b7, result);
    }
#elif K_UNROLL >= 4
    for (; kk + 64 <= push_constants.k; kk += 64)
    {
        let a0 = loadA(layout_a, tile_row, kk + 0);
        let b0 = loadB(layout_b, tile_col, kk + 0);
        let a1 = loadA(layout_a, tile_row, kk + 16);
        let b1 = loadB(layout_b, tile_col, kk + 16);
        let a2 = loadA(layout_a, tile_row, kk + 32);
        let b2 = loadB(layout_b, tile_col, kk + 32);
        let a3 = loadA(layout_a, tile_row, kk + 48);
        let b3 = loadB(layout_b, tile_col, kk + 48);
        result = coopMatMulAdd<float, false>(a0, b0, result);
        result = coopMatMulAdd<float, false>(a1, b1, result);
        result = coopMatMulAdd<float, false>(a2, b2, result);
        result = coopMatMulAdd<float, false>(a3, b3, result);
    }
#elif K_UNROLL >= 2
    for (; kk + 32 <= push_constants.k; kk += 32)
    {
        let a0 = loadA(layout_a, tile_row, kk + 0);
        let b0 = loadB(layout_b, tile_col, kk + 0);
        let a1 = loadA(layout_a, tile_row, kk + 16);
        let b1 = loadB(layout_b, tile_col, kk + 16);
        result = coopMatMulAdd<float, false>(a0, b0, result);
        result = coopMatMulAdd<float, false>(a1, b1, result);
    }
#endif

    for (; kk < push_constants.k; kk += 16)
    {
        result = coopMatMulAdd<float, false>(loadA(layout_a, tile_row, kk), loadB(layout_b, tile_col, kk), result);
    }

    let c_slice = layout_c.Slice(tile_row, 128, tile_col, 128);
    result.Store(c_buffer, 0, c_slice);
}
