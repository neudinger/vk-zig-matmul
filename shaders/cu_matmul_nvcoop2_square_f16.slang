// CUDA-target Slang cooperative-matrix matmul shader.
//
// Computes:
//   C = A * B
//
// A: m x k, row-major half
// B: k x n, row-major half
// C: m x n, row-major float
//
// CUDA-target variant:
//   - No TensorLayout / TensorView.
//   - Uses ordinary CoopMat::Load/Store with RowMajor layout.
//   - Uses CUDA-supported subgroup/warp-level m16n16k16 fragments.
//   - One workgroup has 32 threads.
//   - One workgroup computes:
//       N_TILES=1: one 16x16 C tile
//       N_TILES=2: two adjacent 16x16 C tiles, i.e. 16x32
//
// Host dispatch for this CUDA version should be:
//   x = ceil(n / (16 * N_TILES))
//   y = ceil(m / 16)
//
// Host constraints, matching the original shader style:
//   - k must be a multiple of 16.
//   - m and n should be padded or dispatch-clamped to multiples of the tile size.
//   - a_stride and b_stride should be multiples of 8 half elements.
//   - c_stride should be a multiple of 4 float elements.

using namespace linalg;

#ifndef K_UNROLL
#define K_UNROLL 1
#endif

#ifndef N_TILES
#define N_TILES 1
#endif

// static_assert(N_TILES == 1 || N_TILES == 2);

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

// CUDA CoopMat path:
//   A:   16x16 half
//   B:   16x16 half
//   Acc: 16x16 float
//
// This is the CUDA/WMMA-friendly m16n16k16 shape.
typealias AccMat = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;
typealias AMat    = CoopMat<half,  MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias BMat    = CoopMat<half,  MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;

AMat loadA(uint tile_row, uint kk)
{
    uint element = tile_row * push_constants.a_stride + kk;

    return AMat::Load<CoopMatMatrixLayout.RowMajor>(
        a_buffer,
        element,
        push_constants.a_stride);
}

BMat loadB(uint tile_col, uint kk)
{
    uint element = kk * push_constants.b_stride + tile_col;

    return BMat::Load<CoopMatMatrixLayout.RowMajor>(
        b_buffer,
        element,
        push_constants.b_stride);
}

void storeC(AccMat value, uint tile_row, uint tile_col)
{
    uint element = tile_row * push_constants.c_stride + tile_col;

    value.Store<CoopMatMatrixLayout.RowMajor>(
        c_buffer,
        element,
        push_constants.c_stride);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 workgroup_id : SV_GroupID)
{
    uint tile_row = workgroup_id.y * 16;
    uint tile_col = workgroup_id.x * (16 * N_TILES);

    AccMat result;
    result.fill(0.0f);

#if N_TILES >= 2
    AccMat result1;
    result1.fill(0.0f);
#endif

    uint kk = 0;

#if K_UNROLL >= 8
    for (; kk + 128 <= push_constants.k; kk += 128)
    {
        let a0 = loadA(tile_row, kk + 0);
        result = coopMatMulAdd<float, false>(a0, loadB(tile_col, kk + 0), result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a0, loadB(tile_col + 16, kk + 0), result1);
#endif

        let a1 = loadA(tile_row, kk + 16);
        result = coopMatMulAdd<float, false>(a1, loadB(tile_col, kk + 16), result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a1, loadB(tile_col + 16, kk + 16), result1);
#endif

        let a2 = loadA(tile_row, kk + 32);
        result = coopMatMulAdd<float, false>(a2, loadB(tile_col, kk + 32), result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a2, loadB(tile_col + 16, kk + 32), result1);
#endif

        let a3 = loadA(tile_row, kk + 48);
        result = coopMatMulAdd<float, false>(a3, loadB(tile_col, kk + 48), result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a3, loadB(tile_col + 16, kk + 48), result1);
#endif

        let a4 = loadA(tile_row, kk + 64);
        result = coopMatMulAdd<float, false>(a4, loadB(tile_col, kk + 64), result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a4, loadB(tile_col + 16, kk + 64), result1);
#endif

        let a5 = loadA(tile_row, kk + 80);
        result = coopMatMulAdd<float, false>(a5, loadB(tile_col, kk + 80), result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a5, loadB(tile_col + 16, kk + 80), result1);
#endif

        let a6 = loadA(tile_row, kk + 96);
        result = coopMatMulAdd<float, false>(a6, loadB(tile_col, kk + 96), result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a6, loadB(tile_col + 16, kk + 96), result1);
#endif

        let a7 = loadA(tile_row, kk + 112);
        result = coopMatMulAdd<float, false>(a7, loadB(tile_col, kk + 112), result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a7, loadB(tile_col + 16, kk + 112), result1);
#endif
    }
#elif K_UNROLL >= 4
    for (; kk + 64 <= push_constants.k; kk += 64)
    {
        let a0 = loadA(tile_row, kk + 0);
        result = coopMatMulAdd<float, false>(a0, loadB(tile_col, kk + 0), result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a0, loadB(tile_col + 16, kk + 0), result1);
#endif

        let a1 = loadA(tile_row, kk + 16);
        result = coopMatMulAdd<float, false>(a1, loadB(tile_col, kk + 16), result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a1, loadB(tile_col + 16, kk + 16), result1);
#endif

        let a2 = loadA(tile_row, kk + 32);
        result = coopMatMulAdd<float, false>(a2, loadB(tile_col, kk + 32), result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a2, loadB(tile_col + 16, kk + 32), result1);
#endif

        let a3 = loadA(tile_row, kk + 48);
        result = coopMatMulAdd<float, false>(a3, loadB(tile_col, kk + 48), result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a3, loadB(tile_col + 16, kk + 48), result1);
#endif
    }
#elif K_UNROLL >= 2
    for (; kk + 32 <= push_constants.k; kk += 32)
    {
        let a0 = loadA(tile_row, kk + 0);
        result = coopMatMulAdd<float, false>(a0, loadB(tile_col, kk + 0), result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a0, loadB(tile_col + 16, kk + 0), result1);
#endif

        let a1 = loadA(tile_row, kk + 16);
        result = coopMatMulAdd<float, false>(a1, loadB(tile_col, kk + 16), result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a1, loadB(tile_col + 16, kk + 16), result1);
#endif
    }
#endif

    for (; kk < push_constants.k; kk += 16)
    {
        let a_tail = loadA(tile_row, kk);

        result = coopMatMulAdd<float, false>(
            a_tail,
            loadB(tile_col, kk),
            result);

#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(
            a_tail,
            loadB(tile_col + 16, kk),
            result1);
#endif
    }

    storeC(result, tile_row, tile_col);

#if N_TILES >= 2
    storeC(result1, tile_row, tile_col + 16);
#endif
}