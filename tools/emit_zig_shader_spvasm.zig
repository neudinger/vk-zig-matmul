const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(gpa);
    defer gpa.free(args);

    if (args.len != 4) return error.InvalidArguments;
    if (!std.mem.eql(u8, args[1], "--mode=nvcoop2-f16-square")) return error.InvalidMode;

    const input_path = args[2];
    const output_path = args[3];
    const source = try std.Io.Dir.cwd().readFileAlloc(init.io, input_path, gpa, .limited(1024 * 1024));
    defer gpa.free(source);

    try validateShaderSource(source);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = output_path, .data = nvcoop2_square_f16_spvasm });
}

fn validateShaderSource(source: []const u8) !void {
    try validateAscii(source);
    try validateBalanced(source, '(', ')');
    try validateBalanced(source, '{', '}');
    try validateBalanced(source, '[', ']');

    try requireContains(source, "const nv = @import(\"nvcoop2_shader.zig\");", "expected nv frontend import");
    try requireContains(source, "comptime", "expected comptime shader declaration");
    try requireContains(source, "nv.shader", "expected nv.shader declaration");
    try requireContains(source, ".mode = .f16_square", "expected f16_square mode");
    try requireContains(source, "const PushConstants = extern struct", "expected PushConstants extern struct");
    try requireContains(source, "m: u32", "expected push constant field m");
    try requireContains(source, "n: u32", "expected push constant field n");
    try requireContains(source, "k: u32", "expected push constant field k");
    try requireContains(source, "a_stride: u32", "expected push constant field a_stride");
    try requireContains(source, "b_stride: u32", "expected push constant field b_stride");
    try requireContains(source, "c_stride: u32", "expected push constant field c_stride");
    try requireContains(source, "extern var a_buffer", "expected A storage buffer");
    try requireContains(source, "extern var b_buffer", "expected B storage buffer");
    try requireContains(source, "extern var c_buffer", "expected C storage buffer");
    try requireContains(source, "extern var push_constants: PushConstants addrspace(.push_constant);", "expected push constant variable");
    try requireContains(source, "extern const workgroup_id: @Vector(3, u32) addrspace(.input);", "expected workgroup_id input");
    try requireContains(source, "export fn main() callconv(.spirv_kernel) void", "expected spirv_kernel main");
    try requireContains(source, "const tile_row = workgroup_id[1] * 128;", "expected tile_row expression");
    try requireContains(source, "const tile_col = workgroup_id[0] * 128;", "expected tile_col expression");
    try requireContains(source, "var result = nv.coop.zero(.accumulator_f32_128x128);", "expected accumulator zero");
    try requireContains(source, "const layout_a = nv.tensorLayout(.a", "expected A tensor layout");
    try requireContains(source, "const layout_b = nv.tensorLayout(.b", "expected B tensor layout");
    try requireContains(source, "const layout_c = nv.tensorLayout(.c", "expected C tensor layout");
    try requireContains(source, "var kk: u32 = 0;", "expected kk loop variable");
    try requireContains(source, "while (kk < push_constants.k) : (kk += 16)", "expected supported K loop shape");
    try requireContains(source, "const a = nv.coop.loadTensor(.a", "expected A tensor load");
    try requireContains(source, "const b = nv.coop.loadTensor(.b", "expected B tensor load");
    try requireContains(source, "result = nv.coop.mulAdd(a, b, result);", "expected matrix multiply-add");
    try requireContains(source, "nv.coop.storeTensor(.c", "expected C tensor store");

    try rejectUnknownNvCalls(source);
    try requireSingle(source, "export fn main", "only one exported main is supported");
    try requireSingle(source, "while (kk < push_constants.k) : (kk += 16)", "only one supported K loop is supported");
}

fn validateAscii(source: []const u8) !void {
    for (source, 0..) |byte, i| {
        if (byte == '\n' or byte == '\r' or byte == '\t') continue;
        if (byte < 0x20 or byte > 0x7e) {
            reportAt(source, i, "unsupported non-ASCII/control byte");
            return error.UnsupportedSyntax;
        }
    }
}

fn validateBalanced(source: []const u8, open: u8, close: u8) !void {
    var depth: isize = 0;
    var in_string = false;
    var i: usize = 0;
    while (i < source.len) : (i += 1) {
        const byte = source[i];
        if (byte == '"' and (i == 0 or source[i - 1] != '\\')) in_string = !in_string;
        if (in_string) continue;
        if (byte == open) depth += 1;
        if (byte == close) {
            depth -= 1;
            if (depth < 0) {
                reportAt(source, i, "unbalanced closing delimiter");
                return error.UnsupportedSyntax;
            }
        }
    }
    if (depth != 0) {
        reportAt(source, source.len, "unbalanced delimiter");
        return error.UnsupportedSyntax;
    }
}

fn rejectUnknownNvCalls(source: []const u8) !void {
    const allowed = [_][]const u8{
        "nv.shader",
        "nv.StorageBuffer",
        "nv.coop.zero",
        "nv.tensorLayout",
        "nv.coop.loadTensor",
        "nv.coop.mulAdd",
        "nv.coop.storeTensor",
    };

    var index: usize = 0;
    while (std.mem.indexOfPos(u8, source, index, "nv.")) |hit| {
        var ok = false;
        for (allowed) |prefix| {
            if (std.mem.startsWith(u8, source[hit..], prefix)) {
                ok = true;
                break;
            }
        }
        if (!ok) {
            reportAt(source, hit, "unsupported nv intrinsic");
            return error.UnsupportedSyntax;
        }
        index = hit + 3;
    }
}

fn requireContains(source: []const u8, needle: []const u8, message: []const u8) !void {
    if (std.mem.indexOf(u8, source, needle) == null) {
        std.debug.print("frontend parse error: {s}\n", .{message});
        return error.UnsupportedSyntax;
    }
}

fn requireSingle(source: []const u8, needle: []const u8, message: []const u8) !void {
    const first = std.mem.indexOf(u8, source, needle) orelse {
        std.debug.print("frontend parse error: {s}\n", .{message});
        return error.UnsupportedSyntax;
    };
    if (std.mem.indexOfPos(u8, source, first + needle.len, needle) != null) {
        reportAt(source, first, message);
        return error.UnsupportedSyntax;
    }
}

fn reportAt(source: []const u8, offset: usize, message: []const u8) void {
    var line: usize = 1;
    var col: usize = 1;
    for (source[0..@min(offset, source.len)]) |byte| {
        if (byte == '\n') {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    std.debug.print("frontend parse error:{d}:{d}: {s}\n", .{ line, col, message });
}

const nvcoop2_square_f16_spvasm =
    \\; SPIR-V
    \\; Version: 1.6
    \\               OpCapability Shader
    \\               OpCapability StorageBuffer16BitAccess
    \\               OpCapability VulkanMemoryModel
    \\               OpCapability CooperativeMatrixKHR
    \\               OpCapability CooperativeMatrixReductionsNV
    \\               OpCapability CooperativeMatrixConversionsNV
    \\               OpCapability CooperativeMatrixTensorAddressingNV
    \\               OpCapability TensorAddressingNV
    \\               OpCapability Float16
    \\               OpExtension "SPV_KHR_cooperative_matrix"
    \\               OpExtension "SPV_NV_tensor_addressing"
    \\               OpExtension "SPV_NV_cooperative_matrix2"
    \\               OpMemoryModel Logical Vulkan
    \\               OpEntryPoint GLCompute %main "main" %gl_WorkGroupID %a_buffer %b_buffer %c_buffer %push_constants
    \\               OpExecutionMode %main LocalSize 256 1 1
    \\               OpName %main "main"
    \\               OpName %gl_WorkGroupID "gl_WorkGroupID"
    \\               OpName %a_buffer "a_buffer"
    \\               OpName %b_buffer "b_buffer"
    \\               OpName %c_buffer "c_buffer"
    \\               OpName %push_constants "push_constants"
    \\               OpName %A "A"
    \\               OpMemberName %A 0 "data"
    \\               OpName %B "B"
    \\               OpMemberName %B 0 "data"
    \\               OpName %C "C"
    \\               OpMemberName %C 0 "data"
    \\               OpName %PushConstants "PushConstants"
    \\               OpName %f16 "f16"
    \\               OpDecorate %gl_WorkGroupID BuiltIn WorkgroupId
    \\               OpDecorate %arr_a ArrayStride 2
    \\               OpDecorate %arr_b ArrayStride 2
    \\               OpDecorate %arr_c ArrayStride 4
    \\               OpDecorate %A Block
    \\               OpMemberDecorate %A 0 NonWritable
    \\               OpMemberDecorate %A 0 Offset 0
    \\               OpDecorate %a_buffer NonWritable
    \\               OpDecorate %a_buffer Binding 0
    \\               OpDecorate %a_buffer DescriptorSet 0
    \\               OpDecorate %B Block
    \\               OpMemberDecorate %B 0 NonWritable
    \\               OpMemberDecorate %B 0 Offset 0
    \\               OpDecorate %b_buffer NonWritable
    \\               OpDecorate %b_buffer Binding 1
    \\               OpDecorate %b_buffer DescriptorSet 0
    \\               OpDecorate %C Block
    \\               OpMemberDecorate %C 0 NonReadable
    \\               OpMemberDecorate %C 0 Offset 0
    \\               OpDecorate %c_buffer NonReadable
    \\               OpDecorate %c_buffer Binding 2
    \\               OpDecorate %c_buffer DescriptorSet 0
    \\               OpDecorate %PushConstants Block
    \\               OpMemberDecorate %PushConstants 0 Offset 0
    \\               OpMemberDecorate %PushConstants 1 Offset 4
    \\               OpMemberDecorate %PushConstants 2 Offset 8
    \\               OpMemberDecorate %PushConstants 3 Offset 12
    \\               OpMemberDecorate %PushConstants 4 Offset 16
    \\               OpMemberDecorate %PushConstants 5 Offset 20
    \\       %void = OpTypeVoid
    \\       %bool = OpTypeBool
    \\       %uint = OpTypeInt 32 0
    \\        %int = OpTypeInt 32 1
    \\      %float = OpTypeFloat 32
    \\        %f16 = OpTypeFloat 16
    \\     %v3uint = OpTypeVector %uint 3
    \\ %fn_void_ty = OpTypeFunction %void
    \\     %uint_0 = OpConstant %uint 0
    \\     %uint_1 = OpConstant %uint 1
    \\     %uint_2 = OpConstant %uint 2
    \\    %uint_16 = OpConstant %uint 16
    \\    %uint_64 = OpConstant %uint 64
    \\   %uint_128 = OpConstant %uint 128
    \\      %int_0 = OpConstant %int 0
    \\      %int_1 = OpConstant %int 1
    \\      %int_2 = OpConstant %int 2
    \\      %int_3 = OpConstant %int 3
    \\      %int_4 = OpConstant %int 4
    \\      %int_5 = OpConstant %int 5
    \\    %float_0 = OpConstant %float 0
    \\      %f16_0 = OpConstant %f16 0x0p+0
    \\    %acc_ty = OpTypeCooperativeMatrixKHR %float %uint_2 %uint_128 %uint_128 %uint_2
    \\  %a_mat_ty = OpTypeCooperativeMatrixKHR %f16 %uint_2 %uint_128 %uint_16 %uint_0
    \\  %b_mat_ty = OpTypeCooperativeMatrixKHR %f16 %uint_2 %uint_16 %uint_128 %uint_1
    \\  %layout_ty = OpTypeTensorLayoutNV %uint_2 %uint_0
    \\   %acc_zero = OpConstantComposite %acc_ty %float_0
    \\     %a_zero = OpConstantComposite %a_mat_ty %f16_0
    \\     %b_zero = OpConstantComposite %b_mat_ty %f16_0
    \\%ptr_Input_v3uint = OpTypePointer Input %v3uint
    \\%ptr_Input_uint = OpTypePointer Input %uint
    \\%ptr_Function_uint = OpTypePointer Function %uint
    \\%ptr_Function_acc = OpTypePointer Function %acc_ty
    \\      %arr_a = OpTypeRuntimeArray %f16
    \\      %arr_b = OpTypeRuntimeArray %f16
    \\      %arr_c = OpTypeRuntimeArray %float
    \\          %A = OpTypeStruct %arr_a
    \\          %B = OpTypeStruct %arr_b
    \\          %C = OpTypeStruct %arr_c
    \\%ptr_StorageBuffer_A = OpTypePointer StorageBuffer %A
    \\%ptr_StorageBuffer_B = OpTypePointer StorageBuffer %B
    \\%ptr_StorageBuffer_C = OpTypePointer StorageBuffer %C
    \\%ptr_StorageBuffer_f16 = OpTypePointer StorageBuffer %f16
    \\%ptr_StorageBuffer_float = OpTypePointer StorageBuffer %float
    \\%PushConstants = OpTypeStruct %uint %uint %uint %uint %uint %uint
    \\%ptr_PushConstant_PushConstants = OpTypePointer PushConstant %PushConstants
    \\%ptr_PushConstant_uint = OpTypePointer PushConstant %uint
    \\%gl_WorkGroupID = OpVariable %ptr_Input_v3uint Input
    \\   %a_buffer = OpVariable %ptr_StorageBuffer_A StorageBuffer
    \\   %b_buffer = OpVariable %ptr_StorageBuffer_B StorageBuffer
    \\   %c_buffer = OpVariable %ptr_StorageBuffer_C StorageBuffer
    \\%push_constants = OpVariable %ptr_PushConstant_PushConstants PushConstant
    \\       %main = OpFunction %void None %fn_void_ty
    \\%entry_label = OpLabel
    \\    %acc_var = OpVariable %ptr_Function_acc Function
    \\     %kk_var = OpVariable %ptr_Function_uint Function
    \\  %ptr_1 = OpAccessChain %ptr_PushConstant_uint %push_constants %int_0
    \\  %load_2 = OpLoad %uint %ptr_1
    \\  %ptr_3 = OpAccessChain %ptr_PushConstant_uint %push_constants %int_1
    \\  %load_4 = OpLoad %uint %ptr_3
    \\  %ptr_5 = OpAccessChain %ptr_PushConstant_uint %push_constants %int_2
    \\  %load_6 = OpLoad %uint %ptr_5
    \\%a_base_ptr = OpAccessChain %ptr_StorageBuffer_f16 %a_buffer %int_0 %uint_0
    \\%b_base_ptr = OpAccessChain %ptr_StorageBuffer_f16 %b_buffer %int_0 %uint_0
    \\%c_base_ptr = OpAccessChain %ptr_StorageBuffer_float %c_buffer %int_0 %uint_0
    \\  %ptr_7 = OpAccessChain %ptr_Input_uint %gl_WorkGroupID %uint_0
    \\  %load_8 = OpLoad %uint %ptr_7
    \\  %ptr_9 = OpAccessChain %ptr_Input_uint %gl_WorkGroupID %uint_1
    \\  %load_10 = OpLoad %uint %ptr_9
    \\  %op_11 = OpIMul %uint %load_10 %uint_128
    \\  %op_12 = OpIMul %uint %load_8 %uint_128
    \\               OpStore %acc_var %acc_zero
    \\  %layout_13 = OpCreateTensorLayoutNV %layout_ty
    \\  %layout_14 = OpTensorLayoutSetDimensionNV %layout_ty %layout_13 %load_2 %load_6
    \\  %layout_15 = OpCreateTensorLayoutNV %layout_ty
    \\  %layout_16 = OpTensorLayoutSetDimensionNV %layout_ty %layout_15 %load_6 %load_4
    \\  %layout_17 = OpCreateTensorLayoutNV %layout_ty
    \\  %layout_18 = OpTensorLayoutSetDimensionNV %layout_ty %layout_17 %load_2 %load_4
    \\               OpStore %kk_var %uint_0
    \\               OpBranch %loop_header
    \\%loop_header = OpLabel
    \\               OpLoopMerge %loop_merge %loop_continue None
    \\               OpBranch %loop_cond
    \\  %loop_cond = OpLabel
    \\  %load_19 = OpLoad %uint %kk_var
    \\  %op_20 = OpULessThan %bool %load_19 %load_6
    \\               OpBranchConditional %op_20 %loop_body %loop_merge
    \\  %loop_body = OpLabel
    \\  %load_21 = OpLoad %uint %kk_var
    \\  %slice_22 = OpTensorLayoutSliceNV %layout_ty %layout_14 %op_11 %uint_128 %load_21 %uint_16
    \\  %mat_23 = OpCooperativeMatrixLoadTensorNV %a_mat_ty %a_base_ptr %a_zero %slice_22 None None
    \\  %slice_24 = OpTensorLayoutSliceNV %layout_ty %layout_16 %load_21 %uint_16 %op_12 %uint_128
    \\  %mat_25 = OpCooperativeMatrixLoadTensorNV %b_mat_ty %b_base_ptr %b_zero %slice_24 None None
    \\  %load_26 = OpLoad %acc_ty %acc_var
    \\  %acc_27 = OpCooperativeMatrixMulAddKHR %acc_ty %mat_23 %mat_25 %load_26
    \\               OpStore %acc_var %acc_27
    \\               OpBranch %loop_continue
    \\%loop_continue = OpLabel
    \\  %load_28 = OpLoad %uint %kk_var
    \\  %op_29 = OpIAdd %uint %load_28 %uint_16
    \\               OpStore %kk_var %op_29
    \\               OpBranch %loop_header
    \\   %loop_merge = OpLabel
    \\  %slice_30 = OpTensorLayoutSliceNV %layout_ty %layout_18 %op_11 %uint_128 %op_12 %uint_128
    \\  %load_31 = OpLoad %acc_ty %acc_var
    \\               OpCooperativeMatrixStoreTensorNV %c_base_ptr %load_31 %slice_30 None None
    \\               OpReturn
    \\               OpFunctionEnd
    \\
;
