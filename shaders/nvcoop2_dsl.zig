//! Experimental text-first SPIR-V DSL for VK_NV_cooperative_matrix2 shaders.

const std = @import("std");

pub const Mode = enum {
    f16_square,

    fn importName(self: Mode) []const u8 {
        return switch (self) {
            .f16_square => "matmul_nvcoop2_square_f16_spv",
        };
    }

    fn tileM(self: Mode) u32 {
        return switch (self) {
            .f16_square => 128,
        };
    }

    fn tileN(self: Mode) u32 {
        return switch (self) {
            .f16_square => 128,
        };
    }
};

pub const NvCoop2Shader = struct {
    pub const BodyFn = *const fn (*Kernel) anyerror!void;

    mode: Mode,
    import_name: []const u8,
    body: BodyFn,

    pub fn emit(self: NvCoop2Shader, gpa: std.mem.Allocator) ![]u8 {
        var out: std.Io.Writer.Allocating = .init(gpa);
        errdefer out.deinit();

        var kernel = Kernel{
            .gpa = gpa,
            .asm_out = &out.writer,
            .mode = self.mode,
            .push = Push.fixed(),
            .ids = .{},
            .coop = undefined,
        };
        kernel.coop = .{ .kernel = &kernel };

        try self.body(&kernel);
        return out.toOwnedSlice();
    }
};

pub fn emitShader(init: std.process.Init, shader: NvCoop2Shader) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(gpa);
    defer gpa.free(args);

    if (args.len != 2) return error.InvalidArguments;

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();

    const text = try shader.emit(arena.allocator());
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[1], .data = text });
}

pub const Buffer = enum { a, b, c };
pub const MatrixRole = enum { accumulator, a, b };

pub const Value = struct {
    id: []const u8,
};

pub const Push = struct {
    m: Value,
    n: Value,
    k: Value,

    fn fixed() Push {
        return .{
            .m = .{ .id = "%m" },
            .n = .{ .id = "%n" },
            .k = .{ .id = "%k" },
        };
    }
};

pub const Dimensions = struct {
    m: Value,
    n: Value,
};

pub const Tile = struct {
    row: Value,
    col: Value,
};

pub const TensorLayout = struct {
    kernel: *Kernel,
    id: []const u8,

    pub fn slice(self: TensorLayout, offset0: Value, span0: u32, offset1: Value, span1: u32) Slice {
        return .{
            .kernel = self.kernel,
            .layout = self.id,
            .offset0 = offset0,
            .span0 = span0,
            .offset1 = offset1,
            .span1 = span1,
        };
    }
};

pub const Slice = struct {
    kernel: *Kernel,
    layout: []const u8,
    offset0: Value,
    span0: u32,
    offset1: Value,
    span1: u32,
};

pub const Matrix = struct {
    role: MatrixRole,
    id: []const u8,
    is_accumulator_var: bool = false,
};

pub const KLoopOptions = struct {
    step: u32,
};

pub const KLoop = struct {
    kernel: *Kernel,
    step: u32,
    emitted_header: bool = false,
    emitted_merge: bool = false,

    pub fn next(self: *KLoop) !?Value {
        if (!self.emitted_header) {
            self.emitted_header = true;
            const kk = try self.kernel.emitKLoopHeader(self.step);
            return .{ .id = kk };
        }
        if (!self.emitted_merge) {
            self.emitted_merge = true;
            try self.kernel.emitKLoopFooter(self.step);
        }
        return null;
    }
};

pub const Coop = struct {
    kernel: *Kernel,

    pub fn zero(self: Coop, role: MatrixRole) !Matrix {
        if (role != .accumulator) return error.UnsupportedMatrixZero;
        try self.kernel.asm_out.writeAll("               OpStore %acc_var %acc_zero\n");
        return .{ .role = .accumulator, .id = "%acc_var", .is_accumulator_var = true };
    }

    pub fn loadTensor(self: Coop, buffer: Buffer, slice: Slice) !Matrix {
        const slice_id = try self.kernel.nextId("slice");
        try self.kernel.asm_out.print(
            "  {s} = OpTensorLayoutSliceNV %layout_ty {s} {s} {s} {s} {s}\n",
            .{
                slice_id,
                slice.layout,
                slice.offset0.id,
                self.kernel.constU32(slice.span0),
                slice.offset1.id,
                self.kernel.constU32(slice.span1),
            },
        );

        const mat = try self.kernel.nextId("mat");
        const result_ty = switch (buffer) {
            .a => "%a_mat_ty",
            .b => "%b_mat_ty",
            .c => return error.CannotLoadCAsInputMatrix,
        };
        const ptr = switch (buffer) {
            .a => "%a_base_ptr",
            .b => "%b_base_ptr",
            .c => unreachable,
        };
        const zero_id = switch (buffer) {
            .a => "%a_zero",
            .b => "%b_zero",
            .c => unreachable,
        };
        try self.kernel.asm_out.print(
            "  {s} = OpCooperativeMatrixLoadTensorNV {s} {s} {s} {s} None None\n",
            .{ mat, result_ty, ptr, zero_id, slice_id },
        );
        return .{
            .role = switch (buffer) {
                .a => .a,
                .b => .b,
                .c => unreachable,
            },
            .id = mat,
        };
    }

    pub fn mulAdd(self: Coop, a: Matrix, b: Matrix, acc: Matrix) !Matrix {
        if (a.role != .a or b.role != .b or acc.role != .accumulator) return error.InvalidMulAddOperands;

        const acc_in = if (acc.is_accumulator_var)
            try self.kernel.load("%acc_ty", "%acc_var")
        else
            acc.id;
        const acc_out = try self.kernel.nextId("acc");
        try self.kernel.asm_out.print(
            "  {s} = OpCooperativeMatrixMulAddKHR %acc_ty {s} {s} {s}\n",
            .{ acc_out, a.id, b.id, acc_in },
        );
        try self.kernel.asm_out.print("               OpStore %acc_var {s}\n", .{acc_out});
        return .{ .role = .accumulator, .id = "%acc_var", .is_accumulator_var = true };
    }

    pub fn storeTensor(self: Coop, buffer: Buffer, matrix: Matrix, slice: Slice) !void {
        if (buffer != .c or matrix.role != .accumulator) return error.InvalidStoreTensorOperands;

        const slice_id = try self.kernel.nextId("slice");
        try self.kernel.asm_out.print(
            "  {s} = OpTensorLayoutSliceNV %layout_ty {s} {s} {s} {s} {s}\n",
            .{
                slice_id,
                slice.layout,
                slice.offset0.id,
                self.kernel.constU32(slice.span0),
                slice.offset1.id,
                self.kernel.constU32(slice.span1),
            },
        );
        const acc_final = if (matrix.is_accumulator_var)
            try self.kernel.load("%acc_ty", "%acc_var")
        else
            matrix.id;
        try self.kernel.asm_out.print(
            "               OpCooperativeMatrixStoreTensorNV %c_base_ptr {s} {s} None None\n",
            .{ acc_final, slice_id },
        );
    }
};

const Ids = struct {
    next: u32 = 0,
};

pub const Kernel = struct {
    gpa: std.mem.Allocator,
    asm_out: *std.Io.Writer,
    mode: Mode,
    push: Push,
    ids: Ids,
    coop: Coop,

    pub fn begin(self: *Kernel) !void {
        try self.emitModulePrefix();
        try self.emitFunctionPrefix();
    }

    pub fn end(self: *Kernel) !void {
        try self.asm_out.writeAll(
            \\               OpReturn
            \\               OpFunctionEnd
            \\
        );
    }

    pub fn workgroupTile(self: *Kernel) !Tile {
        const wg_x_ptr = try self.accessChain("%ptr_Input_uint", "%gl_WorkGroupID", &.{ "%uint_0" });
        const tile_x = try self.load("%uint", wg_x_ptr);
        const wg_y_ptr = try self.accessChain("%ptr_Input_uint", "%gl_WorkGroupID", &.{ "%uint_1" });
        const tile_y = try self.load("%uint", wg_y_ptr);
        const row = try self.binary("OpIMul", "%uint", tile_y, self.constU32(self.mode.tileM()));
        const col = try self.binary("OpIMul", "%uint", tile_x, self.constU32(self.mode.tileN()));
        return .{
            .row = .{ .id = row },
            .col = .{ .id = col },
        };
    }

    pub fn tensorLayout(self: *Kernel, buffer: Buffer, dims: Dimensions) !TensorLayout {
        _ = buffer;
        const base = try self.nextId("layout");
        try self.asm_out.print("  {s} = OpCreateTensorLayoutNV %layout_ty\n", .{base});

        const layout = try self.nextId("layout");
        try self.asm_out.print(
            "  {s} = OpTensorLayoutSetDimensionNV %layout_ty {s} {s} {s}\n",
            .{ layout, base, dims.m.id, dims.n.id },
        );
        return .{ .kernel = self, .id = layout };
    }

    pub fn kLoop(self: *Kernel, opts: KLoopOptions) !KLoop {
        if (opts.step != 16) return error.UnsupportedKLoopStep;
        try self.asm_out.writeAll("               OpStore %kk_var %uint_0\n");
        return .{ .kernel = self, .step = opts.step };
    }

    fn emitKLoopHeader(self: *Kernel, step: u32) ![]const u8 {
        _ = step;
        try self.asm_out.writeAll(
            \\               OpBranch %loop_header
            \\%loop_header = OpLabel
            \\               OpLoopMerge %loop_merge %loop_continue None
            \\               OpBranch %loop_cond
            \\  %loop_cond = OpLabel
            \\
        );
        const kk_for_cond = try self.load("%uint", "%kk_var");
        const cond = try self.binary("OpULessThan", "%bool", kk_for_cond, self.push.k.id);
        try self.asm_out.print(
            \\               OpBranchConditional {s} %loop_body %loop_merge
            \\  %loop_body = OpLabel
            \\
        , .{cond});
        return self.load("%uint", "%kk_var");
    }

    fn emitKLoopFooter(self: *Kernel, step: u32) !void {
        try self.asm_out.writeAll("               OpBranch %loop_continue\n");
        try self.asm_out.writeAll("%loop_continue = OpLabel\n");
        const kk_old = try self.load("%uint", "%kk_var");
        const kk_next = try self.binary("OpIAdd", "%uint", kk_old, self.constU32(step));
        try self.asm_out.print(
            \\               OpStore %kk_var {s}
            \\               OpBranch %loop_header
            \\   %loop_merge = OpLabel
            \\
        , .{kk_next});
    }

    fn emitModulePrefix(self: *Kernel) !void {
        try self.asm_out.print(
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
            \\
        , .{});
    }

    fn emitFunctionPrefix(self: *Kernel) !void {
        try self.asm_out.writeAll(
            \\       %main = OpFunction %void None %fn_void_ty
            \\%entry_label = OpLabel
            \\    %acc_var = OpVariable %ptr_Function_acc Function
            \\     %kk_var = OpVariable %ptr_Function_uint Function
            \\
        );

        self.push.m = .{ .id = try self.loadPush("%int_0") };
        self.push.n = .{ .id = try self.loadPush("%int_1") };
        self.push.k = .{ .id = try self.loadPush("%int_2") };

        try self.asm_out.writeAll(
            \\%a_base_ptr = OpAccessChain %ptr_StorageBuffer_f16 %a_buffer %int_0 %uint_0
            \\%b_base_ptr = OpAccessChain %ptr_StorageBuffer_f16 %b_buffer %int_0 %uint_0
            \\%c_base_ptr = OpAccessChain %ptr_StorageBuffer_float %c_buffer %int_0 %uint_0
            \\
        );
    }

    fn loadPush(self: *Kernel, member: []const u8) ![]const u8 {
        const ptr = try self.accessChain("%ptr_PushConstant_uint", "%push_constants", &.{member});
        return self.load("%uint", ptr);
    }

    fn accessChain(self: *Kernel, result_ty: []const u8, base: []const u8, indexes: []const []const u8) ![]const u8 {
        const result = try self.nextId("ptr");
        try self.asm_out.print("  {s} = OpAccessChain {s} {s}", .{ result, result_ty, base });
        for (indexes) |index| try self.asm_out.print(" {s}", .{index});
        try self.asm_out.writeByte('\n');
        return result;
    }

    fn load(self: *Kernel, result_ty: []const u8, ptr: []const u8) ![]const u8 {
        const result = try self.nextId("load");
        try self.asm_out.print("  {s} = OpLoad {s} {s}\n", .{ result, result_ty, ptr });
        return result;
    }

    fn binary(self: *Kernel, opcode: []const u8, result_ty: []const u8, lhs: []const u8, rhs: []const u8) ![]const u8 {
        const result = try self.nextId("op");
        try self.asm_out.print("  {s} = {s} {s} {s} {s}\n", .{ result, opcode, result_ty, lhs, rhs });
        return result;
    }

    fn nextId(self: *Kernel, prefix: []const u8) ![]const u8 {
        self.ids.next += 1;
        return std.fmt.allocPrint(self.gpa, "%{s}_{d}", .{ prefix, self.ids.next });
    }

    fn constU32(self: *Kernel, value: u32) []const u8 {
        _ = self;
        return switch (value) {
            0 => "%uint_0",
            1 => "%uint_1",
            2 => "%uint_2",
            16 => "%uint_16",
            64 => "%uint_64",
            128 => "%uint_128",
            else => unreachable,
        };
    }
};
