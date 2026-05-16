pub const c = struct {
    pub const VkBool32 = u32;
    pub const VkFlags = u32;
    pub const VkDeviceSize = u64;
    pub const VkMemoryPropertyFlags = VkFlags;
    pub const VkMemoryMapFlags = VkFlags;
    pub const VkBufferUsageFlags = VkFlags;
    pub const VkBufferCreateFlags = VkFlags;
    pub const VkDeviceCreateFlags = VkFlags;
    pub const VkDeviceQueueCreateFlags = VkFlags;
    pub const VkInstanceCreateFlags = VkFlags;
    pub const VkQueueFlags = VkFlags;
    pub const VkShaderStageFlags = VkFlags;
    pub const VkDescriptorPoolCreateFlags = VkFlags;
    pub const VkDescriptorSetLayoutCreateFlags = VkFlags;
    pub const VkPipelineLayoutCreateFlags = VkFlags;
    pub const VkPipelineCreateFlags = VkFlags;
    pub const VkCommandPoolCreateFlags = VkFlags;
    pub const VkCommandBufferUsageFlags = VkFlags;
    pub const VkFenceCreateFlags = VkFlags;
    pub const VkAccessFlags = VkFlags;
    pub const VkDependencyFlags = VkFlags;
    pub const VkPipelineStageFlags = VkFlags;

    pub const VkResult = i32;
    pub const VkStructureType = u32;
    pub const VkPhysicalDeviceType = u32;
    pub const VkSharingMode = u32;
    pub const VkDescriptorType = u32;
    pub const VkPipelineBindPoint = u32;
    pub const VkCommandBufferLevel = u32;
    pub const VkComponentTypeKHR = u32;
    pub const VkScopeKHR = u32;

    const VkInstance_T = opaque {};
    const VkPhysicalDevice_T = opaque {};
    const VkDevice_T = opaque {};
    const VkQueue_T = opaque {};
    const VkBuffer_T = opaque {};
    const VkDeviceMemory_T = opaque {};
    const VkShaderModule_T = opaque {};
    const VkDescriptorSetLayout_T = opaque {};
    const VkPipelineLayout_T = opaque {};
    const VkPipeline_T = opaque {};
    const VkPipelineCache_T = opaque {};
    const VkDescriptorPool_T = opaque {};
    const VkDescriptorSet_T = opaque {};
    const VkCommandPool_T = opaque {};
    const VkCommandBuffer_T = opaque {};
    const VkFence_T = opaque {};

    pub const VkInstance = ?*VkInstance_T;
    pub const VkPhysicalDevice = ?*VkPhysicalDevice_T;
    pub const VkDevice = ?*VkDevice_T;
    pub const VkQueue = ?*VkQueue_T;
    pub const VkBuffer = ?*VkBuffer_T;
    pub const VkDeviceMemory = ?*VkDeviceMemory_T;
    pub const VkShaderModule = ?*VkShaderModule_T;
    pub const VkDescriptorSetLayout = ?*VkDescriptorSetLayout_T;
    pub const VkPipelineLayout = ?*VkPipelineLayout_T;
    pub const VkPipeline = ?*VkPipeline_T;
    pub const VkPipelineCache = ?*VkPipelineCache_T;
    pub const VkDescriptorPool = ?*VkDescriptorPool_T;
    pub const VkDescriptorSet = ?*VkDescriptorSet_T;
    pub const VkCommandPool = ?*VkCommandPool_T;
    pub const VkCommandBuffer = ?*VkCommandBuffer_T;
    pub const VkFence = ?*VkFence_T;

    pub const PFN_vkVoidFunction = ?*const fn () callconv(.c) void;

    pub const VK_SUCCESS: VkResult = 0;
    pub const VK_ERROR_EXTENSION_NOT_PRESENT: VkResult = -7;
    pub const VK_ERROR_FEATURE_NOT_PRESENT: VkResult = -8;
    pub const VK_TRUE: VkBool32 = 1;
    pub const VK_FALSE: VkBool32 = 0;
    pub const VK_API_VERSION_1_2: u32 = (1 << 22) | (2 << 12);
    pub const VK_PHYSICAL_DEVICE_TYPE_CPU: VkPhysicalDeviceType = 4;

    pub const VK_STRUCTURE_TYPE_APPLICATION_INFO: VkStructureType = 0;
    pub const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: VkStructureType = 1;
    pub const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: VkStructureType = 2;
    pub const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: VkStructureType = 3;
    pub const VK_STRUCTURE_TYPE_SUBMIT_INFO: VkStructureType = 4;
    pub const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: VkStructureType = 5;
    pub const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: VkStructureType = 8;
    pub const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: VkStructureType = 12;
    pub const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: VkStructureType = 16;
    pub const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: VkStructureType = 18;
    pub const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: VkStructureType = 29;
    pub const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: VkStructureType = 30;
    pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: VkStructureType = 32;
    pub const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: VkStructureType = 33;
    pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: VkStructureType = 34;
    pub const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: VkStructureType = 35;
    pub const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: VkStructureType = 39;
    pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: VkStructureType = 40;
    pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: VkStructureType = 42;
    pub const VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER: VkStructureType = 44;
    pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES: VkStructureType = 1000082000;
    pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES: VkStructureType = 1000083000;
    pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR: VkStructureType = 1000141000;
    pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR: VkStructureType = 1000506000;
    pub const VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR: VkStructureType = 1000506001;

    pub const VK_QUEUE_COMPUTE_BIT: VkQueueFlags = 0x00000002;
    pub const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: VkBufferUsageFlags = 0x00000001;
    pub const VK_BUFFER_USAGE_TRANSFER_DST_BIT: VkBufferUsageFlags = 0x00000002;
    pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: VkBufferUsageFlags = 0x00000020;
    pub const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: VkMemoryPropertyFlags = 0x00000001;
    pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: VkMemoryPropertyFlags = 0x00000002;
    pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: VkMemoryPropertyFlags = 0x00000004;
    pub const VK_SHADER_STAGE_COMPUTE_BIT: VkShaderStageFlags = 0x00000020;
    pub const VK_SHARING_MODE_EXCLUSIVE: VkSharingMode = 0;
    pub const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: VkDescriptorType = 7;
    pub const VK_PIPELINE_BIND_POINT_COMPUTE: VkPipelineBindPoint = 1;
    pub const VK_COMMAND_BUFFER_LEVEL_PRIMARY: VkCommandBufferLevel = 0;
    pub const VK_ACCESS_TRANSFER_READ_BIT: VkAccessFlags = 0x00000800;
    pub const VK_ACCESS_TRANSFER_WRITE_BIT: VkAccessFlags = 0x00001000;
    pub const VK_ACCESS_SHADER_READ_BIT: VkAccessFlags = 0x00000020;
    pub const VK_ACCESS_SHADER_WRITE_BIT: VkAccessFlags = 0x00000040;
    pub const VK_PIPELINE_STAGE_TRANSFER_BIT: VkPipelineStageFlags = 0x00001000;
    pub const VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT: VkPipelineStageFlags = 0x00000800;
    pub const VK_QUEUE_FAMILY_IGNORED: u32 = 0xffffffff;

    pub const VK_COMPONENT_TYPE_FLOAT16_KHR: VkComponentTypeKHR = 0;
    pub const VK_COMPONENT_TYPE_FLOAT32_KHR: VkComponentTypeKHR = 1;
    pub const VK_COMPONENT_TYPE_BFLOAT16_KHR: VkComponentTypeKHR = 1000141000;
    pub const VK_SCOPE_SUBGROUP_KHR: VkScopeKHR = 3;

    pub const VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME = "VK_KHR_cooperative_matrix";
    pub const VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME = "VK_KHR_shader_bfloat16";

    pub fn VK_MAKE_VERSION(major: u32, minor: u32, patch: u32) u32 {
        return (major << 22) | (minor << 12) | patch;
    }

    pub fn VK_VERSION_MAJOR(version: u32) u32 {
        return version >> 22;
    }

    pub fn VK_VERSION_MINOR(version: u32) u32 {
        return (version >> 12) & 0x3ff;
    }

    pub fn VK_VERSION_PATCH(version: u32) u32 {
        return version & 0xfff;
    }

    pub const VkAllocationCallbacks = opaque {};

    pub const VkApplicationInfo = extern struct {
        sType: VkStructureType,
        pNext: ?*const anyopaque,
        pApplicationName: ?[*:0]const u8,
        applicationVersion: u32,
        pEngineName: ?[*:0]const u8,
        engineVersion: u32,
        apiVersion: u32,
    };

    pub const VkInstanceCreateInfo = extern struct {
        sType: VkStructureType,
        pNext: ?*const anyopaque,
        flags: VkInstanceCreateFlags,
        pApplicationInfo: ?*const VkApplicationInfo,
        enabledLayerCount: u32,
        ppEnabledLayerNames: ?[*]const [*:0]const u8,
        enabledExtensionCount: u32,
        ppEnabledExtensionNames: ?[*]const [*:0]const u8,
    };

    pub const VkPhysicalDeviceProperties = extern struct {
        bytes: [4096]u8,
    };

    pub const VkExtensionProperties = extern struct {
        extensionName: [256]u8,
        specVersion: u32,
    };

    pub const VkPhysicalDevice16BitStorageFeatures = extern struct {
        sType: VkStructureType,
        pNext: ?*anyopaque,
        storageBuffer16BitAccess: VkBool32,
        uniformAndStorageBuffer16BitAccess: VkBool32,
        storagePushConstant16: VkBool32,
        storageInputOutput16: VkBool32,
    };

    pub const VkPhysicalDeviceShaderFloat16Int8Features = extern struct {
        sType: VkStructureType,
        pNext: ?*anyopaque,
        shaderFloat16: VkBool32,
        shaderInt8: VkBool32,
    };

    pub const VkPhysicalDeviceShaderBfloat16FeaturesKHR = extern struct {
        sType: VkStructureType,
        pNext: ?*anyopaque,
        shaderBFloat16Type: VkBool32,
        shaderBFloat16DotProduct: VkBool32,
        shaderBFloat16CooperativeMatrix: VkBool32,
    };

    pub const VkPhysicalDeviceCooperativeMatrixFeaturesKHR = extern struct {
        sType: VkStructureType,
        pNext: ?*anyopaque,
        cooperativeMatrix: VkBool32,
        cooperativeMatrixRobustBufferAccess: VkBool32,
    };

    pub const VkCooperativeMatrixPropertiesKHR = extern struct {
        sType: VkStructureType,
        pNext: ?*anyopaque,
        MSize: u32,
        NSize: u32,
        KSize: u32,
        AType: VkComponentTypeKHR,
        BType: VkComponentTypeKHR,
        CType: VkComponentTypeKHR,
        ResultType: VkComponentTypeKHR,
        saturatingAccumulation: VkBool32,
        scope: VkScopeKHR,
    };

    pub const VkMemoryType = extern struct {
        propertyFlags: VkMemoryPropertyFlags,
        heapIndex: u32,
    };

    pub const VkMemoryHeap = extern struct {
        size: VkDeviceSize,
        flags: VkFlags,
    };

    pub const VkPhysicalDeviceMemoryProperties = extern struct {
        memoryTypeCount: u32,
        memoryTypes: [32]VkMemoryType,
        memoryHeapCount: u32,
        memoryHeaps: [16]VkMemoryHeap,
    };

    pub const VkExtent3D = extern struct {
        width: u32,
        height: u32,
        depth: u32,
    };

    pub const VkQueueFamilyProperties = extern struct {
        queueFlags: VkQueueFlags,
        queueCount: u32,
        timestampValidBits: u32,
        minImageTransferGranularity: VkExtent3D,
    };

    pub const VkDeviceQueueCreateInfo = extern struct {
        sType: VkStructureType,
        pNext: ?*const anyopaque,
        flags: VkDeviceQueueCreateFlags,
        queueFamilyIndex: u32,
        queueCount: u32,
        pQueuePriorities: ?[*]const f32,
    };

    pub const VkDeviceCreateInfo = extern struct {
        sType: VkStructureType,
        pNext: ?*const anyopaque,
        flags: VkDeviceCreateFlags,
        queueCreateInfoCount: u32,
        pQueueCreateInfos: ?[*]const VkDeviceQueueCreateInfo,
        enabledLayerCount: u32,
        ppEnabledLayerNames: ?[*]const [*:0]const u8,
        enabledExtensionCount: u32,
        ppEnabledExtensionNames: ?[*]const [*:0]const u8,
        pEnabledFeatures: ?*const anyopaque,
    };

    pub const VkBufferCreateInfo = extern struct {
        sType: VkStructureType,
        pNext: ?*const anyopaque,
        flags: VkBufferCreateFlags,
        size: VkDeviceSize,
        usage: VkBufferUsageFlags,
        sharingMode: VkSharingMode,
        queueFamilyIndexCount: u32,
        pQueueFamilyIndices: ?[*]const u32,
    };

    pub const VkMemoryRequirements = extern struct {
        size: VkDeviceSize,
        alignment: VkDeviceSize,
        memoryTypeBits: u32,
    };

    pub const VkMemoryAllocateInfo = extern struct {
        sType: VkStructureType,
        pNext: ?*const anyopaque,
        allocationSize: VkDeviceSize,
        memoryTypeIndex: u32,
    };

    pub const VkShaderModuleCreateInfo = extern struct {
        sType: VkStructureType,
        pNext: ?*const anyopaque,
        flags: VkFlags,
        codeSize: usize,
        pCode: [*]const u32,
    };

    pub const VkDescriptorSetLayoutBinding = extern struct {
        binding: u32,
        descriptorType: VkDescriptorType,
        descriptorCount: u32,
        stageFlags: VkShaderStageFlags,
        pImmutableSamplers: ?*const anyopaque,
    };

    pub const VkDescriptorSetLayoutCreateInfo = extern struct {
        sType: VkStructureType,
        pNext: ?*const anyopaque,
        flags: VkDescriptorSetLayoutCreateFlags,
        bindingCount: u32,
        pBindings: ?[*]const VkDescriptorSetLayoutBinding,
    };

    pub const VkPushConstantRange = extern struct {
        stageFlags: VkShaderStageFlags,
        offset: u32,
        size: u32,
    };

    pub const VkPipelineLayoutCreateInfo = extern struct {
        sType: VkStructureType,
        pNext: ?*const anyopaque,
        flags: VkPipelineLayoutCreateFlags,
        setLayoutCount: u32,
        pSetLayouts: ?[*]const VkDescriptorSetLayout,
        pushConstantRangeCount: u32,
        pPushConstantRanges: ?[*]const VkPushConstantRange,
    };

    pub const VkPipelineShaderStageCreateInfo = extern struct {
        sType: VkStructureType,
        pNext: ?*const anyopaque,
        flags: VkFlags,
        stage: VkShaderStageFlags,
        module: VkShaderModule,
        pName: [*:0]const u8,
        pSpecializationInfo: ?*const anyopaque,
    };

    pub const VkComputePipelineCreateInfo = extern struct {
        sType: VkStructureType,
        pNext: ?*const anyopaque,
        flags: VkPipelineCreateFlags,
        stage: VkPipelineShaderStageCreateInfo,
        layout: VkPipelineLayout,
        basePipelineHandle: VkPipeline,
        basePipelineIndex: i32,
    };

    pub const VkDescriptorPoolSize = extern struct {
        type: VkDescriptorType,
        descriptorCount: u32,
    };

    pub const VkDescriptorPoolCreateInfo = extern struct {
        sType: VkStructureType,
        pNext: ?*const anyopaque,
        flags: VkDescriptorPoolCreateFlags,
        maxSets: u32,
        poolSizeCount: u32,
        pPoolSizes: ?[*]const VkDescriptorPoolSize,
    };

    pub const VkDescriptorSetAllocateInfo = extern struct {
        sType: VkStructureType,
        pNext: ?*const anyopaque,
        descriptorPool: VkDescriptorPool,
        descriptorSetCount: u32,
        pSetLayouts: ?[*]const VkDescriptorSetLayout,
    };

    pub const VkDescriptorBufferInfo = extern struct {
        buffer: VkBuffer,
        offset: VkDeviceSize,
        range: VkDeviceSize,
    };

    pub const VkWriteDescriptorSet = extern struct {
        sType: VkStructureType,
        pNext: ?*const anyopaque,
        dstSet: VkDescriptorSet,
        dstBinding: u32,
        dstArrayElement: u32,
        descriptorCount: u32,
        descriptorType: VkDescriptorType,
        pImageInfo: ?*const anyopaque,
        pBufferInfo: ?[*]const VkDescriptorBufferInfo,
        pTexelBufferView: ?*const anyopaque,
    };

    pub const VkCommandPoolCreateInfo = extern struct {
        sType: VkStructureType,
        pNext: ?*const anyopaque,
        flags: VkCommandPoolCreateFlags,
        queueFamilyIndex: u32,
    };

    pub const VkCommandBufferAllocateInfo = extern struct {
        sType: VkStructureType,
        pNext: ?*const anyopaque,
        commandPool: VkCommandPool,
        level: VkCommandBufferLevel,
        commandBufferCount: u32,
    };

    pub const VkCommandBufferBeginInfo = extern struct {
        sType: VkStructureType,
        pNext: ?*const anyopaque,
        flags: VkCommandBufferUsageFlags,
        pInheritanceInfo: ?*const anyopaque,
    };

    pub const VkBufferCopy = extern struct {
        srcOffset: VkDeviceSize,
        dstOffset: VkDeviceSize,
        size: VkDeviceSize,
    };

    pub const VkBufferMemoryBarrier = extern struct {
        sType: VkStructureType,
        pNext: ?*const anyopaque,
        srcAccessMask: VkAccessFlags,
        dstAccessMask: VkAccessFlags,
        srcQueueFamilyIndex: u32,
        dstQueueFamilyIndex: u32,
        buffer: VkBuffer,
        offset: VkDeviceSize,
        size: VkDeviceSize,
    };

    pub const VkFenceCreateInfo = extern struct {
        sType: VkStructureType,
        pNext: ?*const anyopaque,
        flags: VkFenceCreateFlags,
    };

    pub const VkSubmitInfo = extern struct {
        sType: VkStructureType,
        pNext: ?*const anyopaque,
        waitSemaphoreCount: u32,
        pWaitSemaphores: ?*const anyopaque,
        pWaitDstStageMask: ?[*]const VkFlags,
        commandBufferCount: u32,
        pCommandBuffers: ?[*]const VkCommandBuffer,
        signalSemaphoreCount: u32,
        pSignalSemaphores: ?*const anyopaque,
    };
};
