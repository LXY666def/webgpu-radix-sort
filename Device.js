class Device {
    constructor() {
        this._device = null;
        this._adapter = null;

    }

    async initialize() {
        if (!navigator.gpu) {
            console.error("WebGPU not supported.");
            return false;
        }
        try {
            this._adapter = await navigator.gpu.requestAdapter({
                powerPreference: "high-performance" // optional：high-performance || low-power
            });

            if (!this._adapter.features.has("subgroups")) {
                throw new Error("Subgroups support is not available");
            }

            if (this._adapter) {
                const adapterInfo = await this._adapter.info;
                console.log("Architecture:", adapterInfo.architecture,
                    "\nVendor:", adapterInfo.vendor);
            } else {
                console.error("Couldn't request WebGPU adapter.");
                return false;
            }

            console.log("Adapter Limits", 
                "\nmaxComputeWorkgroupSizeX", this._adapter.limits.maxComputeWorkgroupSizeX,
                "\nmaxComputeInvocationsPerWorkgroup", this._adapter.limits.maxComputeInvocationsPerWorkgroup,
                "\nmaxComputeWorkgroupStorageSize", this._adapter.limits.maxComputeWorkgroupStorageSize,
                "\nsubgroupMinSize", this._adapter.info.subgroupMinSize,
                "\nsubgroupMaxSize", this._adapter.info.subgroupMaxSize,
            );

            this._device = await this._adapter.requestDevice({
                requiredFeatures:  ["subgroups"],
                requiredLimits: {
                    maxComputeWorkgroupSizeX: 512,
                    maxComputeInvocationsPerWorkgroup: 512,
                    maxComputeWorkgroupStorageSize: 20480,
                }
            });

            console.log("Device Limits", 
                "\nmaxComputeWorkgroupSizeX", this._device.limits.maxComputeWorkgroupSizeX,
                "\nmaxComputeInvocationsPerWorkgroup", this._adapter.limits.maxComputeInvocationsPerWorkgroup,
                "\nmaxComputeWorkgroupStorageSize", this._adapter.limits.maxComputeWorkgroupStorageSize,
                "\nmaxComputeWorkgroupsPerDimension", this._adapter.limits.maxComputeWorkgroupsPerDimension,
                "\nmaxBindGroups", this._device.limits.maxBindGroups,
                "\nmaxBindingsPerBindGroup", this._device.limits.maxBindingsPerBindGroup,
                "\nmaxDynamicUniformBuffersPerPipelineLayout", this._device.limits.maxDynamicUniformBuffersPerPipelineLayout,
                "\nminUniformBufferOffsetAlignment", this._device.limits.minUniformBufferOffsetAlignment,
                "\nminStorageBufferOffsetAlignment", this._device.limits.minStorageBufferOffsetAlignment,
            );

            console.log("[Device]: init successfully");
            return true;
        } catch (error) {
            console.error("Fail to create logical device", error);
            return false;
        }
    }

    createBufferAndFill(usage, data) {
        const buffer = this._device.createBuffer({
            size: data.byteLength,
            usage: usage,
            mappedAtCreation: true,
        });
    
        new Uint32Array(buffer.getMappedRange()).set(data);
        buffer.unmap();
        return buffer;
    }

    createBuffer(size, usage) {
        return this._device.createBuffer({ size: size, usage: usage });
    }

    get device() {
        return this._device;
    }

    get adapter() {
        return this._adapter;
    }
}

export default Device;