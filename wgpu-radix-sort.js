import upsweep_comp_wgsl from "./shaders/upsweep_comp_wgsl.js";
import spine_comp_wgsl from "./shaders/spine_comp_wgsl.js";
import downsweep_comp_wgsl from "./shaders/downsweep_comp_wgsl.js";
import checker_comp_wgsl from "./shaders/checker_comp_wgsl.js";

class VrdxSorterStorageRequirements {
    constructor() {
        this.size = 0;
        this.usage = 0;
    }
}

class RadixSorter {
    static RADIX = 256;
    static WORKGROUP_SIZE = 512;
    static PARTITION_DIVISION = 8;
    static PARTITION_SIZE = RadixSorter.PARTITION_DIVISION * RadixSorter.WORKGROUP_SIZE;

    static RoundUp(a, b) {
        return Math.ceil((a + b - 1) / b);
    }
    static AlignUp(a, aligment) {
        return Math.floor((a + aligment - 1) / aligment) * aligment;
    }

    GlobalHistogramSize() {
        let globalHistogramSize = Uint32Array.BYTES_PER_ELEMENT * 4 * RadixSorter.RADIX;
        globalHistogramSize = RadixSorter.AlignUp(globalHistogramSize, this._dynamicStorageAlignment);
        return globalHistogramSize;
    }

    HistogramSize(elementCount) {
        let globalHistogramSize = this.GlobalHistogramSize();
        let partitionHistogramSize = (1 + RadixSorter.RoundUp(elementCount, RadixSorter.PARTITION_SIZE) * RadixSorter.RADIX) * Uint32Array.BYTES_PER_ELEMENT;
        partitionHistogramSize = RadixSorter.AlignUp(partitionHistogramSize, this._dynamicStorageAlignment);
        return globalHistogramSize + partitionHistogramSize;
    }

    InoutSize(elementCount) {
        return RadixSorter.AlignUp(elementCount * Uint32Array.BYTES_PER_ELEMENT, this._dynamicStorageAlignment);
    }

    constructor(adapter_, device_) {
        this._adapter = adapter_;
        this._device = device_;
        this._dynamicUniformAlignment = this._device.limits.minUniformBufferOffsetAlignment;
        this._dynamicStorageAlignment = this._device.limits.minStorageBufferOffsetAlignment;
        this._maxComputeWorkgroupsPerDimension = this._device.limits.maxComputeWorkgroupsPerDimension;

        this._passBuffer = null;
        this._bindGroup0 = null;
        this._bindGroup1 = null;

        // bindGroupLayout
        this._bindGroupLayout = this._device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage', },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage', },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage', },
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage', },
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage', },
                },
                {
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage', },
                },
                {
                    binding: 6,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage', },
                },
                {
                    binding: 7,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: this._dynamicUniformAlignment },
                },
            ],
        });

        // pipeline layout
        this._pipelineLayout = this._device.createPipelineLayout({
            bindGroupLayouts: [null, null, null, this._bindGroupLayout],
        });
        
        // pipelines
        {
            const shaderModule = this._device.createShaderModule({
                code: upsweep_comp_wgsl,
            });
            this._upsweepPipeline = this._device.createComputePipeline({
                layout: this._pipelineLayout,
                compute: {
                    module: shaderModule,
                    entryPoint: "main",
                },
            });
        }
        {
            const shaderModule = this._device.createShaderModule({
                code: spine_comp_wgsl,
            });
            this._spinePipeline = this._device.createComputePipeline({
                layout: this._pipelineLayout,
                compute: {
                    module: shaderModule,
                    entryPoint: "main",
                },
            });
        }
        {
            const shaderModule = this._device.createShaderModule({
                code: downsweep_comp_wgsl,
            });
            this._downsweepPipeline = this._device.createComputePipeline({
                layout: this._pipelineLayout,
                compute: {
                    module: shaderModule,
                    entryPoint: "main",
                },
            });
        }
    }

    destroy() {
        if (this._upsweepPipeline) {
            this._upsweepPipeline.destroy();
        }
        if (this._spinePipeline) {
            this._spinePipeline.destroy();
        }
        if (this._downsweepPipeline) {
            this._downsweepPipeline.destroy();
        }
        if (this._pipelineLayout) {
            this._pipelineLayout.destroy();
        }
        if (this._bindGroupLayout) {
            this._bindGroupLayout.destroy();
        }
        if (this._bindGroup0) {
            this._bindGroup0.destroy();
        }
        if (this._bindGroup1) {
            this._bindGroup1.destroy();
        }
        if (this._passBuffer) {
            this._passBuffer.destroy();
        }
        if (this._resultBuffer) {
            this._resultBuffer.destroy();
        }
        if (this._bindGroupLayout_checker) {
            this._bindGroupLayout_checker.destroy();
        }
        if (this._bindGroup_checker) {
            this._bindGroup_checker.destroy();
        }
        if (this._pipelineLayout_checker) {
            this._pipelineLayout_checker.destroy();
        }
        if (this._checkerPipeline) {
            this._checkerPipeline.destroy();
        }
        if (this._stagingBuffer) {
            this._stagingBuffer.destroy();
        }

        this._upsweepPipeline = null;
        this._spinePipeline = null;
        this._downsweepPipeline = null;
        this._pipelineLayout = null;
        this._bindGroupLayout = null;
        this._bindGroup0 = null;
        this._bindGroup1 = null;
        this._passBuffer = null;

        this._resultBuffer = null;
        this._stagingBuffer = null;
        this._bindGroupLayout_checker = null;
        this._bindGroup_checker = null;
        this._pipelineLayout_checker = null;
        this._checkerPipeline = null;
    }

    vrdxGetSorterStorageRequirements(maxElementCount) {
        let req = new VrdxSorterStorageRequirements();
        req.size = this.InoutSize(maxElementCount) + this.HistogramSize(maxElementCount);
        req.usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
        return req;
    }

    vrdxGetSorterKeyValueStorageRequirements(maxElementCount) {
        let req = new VrdxSorterStorageRequirements();
        req.size = 2 * this.InoutSize(maxElementCount) + this.HistogramSize(maxElementCount);
        req.usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
        return req;
    }

    createChecker(indirectBuffer, indirectOffset,
        keysBuffer, keysOffset,) {
        // bindGroupLayout
        this._bindGroupLayout_checker = this._device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage', },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage', },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage', },
                },
            ],
        });

        // pipeline layout
        this._pipelineLayout_checker = this._device.createPipelineLayout({
            bindGroupLayouts: [null, null, null, this._bindGroupLayout_checker],
        });
        
        // pipelines
        {
            const shaderModule = this._device.createShaderModule({
                code: checker_comp_wgsl,
            });
            this._checkerPipeline = this._device.createComputePipeline({
                layout: this._pipelineLayout_checker,
                compute: {
                    module: shaderModule,
                    entryPoint: "main",
                },
            });
        }
        this._resultBuffer = this._device.createBuffer({label: "result buffer", size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST});
        this._stagingBuffer = this._device.createBuffer({label: "sorter staging buffer", size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
        this._bindGroup_checker = this._device.createBindGroup({
            layout: this._bindGroupLayout_checker,
            entries: [
                {
                    binding: 0, // ElementCount
                    resource: {
                        buffer: indirectBuffer,
                        offset: indirectOffset,
                    },
                },
                {
                    binding: 1, // KeysIn
                    resource: {
                        buffer: keysBuffer,
                        offset: keysOffset,
                    },
                },
                {
                    binding: 2, // check result
                    resource: {
                        buffer: this._resultBuffer,
                        offset: 0,
                        size: 4,
                    },
                },
            ],
        });
    }

    createBindGroup(elementCount,
        indirectBuffer, indirectOffset,
        keysBuffer, keysOffset, 
        valuesBuffer, valuesOffset, 
        storageBuffer, storageOffset
    )
    {
        this._passBuffer = this._device.createBuffer({
            label: "radix pass buffer",
            size: 4 * this._dynamicUniformAlignment,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const u32Array = new Uint32Array(1);
        for (let i = 0; i < 4; i++) {
            u32Array[0] = i;
            this._device.queue.writeBuffer(this._passBuffer, i * this._dynamicUniformAlignment, u32Array);
        }

        const globalHistogramSize = this.GlobalHistogramSize();
        const histogramSize = this.HistogramSize(elementCount);
        const inoutSize = this.InoutSize(elementCount);
    
        const histogramOffset = storageOffset;
        const inoutOffset = histogramOffset + histogramSize;


        this._bindGroup0 = this._device.createBindGroup({
            layout: this._bindGroupLayout,
            entries: [
                {
                    binding: 0, // ElementCount
                    resource: {
                        buffer: indirectBuffer,
                        offset: indirectOffset,
                    },
                },
                {
                    binding: 1, // GlobalHistogram
                    resource: {
                        buffer: storageBuffer,
                        offset: histogramOffset,
                        size: globalHistogramSize,
                    },
                },
                {
                    binding: 2, // PartitionHistogram
                    resource: {
                        buffer: storageBuffer,
                        offset: histogramOffset + globalHistogramSize,
                        size: histogramSize - globalHistogramSize,
                    },
                },
                {
                    binding: 3, // KeysIn
                    resource: {
                        buffer: keysBuffer,
                        offset: keysOffset,
                    },
                },
                {
                    binding: 4, // KeysOut
                    resource: {
                        buffer: storageBuffer,
                        offset: inoutOffset,
                        size: inoutSize,
                    },
                },
                {
                    binding: 5, // ValuesIn
                    resource: {
                        buffer: valuesBuffer,
                        offset: valuesOffset,
                    },
                },
                {
                    binding: 6, // ValuesOut
                    resource: {
                        buffer: storageBuffer,
                        offset: inoutOffset + inoutSize,
                        size: inoutSize,
                    },
                },
                {
                    binding: 7, // pass
                    resource: {
                        buffer: this._passBuffer,
                        offset: 0,
                        size: this._dynamicUniformAlignment
                    },
                },
            ],
        });
        this._bindGroup1 = this._device.createBindGroup({
            layout: this._bindGroupLayout,
            entries: [
                {
                    binding: 0, // ElementCount
                    resource: {
                        buffer: indirectBuffer,
                        offset: indirectOffset,
                    },
                },
                {
                    binding: 1, // GlobalHistogram
                    resource: {
                        buffer: storageBuffer,
                        offset: histogramOffset,
                        size: globalHistogramSize,
                    },
                },
                {
                    binding: 2, // PartitionHistogram
                    resource: {
                        buffer: storageBuffer,
                        offset: histogramOffset + globalHistogramSize,
                        size: histogramSize - globalHistogramSize,
                    },
                },
                {
                    binding: 4, // KeysIn
                    resource: {
                        buffer: keysBuffer,
                        offset: keysOffset,
                    },
                },
                {
                    binding: 3, // KeysOut
                    resource: {
                        buffer: storageBuffer,
                        offset: inoutOffset,
                        size: inoutSize,
                    },
                },
                {
                    binding: 6, // ValuesIn
                    resource: {
                        buffer: valuesBuffer,
                        offset: valuesOffset,
                    },
                },
                {
                    binding: 5, // ValuesOut
                    resource: {
                        buffer: storageBuffer,
                        offset: inoutOffset + inoutSize,
                        size: inoutSize,
                    },
                },
                {
                    binding: 7, // pass
                    resource: {
                        buffer: this._passBuffer,
                        offset: 0,
                        size: this._dynamicUniformAlignment
                    },
                },
            ],
        });
    }

    gpuSort(commandEncoder, elementCount, 
        storageBuffer, storageOffset) 
    {
        const partitionCount = RadixSorter.RoundUp(elementCount, RadixSorter.PARTITION_SIZE);
        const histogramOffset = storageOffset;

        commandEncoder.clearBuffer(
            storageBuffer,
            histogramOffset,
            4 * RadixSorter.RADIX * Uint32Array.BYTES_PER_ELEMENT,
        );

        const computePass = commandEncoder.beginComputePass();
        for (let i = 0; i < 4; ++i) {
            computePass.setBindGroup(3, i % 2 == 0 ? this._bindGroup0 : this._bindGroup1, [i * this._dynamicUniformAlignment]);
        
            // Upsweep
            computePass.setPipeline(this._upsweepPipeline);
            computePass.dispatchWorkgroups(partitionCount, 1, 1);
        
            // Spine
            computePass.setPipeline(this._spinePipeline);
            computePass.dispatchWorkgroups(RadixSorter.RADIX, 1, 1);
        
            // Downsweep
            computePass.setPipeline(this._downsweepPipeline);
            computePass.dispatchWorkgroups(partitionCount, 1, 1);
        }

        computePass.end();
    }

    gpuCheck(commandEncoder, elementCount) 
    {
        commandEncoder.clearBuffer(
            this._resultBuffer,
            0,
            4,
        );
        const computePass = commandEncoder.beginComputePass();

        computePass.setBindGroup(3, this._bindGroup_checker);
        computePass.setPipeline(this._checkerPipeline);
        computePass.dispatchWorkgroups(Math.min(this._maxComputeWorkgroupsPerDimension, RadixSorter.RoundUp(elementCount, RadixSorter.WORKGROUP_SIZE)), 1, 1);

        computePass.end();

        commandEncoder.copyBufferToBuffer(this._resultBuffer, 0, this._stagingBuffer, 0, 4);
    }

    async checkResult() {
        await this._stagingBuffer.mapAsync(GPUMapMode.READ);
        const mappedData = new Uint32Array(this._stagingBuffer.getMappedRange());
        const newData = mappedData[0];
        this._stagingBuffer.unmap();
        return newData;
    }
}

export {RadixSorter, VrdxSorterStorageRequirements};