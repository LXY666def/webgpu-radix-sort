# webgpu-radix-sort
This is a WebGPU version of [vulkan_radix_sort](https://github.com/jaesung-cs/vulkan_radix_sort/tree/3faac413f643ca799f42950af6f7fb9c12913003). For best performance, please make sure your browser supports feature ***"subgroups"***.
## TODO
- [x] u32 key-value sorting supported now
- [ ] u32 key sorting supported now
## Benchmark
...
## Implementation
`wgpu-radix-sort.js` supports two computePass:  
* **RadixSorter.gpuSort(commandEncoder, elementCount, storageBuffer, storageOffset)**
  * desc: perform gpu based radix sort
  * param:
    * `commandEncoder`: commandEncoder to encode commands.
    * `elementCount`: count of elements to be sorted. (or this value may be bigger than the value in param `indirectBuffer` from `RadixSorter.createBindGroup`, in this way the real count of elements to be sorted is the value in `indirectBuffer`)
    * `storageBuffer`: additional memory space reserved for RadixSorter.
    * `storageOffset`: offset of `storageBuffer`.
* **RadixSorter.gpuCheck(commandEncoder, elementCount)**
  * desc: check if a sequence is in order using gpu
  * param:
    * `commandEncoder`: commandEncoder to encode commands.
    * `elementCount`: count of elements to be sorted. (or this value may be bigger than the value in param `indirectBuffer` from `RadixSorter.createBindGroup`, in this way the real count of elements to be sorted is the value in `indirectBuffer`)

## Sample
```js
import Device from './webgpu-radix-sort/Device.js';
import {RadixSorter, VrdxSorterStorageRequirements} from './webgpu-radix-sort/wgpu-radix-sort.js'

function generateAndShuffleArray(n) {
    let array = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
        array[i] = i + 1;
    }

    for (let i = n - 1; i > 0; i--) {
        let j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }

    return array;
}

async function main() {
    const gpuDevice = new Device();

    if (await gpuDevice.initialize()) {
        
    } else {
        console.error("fail to init WebGPU");
    }

    const radixSorter = new RadixSorter(gpuDevice.adapter, gpuDevice.device);
    const num = 128;

    /* query requirements for and create storage buffer */
    const reqs = radixSorter.vrdxGetSorterKeyValueStorageRequirements(num);
    const storageBuffer = gpuDevice.createBuffer(reqs.size, reqs.usage);

    /* create indirect buffer which stores the total count of elements to be sorted (should be of 'storage' buffer type) */
    const numBuffer = gpuDevice.createBufferAndFill(GPUBufferUsage.STORAGE, new Uint32Array([num]));

    const data = generateAndShuffleArray(num);
    // console.log(data);
    
    const stagingBuffer = gpuDevice.createBuffer(4 * num, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);

    /* create key and value buffer (should be of 'storage' buffer type) */
    const keysBuffer = gpuDevice.createBuffer(4 * num, GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const valsBuffer = gpuDevice.createBuffer(4 * num, GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    gpuDevice.device.queue.writeBuffer(keysBuffer, 0, data);
    gpuDevice.device.queue.writeBuffer(valsBuffer, 0, data);

    /* create bindgroups and checker (num >= value in numBuffer) */
    radixSorter.createBindGroup(num, numBuffer, 0, keysBuffer, 0, valsBuffer, 0, storageBuffer, 0);
    radixSorter.createChecker(numBuffer, 0, keysBuffer, 0);

    const commandEncoder = gpuDevice.device.createCommandEncoder();
    /* sort and check */
    radixSorter.gpuSort(commandEncoder, num, storageBuffer, 0);
    radixSorter.gpuCheck(commandEncoder, num);
    commandEncoder.copyBufferToBuffer(valsBuffer, 0, stagingBuffer, 0, 4 * num);
    const commandBuffer = commandEncoder.finish();
    gpuDevice.device.queue.submit([commandBuffer]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const mappedData = new Uint32Array(stagingBuffer.getMappedRange());
    // console.log("Staging Buffer Data:", mappedData);
    stagingBuffer.unmap();

    /* claim check result: pass if result == 0 else fail */
    const result = await radixSorter.checkResult();
    console.log("check result:", result);

}

main();
```