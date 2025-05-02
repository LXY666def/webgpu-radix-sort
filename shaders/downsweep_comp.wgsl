enable subgroups;

const RADIX: u32 = 256;
const WORKGROUP_SIZE: u32 = 512;
const PARTITION_DIVISION: u32 = 8;
const PARTITION_SIZE: u32 = PARTITION_DIVISION * WORKGROUP_SIZE;

@group(3) @binding(0) var<storage, read> ElementCount: array<u32>;
@group(3) @binding(1) var<storage, read_write> GlobalHistogram: array<u32>;
@group(3) @binding(2) var<storage, read_write> PartitionHistogram: array<u32>;
@group(3) @binding(3) var<storage, read_write> KeysIn: array<u32>;
@group(3) @binding(4) var<storage, read_write> KeysOut: array<u32>;
@group(3) @binding(5) var<storage, read_write> ValuesIn: array<u32>;
@group(3) @binding(6) var<storage, read_write> ValuesOut: array<u32>;
@group(3) @binding(7) var<uniform> _pass: u32;

const SHMEM_SIZE: u32 = PARTITION_SIZE;

var<workgroup> localHistogram: array<atomic<u32>, SHMEM_SIZE>;
var<workgroup> localHistogramSum: array<u32, RADIX>;

fn GetExclusiveSubgroupMask(id: u32) -> vec4<u32> {
    let shift = (1u << extractBits(id, 0, 5)) - 1u;  // (1 << (id % 32)) - 1
    let x = i32(id) >> 5;
    return vec4<u32>(
        (shift & u32((-1 -x) >> 31)) | u32((0 - x) >> 31),
        (shift & u32((0 - x) >> 31)) | u32((1 - x) >> 31),
        (shift & u32((1 - x) >> 31)) | u32((2 - x) >> 31),
        (shift & u32((2 - x) >> 31)) | u32((3 - x) >> 31)
    );
}

fn GetBitCount(value: vec4<u32>) -> u32 {
    let result = countOneBits(value);
    return result.x + result.y + result.z + result.w;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(workgroup_id) workgroupId: vec3<u32>,
    @builtin(subgroup_invocation_id) threadIndex: u32,
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(subgroup_size) subgroupSize: u32,
) {
    let numSubgroups = WORKGROUP_SIZE / subgroupSize;
    let subgroupIndex = lid.x / subgroupSize;
    let index = subgroupIndex * subgroupSize + threadIndex;
    let subgroupMask = GetExclusiveSubgroupMask(threadIndex);
    
    let partitionIndex = workgroupId.x;
    let partitionStart = partitionIndex * PARTITION_SIZE;

    let elementCount = ElementCount[0];

    if (partitionStart >= elementCount) {
        return;
    }

    if (index < RADIX) {
        for (var i: u32 = 0u; i < numSubgroups; i = i + 1u) {
            atomicStore(&localHistogram[numSubgroups * index + i], 0u);
        }
    }
    workgroupBarrier();

    var localKeys: array<u32, PARTITION_DIVISION>;
    var localRadix: array<u32, PARTITION_DIVISION>;
    var localOffsets: array<u32, PARTITION_DIVISION>;
    var subgroupHistogram: array<u32, PARTITION_DIVISION>;
    var localValues: array<u32, PARTITION_DIVISION>;

    for (var i: u32 = 0u; i < PARTITION_DIVISION; i = i + 1u) {
        let keyIndex = partitionStart + (PARTITION_DIVISION * subgroupSize) * subgroupIndex + i * subgroupSize + threadIndex;
        let key = select(0xffffffff, KeysIn[keyIndex], keyIndex < elementCount);
        localKeys[i] = key;

        let value = select(0u, ValuesIn[keyIndex], keyIndex < elementCount);
        localValues[i] = value;

        let radix = extractBits(key, _pass * 8, 8);
        localRadix[i] = radix;

        var mask = subgroupBallot(true);
        for (var j: u32 = 0u; j < 8u; j = j + 1u) {
            let digit = (radix >> j) & 1u;
            let ballot = subgroupBallot(digit == 1u);
            mask &= vec4<u32>(digit - 1u) ^ ballot;
        }

        let subgroupOffset = GetBitCount(subgroupMask & mask);
        let radixCount = GetBitCount(mask);

        if (subgroupOffset == 0u) {
            atomicAdd(&localHistogram[numSubgroups * radix + subgroupIndex], radixCount);
            subgroupHistogram[i] = radixCount;
        } else {
            subgroupHistogram[i] = 0u;
        }

        localOffsets[i] = subgroupOffset;
    }
    workgroupBarrier();
    var loop_i = index;
    for(var i: u32 = 0; i < RADIX * numSubgroups / WORKGROUP_SIZE; i = i + 1) {
        let v = select(0u, atomicLoad(&localHistogram[loop_i]), loop_i < RADIX * numSubgroups);
        let sum = subgroupAdd(v);
        let excl = subgroupExclusiveAdd(v);
        if (loop_i < RADIX * numSubgroups) {
            atomicStore(&localHistogram[loop_i], excl);
            if (threadIndex == 0u) {
                localHistogramSum[loop_i / subgroupSize] = sum;
            }
        }
        loop_i += WORKGROUP_SIZE;
    }

    workgroupBarrier();

    let intermediateOffset0 = RADIX * numSubgroups / subgroupSize;
    let v = select(0u, localHistogramSum[index], index < intermediateOffset0);
    let sum = subgroupAdd(v);
    let excl = subgroupExclusiveAdd(v);
    if (index < intermediateOffset0) {
        localHistogramSum[index] = excl;
        if (threadIndex == 0u) {
            localHistogramSum[intermediateOffset0 + index / subgroupSize] = sum;
        }
    }
    workgroupBarrier();

    let intermediateSize1 = max(RADIX * numSubgroups / subgroupSize / subgroupSize, 1u);
    let v1 = select(0u, localHistogramSum[intermediateOffset0 + index], index < intermediateSize1);
    let excl1 = subgroupExclusiveAdd(v1);
    if (index < intermediateSize1) {
        localHistogramSum[intermediateOffset0 + index] = excl1;
    }
    workgroupBarrier();

    if (index < intermediateOffset0) {
        localHistogramSum[index] += localHistogramSum[intermediateOffset0 + index / subgroupSize];
    }
    workgroupBarrier();

    for (var i: u32 = index; i < RADIX * numSubgroups; i = i + WORKGROUP_SIZE) {
        atomicAdd(&localHistogram[i], localHistogramSum[i / subgroupSize]);
    }
    workgroupBarrier();

    for (var i: u32 = 0u; i < PARTITION_DIVISION; i = i + 1u) {
        let radix = localRadix[i];
        localOffsets[i] += atomicLoad(&localHistogram[numSubgroups * radix + subgroupIndex]);

        workgroupBarrier();
        if (subgroupHistogram[i] > 0u) {
            atomicAdd(&localHistogram[numSubgroups * radix + subgroupIndex], subgroupHistogram[i]);
        }
        workgroupBarrier();
    }

    if (index < RADIX) {
        let v = select(atomicLoad(&localHistogram[numSubgroups * index - 1u]), 0u, index == 0u);
        localHistogramSum[index] = GlobalHistogram[RADIX * _pass + index] + PartitionHistogram[RADIX * partitionIndex + index] - v;
    }
    workgroupBarrier();

    for (var i: u32 = 0u; i < PARTITION_DIVISION; i = i + 1u) {
        atomicStore(&localHistogram[localOffsets[i]], localKeys[i]);
    }
    workgroupBarrier();

    for (var i: u32 = index; i < PARTITION_SIZE; i = i + WORKGROUP_SIZE) {
        let key = atomicLoad(&localHistogram[i]);
        let radix = extractBits(key, _pass * 8, 8);
        let dstOffset = localHistogramSum[radix] + i;
        if (dstOffset < elementCount) {
            KeysOut[dstOffset] = key;
        }
        localKeys[i / WORKGROUP_SIZE] = dstOffset;
    }

    workgroupBarrier();

    for (var i: u32 = 0u; i < PARTITION_DIVISION; i = i + 1u) {
        atomicStore(&localHistogram[localOffsets[i]], localValues[i]);
    }
    workgroupBarrier();

    for (var i: u32 = index; i < PARTITION_SIZE; i = i + WORKGROUP_SIZE) {
        let value = atomicLoad(&localHistogram[i]);
        let dstOffset = localKeys[i / WORKGROUP_SIZE];
        if (dstOffset < elementCount) {
            ValuesOut[dstOffset] = value;
        }
    }
}