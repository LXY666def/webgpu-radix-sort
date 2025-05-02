const upsweep_comp_wgsl = `
enable subgroups;

@group(3) @binding(0) var<storage, read> ElementCount: array<u32>;
@group(3) @binding(1) var<storage, read_write> GlobalHistogram: array<atomic<u32>>;
@group(3) @binding(2) var<storage, read_write> PartitionHistogram: array<u32>;
@group(3) @binding(3) var<storage, read_write> KeysIn: array<u32>;
@group(3) @binding(7) var<uniform> _pass: u32;

const RADIX: u32 = 256;
const WORKGROUP_SIZE: u32 = 512;
const PARTITION_DIVISION: u32 = 8;
const PARTITION_SIZE: u32 = PARTITION_DIVISION * WORKGROUP_SIZE;

var<workgroup> localHistogram: array<atomic<u32>, RADIX>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
  @builtin(subgroup_invocation_id) threadIndex: u32,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(subgroup_size) subgroupSize: u32,
  @builtin(workgroup_id) workgroup_id: vec3u,
) {
    let subgroupIndex = lid.x / subgroupSize;
    let index = subgroupIndex * subgroupSize + threadIndex;
    let elementCount = ElementCount[0];
    let partitionIndex = workgroup_id.x;
    let partitionStart = partitionIndex * PARTITION_SIZE;

    // Discard all workgroup invocations
    if (partitionStart >= elementCount) {
        return;
    }

    if (index < RADIX) {
        atomicStore(&localHistogram[index], 0u);
    }
    workgroupBarrier();

    // Local histogram
    for (var i: u32 = 0u; i < PARTITION_DIVISION; i = i + 1u) {
        let keyIndex = partitionStart + WORKGROUP_SIZE * i + index;
        let key = select(0xffffffffu, KeysIn[keyIndex], keyIndex < elementCount);
        let radix = extractBits(key, 8 * _pass, 8);
        atomicAdd(&localHistogram[radix], 1u);
    }
    workgroupBarrier();

    if (index < RADIX) {
        // Set to partition histogram
        PartitionHistogram[RADIX * partitionIndex + index] = atomicLoad(&localHistogram[index]);

        // Add to global histogram
        atomicAdd(&GlobalHistogram[RADIX * _pass + index], atomicLoad(&localHistogram[index]));
    }
}`;
export default upsweep_comp_wgsl;
