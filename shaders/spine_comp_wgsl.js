const spine_comp_wgsl = `
enable subgroups;

const RADIX: u32 = 256;
const MAX_SUBGROUP_SIZE: u32 = 128;
const WORKGROUP_SIZE: u32 = 512;
const PARTITION_DIVISION: u32 = 8;
const PARTITION_SIZE: u32 = PARTITION_DIVISION * WORKGROUP_SIZE;

@group(3) @binding(0) var<storage, read> ElementCount: array<u32>;
@group(3) @binding(1) var<storage, read_write> GlobalHistogram: array<u32>;
@group(3) @binding(2) var<storage, read_write> PartitionHistogram: array<u32>;
@group(3) @binding(7) var<uniform> _pass: u32;

var<workgroup> reduction: u32;
var<workgroup> intermediate: array<u32, MAX_SUBGROUP_SIZE>;

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
    let radix = workgroupId.x;

    let elementCount = ElementCount[0];
    let partitionCount = (elementCount + PARTITION_SIZE - 1u) / PARTITION_SIZE;

    if (index == 0u) {
        reduction = 0u;
    }
    workgroupBarrier();

    for (var i: u32 = 0u; WORKGROUP_SIZE * i < partitionCount; i = i + 1u) {
        let partitionIndex = WORKGROUP_SIZE * i + index;
        var value = select(0u, PartitionHistogram[RADIX * partitionIndex + radix], partitionIndex < partitionCount);
        var excl = subgroupExclusiveAdd(value) + reduction;
        let sum = subgroupAdd(value);

        if (subgroupElect()) {
            intermediate[subgroupIndex] = sum;
        }
        workgroupBarrier();

        let excl_1 = subgroupExclusiveAdd(select(0u, intermediate[index], index < numSubgroups));
        let sum_1 = subgroupAdd(select(0u, intermediate[index], index < numSubgroups));
        if (index < numSubgroups) {
            intermediate[index] = excl_1;

            if (index == 0u) {
                reduction += sum_1;
            }
        }
        workgroupBarrier();

        if (partitionIndex < partitionCount) {
            excl += intermediate[subgroupIndex];
            PartitionHistogram[RADIX * partitionIndex + radix] = excl;
        }
        workgroupBarrier();
    }

    if (workgroupId.x == 0u) {
        let value = select(0u, GlobalHistogram[RADIX * _pass + index], index < RADIX);
        var excl = subgroupExclusiveAdd(value);
        let sum = subgroupAdd(value);
        if (subgroupElect() && index < RADIX) {
            intermediate[subgroupIndex] = sum;
        }
        workgroupBarrier();
        let excl_1 = subgroupExclusiveAdd(select(0u, intermediate[index], index < RADIX / subgroupSize));
        if (index < RADIX / subgroupSize) {
            intermediate[index] = excl_1;
        }
        workgroupBarrier();
        if (index < RADIX) {
            excl += intermediate[subgroupIndex];
            GlobalHistogram[RADIX * _pass + index] = excl;
        }
    }
}`;
export default spine_comp_wgsl;
