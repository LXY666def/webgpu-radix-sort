const checker_comp_wgsl = `
@group(3) @binding(0) var<storage, read> ElementCount: array<u32>;
@group(3) @binding(1) var<storage, read_write> KeysIn: array<u32>;
@group(3) @binding(2) var<storage, read_write> Result: array<atomic<u32>>;

const WORKGROUP_SIZE: u32 = 512;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(num_workgroups) workgroup_num: vec3u,
) {
    let total_thread = workgroup_num.x * WORKGROUP_SIZE;
    let total_element = ElementCount[0];
    let workLoadPerThread = u32(ceil(f32(total_element) / f32(total_thread)));
    let thread_index = gid.x;
    for (var i: u32 = 0; i < workLoadPerThread && thread_index + i < total_element - 1; i = i + 1u) {
        if (KeysIn[thread_index + i] > KeysIn[thread_index + i + 1]) {
            atomicAdd(&Result[0], 1u);
        }
    }
}`;
export default checker_comp_wgsl;
