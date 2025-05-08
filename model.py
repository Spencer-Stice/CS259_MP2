import math
import csv



def conv_perf_model_nominal(input_H, input_W, Ni, Kx, Ky, Nn, block_x, block_y, block_z, tile_in, core_x, core_y,
                            dtype_size_bytes,
                            dram_bandwidth_GBs=652.8,
                            peak_flops_TFLOPs=13.8):


    # HW specific params for titan v
    shared_mem_per_SM = 96 * 1024 
    registers_per_SM = 65536           
    threads_per_SM = 2048               
    max_blocks_per_SM = 32


    shared_mem_per_block = (block_x * block_y * block_z + Kx * Ky * block_z * tile_in) * dtype_size_bytes  # input + filters in bytes
    threads_per_block = block_x * block_y * block_z                            # 576 threads

    # Estimate ~30 regs per thread
    registers_per_thread = 30

    # Constraint calculations
    limit_shared = math.floor(shared_mem_per_SM / shared_mem_per_block)
    limit_regs = math.floor(registers_per_SM / (registers_per_thread * threads_per_block))
    limit_threads = math.floor(threads_per_SM / threads_per_block)

    blocks_per_SM = min(limit_shared, limit_regs, limit_threads, max_blocks_per_SM)

    threads_per_sm_program = blocks_per_SM * threads_per_block
    sm_thread_occupancy = threads_per_sm_program / threads_per_SM


    flops_per_output = Kx * Ky * Ni * 2  # 2 FLOPs per MAC
    flops_per_output_tile = Kx * Ky * tile_in * 2  # 2 FLOPs per MAC

    # Total output elements (same as input due to padding = 1)
    output_H, output_W = input_H, input_W
    num_outputs = output_H * output_W * Nn

    # Tiling dimensions
    til_num_input_bytes = (block_x * block_y * block_z) * dtype_size_bytes
    tile_synapse_bytes = ((Kx * Ky * block_z) * tile_in) * dtype_size_bytes
    tile_shared_outputs = core_x * core_y * block_z
    shared_bw = 1000e9 #1000 GB/s
    tile_op_intensity = (tile_shared_outputs * flops_per_output_tile) / (til_num_input_bytes + tile_synapse_bytes)
    tops_shared = tile_op_intensity * shared_bw / 1e12


    # Total FLOPs
    total_flops = num_outputs * flops_per_output

    # Total bytes
    input_bytes = (input_H + 2) * (input_W + 2) * Ni * dtype_size_bytes
    filter_bytes = Kx * Ky * Ni * Nn * dtype_size_bytes
    output_bytes = output_H * output_W * Nn * dtype_size_bytes
    total_bytes_dram = input_bytes + filter_bytes + output_bytes

    # Operational intensity
    operational_intensity_dram = total_flops / total_bytes_dram

    l2_filter_reuse_fraction = 0.98
    l2_input_reuse_fraction = 0.75  

    adjusted_dram_bytes = output_bytes + input_bytes * (1 - l2_input_reuse_fraction) + filter_bytes * (1 - l2_filter_reuse_fraction)

    Oi_L2 = total_flops / adjusted_dram_bytes
    TOps_L2 = Oi_L2 * dram_bandwidth_GBs * 1e9 / 1e12
    




    # Times
    dram_bandwidth_Bps = dram_bandwidth_GBs * 1e9
    peak_flops = peak_flops_TFLOPs * 1e12

    memory_bound_time = total_bytes_dram / dram_bandwidth_Bps
    compute_bound_time = total_flops / peak_flops

    # TOps
    tops_mem_bound = total_flops / memory_bound_time / 1e12
    tops_compute_bound = total_flops / compute_bound_time / 1e12  # will equal peak TFLOPs

    max_tops = min(tops_mem_bound, tops_compute_bound, tops_shared)


    active_thread_ratio = (core_x * core_y * block_z) / (block_x * block_y * block_z)

    #constant estimations
    inst_mix_ratio = 0.7    #indexing, control flow, etc.
    mem_inefficiency = 0.8  #uncoalesced memory accesses, bank conflicts, etc.
    sync_overhead = 0.9     #synchthreads


    achieved_tops = max_tops * sm_thread_occupancy * active_thread_ratio * inst_mix_ratio * mem_inefficiency * sync_overhead

    return {
        "Total FLOPs": total_flops,
        "Total DRAM bytes": total_bytes_dram,
        "Operational Intensity (DRAM)": operational_intensity_dram,
        "Operational Intensity (shared memory)": tile_op_intensity,
        "Operational Intensity (L2-adjusted)": Oi_L2,
        "Memory-bound time (s)": memory_bound_time,
        "Compute-bound time (s)": compute_bound_time,
        "Max TOps (DRAM-bound)": tops_mem_bound,
        "Max TOps (compute-bound)": tops_compute_bound,
        "Max TOps (scratchpad-bound)": tops_shared,
        "Max TOps (L2-adjusted)": TOps_L2,
        "Max TOps (overall)": max_tops,
        "Achieved TOps": achieved_tops,
        "Max blocks per SM": blocks_per_SM,
        "Per SM occupancy": sm_thread_occupancy,
    }

rows = [
    (72, 4, 16),
    (72, 9, 4),
    (72, 12, 4),
    (200, 4, 16),
    (200, 5, 8),
    (200, 8, 4),
    (320, 4, 16),
    (320, 5, 8),
    (320, 8, 4),
]

results_list = []

for inp_spat, block_spat, z_tile in rows:
    results = conv_perf_model_nominal(
        input_H=inp_spat, input_W=inp_spat, Ni=64,
        Kx=3, Ky=3, Nn=64, block_x=block_spat, block_y=block_spat, block_z=z_tile,
        tile_in=z_tile, core_x=block_spat - 2, core_y=block_spat - 2,
        dtype_size_bytes=4
    )

    row_data = {
        "input_spat": inp_spat,
        "block_spat": block_spat,
        "z_tile": z_tile,
        **results  # Merge the performance model results into the row
    }
    results_list.append(row_data)

# Write to CSV
fieldnames = list(results_list[0].keys())
with open("model_preds.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results_list)

print("Results written to performance_results.csv")