struct ParticleMap {
    RWStructuredBuffer<uint>  data;
    RWStructuredBuffer<uint>  sorted_data;
    RWStructuredBuffer<uint2> data_indices; // cell index, index in cell
    RWStructuredBuffer<uint>  data_counters; // data size, prefix sum
    RWStructuredBuffer<uint>  cell_data_offsets;
    RWStructuredBuffer<uint>  cell_counters;
    uniform int2 grid_size;
    uniform int2 grid_offset;
    uniform uint capacity;

    // get an element at an index. use getRange() to get an index range for a cell.
    uint get(const uint i)
    {
        return sorted_data[i];
    }

    // get the index range of elements in a cell
    uint2 get_range(int2 cell)
    {
        cell += grid_offset;
        if (any(cell < 0) || any(cell >= grid_size))
            return -1;
        const uint offset = cell_data_offsets[cell.y * grid_size.x + cell.x];
        return uint2(offset, offset + cell_counters[cell.y * grid_size.x + cell.x]);
    }

    bool insert(int2 cell, const uint particle_index)
    {
        cell += grid_offset;
        if (any(cell < 0) || any(cell >= grid_size))
            return false;
        
        uint data_index;
        InterlockedAdd(data_counters[0], 1, data_index);
        if (data_index >= capacity)
            return false;

        uint index_in_cell;
        InterlockedAdd(cell_counters[cell.y * grid_size.x + cell.x], 1, index_in_cell);

        data[data_index] = particle_index;
        data_indices[data_index] = uint2(cell.y * grid_size.x + cell.x, index_in_cell);

        return true;
    }
};