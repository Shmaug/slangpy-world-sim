#pragma once

struct ParticleMap
{
    RWStructuredBuffer<float4> data;
    RWStructuredBuffer<float4> sorted_data;
    RWStructuredBuffer<uint2>  data_indices;
    RWStructuredBuffer<uint>   cell_counters;
    RWStructuredBuffer<uint>   cell_offsets;
    RWStructuredBuffer<uint>   counters;
    uint max_cells;
    uint max_data;

    // get the index range of elements in a cell
    uint2 get_range(const uint cell)
    {
        const uint offset = cell_offsets[cell];
        return uint2(offset, offset + cell_counters[cell]);
    }

    // get an element at an index. use getRange() to get an index range for a cell.
    float4 get(const uint i)
    {
        return sorted_data[i];
    }

    bool insert(const uint cell, const float4 v)
    {
        uint i;
        InterlockedAdd(counters[0], 1, i);
        if (i >= max_data)
        {
            return false;
        }

        uint indexInCell;
        InterlockedAdd(cell_counters[cell], 1, indexInCell);

        data_indices[i] = uint2(cell, indexInCell);
        data[i] = v;

        return true;
    }
};
