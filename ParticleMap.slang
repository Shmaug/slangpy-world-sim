#pragma once

struct Particle
{
    float pressure;
    uint  packed_pos;
    float2 velocity;

    // [0-1) position within the grid cell
    property float2 pos_in_cell
    {
        get { return unpackUnorm2x16ToFloat(packed_pos); }
        set { packed_pos = packUnorm2x16(newValue); }
    }
};

struct ParticleMap
{
    RWStructuredBuffer<Particle> data;
    RWStructuredBuffer<Particle> sorted_data;
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
    Particle get(const uint i)
    {
        return sorted_data[i];
    }

    bool insert(const uint cell, const Particle particle)
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
        data[i] = particle;

        return true;
    }
};
