import FluidGrid;
uniform FluidGrid grid;

StructuredBuffer<float> current_pressure;

[shader("compute")]
[numthreads(8, 8, 1)]
void pressure_correction(uint3 thread : SV_DispatchThreadID)
{
    uint2 cell = thread.xy;
    if (any(cell >= grid.resolution))
        return;

    uint w, h, mips;
    grid.pressure.GetDimensions(0, w, h, mips);
    float target = grid.pressure.Load(uint3(0, 0, mips - 1));
    float curr = current_pressure[0];
    grid.pressure_rw[cell] *= curr > 0 ? target / curr : 0;
}