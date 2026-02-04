// PCG (https://www.pcg-random.org/)
uint pcg(uint v) {
	uint state = v * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

struct RNG {
	uint mState;

    __init(const uint seed) {
        mState = seed;
	}

	[mutating]
	void skipNext(const uint n = 1) {
		mState += n;
	}

	[mutating]
    uint nextUint() {
        skipNext(1);
        return pcg(mState);
    }

    [mutating]
    float nextFloat() {
        return (asfloat(0x3f800000 | ((nextUint()) >> 9)) - 1);
    }
};

uint sampleArrayIndex(inout RNG rng, const uint count)
{
    return min(uint(rng.nextFloat() * count), count - 1);
}

float2 sampleBoxMuller(inout RNG rng)
{
    let angle = 2 * 3.14159265 * rng.nextFloat();
    return sqrt(-2.0f * log(1 - rng.nextFloat())) * float2(cos(angle), sin(angle));
}