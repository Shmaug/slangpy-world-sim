static const float M_E        = 2.71828182845904523536;  // e
static const float M_LOG2E    = 1.44269504088896340736;  // log2(e)
static const float M_LOG10E   = 0.434294481903251827651; // log10(e)
static const float M_LN2      = 0.693147180559945309417; // ln(2)
static const float M_LN10     = 2.30258509299404568402;  // ln(10)
static const float M_PI       = 3.14159265358979323846;  // pi
static const float M_PI_2     = 1.57079632679489661923;  // pi/2
static const float M_PI_4     = 0.785398163397448309616; // pi/4
static const float M_1_PI     = 0.318309886183790671538; // 1/pi
static const float M_2_PI     = 0.636619772367581343076; // 2/pi
static const float M_1_SQRTPI = 0.564189583547756279280; // 1/sqrt(pi)
static const float M_2_SQRTPI = 1.12837916709551257390;  // 2/sqrt(pi)
static const float M_SQRT2    = 1.41421356237309504880;  // sqrt(2)
static const float M_SQRT1_2  = 0.707106781186547524401; // 1/sqrt(2)
static const float M_2PI     = 6.28318530717958647693;  // 2pi
static const float M_4PI     = 12.5663706143591729539;  // 4pi
static const float M_4_PI    = 1.27323954473516268615;  // 4/pi
static const float M_1_2PI   = 0.159154943091895335769; // 1/2pi
static const float M_1_4PI   = 0.079577471545947667884; // 1/4pi
static const float M_SQRT3   = 1.732050807568877293527; // sqrt(3)
static const float M_SQRTPI  = 1.77245385090551602730;  // sqrt(pi)
static const float M_1_SQRT2 = 0.707106781186547524401; // 1/sqrt(2)

float sqr(float a)
{
    return a * a;
}
float luminance(float3 rgb)
{
    return dot(rgb, float3(0.2126f, 0.7152f, 0.0722f));
}

bool intersect_sphere(float3 origin, float3 direction, float radius, out float2 t)
{
    // precise version
    float r2 = radius * radius;

    float DD = dot(direction, direction);
    float Df = dot(origin, direction);
    
    float DD2 = DD*DD;

    float3 l = DD * origin - direction*Df;
    float a2_r2 = DD2*r2;
    
    float ll = dot(l,l);

    if (a2_r2 < ll)
    {
        t = -1;
        return false;
    }
    
    float det = a2_r2 - ll;
    
    float rcp_a = 1.0 / DD;
    
    det = sqrt(det * rcp_a);
    
    t = (-Df + float2(-det, det)) * rcp_a;
    return t.y > 0;
}

float3 uv_to_xyz(float2 uv) {
    return float3(
        sin(uv.y * M_PI) * sin(uv.x * M_2PI),
        cos(uv.y * M_PI),
        sin(uv.y * M_PI) * cos(uv.x * M_2PI)
    );
}
float2 xyz_to_uv(float3 xyz)
{
    return float2(
        -atan2(xyz.x, -xyz.z) * M_1_2PI + 0.5,
        acos(xyz.y) * M_1_PI
    );
}
