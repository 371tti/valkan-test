#pragma once

float4 sample_texture(Sampler2D texture_sampler, float2 uv)
{
    return texture_sampler.Sample(uv);
}

float4 sample_texture(SamplerCube texture_sampler, float3 direction)
{
    return texture_sampler.Sample(direction);
}

float4 sample_texture_level(Sampler2D texture_sampler, float2 uv, float level)
{
    return texture_sampler.SampleLevel(uv, level);
}

float4 load_texel(Sampler2D texture_sampler, int2 coordinate, int level)
{
    return texture_sampler.Load(int3(coordinate, level));
}

int2 texture_dimensions(Sampler2D texture_sampler, int level)
{
    uint width;
    uint height;
    uint mip_count;
    texture_sampler.GetDimensions(uint(level), width, height, mip_count);
    return int2(width, height);
}
