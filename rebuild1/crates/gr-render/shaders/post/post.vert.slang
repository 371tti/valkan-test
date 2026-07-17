#include "slang_compat.slang"

layout(location = 0) out float2 frag_uv;

static const float2 positions[3] = float2[](
    float2(-1.0, -1.0),
    float2( 3.0, -1.0),
    float2(-1.0,  3.0)
);

static const float2 uvs[3] = float2[](
    float2(0.0, 0.0),
    float2(2.0, 0.0),
    float2(0.0, 2.0)
);

[shader("vertex")]
void main(uint vertex_index : SV_VertexID, out float4 out_position : SV_Position) {
    frag_uv = uvs[vertex_index];
    out_position = float4(positions[vertex_index], 0.0, 1.0);
}
