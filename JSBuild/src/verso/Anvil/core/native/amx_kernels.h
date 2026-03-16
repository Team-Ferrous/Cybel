#pragma once

namespace anvil::native {

bool amx_compiled();
bool amx_runtime_available();
bool amx_matmul_f32(
    const float* a,
    const float* b,
    float* c,
    int m,
    int k,
    int n
);

}  // namespace anvil::native

extern "C" {

int anvil_compiled_with_amx();
int anvil_runtime_amx_available();

}
