#ifndef QSG_MPS_ENTANGLE_H_
#define QSG_MPS_ENTANGLE_H_

#include <vector>
#include "hnn_simd_common.h"

namespace highnoon {
namespace ops {

/**
 * @brief Computes fused MPS context entanglement.
 * 
 * Performs O(N·χ²) contraction of Matrix Product State context entanglement.
 * 
 * @param embeddings Input embeddings [batch_size, seq_len, embedding_dim]
 * @param site_weights MPS site tensors [batch_size, seq_len, bond_dim, phys_dim, bond_dim]
 * @param context_out Output entangled context [batch_size, seq_len, embedding_dim]
 * @param entropy_out Output bond entropy [batch_size, seq_len-1]
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param embedding_dim Embedding dimension
 * @param bond_dim Bond dimension (chi)
 * @param phys_dim Physical dimension (d)
 */
void qsg_mps_context_entangle(
    const float* embeddings,
    const float* site_weights,
    float* context_out,
    float* entropy_out,
    int batch_size,
    int seq_len,
    int embedding_dim,
    int bond_dim,
    int phys_dim
);

} // namespace ops
} // namespace highnoon

#endif // QSG_MPS_ENTANGLE_H_
