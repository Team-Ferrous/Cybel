module KernelLib = {
  -- thermal_conservation_loss.fut
  -- 3D Laplacian for interior points of a 3D tensor
  let laplacian3d (T: [][][]f32) (dx: f32) (dy: f32) (dz: f32) : [][][]f32 =
    let nx = length T
    let ny = length (T[0])
    let nz = length (T[0][0])
    -- interior: 1..nx-2, 1..ny-2, 1..nz-2
    in [i | i <- 1..nx-2]    |> map (\ix ->
        [j | j <- 1..ny-2]   |> map (\jy ->
          [k | k <- 1..nz-2] |> map (\kz ->
            let d2x = (T[ix+1][jy][kz] - 2.0f32 * T[ix][jy][kz] + T[ix-1][jy][kz]) / (dx*dx)
            let d2y = (T[ix][jy+1][kz] - 2.0f32 * T[ix][jy][kz] + T[ix][jy-1][kz]) / (dy*dy)
            let d2z = (T[ix][jy][kz+1] - 2.0f32 * T[ix][jy][kz] + T[ix][jy][kz-1]) / (dz*dz)
            in  d2x + d2y + d2z
          )
        )
      )

  -- Main thermal conservation loss
  let thermal_conservation_loss
    (T_pred: [][][][]f32)      -- [batch, nx, ny, nz]
    (Q_joule: [][][][]f32)
    (k_th: [][][][]f32)        -- can be scalar broadcasted if needed
    (dx: f32) (dy: f32) (dz: f32)
    (lambda_thermal: f32) (epsilon_th: f32)
    : f32 =
    let batch_size = length T_pred
    in [b | b <- 0..batch_size-1] |> map (\b ->
        let lap = laplacian3d T_pred[b] dx dy dz
        let Q_int = Q_joule[b][1..length lap+0][1..length (lap[0])+0][1..length (lap[0][0])+0]
        let k_int = k_th[b][1..length lap+0][1..length (lap[0])+0][1..length (lap[0][0])+0]
        let residual = map3 (\t q k -> t + q/k) lap Q_int k_int
        let excess = map3 (\r _ _ -> f32.max 0.0 (abs r - epsilon_th)) residual residual residual
        let pointwise = map3 (\e _ _ -> e*e) excess excess excess
        in reduce (+) 0.0 (concat (concat pointwise)) -- mean can be done by dividing by total points
      )
    |> reduce (+) 0.0
    |> (\s -> lambda_thermal * s / f32((batch_size * (length (T_pred[0])-2) * (length (T_pred[0][0])-2) * (length (T_pred[0][0][0])-2))))

  -- thermal_boundary_loss.fut
  -- Compute thermal boundary loss
  let thermal_boundary_loss
    (T_pred: [][][][]f32)       -- [batch, nx, ny, nz]
    (T_boundary: [][][][]f32)
    (boundary_mask: [][][][]f32)
    (lambda_bc: f32) (reduction: string)
    : f32 =
    let batch_size = length T_pred
    let nx = length (T_pred[0])
    let ny = length (T_pred[0][0])
    let nz = length (T_pred[0][0][0])

    -- Compute squared error, masked
    let masked_error =
      [b | b <- 0..batch_size-1] |> map (\b ->
        [i | i <- 0..nx-1] |> map (\i ->
          [j | j <- 0..ny-1] |> map (\j ->
            [k | k <- 0..nz-1] |> map (\k ->
              let e = T_pred[b][i][j][k] - T_boundary[b][i][j][k]
              in (e*e) * boundary_mask[b][i][j][k]
            )
          )
        )
      )
      in
        if reduction == "none" then
          -- just return sum over interior? For 'none' we can flatten to same shape
          reduce (+) 0.0 (concat (concat (concat masked_error)))
        else
          let total_mask = reduce (+) 0.0 (concat (concat (concat boundary_mask))) + 1e-12
          let total_error = reduce (+) 0.0 (concat (concat (concat masked_error)))
          in if reduction == "sum" then
              lambda_bc * total_error
            else  -- 'mean'
              lambda_bc * total_error / total_mask

  
  -- ===================================================================
-- VAE losses in Futhark
-- ===================================================================

-- Mean squared error per sample (with optional validity mask)
let reconstruction_loss (x_true: [][][]f32) (x_pred: [][][]f32) (mask: [][]i32) : []f32 =
  map2 (\true pred ->
    let valid_mask = map f32 mask
    let sq_diff = map2 (\t p -> map2 (\tt pp -> (tt - pp)*(tt - pp)) t p) true pred
    let masked = map2 (\row m -> map (\v -> v*m) row m) sq_diff valid_mask
    let sum_masked = f32.reduce (+) 0.0 (concat masked)
    let num_valid = f32.reduce (+) 0.0 (concat valid_mask)
    sum_masked / max 1.0 num_valid
  ) x_true x_pred

-- KL divergence per sample: KL(N(mu, sigma) || N(0,1))
let kl_divergence_loss (mu: [][]f32) (logvar: [][]f32) : []f32 =
  map2 (\m lv ->
    f32.reduce (+) 0.0 (map2 (\mu_i logv_i -> -0.5 * (1.0 + logv_i - mu_i*mu_i - exp logv_i)) m lv)
  ) mu logvar

-- Physics penalty (minimum spacing)
let physics_penalty_loss
  (node_features: [][][]f32)
  (validity_mask: [][]i32)
  (d_min: f32)
  : []f32 =
  map2 (\nodes mask ->
    let positions = map (\feat -> feat[0..2]) nodes
    let max_N = length positions
    let distances =
      map (\i -> map (\j ->
        let dx = positions[i][0] - positions[j][0]
        let dy = positions[i][1] - positions[j][1]
        let dz = positions[i][2] - positions[j][2]
        sqrt(dx*dx + dy*dy + dz*dz)
      ) (iota max_N)) (iota max_N)
    let mask_2d =
      map (\i -> map (\j ->
        let valid = mask[i] * mask[j]
        let not_self = if i == j then 0 else 1
        f32(valid * not_self)
      ) (iota max_N)) (iota max_N)
    let violations =
      map (\i -> map (\j -> max(0.0, d_min - distances[i][j])) (iota max_N)) (iota max_N)
    let masked_violations =
      map2 (\v_row m_row -> map2 (*) v_row m_row) violations mask_2d
    let total_violation = f32.reduce (+) 0.0 (concat masked_violations)
    let num_pairs = f32.reduce (+) 0.0 (concat mask_2d)
    total_violation / max 1.0 num_pairs
  ) node_features validity_mask

  -- Total VAE loss per sample
  let vae_total_loss
    (node_features_true: [][][]f32)
    (node_features_pred: [][][]f32)
    (validity_mask: [][]i32)
    (mu: [][]f32)
    (logvar: [][]f32)
    (beta: f32)
    (lambda_physics: f32)
    (d_min: f32)
    : ([]f32, []f32, []f32, []f32) =
    let recon = reconstruction_loss node_features_true node_features_pred validity_mask
    let kl = kl_divergence_loss mu logvar
    let phys = physics_penalty_loss node_features_pred validity_mask d_min
    let total = map4 (\r k p _ -> r + beta*k + lambda_physics*p) recon kl phys recon  -- dummy last arg to satisfy map4
    in (total, recon, kl, phys)

  -- ===================================================================
  -- Optional: Linear MMD for features (simpler than RBF in Futhark)
  -- ===================================================================
  let mmd_loss_linear (source: [][]f32) (target: [][]f32) : f32 =
    let mean_s = map (\i -> f32.reduce (+) 0.0 (map (\row -> row[i]) source) / f32.length source) (iota (length source[0]))
    let mean_t = map (\i -> f32.reduce (+) 0.0 (map (\row -> row[i]) target) / f32.length target) (iota (length target[0]))
    f32.reduce (+) 0.0 (map2 (\ms mt -> (ms - mt)*(ms - mt)) mean_s mean_t)
    
  let u_em (E: [][][][][f32]) (B: [][][][][f32]) (epsilon_0: f32) (mu_0: f32) : [][][][]f32 =
  map (\b -> map (\i -> map (\j -> map (\k ->
    let E_sq = reduce (+) 0.0 (map (\c -> E[b][i][j][k][c]*E[b][i][j][k][c]) (0..2)))
    let B_sq = reduce (+) 0.0 (map (\c -> B[b][i][j][k][c]*B[b][i][j][k][c]) (0..2)))
    0.5 * (epsilon_0 * E_sq + B_sq / mu_0)
  ) (0..length E[b][i][j]-1)) (0..length E[b][i]-1)) (0..length E[b]-1))

  let poynting_vector (E: [][][][][f32]) (B: [][][][][f32]) (mu_0: f32) : [][][][][f32] =
    map (\b -> map (\i -> map (\j -> map (\k ->
      let S_x = (E[b][i][j][k][1]*B[b][i][j][k][2] - E[b][i][j][k][2]*B[b][i][j][k][1])/mu_0
      let S_y = (E[b][i][j][k][2]*B[b][i][j][k][0] - E[b][i][j][k][0]*B[b][i][j][k][2])/mu_0
      let S_z = (E[b][i][j][k][0]*B[b][i][j][k][1] - E[b][i][j][k][1]*B[b][i][j][k][0])/mu_0
      [S_x, S_y, S_z]
    ) (0..length E[b][i][j]-1)) (0..length E[b][i]-1)) (0..length E[b]-1))

  let divergence (S: [][][][][f32]) (dx: f32) (dy: f32) (dz: f32) : [][][]f32 =
    map (\b -> map (\i -> map (\j ->
      map (\k ->
        let dSx_dx = (S[b][i+1][j][k][0] - S[b][i-1][j][k][0]) / (2.0*dx)
        let dSy_dy = (S[b][i][j+1][k][1] - S[b][i][j-1][k][1]) / (2.0*dy)
        let dSz_dz = (S[b][i][j][k+1][2] - S[b][i][j][k-1][2]) / (2.0*dz)
        dSx_dx + dSy_dy + dSz_dz
      ) (1..length S[b][i][j]-2))
    ) (1..length S[b][i]-2)) (1..length S[b]-2))

  let ohmic_dissipation (J: [][][][][f32]) (E: [][][][][f32]) : [][][]f32 =
    map (\b -> map (\i -> map (\j ->
      map (\k -> reduce (+) 0.0 (map (\c -> J[b][i][j][k][c]*E[b][i][j][k][c]) (0..2)))
    ) (0..length J[b][i][j]-1)) (0..length J[b][i]-1)) (0..length J-1))
    let em_energy_loss (E: [][][][][f32]) (B: [][][][][f32]) (J: [][][][][f32])
                      (dx: f32) (dy: f32) (dz: f32) (lambda_em: f32) (reduction: string) : f32 =
      let S = poynting_vector E B 1.257e-6
      let div_S = divergence S dx dy dz
      let J_dot_E = ohmic_dissipation J E
      let residual =
        map2 (\d j -> map2 (\di ji -> map2 (\dij jij -> dij + jij) di ji) d j) div_S J_dot_E
      let pointwise_loss = map (\b -> map (\i -> map (\j -> map (\k -> residual[b][i][j][k]*residual[b][i][j][k]) ... )))  -- flatten & square
      -- Reduction: sum/mean/none

  let rbf (x: [n][d]f32) (y: [m][d]f32) (sigma: f32) : [n][m]f32 =
    map (\xi ->
      map (\yi ->
        let dist_sq = reduce (+) 0.0 (map2 (\a b -> (a-b)*(a-b)) xi yi)
        in exp(-dist_sq / (2.0*sigma*sigma))
      ) y
    ) x

  let linear (x: [n][d]f32) (y: [m][d]f32) : [n][m]f32 =
    map (\xi ->
      map (\yi ->
        reduce (+) 0.0 (map2 (*) xi yi)
      ) y
    ) x

  let polynomial (x: [n][d]f32) (y: [m][d]f32)
                 (c: f32) (p: i32) : [n][m]f32 =
    map (\xi ->
      map (\yi ->
        let dot = reduce (+) 0.0 (map2 (*) xi yi)
        in (dot + c) ** f32.p
      ) y
    ) x

  type Operation =
    | Laplacian
    | Sum
    | None

  entry dispatch_kernel
        (kind: i32)
        (x: [] [] f32)
        (y: [] [] f32)
        (sigma: f32)
        (c: f32)
        (p: i32)
        : [][]f32 =
    if kind == 0 then rbf x y sigma
    else if kind == 1 then linear x y
    else if kind == 2 then polynomial x y c p
    else replicate (length x) (replicate (length y) 0.0)
}