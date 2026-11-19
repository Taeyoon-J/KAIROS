# The model, against the code

Every equation below is followed by the file and the expression that implements it. Symbols are
the ones used in the write-up; the code's own names are given beside them.

Shapes: an image is `[128, 128, 3]` uint8, pooled to `N = 64 x 64 = 4096` patches with 8
features each. Every patch carries `a = 1..4` oscillator components.

## 1. From image to patch activations

`vendor/snn_kuramoto_bidirectional/input_layer_generator.py` (`CNNFeatureEncoder`),
`gamma_initializer.py` (`FeaturePatchGammaInitializer`).

One 3x3 convolution with 8 trainable filters, no activation, standardised by a scalar mean and
std held in the model's `feature_mean` / `feature_std` buffers, clipped to `[-3, 3]`, then
adaptively average-pooled to the 64x64 grid. `FreshModel.encode` is the whole path.

## 2. Patch connection graph

`vendor/snn_kuramoto_bidirectional/graph_generator.py` (`ImageConditionedGraph`),
`vendor/full_benchmark/geodesic.py` (`distance_power2`).

    S_ij = z_i . z_j / tau - r d(i,j)
    B_ij = G softmax_{j in N_i}(S_ij),   A_ij = (B_ij + B_ji) / 2

`N_i` is the 512 highest-scoring neighbours (`graph_top_k=512` at grid 64). `tau`, `G` and `r`
come from trainable scalars; `d(i,j)` is the squared-geodesic distance, whose contrast is an
architectural constant here — `FreshModel.__init__` sets `graph.geodesic_contrast.requires_grad_(False)`
and overrides `_geodesic_distance` with the fixed-8 form.

## 3. Input phase

`FreshModel` -> `S2NetCore`, `vendor/snn_kuramoto_bidirectional/s2net_cls.py`.

    u_i = W_drive gamma_i,   phi_i = pi tanh(W_phi u_i_standardised),   theta_i(0) = phi_i

The drive and the graph are computed once and held fixed for the whole rollout.

## 4. Kuramoto layer

`vendor/snn_kuramoto_bidirectional/kuramoto_layer.py` (`graphVectorKuramoto`), called once per
step from `sw240/integration.py`.

    theta_i(t+1) = theta_i(t) + dt [ omega_i + f phi_i
                                   + kappa_i * sin(phi_i - theta_i(t))
                                   + (K/N) sum_j Abar_ij sin(theta_j(t) - theta_i(t) - alpha_ij) ]

`K = N`, `dt = 0.1`. The phase lag `alpha_ij` is built in the same file from a trainable
directional matrix and the normalised inverse connection cost, and is shared by all four
components.

## 5. Sinusoidal gating

`vendor/snn_kuramoto_bidirectional/sinusoidal_gating.py`, called with `gate_mode='raw'`
(`sw240/model.py: default_config`) and `phase_delay_steps = 2` (`s2net_cls.py:75`).

    thetabar_i(t) = mean_a theta_ia(t)                           delayed.mean(dim=-1)
    g_i(t) = (1 + sin(thetabar_i(t - d))) / 2,   d = 2           mask
    x_i(t) = g_i(t) sin(theta_i(t))                              gamma_wave_t

The phases are reduced over components *before* the sine, and the mean is arithmetic, matching
the phase-locking value the loss reads. `'raw'` means the gate is passed on unsquashed: the
`'sigmoid'` mode would compress it to [0.5, 0.731], where the membrane's zero-gate freeze can
never trigger and the gate scales spikes instead of switching them.

## 6. Dendrite and membrane, gated

`vendor/snn_kuramoto_bidirectional/sw240/integration.py`, `PhaseStateIntegration.__call__` —
this is the step that supersedes the plain `dendric_layer` / `membrane_layer` forwards.

    v_ia(t)    = U [x_ia(t); s_ia(t-1)] + b                      dense
    hhat_ia(t) = beta_i * h_ia(t-1) + (1 - beta_i) * v_ia(t)     native_h
    h_ia(t)    = h_ia(t-1) + (1 - eta_a (1 - g_i(t))) (hhat_ia(t) - h_ia(t-1))
                                                                 integrated_h
    H_ia(t)    = sum_b h_iab(t)                                  h_wave

`integrated_h` is written in the code as `native_h + a_d * (u - 1) * (native_h - prev_h)`, which
is the same expression rearranged; **`a_d` is `eta_a`** and `u` is the gate. `beta_i` is
`sigmoid(dendrite.tau_n)`.

    Vhat_ia(t) = lambda_i V_ia(t-1) + (1 - lambda_i) H_ia(t) - theta_a s_ia(t-1)   native_mem
    V_ia(t)    = V_ia(t-1) + (1 - xi_a (1 - g_i(t))) (Vhat_ia(t) - V_ia(t-1))      integrated_mem
    e_ia(t)    = 1[V_ia(t) > theta_a]                                              events
    s_ia(t)    = g_i(t) e_ia(t)                                                    emitted

**`a_m` is `xi_a`**, `lambda_i` is `sigmoid(membrane.tau_m)`, and the firing threshold is
`theta_a = BASE_THRESHOLD * exp(b_a)` with `BASE_THRESHOLD = 0.06` and `b_a` trainable per
component. `torch.where(folded_gate == 0, prev_mem, integrated_mem)` keeps the previous voltage
when the gate is exactly zero. The binary step uses a multi-Gaussian surrogate gradient
(`membrane_layer.ActFun_adp`).

The rollout is 1024 steps (`sw240/rollout.py`, `late_rollout`), with gradients tracked only over
the last 64 (`live_tail_steps`); the earlier steps run detached and still contribute to the
statistics the loss reads.

## 7. Affinity and masks

`vendor/snn_kuramoto_bidirectional/sw240/correlation.py` (`native_component_rho_q`), then
`spike_classifier.py` (`spike_synchrony_components`) and `sw240/partition.py`.

    rho_ija = clip( sum_m r_ia,m r_ja,m , -1, 1 )      r: centred, L2-normalised traces
    Q_ij    = prod_a max(rho_ija, 0)

over the retained window `m = 1..512`, the second half of the rollout. One non-positive
component correlation sets the affinity to zero.

Masks: link patch pairs with `Q_ij >= 0.5`, take connected components, **discard components
smaller than 8 patches** (a floor selected on CLEVR validation; `--min-group 8`), treat the
largest remaining component as background, then merge spatially adjacent groups while

    E_ij = (1 + N Abar_ij / G) log( Q'_ij / (1 - Q'_ij) )

summed over cross-group pairs is positive. Every patch enters this step as a candidate: there
is **no firing-rate test**, so the foreground is defined entirely by the synchrony structure.
In `spike_classifier.py` this is the `background='largest_component'` branch, which ignores
`foreground_threshold` altogether.

## 8. What is trained

`FreshModel.parameter_groups` declares six groups at lr 3e-5 and the encoder at 3e-6: encoder,
graph, kuramoto, dendrite, membrane, sensory projection, plus the integration parameters
(`a_d`, `a_m`, `b`). Variant 253 has `head=None` and the training objective has no
reconstruction term and no mask supervision (`vendor/full_benchmark/loss.py`,
`sw240/loss.py`). 16,830,877 trainable parameters.
