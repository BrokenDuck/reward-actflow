def actflow(FlowModel m, Verifier v, timestep s, num_iter T):
    D = {}
    for t in range(T):
        Update surrogate uncertainty \sigma_t from D
        Self-generate x_{t+1} according to
            p_t\in argmax_q E_{x\sim q}[\sigma_t(\phi_s^t(x))]-\beta KL(q\Vert p_1^{\theta_t})
        Query_verifier: y_{t+1}=v(x_{t+1})
        D += (x_{t+1}, y_{t+1})
        \theta = UpdateFlow(\theta, D)

