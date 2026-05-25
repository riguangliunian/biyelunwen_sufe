"""
半监督双重稳健 Kink 回归模型处理效应检验 - 严格按论文实现
Semi-supervised Doubly Robust Test in Kink Regression Model with Treatment Effect

论文: Feixiang Liu & Xu Liu (上海财经大学, 2026)

所有偏差已修正:
1. ✅ Score函数 psi1 = V*(A-pi)*(Y-h) — 与论文公式(3)一致
2. ✅ 基线模型 h=E[Y|X,W] 不含A — 论文第89行明确 h(X,W;zeta)=E[Y|X,W]
3. ✅ zeta估计方程合并标记+未标记数据 — 论文Psi_3(第149-171行)
4. ✅ 搜索空间D用X的次序统计量 — 论文第229-231行
5. ✅ Bootstrap使用adjusted IF psi*_SS — 论文公式(9)
6. ✅ 方差使用adjusted IF的样本方差 — 论文3.3节
7. ✅ NW核估计不做交叉拟合 — 论文公式(5)未提及
8. ✅ Pooled倾向分数 — 与论文一致
9. ✅ lambda = n/M — 与论文一致
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 数据生成 (公式1)
# Y = alpha0 + alpha1*X + tau*A*(X-delta)*1(X>=delta) + mu(W) + epsilon
# ============================================================
def generate_data(n, N, tau, delta, seed=None):
    rng = np.random.RandomState(seed)
    M = n + N
    X = rng.uniform(-2, 2, size=M)
    W = rng.normal(0, 1, size=(M, 2))
    eta_true = np.array([0.2, 0.3, -0.2, 0.1])
    logit_pi = eta_true[0] + eta_true[1]*X + eta_true[2]*W[:,0] + eta_true[3]*W[:,1]
    pi = 1.0 / (1.0 + np.exp(-logit_pi))
    A = rng.binomial(1, pi, size=M)
    Y = 1.0 + 0.5*X + tau*A*(X-delta)*(X>=delta).astype(float) + 0.3*W[:,0] + 0.2*W[:,1] + rng.normal(0,1,M)
    return {
        'X_L': X[:n], 'W_L': W[:n], 'A_L': A[:n], 'Y_L': Y[:n],
        'X_U': X[n:], 'W_U': W[n:], 'A_U': A[n:], 'Y_U': Y[n:],
        'X': X, 'W': W, 'A': A, 'Y': Y,
        'n': n, 'N': N, 'M': M,
        'delta_true': delta, 'tau_true': tau
    }


# ============================================================
# Logistic回归 (IRLS) — 用于估计倾向分数
# ============================================================
def logistic_fit(X, A, max_iter=300, tol=1e-7):
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = X @ beta
        pi = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        W_diag = np.maximum(pi * (1 - pi), 1e-10)
        z = eta + (A - pi) / W_diag
        XWX = X.T @ (W_diag[:, None] * X) + 1e-8 * np.eye(p)
        XWz = X.T @ (W_diag * z)
        beta_new = np.linalg.solve(XWX, XWz)
        if np.max(np.abs(beta_new - beta)) < tol:
            break
        beta = beta_new
    return beta


def logistic_predict(X, beta):
    return 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))


# ============================================================
# 倾向分数模型 — 在L∪U上估计 (论文3.1节)
# pi(X,W;eta) = P(A=1|X,W), 只依赖(A,X,W), 不需要Y
# 估计方程 Psi_2(eta) = (1/M) sum_{k=1}^M D_eta_k * {A_k - pi_k} = 0
# 其中 D_eta = [pi(1-pi)]^{-1} * d_pi/d_eta = X_pi (logistic有效score)
# ============================================================
def fit_propensity_pooled(data):
    """在L∪U上拟合倾向分数 (论文: maximize binomial log-likelihood over L∪U)"""
    n, N, M = data['n'], data['N'], data['M']
    X_pi_all = np.column_stack([np.ones(M), data['X'], data['W']])
    beta_pi = logistic_fit(X_pi_all, data['A'])
    pi_L = logistic_predict(np.column_stack([np.ones(n), data['X_L'], data['W_L']]), beta_pi)
    pi_U = logistic_predict(np.column_stack([np.ones(N), data['X_U'], data['W_U']]), beta_pi)
    return pi_L, pi_U, beta_pi


def fit_propensity_labeled(data):
    """仅标记数据 (全监督对照)"""
    n = data['n']
    X_pi_L = np.column_stack([np.ones(n), data['X_L'], data['W_L']])
    beta_pi = logistic_fit(X_pi_L, data['A_L'])
    pi_L = logistic_predict(X_pi_L, beta_pi)
    return pi_L, beta_pi


# ============================================================
# 基线结果模型 h(X,W;zeta) = E[Y|X,W] — 修正: 不含A!
# 论文第89行: h(Xi,Wi;zeta) = E[Yi|Xi,Wi]
# 估计方程 Psi_3(zeta) 合并标记和未标记数据:
#   Psi_3(zeta) = (1/n) sum_i D_zeta_i*(Y_i - h_i)
#               + (1/N) sum_j D_zeta_j*(m_hat_j - h_j) = 0
# 其中 D_zeta = d_h/d_zeta = X_h (OLS score)
# ============================================================
def fit_outcome_model_ss(data, pi_L, pi_U, m_U, bw_factor=1.0):
    """
    半监督基线模型: 用估计方程 Psi_3 合并标记+未标记数据求解 zeta

    h(X,W;zeta) = E[Y|X,W], 设计矩阵 X_h = [1, X, W] (不含A)

    Psi_3(zeta) = (1/n)*X_h_L'*(Y_L - X_h_L*zeta)
               + (1/N)*X_h_U'*(m_U - X_h_U*zeta) = 0

    解: zeta = [X_h_L'*X_h_L/n + X_h_U'*X_h_U/N]^{-1}
              * [X_h_L'*Y_L/n + X_h_U'*m_U/N]
    """
    n, N = data['n'], data['N']

    # 设计矩阵 X_h = [1, X, W] (不含A)
    X_h_L = np.column_stack([np.ones(n), data['X_L'], data['W_L']])
    X_h_U = np.column_stack([np.ones(N), data['X_U'], data['W_U']])

    # NW估计m_U需要先有一个初始h来做单指标, 用标记数据的OLS初始化
    gamma_init = np.linalg.lstsq(X_h_L, data['Y_L'], rcond=None)[0]
    h_L_init = X_h_L @ gamma_init
    h_U_init = X_h_U @ gamma_init

    # NW核估计 (公式5)
    h_std = np.std(h_L_init)
    bandwidth = bw_factor * h_std * n**(-0.2)
    bandwidth = max(bandwidth, 0.05)
    m_U = nadaraya_watson_si(h_L_init, data['Y_L'], h_U_init, bandwidth)

    # 求解合并估计方程 Psi_3(zeta) = 0
    # (1/n)*X_h_L'*(Y_L - X_h_L*zeta) + (1/N)*X_h_U'*(m_U - X_h_U*zeta) = 0
    # => [X_h_L'X_h_L/n + X_h_U'X_h_U/N] * zeta = X_h_L'Y_L/n + X_h_U'm_U/N
    A_mat = X_h_L.T @ X_h_L / n + X_h_U.T @ X_h_U / N
    b_vec = X_h_L.T @ data['Y_L'] / n + X_h_U.T @ m_U / N

    try:
        gamma_ss = np.linalg.solve(A_mat, b_vec)
    except np.linalg.LinAlgError:
        gamma_ss = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]

    h_L = X_h_L @ gamma_ss
    h_U = X_h_U @ gamma_ss

    return h_L, h_U, gamma_ss, X_h_L, X_h_U, bandwidth


def fit_outcome_model_sup(data):
    """全监督基线模型: 仅标记数据, h(X,W)=E[Y|X,W] (不含A)"""
    n = data['n']
    X_h_L = np.column_stack([np.ones(n), data['X_L'], data['W_L']])
    gamma = np.linalg.lstsq(X_h_L, data['Y_L'], rcond=None)[0]
    h_L = X_h_L @ gamma
    return h_L, gamma, X_h_L


# ============================================================
# NW核估计 (公式5) — 不做交叉拟合, 按论文原文
# m(s) = sum_i Y_i * K_h(s - h_i) / sum_i K_h(s - h_i)
# 单指标 S = h(X,W;zeta_hat), 高斯核
# ============================================================
def nadaraya_watson_si(h_train, Y_train, h_eval, bandwidth):
    diff = (h_eval[:, None] - h_train[None, :]) / bandwidth
    K_vals = np.exp(-0.5 * diff**2) / (bandwidth * np.sqrt(2 * np.pi))
    denom = np.maximum(K_vals.sum(axis=1), 1e-12)
    return (K_vals @ Y_train) / denom


# ============================================================
# Score函数 (公式3, 6)
# ============================================================
def compute_score_labeled(X, A, Y, delta, pi_hat, h_hat):
    """公式(3): psi1(Z;delta) = (X-delta)*1(X>=delta) * {A-pi} * {Y-h}"""
    V = (X - delta) * (X >= delta).astype(float)
    return V * (A - pi_hat) * (Y - h_hat)


def compute_score_unlabeled(X, A, delta, pi_hat, m_hat, h_hat):
    """公式(6): psi1_bar = (X-delta)*1(X>=delta) * {A-pi} * {m-h}"""
    V = (X - delta) * (X >= delta).astype(float)
    return V * (A - pi_hat) * (m_hat - h_hat)


# ============================================================
# Adjusted Influence Function (论文3.2节, 用于公式9)
#
# 论文: psi*_SS 将 (eta_hat_M, zeta_hat_n, m_hat) 的一阶估计误差
#       从score函数中投影出去
#
# D_eta(X,W) = [pi(1-pi)]^{-1} * d_pi/d_eta = X_pi (logistic)
# D_zeta(X,W) = d_h/d_zeta = X_h (线性模型, 不含A)
#
# 标记数据 G_L = [X_pi_L*(A_L-pi_L), X_h_L*(Y_L-h_L)]
# 未标记数据 G_U = [X_pi_U*(A_U-pi_U), X_h_U*(m_U-h_U)]
#
# 投影: psi* = psi - G @ c, 其中 c 通过回归求解
# ============================================================
def compute_adjusted_IF(psi_L, psi_U, A_L, pi_L, h_L, Y_L, X_pi_L, X_h_L,
                         A_U, pi_U, h_U, m_U, X_pi_U, X_h_U):
    """计算 adjusted influence function psi*_SS"""
    # 标记数据 nuisance score
    G_L = np.column_stack([
        X_pi_L * (A_L - pi_L)[:, None],     # D_eta * (A-pi)
        X_h_L * (Y_L - h_L)[:, None]         # D_zeta * (Y-h)
    ])

    # 未标记数据 nuisance score
    G_U = np.column_stack([
        X_pi_U * (A_U - pi_U)[:, None],      # D_eta * (A-pi)
        X_h_U * (m_U - h_U)[:, None]          # D_zeta * (m-h)
    ])

    # 分别投影 (标记/未标记数据各自的回归)
    c_L = np.linalg.lstsq(G_L, psi_L, rcond=None)[0]
    c_U = np.linalg.lstsq(G_U, psi_U, rcond=None)[0]

    psi_star_L = psi_L - G_L @ c_L
    psi_star_U = psi_U - G_U @ c_U

    return psi_star_L, psi_star_U


# ============================================================
# 搜索空间 D — 论文第229-231行:
# D = {delta(k) : k = ceil(M*theta), ..., floor(M*(1-theta))}
# delta(k) 是 {X_k}_{k=1}^M 的第k个次序统计量
# ============================================================
def build_search_grid(X_pooled, M, theta=0.05, n_grid=20):
    """用pooled数据的次序统计量构造搜索空间D"""
    X_sorted = np.sort(X_pooled)
    k_lo = int(np.ceil(M * theta))
    k_hi = int(np.floor(M * (1 - theta)))
    # 从次序统计量中均匀抽取n_grid个点
    indices = np.linspace(k_lo, k_hi, n_grid, dtype=int)
    indices = np.clip(indices, 0, M - 1)
    return X_sorted[indices]


# ============================================================
# 半监督检验 (公式7-8)
# ============================================================
def compute_ss_test(data, grid_D, lam=None, bw_factor=1.0):
    n, N, M = data['n'], data['N'], data['M']
    if lam is None:
        lam = n / M

    X_L, W_L, A_L, Y_L = data['X_L'], data['W_L'], data['A_L'], data['Y_L']
    X_U, W_U, A_U = data['X_U'], data['W_U'], data['A_U']

    # Step 1: Pooled倾向分数
    pi_L, pi_U, beta_pi = fit_propensity_pooled(data)
    X_pi_L = np.column_stack([np.ones(n), X_L, W_L])
    X_pi_U = np.column_stack([np.ones(N), X_U, W_U])

    # Step 2: 基线模型 h(X,W)=E[Y|X,W] (不含A) + NW估计m
    # 用合并估计方程 Psi_3 求解zeta
    h_L, h_U, gamma_h, X_h_L, X_h_U, bandwidth = fit_outcome_model_ss(
        data, pi_L, pi_U, None, bw_factor=bw_factor)

    # Step 3: NW核估计m (在fit_outcome_model_ss内部已完成)
    m_U = nadaraya_watson_si(h_L, Y_L, h_U, bandwidth)

    # Step 4: 计算score和adjusted IF
    n_grid = len(grid_D)
    scores_L = np.zeros((n_grid, n))
    scores_U = np.zeros((n_grid, N))
    adj_scores_L = np.zeros((n_grid, n))
    adj_scores_U = np.zeros((n_grid, N))

    for idx, delta in enumerate(grid_D):
        scores_L[idx] = compute_score_labeled(X_L, A_L, Y_L, delta, pi_L, h_L)
        scores_U[idx] = compute_score_unlabeled(X_U, A_U, delta, pi_U, m_U, h_U)

        # adjusted influence function (论文3.2节, 用于公式9)
        adj_L, adj_U = compute_adjusted_IF(
            scores_L[idx], scores_U[idx],
            A_L, pi_L, h_L, Y_L, X_pi_L, X_h_L,
            A_U, pi_U, h_U, m_U, X_pi_U, X_h_U)
        adj_scores_L[idx] = adj_L
        adj_scores_U[idx] = adj_U

    # 半监督score (公式7)
    Psi_SS = lam * scores_L.mean(axis=1) + (1 - lam) * scores_U.mean(axis=1)

    # 方差估计 (论文3.3节):
    # "sigma^2_SS(delta) is consistently computed using the sample variance
    #  of these adjusted influence functions"
    # 合并标记和未标记的adjusted IF, 计算加权样本方差
    # Var(sqrt(n)*Psi_SS) = n * Var(Psi_SS)
    # = n * [lam^2 * Var(adj_L)/n + (1-lam)^2 * Var(adj_U)/N]  (L,U独立)
    # = lam^2 * Var(adj_L) + (n/N)*(1-lam)^2 * Var(adj_U)
    var_adj_L = np.var(adj_scores_L, axis=1, ddof=1)
    var_adj_U = np.var(adj_scores_U, axis=1, ddof=1)
    sigma2_SS = lam**2 * var_adj_L + (n / N) * (1 - lam)**2 * var_adj_U
    sigma2_SS = np.maximum(sigma2_SS, 1e-12)

    # 检验统计量 (公式8)
    T_vals = n * Psi_SS**2 / sigma2_SS
    T_n_SS = np.max(T_vals)
    delta_hat = grid_D[np.argmax(T_vals)]

    return {
        'T_n_SS': T_n_SS, 'delta_hat': delta_hat,
        'T_vals': T_vals, 'Psi_SS': Psi_SS,
        'sigma2_SS': sigma2_SS,
        'scores_L': scores_L, 'scores_U': scores_U,
        'adj_scores_L': adj_scores_L, 'adj_scores_U': adj_scores_U,
        'lam': lam, 'bandwidth': bandwidth,
    }


# ============================================================
# 全监督检验 (对照)
# ============================================================
def compute_sup_test(data, grid_D):
    n = data['n']
    X_L, W_L, A_L, Y_L = data['X_L'], data['W_L'], data['A_L'], data['Y_L']

    pi_L, beta_pi = fit_propensity_labeled(data)
    X_pi_L = np.column_stack([np.ones(n), X_L, W_L])

    h_L, gamma_h, X_h_L = fit_outcome_model_sup(data)

    n_grid = len(grid_D)
    scores = np.zeros((n_grid, n))
    adj_scores = np.zeros((n_grid, n))

    for idx, delta in enumerate(grid_D):
        scores[idx] = compute_score_labeled(X_L, A_L, Y_L, delta, pi_L, h_L)

        # 全监督adjusted IF
        G = np.column_stack([
            X_pi_L * (A_L - pi_L)[:, None],
            X_h_L * (Y_L - h_L)[:, None]
        ])
        c = np.linalg.lstsq(G, scores[idx], rcond=None)[0]
        adj_scores[idx] = scores[idx] - G @ c

    Psi = scores.mean(axis=1)
    sigma2 = np.maximum(np.var(adj_scores, axis=1, ddof=1), 1e-12)

    T_vals = n * Psi**2 / sigma2
    T_sup = np.max(T_vals)
    delta_hat = grid_D[np.argmax(T_vals)]

    return {
        'T_sup': T_sup, 'delta_hat': delta_hat,
        'T_vals': T_vals, 'Psi': Psi,
        'sigma2': sigma2,
        'scores': scores, 'adj_scores': adj_scores,
    }


# ============================================================
# 乘数自助法 (公式9, Algorithm 1) — 使用adjusted IF
#
# 论文公式(9):
# T*_{n,SS} = sup_{delta in D} (1/sigma^2_SS(delta)) *
#   [sqrt(n) * (lambda/n * sum_{i=1}^n xi_i * psi*_SS(Z_i;delta)
#             + (1-lambda)/N * sum_{j=n+1}^M xi_j * psi*_SS(O_j;delta))]^2
# ============================================================
def bootstrap_ss(test_result, n, N, B=500):
    M = n + N
    lam = test_result['lam']
    adj_scores_L = test_result['adj_scores_L']   # 使用adjusted IF
    adj_scores_U = test_result['adj_scores_U']   # 使用adjusted IF
    sigma2 = test_result['sigma2_SS']

    T_star = np.zeros(B)
    for b in range(B):
        xi = np.random.randn(M)
        perturbed = lam * (adj_scores_L * xi[:n][None, :]).mean(axis=1) + \
                    (1 - lam) * (adj_scores_U * xi[n:][None, :]).mean(axis=1)
        T_b = n * perturbed**2 / sigma2
        T_star[b] = np.max(T_b)
    return T_star


def bootstrap_sup(test_result, n, B=500):
    """全监督乘数自助法 — 使用adjusted IF"""
    adj_scores = test_result['adj_scores']
    sigma2 = test_result['sigma2']

    T_star = np.zeros(B)
    for b in range(B):
        xi = np.random.randn(n)
        perturbed = (adj_scores * xi[None, :]).mean(axis=1)
        T_b = n * perturbed**2 / sigma2
        T_star[b] = np.max(T_b)
    return T_star


# ============================================================
# 模拟实验
# ============================================================
def run_simulation(n_rep=200, n=200, N_ratio=5, tau_values=None,
                   delta=0.0, alpha=0.05, B=500):
    if tau_values is None:
        tau_values = [0, 0.1, 0.2, 0.3, 0.5]

    N = N_ratio * n
    M = n + N
    lam = n / M

    results = {tau: {'SS_reject': 0, 'Sup_reject': 0} for tau in tau_values}

    print(f"\n{'='*70}")
    print(f"半监督双重稳健 Kink 检验模拟实验 (严格按论文)")
    print(f"{'='*70}")
    print(f"n={n}, N={N}, N/n={N_ratio}, delta={delta}, alpha={alpha}, B={B}")

    for tau in tau_values:
        print(f"\n--- tau={tau} ({'H0' if tau == 0 else 'H1'}) ---")
        reject_ss = reject_sup = 0

        for rep in range(n_rep):
            seed = rep * 7919 + int(abs(tau * 1000)) % 10000
            data = generate_data(n, N, tau, delta, seed=seed)

            # 搜索空间D: pooled数据X的次序统计量 (论文第229-231行)
            grid_D = build_search_grid(data['X'], M, theta=0.05, n_grid=20)

            try:
                ss = compute_ss_test(data, grid_D, lam)
                T_star = bootstrap_ss(ss, n, N, B=B)
                if ss['T_n_SS'] > np.quantile(T_star, 1 - alpha):
                    reject_ss += 1
            except:
                pass

            try:
                sup = compute_sup_test(data, grid_D)
                T_star = bootstrap_sup(sup, n, B=B)
                if sup['T_sup'] > np.quantile(T_star, 1 - alpha):
                    reject_sup += 1
            except:
                pass

            if (rep + 1) % 50 == 0:
                print(f"  {rep+1}/{n_rep}")

        ss_r = reject_ss / n_rep
        su_r = reject_sup / n_rep
        results[tau] = {'SS_reject': reject_ss, 'Sup_reject': reject_sup}

        label = "Type I Error" if tau == 0 else "Power"
        print(f"  {label}: 半监督={ss_r:.4f}, 全监督={su_r:.4f}, 增益={ss_r-su_r:+.4f}")

    print(f"\n{'='*70}")
    print(f"模拟结果汇总 (n={n}, N={N}, delta={delta})")
    print(f"{'='*70}")
    print(f"{'tau':>6s} | {'半监督':>8s} | {'全监督':>8s} | {'增益':>8s} | {'类型':>8s}")
    print(f"{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for tau in tau_values:
        ss_r = results[tau]['SS_reject'] / n_rep
        su_r = results[tau]['Sup_reject'] / n_rep
        label = "Type I" if tau == 0 else "Power"
        print(f"{tau:>6.2f} | {ss_r:>8.4f} | {su_r:>8.4f} | {ss_r-su_r:>+8.4f} | {label:>8s}")

    return results


# ============================================================
# 单次实验演示
# ============================================================
def single_demo():
    n, N = 500, 2500
    tau, delta = 0.5, 0.0
    data = generate_data(n, N, tau, delta, seed=42)

    print("=" * 70)
    print("单次实验演示")
    print("=" * 70)
    print(f"n={n}, N={N}, tau={tau}, delta={delta}")

    M = n + N
    lam = n / M
    grid_D = build_search_grid(data['X'], M, theta=0.05, n_grid=30)

    ss = compute_ss_test(data, grid_D, lam)
    sup = compute_sup_test(data, grid_D)

    T_ss_star = bootstrap_ss(ss, n, N, B=1000)
    T_sup_star = bootstrap_sup(sup, n, B=1000)

    pval_ss = np.mean(T_ss_star > ss['T_n_SS'])
    pval_sup = np.mean(T_sup_star > sup['T_sup'])

    print(f"半监督: T={ss['T_n_SS']:.4f}, delta_hat={ss['delta_hat']:.3f}, p={pval_ss:.4f}")
    print(f"全监督: T={sup['T_sup']:.4f}, delta_hat={sup['delta_hat']:.3f}, p={pval_sup:.4f}")

    print(f"\n{'delta':>8s} | {'T_SS':>8s} | {'T_Sup':>8s}")
    print(f"{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for i in range(0, len(grid_D), 3):
        print(f"{grid_D[i]:>8.3f} | {ss['T_vals'][i]:>8.4f} | {sup['T_vals'][i]:>8.4f}")


if __name__ == '__main__':
    single_demo()
    print("\n\n>>> 模拟实验 (n=200) <<<")
    r1 = run_simulation(n_rep=200, n=200, N_ratio=5,
                         tau_values=[0, 0.3, 0.5, 0.8],
                         delta=0.0, B=500)
    print("\n\n>>> 模拟实验 (n=400) <<<")
    r2 = run_simulation(n_rep=200, n=400, N_ratio=5,
                         tau_values=[0, 0.3, 0.5, 0.8],
                         delta=0.0, B=500)
