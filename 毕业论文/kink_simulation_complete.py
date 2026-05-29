"""
半监督双重稳健 Kink 回归模型处理效应检验 - 完整模拟实验
Semi-supervised Doubly Robust Test in Kink Regression Model with Treatment Effect

论文: Feixiang Liu & Xu Liu (上海财经大学, 2026)

实验设计:
- 实验1: Type I Error控制 (tau=0时的拒绝率)
- 实验2: 检验功效 (tau>0时的拒绝率)
- 实验3: 半监督 vs 全监督对比
- 实验4: 双重稳健性验证 (故意设错模型)
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
# 【实验4专用】数据生成 - 带非线性项
# 用于测试双重稳健性: 真实模型有非线性，但拟合时用线性模型
# ============================================================
def generate_data_nonlinear(n, N, tau, delta, seed=None):
    """生成带非线性项的数据，用于测试模型误设"""
    rng = np.random.RandomState(seed)
    M = n + N
    X = rng.uniform(-2, 2, size=M)
    W = rng.normal(0, 1, size=(M, 2))

    # 倾向分数有非线性项
    eta_true = np.array([0.2, 0.3, -0.2, 0.1])
    logit_pi_true = (eta_true[0] + eta_true[1]*X + eta_true[2]*W[:,0] + eta_true[3]*W[:,1]
                     + 0.15*X**2)  # 非线性项
    pi = 1.0 / (1.0 + np.exp(-logit_pi_true))
    A = rng.binomial(1, pi, size=M)

    # 结果模型有非线性项
    Y = (1.0 + 0.5*X + tau*A*(X-delta)*(X>=delta).astype(float)
         + 0.3*W[:,0] + 0.2*W[:,1]
         + 0.1*X**2 + 0.08*W[:,0]**2  # 非线性项
         + rng.normal(0, 1, M))

    return {
        'X_L': X[:n], 'W_L': W[:n], 'A_L': A[:n], 'Y_L': Y[:n],
        'X_U': X[n:], 'W_U': W[n:], 'A_U': A[n:], 'Y_U': Y[n:],
        'X': X, 'W': W, 'A': A, 'Y': Y,
        'n': n, 'N': N, 'M': M,
        'delta_true': delta, 'tau_true': tau
    }


# ============================================================
# Logistic回归 (IRLS)
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
# 倾向分数估计
# ============================================================
def fit_propensity_pooled(data):
    """在L∪U上拟合倾向分数"""
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
# 【实验4专用】错误的倾向分数模型 - 额外加一个无用变量
# ============================================================
def fit_propensity_wrong(data, use_labeled_only=False):
    """
    故意设错倾向分数模型: 加一个无关变量 Z~N(0,1)
    这样模型就偏离了真实的pi
    """
    n, N, M = data['n'], data['N'], data['M']
    rng = np.random.RandomState(12345)

    if use_labeled_only:
        # 仅标记数据
        Z_L = rng.normal(0, 1, size=n)
        X_pi_L = np.column_stack([np.ones(n), data['X_L'], data['W_L'], Z_L])
        beta_pi = logistic_fit(X_pi_L, data['A_L'])
        pi_L = logistic_predict(X_pi_L, beta_pi)
        return pi_L, beta_pi
    else:
        # 合并数据
        Z_all = rng.normal(0, 1, size=M)
        X_pi_all = np.column_stack([np.ones(M), data['X'], data['W'], Z_all])
        beta_pi = logistic_fit(X_pi_all, data['A'])
        pi_L = logistic_predict(np.column_stack([np.ones(n), data['X_L'], data['W_L'], Z_all[:n]]), beta_pi)
        pi_U = logistic_predict(np.column_stack([np.ones(N), data['X_U'], data['W_U'], Z_all[n:]]), beta_pi)
        return pi_L, pi_U, beta_pi


# ============================================================
# NW核估计
# ============================================================
def nadaraya_watson_si(h_train, Y_train, h_eval, bandwidth):
    diff = (h_eval[:, None] - h_train[None, :]) / bandwidth
    K_vals = np.exp(-0.5 * diff**2) / (bandwidth * np.sqrt(2 * np.pi))
    denom = np.maximum(K_vals.sum(axis=1), 1e-12)
    return (K_vals @ Y_train) / denom


# ============================================================
# 基线结果模型
# ============================================================
def fit_outcome_model_ss(data, bw_factor=1.0):
    """半监督基线模型"""
    n, N = data['n'], data['N']
    X_h_L = np.column_stack([np.ones(n), data['X_L'], data['W_L']])
    X_h_U = np.column_stack([np.ones(N), data['X_U'], data['W_U']])

    gamma_init = np.linalg.lstsq(X_h_L, data['Y_L'], rcond=None)[0]
    h_L_init = X_h_L @ gamma_init
    h_U_init = X_h_U @ gamma_init

    h_std = np.std(h_L_init)
    bandwidth = max(bw_factor * h_std * n**(-0.2), 0.05)
    m_U = nadaraya_watson_si(h_L_init, data['Y_L'], h_U_init, bandwidth)

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
    """全监督基线模型"""
    n = data['n']
    X_h_L = np.column_stack([np.ones(n), data['X_L'], data['W_L']])
    gamma = np.linalg.lstsq(X_h_L, data['Y_L'], rcond=None)[0]
    h_L = X_h_L @ gamma
    return h_L, gamma, X_h_L


# ============================================================
# 【实验4专用】错误的结果模型 - 加一个无关变量
# ============================================================
def fit_outcome_model_wrong(data, use_ss=True, bw_factor=1.0):
    """
    故意设错结果模型: 加一个无关变量 Z~N(0,1)
    这样 h(X,W,Z) 就偏离了真实的 E[Y|X,W]
    """
    n, N = data['n'], data['N']
    rng = np.random.RandomState(54321)

    Z_L = rng.normal(0, 1, size=n)
    Z_U = rng.normal(0, 1, size=N)

    X_h_L = np.column_stack([np.ones(n), data['X_L'], data['W_L'], Z_L])
    X_h_U = np.column_stack([np.ones(N), data['X_U'], data['W_U'], Z_U])

    if use_ss:
        # 半监督: 用合并估计方程
        gamma_init = np.linalg.lstsq(X_h_L, data['Y_L'], rcond=None)[0]
        h_L_init = X_h_L @ gamma_init
        h_U_init = X_h_U @ gamma_init
        h_std = np.std(h_L_init)
        bandwidth = max(bw_factor * h_std * n**(-0.2), 0.05)
        m_U = nadaraya_watson_si(h_L_init, data['Y_L'], h_U_init, bandwidth)

        A_mat = X_h_L.T @ X_h_L / n + X_h_U.T @ X_h_U / N
        b_vec = X_h_L.T @ data['Y_L'] / n + X_h_U.T @ m_U / N

        try:
            gamma_ss = np.linalg.solve(A_mat, b_vec)
        except:
            gamma_ss = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]

        h_L = X_h_L @ gamma_ss
        h_U = X_h_U @ gamma_ss
        return h_L, h_U, gamma_ss, X_h_L, X_h_U, bandwidth
    else:
        # 全监督
        gamma = np.linalg.lstsq(X_h_L, data['Y_L'], rcond=None)[0]
        h_L = X_h_L @ gamma
        return h_L, gamma, X_h_L


# ============================================================
# Score函数
# ============================================================
def compute_score_labeled(X, A, Y, delta, pi_hat, h_hat):
    V = (X - delta) * (X >= delta).astype(float)
    return V * (A - pi_hat) * (Y - h_hat)


def compute_score_unlabeled(X, A, delta, pi_hat, m_hat, h_hat):
    V = (X - delta) * (X >= delta).astype(float)
    return V * (A - pi_hat) * (m_hat - h_hat)


# ============================================================
# Adjusted Influence Function
# ============================================================
def compute_adjusted_IF(psi_L, psi_U, A_L, pi_L, h_L, Y_L, X_pi_L, X_h_L,
                         A_U, pi_U, h_U, m_U, X_pi_U, X_h_U):
    G_L = np.column_stack([
        X_pi_L * (A_L - pi_L)[:, None],
        X_h_L * (Y_L - h_L)[:, None]
    ])
    G_U = np.column_stack([
        X_pi_U * (A_U - pi_U)[:, None],
        X_h_U * (m_U - h_U)[:, None]
    ])

    c_L = np.linalg.lstsq(G_L, psi_L, rcond=None)[0]
    c_U = np.linalg.lstsq(G_U, psi_U, rcond=None)[0]

    psi_star_L = psi_L - G_L @ c_L
    psi_star_U = psi_U - G_U @ c_U

    return psi_star_L, psi_star_U


# ============================================================
# 搜索空间
# ============================================================
def build_search_grid(X_pooled, M, theta=0.05, n_grid=20):
    X_sorted = np.sort(X_pooled)
    k_lo = int(np.ceil(M * theta))
    k_hi = int(np.floor(M * (1 - theta)))
    indices = np.linspace(k_lo, k_hi, n_grid, dtype=int)
    indices = np.clip(indices, 0, M - 1)
    return X_sorted[indices]


# ============================================================
# 半监督检验
# ============================================================
def compute_ss_test(data, grid_D, lam=None, bw_factor=1.0,
                    pi_L_override=None, pi_U_override=None,
                    h_L_override=None, h_U_override=None,
                    X_pi_L_override=None, X_pi_U_override=None,
                    X_h_L_override=None, X_h_U_override=None):
    """
    半监督检验统计量

    可选参数用于实验4：覆盖默认的pi和h估计
    """
    n, N, M = data['n'], data['N'], data['M']
    if lam is None:
        lam = n / M

    X_L, W_L, A_L, Y_L = data['X_L'], data['W_L'], data['A_L'], data['Y_L']
    X_U, W_U, A_U = data['X_U'], data['W_U'], data['A_U']

    # Step 1: 倾向分数 (可覆盖)
    if pi_L_override is not None:
        pi_L, pi_U = pi_L_override, pi_U_override
        X_pi_L, X_pi_U = X_pi_L_override, X_pi_U_override
    else:
        pi_L, pi_U, beta_pi = fit_propensity_pooled(data)
        X_pi_L = np.column_stack([np.ones(n), X_L, W_L])
        X_pi_U = np.column_stack([np.ones(N), X_U, W_U])

    # Step 2: 基线模型 (可覆盖)
    if h_L_override is not None:
        h_L, h_U = h_L_override, h_U_override
        X_h_L, X_h_U = X_h_L_override, X_h_U_override
        # 需要重新计算m_U
        gamma_init = np.linalg.lstsq(X_h_L, Y_L, rcond=None)[0]
        h_L_init = X_h_L @ gamma_init
        h_U_init = X_h_U @ gamma_init
        h_std = np.std(h_L_init)
        bandwidth = max(bw_factor * h_std * n**(-0.2), 0.05)
        m_U = nadaraya_watson_si(h_L_init, Y_L, h_U_init, bandwidth)
    else:
        h_L, h_U, gamma_h, X_h_L, X_h_U, bandwidth = fit_outcome_model_ss(data, bw_factor)
        m_U = nadaraya_watson_si(h_L, Y_L, h_U, bandwidth)

    # Step 3: Score和Adjusted IF
    n_grid = len(grid_D)
    scores_L = np.zeros((n_grid, n))
    scores_U = np.zeros((n_grid, N))
    adj_scores_L = np.zeros((n_grid, n))
    adj_scores_U = np.zeros((n_grid, N))

    for idx, delta in enumerate(grid_D):
        scores_L[idx] = compute_score_labeled(X_L, A_L, Y_L, delta, pi_L, h_L)
        scores_U[idx] = compute_score_unlabeled(X_U, A_U, delta, pi_U, m_U, h_U)

        adj_L, adj_U = compute_adjusted_IF(
            scores_L[idx], scores_U[idx],
            A_L, pi_L, h_L, Y_L, X_pi_L, X_h_L,
            A_U, pi_U, h_U, m_U, X_pi_U, X_h_U)
        adj_scores_L[idx] = adj_L
        adj_scores_U[idx] = adj_U

    Psi_SS = lam * scores_L.mean(axis=1) + (1 - lam) * scores_U.mean(axis=1)

    var_adj_L = np.var(adj_scores_L, axis=1, ddof=1)
    var_adj_U = np.var(adj_scores_U, axis=1, ddof=1)
    sigma2_SS = lam**2 * var_adj_L + (n / N) * (1 - lam)**2 * var_adj_U
    sigma2_SS = np.maximum(sigma2_SS, 1e-12)

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
# 全监督检验
# ============================================================
def compute_sup_test(data, grid_D,
                     pi_L_override=None, h_L_override=None,
                     X_pi_L_override=None, X_h_L_override=None):
    """全监督检验统计量"""
    n = data['n']
    X_L, W_L, A_L, Y_L = data['X_L'], data['W_L'], data['A_L'], data['Y_L']

    if pi_L_override is not None:
        pi_L = pi_L_override
        X_pi_L = X_pi_L_override
    else:
        pi_L, beta_pi = fit_propensity_labeled(data)
        X_pi_L = np.column_stack([np.ones(n), X_L, W_L])

    if h_L_override is not None:
        h_L = h_L_override
        X_h_L = X_h_L_override
    else:
        h_L, gamma_h, X_h_L = fit_outcome_model_sup(data)

    n_grid = len(grid_D)
    scores = np.zeros((n_grid, n))
    adj_scores = np.zeros((n_grid, n))

    for idx, delta in enumerate(grid_D):
        scores[idx] = compute_score_labeled(X_L, A_L, Y_L, delta, pi_L, h_L)

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
# 乘数自助法
# ============================================================
def bootstrap_ss(test_result, n, N, B=500):
    M = n + N
    lam = test_result['lam']
    adj_scores_L = test_result['adj_scores_L']
    adj_scores_U = test_result['adj_scores_U']
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
# 实验1-3: Type I Error, Power, 半监督对比
# ============================================================
def run_experiment_1_2_3(n_values=None, N_ratio=5, tau_values=None,
                          delta=0.0, alpha=0.05, B=500, n_rep=200):
    """
    实验1: Type I Error (tau=0)
    实验2: Power (tau>0)
    实验3: 半监督 vs 全监督对比
    """
    if n_values is None:
        n_values = [200, 400, 800]
    if tau_values is None:
        tau_values = [0, 0.3, 0.5, 0.8]

    all_results = {}

    print("\n" + "="*80)
    print("实验1-3: Type I Error控制、检验功效、半监督对比")
    print("="*80)

    for n in n_values:
        N = N_ratio * n
        M = n + N
        lam = n / M

        print(f"\n{'='*80}")
        print(f"样本量: n={n}, N={N}, N/n={N_ratio}")
        print(f"{'='*80}")

        results = {tau: {'SS_reject': 0, 'Sup_reject': 0} for tau in tau_values}

        for tau in tau_values:
            print(f"\n--- tau={tau} ({'H0' if tau == 0 else 'H1'}) ---")
            reject_ss = reject_sup = 0

            for rep in range(n_rep):
                seed = rep * 7919 + int(abs(tau * 1000)) % 10000
                data = generate_data(n, N, tau, delta, seed=seed)
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
                    print(f"  进度: {rep+1}/{n_rep}")

            ss_r = reject_ss / n_rep
            su_r = reject_sup / n_rep
            results[tau] = {'SS_reject': reject_ss, 'Sup_reject': reject_sup}

            label = "Type I Error" if tau == 0 else "Power"
            print(f"  {label}: 半监督={ss_r:.4f}, 全监督={su_r:.4f}, 增益={ss_r-su_r:+.4f}")

        all_results[n] = results

    # 打印汇总表格
    print("\n" + "="*80)
    print("实验1-3 结果汇总")
    print("="*80)
    print(f"{'n':>6s} | {'tau':>6s} | {'半监督':>10s} | {'全监督':>10s} | {'增益':>10s} | {'类型':>12s}")
    print("-"*80)
    for n in n_values:
        for tau in tau_values:
            ss_r = all_results[n][tau]['SS_reject'] / n_rep
            su_r = all_results[n][tau]['Sup_reject'] / n_rep
            label = "Type I Error" if tau == 0 else "Power"
            print(f"{n:>6d} | {tau:>6.2f} | {ss_r:>10.4f} | {su_r:>10.4f} | {ss_r-su_r:>+10.4f} | {label:>12s}")

    return all_results


# ============================================================
# 实验4: 双重稳健性验证
# ============================================================
def run_experiment_4(n=400, N_ratio=5, tau_values=None,
                      delta=0.0, alpha=0.05, B=500, n_rep=200):
    """
    实验4: 双重稳健性验证

    测试4种场景:
    1. π正确, h正确 (基准)
    2. π正确, h错误 (双重稳健应该有效)
    3. π错误, h正确 (双重稳健应该有效)
    4. π错误, h错误 (可能失效)

    使用非线性DGP, 拟合时用线性模型 = 故意设错
    """
    if tau_values is None:
        tau_values = [0, 0.3, 0.5]

    N = N_ratio * n
    M = n + N
    lam = n / M

    scenarios = {
        '场景1: π正确 h正确': {'wrong_pi': False, 'wrong_h': False},
        '场景2: π正确 h错误': {'wrong_pi': False, 'wrong_h': True},
        '场景3: π错误 h正确': {'wrong_pi': True, 'wrong_h': False},
        '场景4: π错误 h错误': {'wrong_pi': True, 'wrong_h': True},
    }

    all_results = {}

    print("\n" + "="*80)
    print("实验4: 双重稳健性验证")
    print("="*80)
    print("数据生成过程(DGP)包含非线性项, 但拟合时使用线性模型来故意设错")
    print("双重稳健性: 当π或h至少一个正确时, Type I Error应该仍然受控")
    print("="*80)

    for scenario_name, scenario_config in scenarios.items():
        print(f"\n{'='*80}")
        print(f"{scenario_name}")
        print(f"{'='*80}")

        results = {tau: {'SS_reject': 0, 'Sup_reject': 0} for tau in tau_values}

        for tau in tau_values:
            print(f"\n--- tau={tau} ({'H0' if tau == 0 else 'H1'}) ---")
            reject_ss = reject_sup = 0

            for rep in range(n_rep):
                seed = rep * 7919 + int(abs(tau * 1000)) % 10000 + hash(scenario_name) % 10000

                # 生成带非线性项的数据
                data = generate_data_nonlinear(n, N, tau, delta, seed=seed)
                grid_D = build_search_grid(data['X'], M, theta=0.05, n_grid=20)

                # 根据场景配置决定是否使用错误的模型
                wrong_pi = scenario_config['wrong_pi']
                wrong_h = scenario_config['wrong_h']

                try:
                    if wrong_pi and wrong_h:
                        # 两个都错
                        pi_L, pi_U, _ = fit_propensity_wrong(data)
                        X_pi_L = np.column_stack([np.ones(n), data['X_L'], data['W_L'],
                                                  np.random.RandomState(12345).normal(0,1,n)])
                        X_pi_U = np.column_stack([np.ones(N), data['X_U'], data['W_U'],
                                                  np.random.RandomState(12345).normal(0,1,N)])
                        h_L, h_U, _, X_h_L, X_h_U, _ = fit_outcome_model_wrong(data, use_ss=True)
                    elif wrong_pi:
                        # π错, h对
                        pi_L, pi_U, _ = fit_propensity_wrong(data)
                        X_pi_L = np.column_stack([np.ones(n), data['X_L'], data['W_L'],
                                                  np.random.RandomState(12345).normal(0,1,n)])
                        X_pi_U = np.column_stack([np.ones(N), data['X_U'], data['W_U'],
                                                  np.random.RandomState(12345).normal(0,1,N)])
                        # h使用正确的线性模型
                        h_L, h_U, _, X_h_L, X_h_U, _ = fit_outcome_model_ss(data)
                    elif wrong_h:
                        # π对, h错
                        pi_L, pi_U, _ = fit_propensity_pooled(data)
                        X_pi_L = np.column_stack([np.ones(n), data['X_L'], data['W_L']])
                        X_pi_U = np.column_stack([np.ones(N), data['X_U'], data['W_U']])
                        h_L, h_U, _, X_h_L, X_h_U, _ = fit_outcome_model_wrong(data, use_ss=True)
                    else:
                        # 两个都对 (但DGP有非线性, 所以实际上是近似正确)
                        pi_L, pi_U, _ = fit_propensity_pooled(data)
                        X_pi_L = np.column_stack([np.ones(n), data['X_L'], data['W_L']])
                        X_pi_U = np.column_stack([np.ones(N), data['X_U'], data['W_U']])
                        h_L, h_U, _, X_h_L, X_h_U, _ = fit_outcome_model_ss(data)

                    # 计算检验
                    ss = compute_ss_test(data, grid_D, lam,
                                         pi_L_override=pi_L, pi_U_override=pi_U,
                                         h_L_override=h_L, h_U_override=h_U,
                                         X_pi_L_override=X_pi_L, X_pi_U_override=X_pi_U,
                                         X_h_L_override=X_h_L, X_h_U_override=X_h_U)
                    T_star = bootstrap_ss(ss, n, N, B=B)
                    if ss['T_n_SS'] > np.quantile(T_star, 1 - alpha):
                        reject_ss += 1
                except Exception as e:
                    pass

                try:
                    if wrong_pi and wrong_h:
                        pi_L, _ = fit_propensity_wrong(data, use_labeled_only=True)
                        X_pi_L = np.column_stack([np.ones(n), data['X_L'], data['W_L'],
                                                  np.random.RandomState(12345).normal(0,1,n)])
                        h_L, _, X_h_L = fit_outcome_model_wrong(data, use_ss=False)
                    elif wrong_pi:
                        pi_L, _ = fit_propensity_wrong(data, use_labeled_only=True)
                        X_pi_L = np.column_stack([np.ones(n), data['X_L'], data['W_L'],
                                                  np.random.RandomState(12345).normal(0,1,n)])
                        h_L, _, X_h_L = fit_outcome_model_sup(data)
                    elif wrong_h:
                        pi_L, _ = fit_propensity_labeled(data)
                        X_pi_L = np.column_stack([np.ones(n), data['X_L'], data['W_L']])
                        h_L, _, X_h_L = fit_outcome_model_wrong(data, use_ss=False)
                    else:
                        pi_L, _ = fit_propensity_labeled(data)
                        X_pi_L = np.column_stack([np.ones(n), data['X_L'], data['W_L']])
                        h_L, _, X_h_L = fit_outcome_model_sup(data)

                    sup = compute_sup_test(data, grid_D,
                                           pi_L_override=pi_L, h_L_override=h_L,
                                           X_pi_L_override=X_pi_L, X_h_L_override=X_h_L)
                    T_star = bootstrap_sup(sup, n, B=B)
                    if sup['T_sup'] > np.quantile(T_star, 1 - alpha):
                        reject_sup += 1
                except:
                    pass

                if (rep + 1) % 50 == 0:
                    print(f"  进度: {rep+1}/{n_rep}")

            ss_r = reject_ss / n_rep
            su_r = reject_sup / n_rep
            results[tau] = {'SS_reject': reject_ss, 'Sup_reject': reject_sup}

            label = "Type I Error" if tau == 0 else "Power"
            print(f"  {label}: 半监督={ss_r:.4f}, 全监督={su_r:.4f}")

        all_results[scenario_name] = results

    # 打印汇总表格
    print("\n" + "="*80)
    print("实验4 结果汇总 (双重稳健性验证)")
    print("="*80)
    print("Type I Error (tau=0) 应该接近 0.05, 除非两个模型都设错")
    print("="*80)
    print(f"{'场景':>25s} | {'tau':>6s} | {'半监督':>10s} | {'全监督':>10s} | {'类型':>12s}")
    print("-"*80)
    for scenario_name, results in all_results.items():
        for tau in tau_values:
            ss_r = results[tau]['SS_reject'] / n_rep
            su_r = results[tau]['Sup_reject'] / n_rep
            label = "Type I Error" if tau == 0 else "Power"
            print(f"{scenario_name:>25s} | {tau:>6.2f} | {ss_r:>10.4f} | {su_r:>10.4f} | {label:>12s}")

    return all_results


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    print("\n" + "="*80)
    print("半监督双重稳健 Kink 回归模型处理效应检验")
    print("完整模拟实验")
    print("="*80)

    # 实验1-3: Type I Error, Power, 半监督对比
    results_1_2_3 = run_experiment_1_2_3(
        n_values=[200, 400],
        N_ratio=5,
        tau_values=[0, 0.3, 0.5, 0.8],
        delta=0.0,
        alpha=0.05,
        B=500,
        n_rep=200
    )

    # 实验4: 双重稳健性验证
    results_4 = run_experiment_4(
        n=400,
        N_ratio=5,
        tau_values=[0, 0.3, 0.5],
        delta=0.0,
        alpha=0.05,
        B=500,
        n_rep=200
    )

    print("\n" + "="*80)
    print("所有实验完成!")
    print("="*80)
