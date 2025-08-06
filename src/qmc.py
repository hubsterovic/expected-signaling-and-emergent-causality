from typing import Literal
from tqdm import tqdm
import qutip as qt
import numpy as np


def random_Haar_sampled_unitary(d_A: int, d_B: int) -> qt.Qobj:
    random_U = qt.rand_unitary(dimensions=[d_A, d_B], distribution="haar")
    return random_U


def partially_random_Haar_sampled_unitary(
    d_A: int, d_B: int, which: Literal["A", "B"] = "A"
) -> qt.Qobj:
    if which == "A":
        V_A = qt.rand_unitary(d_A, distribution="haar")
        V = qt.tensor(V_A, qt.qeye(d_B))
    elif which == "B":
        V_B = qt.rand_unitary(d_B, distribution="haar")
        V = qt.tensor(qt.qeye(d_A), V_B)
    else:
        raise ValueError("Argument `which` must be 'A' or 'B'.")

    return V


def random_Haar_sampled_bip_COMS(d_A: int, d_B: int) -> list[qt.Qobj]:
    random_U = random_Haar_sampled_unitary(d_A, d_B)
    ket_lambdas = [
        qt.Qobj(random_U[:, i], dims=[[d_A, d_B], [1]]) for i in range(d_A * d_B)
    ]
    return ket_lambdas


def coms_of_rho(coms_ket_lambdas: list[qt.Qobj], rho_in: qt.Qobj) -> qt.Qobj:
    rho_out = qt.qzero_like(rho_in)
    for ket_l in coms_ket_lambdas:
        E_l = qt.ket2dm(ket_l)
        rho_out += E_l @ rho_in @ E_l
    return rho_out


def random_Haar_pure_dm(d_A: int, d_B: int) -> qt.Qobj:
    ket = qt.rand_ket(dimensions=[d_A, d_B], distribution="haar")
    dm = qt.ket2dm(ket)
    return dm


def random_Haar_product_dm(d_A: int, d_B: int) -> qt.Qobj:
    ket_A = qt.rand_ket(dimensions=[d_A], distribution="haar")
    ket_B = qt.rand_ket(dimensions=[d_B], distribution="haar")
    product_ket = qt.tensor(ket_A, ket_B)
    product_dm = qt.ket2dm(product_ket)
    return product_dm


def compute_signaling_X_to_Y(
    U_AB: qt.Qobj,
    V_X: qt.Qobj,
    W_AB: qt.Qobj,
    rho_AB: qt.Qobj,
    coms: list[qt.Qobj],
    direction: Literal["A to B", "B to A"] = "A to B",
) -> float:
    if direction == "A to B":
        ptr_sel = 1
    elif direction == "B to A":
        ptr_sel = 0

    rho_AB_U = U_AB @ rho_AB @ U_AB.dag()
    rho_AB_U_V = V_X @ rho_AB_U @ V_X.dag()
    coms_W = [W_AB @ ket for ket in coms]

    rho_AB_U_W = coms_of_rho(coms_W, rho_AB_U)
    rho_AB_U_V_W = coms_of_rho(coms_W, rho_AB_U_V)

    rho_U_W_Y = rho_AB_U_W.ptrace(ptr_sel)
    rho_U_V_W_Y = rho_AB_U_V_W.ptrace(ptr_sel)

    dist_hs_squared = qt.hilbert_dist(rho_U_W_Y, rho_U_V_W_Y) ** 2
    return dist_hs_squared  # type: ignore


def haar_expected_mc_signaling_X_to_Y(
    N: int,
    d_A: int,
    d_B: int,
    direction: Literal["A to B", "B to A"] = "A to B",
    dm_type: Literal["pure", "product", "mixed"] = "pure",
    fixed_coms: None | list[qt.Qobj] = None,
) -> tuple[float, list[float]]:
    which = "A" if direction == "A to B" else "B"

    coms = fixed_coms or random_Haar_sampled_bip_COMS(d_A, d_B)
    if dm_type == "pure":
        rho = random_Haar_pure_dm(d_A, d_B)
    elif dm_type == "product":
        rho = random_Haar_product_dm(d_A, d_B)
    elif dm_type == "mixed":
        raise NotImplementedError()

    dists: list[float] = []
    for _ in tqdm(
        range(N), desc=f"Computing <S>_{direction}_({d_A=},{d_B=})", leave=False
    ):
        U_AB = random_Haar_sampled_unitary(d_A, d_B)
        W_AB = (
            qt.identity([d_A, d_B])
            if fixed_coms
            else random_Haar_sampled_unitary(d_A, d_B)
        )
        V_X = partially_random_Haar_sampled_unitary(d_A, d_B, which)
        S_X_to_Y = compute_signaling_X_to_Y(
            U_AB=U_AB, V_X=V_X, W_AB=W_AB, rho_AB=rho, coms=coms, direction=direction
        )
        dists.append(S_X_to_Y)
    return (float(np.mean(dists)), dists)


def main():
    N = 5000
    d_A = 10
    d_B = 2

    m, v = haar_expected_mc_signaling_X_to_Y(N, d_A, d_B)
    print(m, v)


if __name__ == "__main__":
    main()
