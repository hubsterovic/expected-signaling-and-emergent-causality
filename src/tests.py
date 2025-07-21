import qutip as qt
import numpy as np
from help import random_Haar_sampled_bip_COMS


def test_orthogonal(psi: qt.Qobj, phi: qt.Qobj, _tol: float = 1e-10) -> bool:
    overlap = psi.overlap(phi)
    return np.abs(overlap) < _tol


def test_valid_COMS(
    coms: list[qt.Qobj], d_A: int, d_B: int, _tol: float = 1e-10
) -> bool:
    expected_dim = d_A * d_B

    if len(coms) != expected_dim:
        return False

    for i, ket_i in enumerate(coms):
        assert ket_i.isket
        assert ket_i.shape == (expected_dim, 1)
        assert np.isclose(ket_i.norm(), 1.0, atol=_tol)
        for j, ket_j in enumerate(coms[i + 1 :], i + 1):
            assert test_orthogonal(ket_i, ket_j, _tol)

    return True


def test_coms():
    d_A = 2
    d_B = 4
    random_coms = random_Haar_sampled_bip_COMS(d_A, d_B)
    assert test_valid_COMS(random_coms, d_A, d_B)


def main():
    test_coms()


if __name__ == "__main__":
    main()
