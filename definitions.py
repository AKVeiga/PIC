import itertools
import numpy as np
import qiskit.quantum_info as qi
from qiskit.visualization import array_to_latex
from IPython.display import display
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn
from kaleidoscope import bloch_sphere

class Basis:
    def __init__(self, data, name=''):
        self.data = data
        self.index = 0
        self.num = len(data)
        self.num_qubits = data[0].num_qubits
        self.dim = 2**self.num_qubits
        self.name = name
        self.pseudoinv = self._pseudoinverse()
        
    def __iter__(self):
        self.index = 0
        return self
        
    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        result = self.data[self.index]
        self.index += 1
        return result
        
    def _pseudoinverse(self):
        matrix = []
        for mat in self.data:
            temp = mat.conjugate().data.reshape(-1,1)
            temp = [element[0] for element in temp]
            matrix.append(np.array(temp))
        matrix = np.array(matrix)
        pseudoinv = np.matmul(
            np.linalg.inv(
                np.matmul(
                    matrix.transpose(), 
                    matrix)
            ), 
            matrix.transpose()
        )
        return pseudoinv

class QSTdata:
    def __init__(self, 
                 runs,
                 fidelity,
                 err_lo=None,
                 err_hi=None,
                 err_min=None,
                 err_max=None):
        self.runs = runs
        self.fidelity = fidelity
        self.err_lo = err_lo
        self.err_hi = err_hi
        self.err_min = err_min
        self.err_max = err_max
        
sicpovm_1qb = Basis(
    [
    qi.Statevector([1,0]).to_operator()/2,
    qi.Statevector([1/np.sqrt(3),
                    np.sqrt(2/3)]).to_operator()/2,
    qi.Statevector([1/np.sqrt(3),
                    np.sqrt(2/3) * np.exp(1j*2*np.pi/3)]
                  ).to_operator()/2,
    qi.Statevector([1/np.sqrt(3),
                    np.sqrt(2/3) * np.exp(1j*4*np.pi/3)]
                  ).to_operator()/2
],
    name='SIC-POVM')

def generate_sicpovm_2qb():
    x = np.sqrt(2 + np.sqrt(5))
    norm = 4*(5 + np.sqrt(5))
    basis = [
        qi.Statevector([x,1,1,1]).to_operator()/norm,
        qi.Statevector([x,1,-1,-1]).to_operator()/norm,
        qi.Statevector([x,-1,1,-1]).to_operator()/norm,
        qi.Statevector([x,-1,-1,1]).to_operator()/norm,
        qi.Statevector([1j,x,1,-1j]).to_operator()/norm,
        qi.Statevector([1j,x,-1,1j]).to_operator()/norm,
        qi.Statevector([-1j,x,1,1j]).to_operator()/norm,
        qi.Statevector([-1j,x,-1,-1j]).to_operator()/norm,
        qi.Statevector([1j,1j,x,-1]).to_operator()/norm,
        qi.Statevector([1j,-1j,x,1]).to_operator()/norm,
        qi.Statevector([-1j,1j,x,1]).to_operator()/norm,
        qi.Statevector([-1j,-1j,x,-1]).to_operator()/norm,
        qi.Statevector([1j,1,-1j,x]).to_operator()/norm,
        qi.Statevector([1j,-1,1j,x]).to_operator()/norm,
        qi.Statevector([-1j,1,1j,x]).to_operator()/norm,
        qi.Statevector([-1j,-1,-1j,x]).to_operator()/norm
    ]
    return Basis(basis, name='SIC-POVM 2-Qubit')

sicpovm_2qb = generate_sicpovm_2qb()
sicpovm = {
    1: sicpovm_1qb,
    2: sicpovm_2qb
}

def generate_pauli_matrices(num):
    pauli = [
        qi.Operator([[1,0],[0,1]]),
        qi.Operator([[0,1],[1,0]]),
        qi.Operator([[0,-1j],[1j,0]]),
        qi.Operator([[1,0],[0,-1]])
    ]
    temp_pauli = pauli
    new_pauli = pauli
    for _ in range(num-1):
        new_pauli = []
        for operator1 in temp_pauli:
            for operator2 in pauli:
                new_pauli.append(operator1.tensor(operator2))
        temp_pauli = new_pauli
    return new_pauli
    
def generate_pauli_nqb(num):
    new_pauli = generate_pauli_matrices(num)
    basis = []
    dim = 2**num
    for operator in new_pauli[1:]:
        basis.append((new_pauli[0] + operator)/dim)
        basis.append((new_pauli[0] - operator)/dim)

    result = qi.Operator(np.zeros((dim, dim)))
    for operator in basis:
        result += operator
    trace = result.data.trace()
    for i, operator in enumerate(basis):
        basis[i] = operator / trace*dim
    return Basis(basis, name=f'{len(basis)}-Pauli')

pauli_1qb = generate_pauli_nqb(1)
pauli_2qb = generate_pauli_nqb(2)
pauli_povm = {
    1: pauli_1qb,
    2: pauli_2qb
}
    
def random_direction(num=4):
    n = []
    for _ in range(num):
        n.append(np.random.uniform(-1, 1, size=3))
        n[-1] /= np.sqrt(np.sum(n[-1]**2))
    return n
    
def atl(x, prefix='', source=False):
    if source:
        print(array_to_latex(x, prefix=prefix, source=True))
    else:
        display(array_to_latex(x, prefix=prefix))

def remove_negatives_and_diagonalize(arr):
    arr[arr < 0] = 0
    diagonal_matrix = np.diag(arr)
    return diagonal_matrix

def measure(probabilities, num=1):
    p_total = abs(1 - sum(probabilities))
    if p_total > 0.01:
        print(f'ERROR: Probabilities sum to 1 +/- {p_total}')

    counts = rng.multinomial(num, probabilities)
    estimators = counts / num
    return estimators

def qst(state, basis, 
        shots_max=400, 
        runs_per_shot=1, 
        step=1, 
        profile=True, 
        calc_errors=True):
    probabilities = []
    for projector in basis.data:
        probabilities.append(
            np.trace(
                np.matmul(
                    state.data, 
                    projector.data
                )
            ).real)
    if profile:
        shots = range(step*basis.num, (shots_max+1), step*basis.num)
    else:
        shots = [shots_max - (shots_max % basis.num)]
    fidelity_avg = []
    err_hi = []
    err_lo = []
    err_max = []
    err_min = []
    print('QST completion %', end=' ')
    for num_shots in shots:
        print(int(100*num_shots / shots_max), end=' ')
        fidelity = []
        for _ in range(runs_per_shot):
            estimator = measure(probabilities, num_shots/basis.num)
            state_reconstruct = qi.DensityMatrix(
                np.matmul(
                    basis.pseudoinv, 
                    estimator
                ).reshape(basis.dim, basis.dim))
            state_valid = make_psd_and_rescale_trace(state_reconstruct)
            fidelity.append(qi.state_fidelity(state_valid, state))
        fidelity = np.array(fidelity)
        avg = np.mean(fidelity)
        if profile:
            fidelity_avg.append(avg)
        else:
            fidelity_avg = avg
        if calc_errors:
            var_hi = fidelity[fidelity > avg]
            var_lo = fidelity[fidelity < avg]
            dev_hi = np.mean(var_hi)
            dev_lo = np.mean(var_lo)
            if profile:
                err_max.append(np.max(fidelity))
                err_min.append(np.min(fidelity))
                err_hi.append(dev_hi)
                err_lo.append(dev_lo)
            else:
                err_max = np.max(fidelity)
                err_min = np.min(fidelity)
                err_hi = dev_hi
                err_lo = dev_lo                
    if profile:
        fidelity_avg = np.array(fidelity_avg)
    print()
    if calc_errors:
        return QSTdata(shots, 
                       fidelity_avg, 
                       err_lo, 
                       err_hi, 
                       err_min, 
                       err_max)
    else:
        return QSTdata(shots, 
                       fidelity_avg)

def make_psd_and_rescale_trace(state):
    y = 0.5*(state.data + state.data.T.conjugate())
    eigvals, eigvecs = np.linalg.eig(y)
    eigvals = remove_negatives_and_diagonalize(eigvals)
    z = np.matmul(eigvecs, 
                  np.matmul(eigvals, 
                            eigvecs.T.conjugate()))
    z /= z.trace()
    return z

def generate_complex_numbers(num):
    angles = 2*np.pi*rng.uniform(size=num)
    z = np.exp(1j * angles)
    return z

def generate_real_complex_pair():
    x = rng.uniform()
    z = np.sqrt(1 - x**2) * generate_complex_numbers(1)
    pair = np.concatenate(([x], z))
    return pair

def generate_complex_pairs(num_pairs):
    z = generate_complex_numbers(2*num_pairs)
    norms = []
    for x in rng.uniform(size=num_pairs):
        norms.append(x)
        norms.append(np.sqrt(1 - x**2))
    return z * np.array(norms)

def normalize(x):
    norm = np.sqrt(np.sum(np.vdot(x, x)))
    return x / norm

def generate_n_qubits(num, entangled=False, density_matrix=False):
    if entangled:
        coefs = np.concatenate((generate_real_complex_pair(),
                                generate_complex_pairs(2**(num-1)-1)))
        coefs = normalize(coefs)
        state = qi.Statevector(coefs)
    else:
        state = qi.Statevector(generate_real_complex_pair())
        for _ in range(num-1):
            new_state = qi.Statevector(generate_real_complex_pair())
            state = state.tensor(new_state)

    if density_matrix:
        state = qi.DensityMatrix(state)
        
    return state

def generate_combinations(lst):
    combinations = []
    for r in range(1, len(lst)):
        combinations.extend([list(comb) for comb \
                             in itertools.combinations(lst, r)])
    return combinations

def list_subsystems(state):
    trace_out = generate_combinations(range(state.num_qubits))
    states_traced = []
    for subsys in trace_out:
        states_traced.append(qi.partial_trace(state, subsys))
    return states_traced, trace_out

def frobenius_norm(a, b):
    res = np.matmul(a.conj().T, b).trace()
    return res

def verify_povm(basis):
    physical = True
    result = np.zeros_like(basis.data[0])
    eigvals = []
    for operator in basis.data:
        result += operator.data
        eigvals.append(np.linalg.eigvals(operator.data))
        for val in eigvals[-1]:
            if abs(np.imag(val)) > 0.0001:
                physical = False
            elif np.real(val) < -0.0001:
                physical = False
    if not physical:
        print('Operators not PSD')
    if not np.allclose(result, np.identity(result.shape[0])):
        physical = False
        print('Operators sum to')
        atl(result)
    if physical:
        print('Valid POVM')
    return physical

def generate_perturbed_sicpovm_1qb(amt=0.1):
    pauli = [
        np.array([[1,0],[0,1]], dtype=complex),
        np.array([[0,1],[1,0]], dtype=complex),
        np.array([[0,-1j],[1j,0]], dtype=complex),
        np.array([[1,0],[0,-1]], dtype=complex)
    ]
    n = random_direction()
    rotation = []
    for ni in n:
        rotation.append(pauli[0]*np.cos(amt*np.pi))
        for nj, sigma in zip(ni, pauli[1:]):
            rotation[-1] -= sigma*1j*np.sin(amt*np.pi)*nj
    basis_vecs = [
        qi.Statevector([1,0]),
        qi.Statevector([1/np.sqrt(3),
                        np.sqrt(2/3)]),
        qi.Statevector([1/np.sqrt(3),
                        np.sqrt(2/3) * np.exp(1j*2*np.pi/3)]),
        qi.Statevector([1/np.sqrt(3),
                        np.sqrt(2/3) * np.exp(1j*4*np.pi/3)])
    ]
    new_povm = []
    bloch_factor = 1
    trace = 0
    for rot, vec in zip(rotation, basis_vecs):
        vec = qi.Statevector(
            np.dot(rot, np.dot(vec.data, rot.conjugate().T))
        )
        bloch_factor = np.abs(vec[0]) / vec.data[0]
        vec *= bloch_factor
        new_povm.append(vec.to_operator())
        new_trace = new_povm[-1].data.trace()
        trace += new_trace
    for i, _ in enumerate(new_povm):
        new_povm[i] /= trace/2
    total = qi.Operator(np.zeros_like(new_povm[0]))
    for proj in new_povm:
        total += proj
    diff = total - np.identity(2)
    physical = True
    for i, _ in enumerate(new_povm):
        new_povm[i] -= diff/4
    new_povm = Basis(new_povm, name=f'SIC-POVM (p={amt})')
    physical = verify_povm(new_povm)
    return new_povm, physical

rng = np.random.default_rng(seed = 12345)

# 1 qubit states
state_1qb = generate_n_qubits(1, 
                              entangled=True, 
                              density_matrix=True)

# 1 qubit mixed array
num_coefs = 100
coefs = np.linspace(0, 1, num_coefs)
var_states = []
for c in coefs:
    var_states.append(
        qi.DensityMatrix(
            [[(1-c)*state_1qb.data[0,0] + c/2, 
              (1-c)*state_1qb.data[0,1]],
             [(1-c)*state_1qb.data[1,0], 
              (1-c)*state_1qb.data[1,1] + c/2]]))

# 2 qubit states
state_2qb_sep = generate_n_qubits(2, 
                                  entangled=False, 
                                  density_matrix=True)
state_2qb_ent = generate_n_qubits(2, 
                                  entangled=True, 
                                  density_matrix=True)
state_bell = qi.DensityMatrix(
    qi.Statevector([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]))
state_mine = qi.DensityMatrix(
    qi.Statevector([1, 0]).tensor(
        qi.Statevector([1/np.sqrt(2), 1/np.sqrt(2)])))

# 3 qubit states
state_3qb_sep = generate_n_qubits(3, 
                                  entangled=False, 
                                  density_matrix=True)
state_3qb_ent = generate_n_qubits(3, 
                                  entangled=True, 
                                  density_matrix=True)
state_ghz = qi.DensityMatrix(
    qi.Statevector([1/np.sqrt(2),0,0,0,0,0,0,1/np.sqrt(2)]))
state_w = qi.DensityMatrix(
    qi.Statevector([0, 1/np.sqrt(3), 
                    1/np.sqrt(3), 0, 
                    1/np.sqrt(3), 0, 
                    0, 0]))

# 4 qubit states
state_4qb = generate_n_qubits(5, entangled=True)

seaborn.set_theme(style='whitegrid', 
                  palette='colorblind')