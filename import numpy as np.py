import numpy as np
import matplotlib.pyplot as plt

def hadamard(n):
    if n == 1:
        return np.array([[1]])
    H = hadamard(n // 2)
    top = np.concatenate((H, H), axis=1)
    bottom = np.concatenate((H, -H), axis=1)
    return np.concatenate((top, bottom), axis=0)

def generate_bits(users, bits_len, seed=None):
    if seed:
        np.random.seed(seed)
    return np.random.choice([1, -1], size=(users, bits_len))

def spread_bits(bits, codes):   
    users, bits_len = bits.shape    
    code_len = codes.shape[1]    
    signals = np.zeros((users, bits_len * code_len))    
    for u in range(users):        
        expanded = np.repeat(bits[u], code_len) * np.tile(codes[u], bits_len)        
        signals[u] = expanded  
    return signals

def awgn(signal, sigma):    
    noise = np.random.normal(0, sigma, size=signal.shape)
    return signal + noise

def despread_and_decode(rx, codes, bits_len): 
    users, code_len = codes.shape    
    rx_reshaped = rx.reshape(bits_len, code_len)    
    decoded = np.zeros((users, bits_len))    
    for u in range(users):        
        corrs = np.dot(rx_reshaped, codes[u])        
        decoded[u] = np.sign(corrs)       
        decoded[u][decoded[u] == 0] = 1    
    return decoded

def compute_ber(bits, decoded):    
    return np.mean(bits != decoded)

# Parameters
users, code_len, bits_len = 3, 8, 500
H = hadamard(code_len)
codes = H[:users]
snr_db_range = np.arange(-5, 15, 2)
ber_results = []

# Simulation loop
for snr_db in snr_db_range:    
    snr_linear = 10**(snr_db/10)   
    sigma = 1/np.sqrt(2*snr_linear)   
    
    bits = generate_bits(users, bits_len, seed=42)    
    signals = spread_bits(bits, codes)   
    tx = signals.sum(axis=0)
    
    rx = awgn(tx, sigma)   
    
    decoded = despread_and_decode(rx, codes, bits_len)
    ber = compute_ber(bits, decoded)    
    ber_results.append(ber)

# Plot BER vs SNR
plt.semilogy(snr_db_range, ber_results, marker='o')
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("CDMA Simulation: BER vs SNR")
plt.grid(True, which="both")
plt.show()

print("Simulation completed!")
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}"
    }
  ]
}