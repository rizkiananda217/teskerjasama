import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fuzzy membership functions
def usia_membership(usia):
    muda = max(0, min(1, (35 - usia) / 15))
    paruh_baya = max(0, min((usia - 30) / 10, (50 - usia) / 10))
    tua = max(0, min(1, (usia - 45) / 15))
    return muda, paruh_baya, tua

def masa_kerja_membership(masa_kerja):
    sedikit = max(0, min(1, (10 - masa_kerja) / 10))
    sedang = max(0, min((masa_kerja - 5) / 5, (20 - masa_kerja) / 5))
    lama = max(0, min(1, (masa_kerja - 15) / 10))
    return sedikit, sedang, lama

def gaji_membership(gaji):
    rendah = max(0, min(1, (1000000 - gaji) / 500000))
    sedang = max(0, min((gaji - 750000) / 250000, (1500000 - gaji) / 500000))
    tinggi = max(0, min(1, (gaji - 1250000) / 500000))
    return rendah, sedang, tinggi

def tunjangan_membership(x):
    """Membership functions for allowance, handles array."""
    kecil = np.maximum(0, np.minimum(1, (500000 - x) / 250000))
    menengah = np.maximum(0, np.minimum((x - 250000) / 250000, (750000 - x) / 250000))
    besar = np.maximum(0, np.minimum(1, (x - 500000) / 250000))
    return kecil, menengah, besar

# Define fuzzy rules
def fuzzy_rules(usia, masa_kerja, gaji):
    muda, paruh_baya, tua = usia_membership(usia)
    sedikit, sedang, lama = masa_kerja_membership(masa_kerja)
    rendah, sedang_gaji, tinggi = gaji_membership(gaji)

    rules = [
        (min(muda, sedikit, rendah), 'kecil'),
        (min(muda, sedang, sedang_gaji), 'menengah'),
        (min(paruh_baya, lama, tinggi), 'besar'),
        (min(tua, lama, tinggi), 'besar')
    ]
    return rules

def defuzzify(rules):
    """Defuzzification using the centroid method."""
    x = np.linspace(0, 1000000, 1000)
    aggregated = np.zeros_like(x)
    
    for weight, label in rules:
        if label == 'kecil':
            aggregated = np.maximum(aggregated, np.minimum(weight, tunjangan_membership(x)[0]))
        elif label == 'menengah':
            aggregated = np.maximum(aggregated, np.minimum(weight, tunjangan_membership(x)[1]))
        elif label == 'besar':
            aggregated = np.maximum(aggregated, np.minimum(weight, tunjangan_membership(x)[2]))

    # Compute the centroid
    return np.sum(x * aggregated) / np.sum(aggregated) if np.sum(aggregated) > 0 else 0

# Contoh data
data = pd.DataFrame({
    'id': [1],
    'nama': ['Ali'],
    'usia': [30],
    'masa kerja': [6],
    'gaji': [750000]
})

# Calculate tunjangan
data['tunjangan'] = data.apply(
    lambda row: defuzzify(fuzzy_rules(row['usia'], row['masa kerja'], row['gaji'])), axis=1
)

# Display the updated data
print(data[['id', 'nama', 'tunjangan']])

# Plot example for first row
example = data.iloc[0]
usia, masa_kerja, gaji = example['usia'], example['masa kerja'], example['gaji']
rules = fuzzy_rules(usia, masa_kerja, gaji)

tunjangan = defuzzify(rules)

x = np.linspace(0, 1000000, 1000)
plt.plot(x, [tunjangan_membership(t)[0] for t in x], label="Kecil")
plt.plot(x, [tunjangan_membership(t)[1] for t in x], label="Menengah")
plt.plot(x, [tunjangan_membership(t)[2] for t in x], label="Besar")
plt.axvline(tunjangan, color='r', linestyle='--', label=f"Defuzzified: {tunjangan:.2f}")
plt.legend()
plt.title("Tunjangan Membership Functions")
plt.xlabel("Tunjangan")
plt.ylabel("Membership")
plt.show()
