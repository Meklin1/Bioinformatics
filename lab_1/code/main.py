from collections import defaultdict
from Bio import SeqIO
import math
import numpy as np
from tabulate import tabulate
from collections import Counter

# Žemiau pateikti kodonų ir atitinkamų aminorūgščių atitikmenys
codon_mappings = {
    'TCA': 'S',
    'TCC': 'S',
    'TCG': 'S',
    'TCT': 'S',
    'TTC': 'F',
    'TTT': 'F',
    'TTA': 'L',
    'TTG': 'L',
    'TAC': 'Y',
    'TAT': 'Y',
    'TAA': '*',
    'TAG': '*',
    'TGC': 'C',
    'TGT': 'C',
    'TGA': '*',
    'TGG': 'W',
    'CTA': 'L',
    'CTC': 'L',
    'CTG': 'L',
    'CTT': 'L',
    'CCA': 'P',
    'CCC': 'P',
    'CCG': 'P',
    'CCT': 'P',
    'CAC': 'H',
    'CAT': 'H',
    'CAA': 'Q',
    'CAG': 'Q',
    'CGA': 'R',
    'CGC': 'R',
    'CGG': 'R',
    'CGT': 'R',
    'ATA': 'I',
    'ATC': 'I',
    'ATT': 'I',
    'ATG': 'M',
    'ACA': 'T',
    'ACC': 'T',
    'ACG': 'T',
    'ACT': 'T',
    'AAC': 'N',
    'AAT': 'N',
    'AAA': 'K',
    'AAG': 'K',
    'AGC': 'S',
    'AGT': 'S',
    'AGA': 'R',
    'AGG': 'R',
    'GTA': 'V',
    'GTC': 'V',
    'GTG': 'V',
    'GTT': 'V',
    'GCA': 'A',
    'GCC': 'A',
    'GCG': 'A',
    'GCT': 'A',
    'GAC': 'D',
    'GAT': 'D',
    'GAA': 'E',
    'GAG': 'E',
    'GGA': 'G',
    'GGC': 'G',
    'GGG': 'G',
    'GGT': 'G'
}

START_CODON = "ATG"
STOP_CODS = ["TAA", "TAG", "TGA"]

from Bio.Seq import Seq

# Funkcija, skirta rasti visas kodonų sekas, kurios prasideda startiniu kodonu ir baigiasi stop kodonu
def find_codon_seq_list(seq):
    rev_seq = str(Seq(seq).reverse_complement())
    codon_seq_list = []

    def process_strand(s):
        codons = [s[i:i + 3] for i in range(0, len(s), 3)]  # Padalija seką į kodonus (po 3 nukleotidus)
        i = 0
        while i < len(codons) - 1:
            if codons[i] == START_CODON:  # Randa starto kodoną
                start_pos = i
                i += 1
                while i < len(codons) - 1 and codons[i] not in STOP_CODS:  # Ieško stop kodono
                    if codons[i] == START_CODON:
                        break                                          
                    i += 1
                if i < len(codons) - 1 and codons[i] in STOP_CODS:  # Randa stop kodoną
                    end_pos = i + 1
                    if((end_pos - start_pos) * 3 >= 100):  # Patikrina, ar ilgis yra daugiau nei 100 nukleotidų                  
                        codon_seq_list.append(codons[start_pos:end_pos])  # Išsaugo seką, jei ji atitinka sąlygą
            i += 1

    process_strand(seq)  # Apdoroja pradinę seką
    process_strand(rev_seq)  # Apdoroja atvirkštinę seką

    return codon_seq_list

# Funkcija, apskaičiuojanti kodonų ar dikodonų dažnius aminorūgščių sekoje
def compute_codon_dicodon_frequencies(sequence, use_dicodon=False):
    amino_acid_sequence = [codon_mappings[codon] for codon in sequence]  # Verčia kodonus į aminorūgščių seką
    amino_acid_freq = defaultdict(int)
    
    for amino_acid_index in range(0, len(amino_acid_sequence)):
        if(use_dicodon):
            if amino_acid_index + 1 < len(amino_acid_sequence):
                dicodon = amino_acid_sequence[amino_acid_index] + amino_acid_sequence[amino_acid_index + 1]
                amino_acid_freq[dicodon] += 1  # Skaičiuoja dikodonų dažnį
        else:
            codon =  amino_acid_sequence[amino_acid_index]
            amino_acid_freq[codon] += 1  # Skaičiuoja vieno kodono dažnį
        
    return amino_acid_freq

# Funkcija, skaičiuojanti kodonų ar dikodonų variaciją tarp įvairių virusinių sekų
def variance_across_sequences(all_sequences, limit = 5, use_dicodon=False):
    all_freqs = defaultdict(list)
    
    for virus_sequences in all_sequences:
        precomputed_freqs = [compute_codon_dicodon_frequencies(seq, use_dicodon) for seq in virus_sequences]  # Apskaičiuoja dažnius kiekvienai sekai
        
        for freqs in precomputed_freqs: 
            for codon, count in freqs.items():
                all_freqs[codon].append(count)
    
    # Apskaičiuoja variaciją kiekvienam kodonui/dikodonui
    variances = {codon: np.var(counts) for codon, counts in all_freqs.items()}
    sorted_variances = sorted(variances.items(), key=lambda x: x[1], reverse=True)  # Rikiuoja pagal didžiausią variaciją

    return sorted_variances[:limit]  # Grąžina kodonus su didžiausia variacija

# Funkcija, apskaičiuojanti euklidinį atstumą tarp dviejų dažnių
def compute_euclidean_distance(freq1, freq2):
    keys = set(freq1.keys()).union(set(freq2.keys()))  # Sujungia visus raktus
    distance = sum([(freq1[k] - freq2[k]) ** 2 for k in keys])  # Apskaičiuoja atstumą
    return math.sqrt(distance)

# Funkcija, apskaičiuojanti atstumo matricą tarp visų sekų pagal kodonų ar dikodonų dažnius
def compute_distance_matrix(all_sequences, use_dicodon=False):  
    all_freqs = []
    for virus_sequence in all_sequences:
        precomputed_freqs = [compute_codon_dicodon_frequencies(seq, use_dicodon) for seq in virus_sequence]
        all_freqs.append(precomputed_freqs)
        
    matrix = []
    for freqs1 in all_freqs:
        flattened_freq_dict_1 = sum((Counter(d) for d in freqs1), Counter())  # Sujungia visus dažnius į vieną
        row = []
        for freqs2 in all_freqs:
            flattened_freq_dict_2 = sum((Counter(d) for d in freqs2), Counter())
            row.append(compute_euclidean_distance(flattened_freq_dict_1, flattened_freq_dict_2))  # Apskaičiuoja atstumą
        matrix.append(row)  
    return matrix

# Funkcija formatuojanti atstumo matricą PHYLIP formatu
def format_phylip(matrix, names):
    n = len(matrix)
    output = [str(n)] 
    for i in range(n):
        output.append(names[i] + " " + " ".join(map(str, matrix[i])))  # Prideda pavadinimus ir atstumus
    return "\n".join(output)

# Įkeliamas failų sąrašas (genomų sekos failai)
file_names = ["mamalian1.fasta", "mamalian2.fasta", "mamalian3.fasta", "mamalian4.fasta", 
              "bacterial1.fasta", "bacterial2.fasta", "bacterial3.fasta", "bacterial4.fasta"]

if __name__ == '__main__':
    codon_seq_list = [] 
    names = [] 
    for file_name in file_names:
        for record in SeqIO.parse("code/viruses/" + file_name, "fasta"):  # Nuskaito kiekvieną seką iš failo
            names.append(record.id)  # Prideda sekos pavadinimą
            seq = str(record.seq)
            codon_seq_list.append(find_codon_seq_list(seq))  # Randa kodonų sekas
            
    codon_distance_matrix = compute_distance_matrix(codon_seq_list)  
    dicodon_distance_matrix = compute_distance_matrix(codon_seq_list, use_dicodon=True)  

    dicodon_variances = variance_across_sequences(codon_seq_list, limit = 5, use_dicodon=True)  # Dikodonų variacijos
    codon_variances = variance_across_sequences(codon_seq_list, limit = 5)  # Kodonų variacijos

    print("Kodonai su didžiausia variacija:")
    print(tabulate(codon_variances, headers=["Codon", "Variance"], tablefmt="pretty"))

    print("Dikodonai su didžiausia variacija:")
    print(tabulate(dicodon_variances, headers=["Dicodon", "Variance"], tablefmt="pretty"))
    
    codon_phylip_output = format_phylip(codon_distance_matrix, names) 
    dicodon_phylip_output = format_phylip(dicodon_distance_matrix, names) 

    # Išsaugo atstumo matricas į tekstinius failus
    with open('codon_distance_matrix.txt', 'w') as f:
        f.write(codon_phylip_output) 

    with open('dicodon_distance_matrix.txt', 'w') as f:
        f.write(dicodon_phylip_output)
