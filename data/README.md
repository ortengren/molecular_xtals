`all_crystals.xyz`: This file contains the relaxed geometries and properties for 2'707 organic crystals. Each entry contains the following information: total energy from QE in Ryd (`dft_energy_ryd`), binding energy (`dft_cohesive_energy_relaxed_configs_ryd`), CCDC ID (`CCDC_ID`), index in the file (`crystal_idx`) and index of the corresponding molecules in the molecule file (`mol_indices`). Each atom carries the corresponding information: position, atomic number (`number`), index of the corresponding molecule (`mol_indices`), index within the corresponding molecule (`atom_index_in_mol`), and the index of the corrsponding unique molecule (`unique_mol_index`).

`all_relaxed_molecules.xyz`: This file contains the relaxed geometries and properties for 3'242 organic crystals. Each entry contains the following information: total energy from QE in Ryd (`dft_energy_ryd`), index in the file (`molecule_idx`) and index of the corresponding crystal in the crystal file (`crystal_idx`). 

`all_relaxed_molecules_tagged.xyz`: This file contains the relaxed geometries and properties for 3'242 organic crystals. Each entry contains the following information: total energy from QE in Ryd (`dft_energy_ryd`), index in the file (`molecule_idx`) and index of the corresponding crystal in the crystal file (`crystal_idx`). This file also contains in the `info` the entries `motif_indices` and `motif_names`, which tell which atoms belong to which motifs, as determined by SMARTS pattern matching. 

`chemiscopes/`: Chemiscope files uploaded to molmotifs.matcloud.xyz.

`pseudo/`: pseudopotentials used for all DFT-PBE-D2 calculations
