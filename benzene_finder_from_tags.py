import numpy as np
import ase
from ase.io import read, write
from scipy.spatial.transform import Rotation
from data_reader import read_tagged_xyz


def get_ellipsoids(xtal, mol_frames, axes):
    coms = []
    quats = []
    dim1 = []
    dim2 = []
    dim3 = []
    mol_ids = list(zip(xtal.arrays["mol_indices"], xtal.arrays["unique_mol_indices"]))
    tmp = []
    for id in mol_ids:
        if id not in tmp:
            tmp.append(id)
    mol_ids = tmp
    # Find the indices of atoms making up each ring
    ring_indices_in_xtal = (
        []
    )  # should have the form [[1, 2, 3, 4, 5, 6], ..., [7, 8, 9, 0, 1, 2]]
    for id in mol_ids:
        # For every molecule, find the indices in the molecule frame (different from the indices in the xtal frame)
        ring_indices_in_mol = get_benzene_carbon_indices(id[0], mol_frames)
        for ring in ring_indices_in_mol:
            atoms_in_ring = []
            for j in range(len(xtal)):
                if len(atoms_in_ring) == 6:
                    break
                m_idx = xtal.arrays["mol_indices"][j]
                u_idx = xtal.arrays["unique_mol_indices"][j]
                if m_idx == id[0] and u_idx == id[1]:
                    a_idx = xtal.arrays["atom_index_in_mol"][j]
                    if a_idx in ring:
                        atoms_in_ring.append(j)
            if len(atoms_in_ring) != 6:
                print(
                    f"empty ring in crystal with crystal_idx {xtal.info["crystal_idx"]}"
                )
                print(f"mol with id {id}")
                print(f"atoms_in_ring: {atoms_in_ring}")
                print(f"ring_indices_in_mol: {ring_indices_in_mol}")
                print(f"ring: {ring}")
                print(f"ring_indices_in_xtal: {ring_indices_in_xtal}")
                print(f"mol_ids: {mol_ids}")
                print()
            if len(atoms_in_ring) == 6:
                ring_indices_in_xtal.append(atoms_in_ring)
    if len(ring_indices_in_xtal) == 0:
        return None
    for i, ring in enumerate(ring_indices_in_xtal):
        cluster = xtal[ring]
        # com = get_periodic_com(cluster)
        # com_atom = ase.Atom("X", position=com)
        dists = cluster.get_all_distances(mic=True)[0]
        farthest_idx = np.argmax(dists)
        dist_vecs = cluster.get_all_distances(mic=True, vector=True)[0]
        bisector = dist_vecs[farthest_idx]
        midpoint = cluster.get_positions()[0] + bisector / 2
        com_atom = ase.Atom("X", midpoint)
        cluster.append(com_atom)
        cluster.wrap()
        vecs = cluster.get_distances(6, [0, 2], mic=True, vector=True)
        cent_axis = np.linalg.cross(vecs[0], vecs[1])
        cent_axis = cent_axis / np.linalg.norm(cent_axis)
        side_axis = np.linalg.cross(cent_axis, vecs[0])
        side_axis = side_axis / np.linalg.norm(side_axis)
        rot = np.asarray([side_axis, vecs[0] / np.linalg.norm(vecs[0]), cent_axis]).T
        # Ensure that matrix is proper, i.e. det(R) = 1
        if np.linalg.det(rot) < 0:
            rot = np.matmul(rot, [[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert np.isclose(
            np.linalg.det(rot), 1, atol=0.01
        ), f"crystal_idx {xtal.info["crystal_idx"]}, ring {i} ({ring}) has rot mat has det {np.linalg.det(rot)}; should be 1"
        # Convert matrix representation to quaternions of the form (W, X, Y, Z)
        quat = Rotation.from_matrix(rot).as_quat()
        quat = np.roll(quat, 1)
        quats.append(quat)
        coms.append(midpoint)
        dim1.append(axes[0])
        dim2.append(axes[1])
        dim3.append(axes[2])
    ell_frame = ase.Atoms(positions=np.vstack(coms), cell=xtal.cell, pbc=xtal.pbc)
    ell_frame.arrays["quaternions"] = np.vstack(quats)
    ell_frame.arrays["c_diameter[1]"] = np.array(dim1).flatten()
    ell_frame.arrays["c_diameter[2]"] = np.array(dim2).flatten()
    ell_frame.arrays["c_diameter[3]"] = np.array(dim3).flatten()
    return ell_frame


def get_benzene_carbons(xtals: [ase.Atoms], mols: [ase.Atoms]) -> [ase.Atoms]:
    result_frames = []
    for xtal in xtals:
        mol_ids = list(
            zip(xtal.arrays["mol_indices"], xtal.arrays["unique_mol_indices"])
        )
        tmp = []
        for id in mol_ids:
            if id not in tmp:
                tmp.append(id)
        mol_ids = tmp
        # Find the indices of atoms making up each ring
        ring_indices_in_xtal = []
        for id in mol_ids:
            # For every molecule, find the indices in the molecule frame (different from the indices in the xtal frame)
            ring_indices_in_mol = get_benzene_carbon_indices(id[0], mols)
            for ring in ring_indices_in_mol:
                atoms_in_ring = []
                for j in range(len(xtal)):
                    if len(atoms_in_ring) == 6:
                        break
                    m_idx = xtal.arrays["mol_indices"][j]
                    u_idx = xtal.arrays["unique_mol_indices"][j]
                    if m_idx == id[0] and u_idx == id[1]:
                        a_idx = xtal.arrays["atom_index_in_mol"][j]
                        if a_idx in ring:
                            atoms_in_ring.append(j)
                ring_indices_in_xtal += atoms_in_ring
        result_frames.append(xtal[ring_indices_in_xtal])
    return result_frames


def get_ell_frames(xtal_frames, mol_frames, axes):
    ell_frames = []
    no_benzenes = []
    for i, frame in enumerate(xtal_frames):
        result = get_ellipsoids(frame, mol_frames, axes)
        if result is None:
            no_benzenes.append(i)
        else:
            ell_frames.append(result)
    return ell_frames, no_benzenes


def get_benzene_carbon_indices(mol_idx: int, mol_frames: [ase.Atoms]):
    mol = mol_frames[mol_idx]
    rings = []
    for m_idx, motif in enumerate(mol.info["motif_names"]):
        # print(motif)
        if motif != "benzene":
            continue
        ring_atom_indices = mol.info["motif_idx"][m_idx]
        c_indices = [i for i in ring_atom_indices if mol[i].number == 6]
        rings.append(c_indices)
    return rings


if __name__ == "__main__":
    mol_frames = read_tagged_xyz(
        "data/all_relaxed_molecules_tagged.xyz", "data/all_relaxed_molecules.xyz"
    )
    xtal_frames = ase.io.read("data/all_crystals.xyz", ":")
    no_benzenes_expected = []
    for i, xtal in enumerate(xtal_frames):
        contains_benzene = False
        for j in np.array(xtal.info["mol_indices"]).flatten():
            if "benzene" in mol_frames[j].info["motif_names"]:
                contains_benzene = True
        if not contains_benzene:
            no_benzenes_expected.append(i)
    ell_frames, no_benzenes = get_ell_frames(
        xtal_frames, mol_frames, axes=[2.1, 2.1, 1.5]
    )
    print("ell_frames length:", len(ell_frames))
    print("Expected no_benzenes:", len(no_benzenes_expected))
    print("no_benzenes length:", len(no_benzenes))
    benzene_xtal_frames = [
        xtal for i, xtal in enumerate(xtal_frames) if i not in no_benzenes_expected
    ]
    # ase.io.write("benzene_xtals.xyz", benzene_xtal_frames)
    ase.io.write("output/benzene_ell_frames.xyz", ell_frames)
