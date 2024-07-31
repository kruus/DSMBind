

def get_seq_coords_and_angles(chain, replace_nonstd=True, allow_nan=False):
    """Extract protein sequence, coordinates, and angles from a ProDy chain.

    Args:
        chain: ProDy chain object

    Returns:
        Returns a tuple (angles, coords, sequence) for the protein chain.
        Returns None if the data fails to parse.
        Example angles returned:
            [[phi, psi, omega, ncac, cacn, cnca, chi1, chi2,...chi12],[...] ...]
    """
    chain = chain.select("protein")
    if chain is None:
        raise NoneStructureError
    chain = chain.copy()

    coords = []
    dihedrals = []
    observed_sequence = ""
    all_residues = list(chain.iterResidues())
    unmodified_sequence = [res.getResname() for res in all_residues]
    is_nonstd = np.asarray([0 for res in all_residues])
    if chain.nonstdaa and replace_nonstd:
        all_residues, unmodified_sequence, is_nonstd = replace_nonstdaas(all_residues)
    prev_res = None

    for res_id, (res, is_modified) in enumerate(zip(all_residues, is_nonstd)):
        if res.getResname() == "XAA":  # Treat unknown amino acid as missing
            continue
        elif not res.stdaa:
            raise NonStandardAminoAcidError

        # Measure basic angles
        bb_angles = measure_phi_psi_omega(res, last_res=res_id == len(all_residues) - 1)
        bond_angles = measure_bond_angles(res, res_id, all_residues)

        # Measure sidechain angles
        all_res_angles = bb_angles + bond_angles + compute_sidechain_dihedrals(
            res, prev_res, allow_nan=allow_nan)

        # Measure coordinates
        rescoords = measure_res_coordinates(res)

        # Update records
        coords.append(rescoords)
        dihedrals.append(all_res_angles)
        prev_res = res
        observed_sequence += res.getSequence()[0]

    for res_id, (res, is_modified) in enumerate(zip(all_residues, is_nonstd)):
        # Standardized non-standard amino acids
        if is_modified:
            prev_coords = coords[res_id - 1] if res_id > 0 else None
            next_coords = coords[res_id + 1] if res_id + 1 < len(coords) else None
            prev_ang = dihedrals[res_id - 1] if res_id > 0 else None
            try:
                res = standardize_residue(res, all_res_angles, prev_coords, next_coords,
                                          prev_ang)
            except ValueError as e:
                pass

    dihedrals_np = np.asarray(dihedrals)
    coords_np = np.concatenate(coords)

    if coords_np.shape[0] != len(observed_sequence) * NUM_COORDS_PER_RES:
        print(
            f"Coords shape {coords_np.shape} does not match len(seq)*{NUM_COORDS_PER_RES} = "
            f"{len(observed_sequence) * NUM_COORDS_PER_RES},\nOBS: {observed_sequence}\n{chain}"
        )
        raise SequenceError

    return dihedrals_np, coords_np, observed_sequence, unmodified_sequence, is_nonstd

# end of code from sidechainnet
