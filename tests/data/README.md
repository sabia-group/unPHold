# README

Naming conventions:
- `uc`: unit cell size relative to the primitive cell
- `sc`: supercell size relative to the unit cell

Available files:
- Each case directory holds `phonopy.yaml`, `force_constants.h5`, and the primitive cell for supercell construction.

## `si/`: silicon bulk

Calculated by FHI-aims:
- `uc_1_sc_2_aims`
- `uc_2_sc_1_aims`

## `graphene/`: monolayer graphene

Calculated by MLIP (MACE-OMAT-0):
- `uc_1_sc_9_mace`
- `vacancy_uc_9_sc_1_mace`
    - With one vacancy

## `blg/`: bilayer graphene

Calculated by MLIP (MACE, model available upon request):
- `AB_uc_1_sc_9_mace`

## `tbg/`: twisted bilayer graphene

Naming: `m` and `r` are the paramaters for determining the twist angle and transformation matix.

Calculated by MLIP (MACE, same model as `blg/`):
- `m_2_r_1_sc_4_mace`
- `m_6_r_1_sc_1_mace`

These cases include an extra `tmat.npz` file (the transformation matrix used to build the supercell).


