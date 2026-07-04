# README

## `si/`

Calculated by FHI-aims:
- `uc_1_sc_2_aims`
- `uc_2_sc_1_aims`

## `graphene/`: monolayer graphene

Calculated by MLIP (MACE-OMAT-0):
- `uc_1_sc_9_mace`
- `vacancy_uc_9_sc_1_mace`
    - The unit cell is 9x9x1, with one vacancy in the cell.

## `blg/`: bilayer graphene

Calculated by MLIP (MACE, model available upon request):
- `AB_uc_1_sc_9_mace`

## `tbg/`: twisted bilayer graphene

Calculated by MLIP (MACE, model available upon request):
- `m_2_r_1_sc_4_mace`
- `m_6_r_1_sc_1_mace`

Each case directory holds `phonopy.yaml`, `force_constants.h5`, `tmat.npz` (tmat_l0/tmat_l1/
twist_angle_deg), and `gp_pc.xyz` (the 2-atom graphene primitive cell used to build that case,
as directly generated from training data).


