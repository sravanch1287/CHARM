# Boston input for training and policy evaluation

The quick paper reproduction command does not need this external dataset.
New training, policy evaluation, Tier 2, and PA experiments need the original
Boston `pathTable.csv`. Obtain this input from the authors/data owner; a public
download location has not been established. Dataset redistribution is not
covered by this repository.

Pass the containing folder with `--data-dir`. The input is read-only. Its
expected SHA-256 is recorded in `configs/input_data.json` and checked before
experiments. Required columns:

```text
txid, rxid, LOS, pathloss_db, doa_theta_deg, doa_phi_deg,
aoa_theta_deg, aoa_phi_deg, toa_sec, Phase_deg
```

Do not silently swap the angle columns or change phase units. The physical
layer preserves the conventions used in the reported experiment. It builds
channels from the ray rows and does not require a separate saved channel matrix.

The reduced instance uses five CU slots, eight DUs, and NRF=3. CU txids are
`[9,1,8,10,5]`; DU rxids are `[7,25,4,5,19,30,10,15]`.
The external dataset and experiment output directories must be separate.
