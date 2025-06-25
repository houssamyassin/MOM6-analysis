#!/bin/bash

# Change the values of the XML variables
./xmlchange MASK_MESH=/glade/work/gmarques/cesm/tx1_4/mesh/tx1_4_mesh_221216_cdf5.nc
./xmlchange OCN_DOMAIN_MESH=/glade/work/gmarques/cesm/tx1_4/mesh/tx1_4_mesh_221216_cdf5.nc
./xmlchange ICE_DOMAIN_MESH=/glade/work/gmarques/cesm/tx1_4/mesh/tx1_4_mesh_221216_cdf5.nc
./xmlchange ROF2OCN_ICE_RMAPNAME=/glade/work/gmarques/cesm/tx1_4/runoff_mapping/map_jra_to_tx1_4_nnsm_e333r100_221217_cdf5.nc
./xmlchange ROF2OCN_LIQ_RMAPNAME=/glade/work/gmarques/cesm/tx1_4/runoff_mapping/map_jra_to_tx1_4_nnsm_e333r100_221217_cdf5.nc
./xmlchange NTASKS_OCN=2560
./xmlchange NTASKS_CPL=256
./xmlchange NTASKS_ATM=256
./xmlchange NTASKS_LND=256
./xmlchange NTASKS_ICE=256
./xmlchange NTASKS_ROF=256
./xmlchange NTASKS_GLC=256
./xmlchange NTASKS_WAV=256
./xmlchange ROOTPE_OCN=256
./xmlchange ATM_NCPL=48
./xmlchange ICE_NCPL=48
./xmlchange LND_NCPL=48
./xmlchange WAV_NCPL=48
./xmlchange OCN_NCPL=24
./xmlchange ROF_NCPL=24

# Print the values of the XML variables
./xmlquery MASK_MESH
./xmlquery OCN_DOMAIN_MESH
./xmlquery ICE_DOMAIN_MESH
./xmlquery ROF2OCN_ICE_RMAPNAME
./xmlquery ROF2OCN_LIQ_RMAPNAME
./xmlquery NTASKS_OCN
./xmlquery NTASKS_CPL
./xmlquery NTASKS_ATM
./xmlquery NTASKS_LND
./xmlquery NTASKS_ICE
./xmlquery NTASKS_ROF
./xmlquery NTASKS_GLC
./xmlquery NTASKS_WAV
./xmlquery ROOTPE_OCN
./xmlquery ATM_NCPL
./xmlquery ICE_NCPL
./xmlquery LND_NCPL
./xmlquery WAV_NCPL
./xmlquery OCN_NCPL
./xmlquery ROF_NCPL

# Copy over use files
cp /glade/derecho/scratch/gmarques/gmom.e23b17.GJRAv4.TL319_t025_hycom1_N75.test_01/user_nl_cice .
ls user_nl_cice

# Copy over MOM input files
cp /glade/derecho/scratch/gmarques/gmom.e23b17.GJRAv4.TL319_t025_hycom1_N75.test_01/SourceMods/src.mom/MOM_input SourceMods/src.mom/
cp /glade/work/yhoussam/cases/gmom.e23.GJRAv3.TL319_t025_hycom1_N75.tx1_4.leith_GM.026/SourceMods/src.mom/diag_table SourceMods/src.mom/
cp /glade/work/gmarques/cesm.cases/G/gmom.e23_a16e.GJRAv4.TL319_t025_zstar_N65.t025_test/SourceMods/src.mom/input.nml SourceMods/src.mom/
ls SourceMods/src.mom/MOM_input
ls SourceMods/src.mom/diag_table
ls SourceMods/src.mom/input.nml

# Change name of run in diag_table
sed -i 's#gmom.e23.GJRAv3.TL319_t025_hycom1_N75.tx1_4.leith_GM.026#'"$(basename "$PWD")"'#g' SourceMods/src.mom/diag_table
head -n 1 SourceMods/src.mom/diag_table
