#!/bin/bash

set -e

# Created 2024-07-30 14:27:23

CASEDIR="/glade/work/yhoussam/cases/gmom.e23.GJRAv3.TL319_t025_hycom1_N75.tx1_4.MEKE_GM.007"

/glade/work/yhoussam/code/cesm2_3_beta17_t025_res/cime/scripts/create_newcase --case gmom.e23.GJRAv3.TL319_t025_hycom1_N75.tx1_4.MEKE_GM.007 --res TL319_t025 --compset 2000_DATM%JRA-1p4-2018_SLND_CICE_MOM6_DROF%JRA-1p4-2018_SGLC_SWAV --run-unsupported --project p93300612

cd "${CASEDIR}"

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

./case.setup

./case.build

./case.build

./case.build

./xmlchange STOP_N=2

./xmlchange STOP_OPTION=nyears

./xmlchange --subgroup case.run JOB_WALLCLOCK_TIME=12:00:00

./xmlchange --subgroup case.st_archive JOB_WALLCLOCK_TIME=06:00:00

./xmlchange RESUBMIT=4

./case.submit

./xmlchange RESUBMIT=1

./case.submit

./xmlchange RESUBMIT=2

./case.submit

./xmlchange RESUBMIT=4

./case.submit

./xmlchange RESUBMIT=4

./case.submit

./xmlchange RESUBMIT=6

./case.submit

./xmlchange RESUBMIT=3

./case.submit

./case.submit

