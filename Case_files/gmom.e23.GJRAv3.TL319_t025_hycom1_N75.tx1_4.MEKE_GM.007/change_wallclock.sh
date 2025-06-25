#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <parameter>"
  exit 1
fi

parameter=$1

if [ "$parameter" == "test" ]; then
  ./xmlchange STOP_N=5
  ./xmlchange STOP_OPTION=ndays
  ./xmlchange --subgroup case.run JOB_WALLCLOCK_TIME=00:20:00
  ./xmlchange --subgroup case.st_archive JOB_WALLCLOCK_TIME=00:20:00
elif [ "$parameter" == "run" ]; then
  ./xmlchange STOP_N=2
  ./xmlchange STOP_OPTION=nyears
  ./xmlchange --subgroup case.run JOB_WALLCLOCK_TIME=12:00:00
  ./xmlchange --subgroup case.st_archive JOB_WALLCLOCK_TIME=06:00:00
else
  echo "Invalid parameter. Please use 'test' or 'run'."
  exit 1
fi

./xmlquery -p STOP
./xmlquery -p WALLCLOCK

# Get the current directory name
current_directory=$(basename "$(pwd)")

# Source and destination directories
source_directory="/glade/derecho/scratch/gmarques/gmom.e23b17.GJRAv4.TL319_t025_hycom1_N75.test_01/run/INPUT"
destination_directory="/glade/derecho/scratch/yhoussam/${current_directory}/run/"

# Copy files from source to destination
cp -r "${source_directory}" "${destination_directory}"

# List the files in the destination directory

echo "ls INPUT"
ls "${destination_directory}"/INPUT
