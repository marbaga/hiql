# tar -czf /lustre/home/mbagatella/hiql/v_hiql.tar v_hiql
command="cp -r /lustre/home/mbagatella/hiql/v_hiql.tar /tmp/v_hiql_$$.tar"
max_retry=20
counter=0
until $command
do
   sleep 0.1
   [[ counter -eq $max_retry ]] && echo "Failed!" && break
   echo "Trying again. Attempt #$counter"
   ((counter++))
done
tar -xf /tmp/v_hiql_$$.tar -C /tmp --one-top-level=v_hiql_$$ --strip-components 1
rm /tmp/v_hiql_$$.tar
sed -i "s#/lustre/home/mbagatella/hiql/v_hiql#/tmp/v_hiql_$$#" /tmp/v_hiql_$$/bin/activate
sed -i "s#/lustre/home/mbagatella/hiql/v_hiql#/tmp/v_hiql_$$#" /tmp/v_hiql_$$/bin/pip
. /tmp/v_hiql_$$/bin/activate
export LD_LIBRARY_PATH=/is/software/nvidia/cudnn-8.6.0-cu11.7/lib:/is/software/nvidia/cuda-11.7/lib64:/is/cluster/fast/mbagatella/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export XLA_PYTHON_CLIENT_MEM_FRACTION=.8
