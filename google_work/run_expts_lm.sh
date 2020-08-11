# Lunch bf jobs on borg
#!/bin/bash
source gbash.sh || exit
DEFINE_string train_cell "vz" " [qo, vz]Cell with accelerators quota."
DEFINE_string evaler_cell "jn" "Cell for evaler/decoder CPU jobs."
# DEFINE_string decoder_jobs "test" "Datasets for decoder jobs."
# DEFINE_string evaler_jobs "test" "Datasets for evaler jobs."
DEFINE_string days "2" "Runs jobs for number of days."
DEFINE_string tpu_type "jellyfish" "Dragonfish, Jellyfish..."
gbash::init_google "$@"
GOOGLE3_ROOT_DIR=$(pwd -P | sed 's/google3\/.*/google3/g');
pushd "${GOOGLE3_ROOT_DIR}"
model_path="lm.lm_one_billion_wds_custom_H."
maxlen=''
SeqL=''
L=2
ESNt='ESN'  # keep esnt as esn cell
TASK='LM'
NORM=1
m01=( ${TASK}_${ESNt}_${SeqL}${maxlen}_topo_scrl_lr_unif_L${L}_n${NORM}_r01 ${model_path}V01)
m02=( ${TASK}_${ESNt}_${SeqL}${maxlen}_topo_scrl_fix_unif_L${L}_n${NORM}_r02 ${model_path}V02)
m03=( ${TASK}_${ESNt}_${SeqL}${maxlen}_topo_scrbd_lr_unif_L${L}_n${NORM}_r03 ${model_path}V03)
m04=( ${TASK}_${ESNt}_${SeqL}${maxlen}_topo_scrbd_fix_unif_L${L}_n${NORM}_r04 ${model_path}V04)
m05=( ${TASK}_${ESNt}_${SeqL}${maxlen}_topo_rot_lr_unif_L${L}_n${NORM}_r05 ${model_path}V05)
m06=( ${TASK}_${ESNt}_${SeqL}${maxlen}_topo_rot_fix_unif_L${L}_n${NORM}_r06 ${model_path}V06)
m07=( ${TASK}_${ESNt}_${SeqL}${maxlen}_topo_randN_lr_unif_L${L}_n${NORM}_r07 ${model_path}V07)
m08=( ${TASK}_${ESNt}_${SeqL}${maxlen}_topo_randN_fix_unif_L${L}_n${NORM}_r08 ${model_path}V08)
m09=( ${TASK}_${ESNt}_${SeqL}${maxlen}_topo_rand3N_lr_unif_L${L}_n${NORM}_r09 ${model_path}V09)
m10=( ${TASK}_${ESNt}_${SeqL}${maxlen}_topo_rand3N_fix_unif_L${L}_n${NORM}_r10 ${model_path}V10)
m11=( ${TASK}_${ESNt}_${SeqL}${maxlen}_ensem_scrlueye_lr_unif_L${L}_n${NORM}_r11 ${model_path}V11)
m12=( ${TASK}_${ESNt}_${SeqL}${maxlen}_ensem_scrlueye_fix_unif_L${L}_n${NORM}_r12 ${model_path}V12)
m13=( ${TASK}_${ESNt}_${SeqL}${maxlen}_lapchain_scale_L${L}_n${NORM}_r13 ${model_path}V13)
m14=( ${TASK}_${ESNt}_${SeqL}${maxlen}_lapchain_spec_L${L}_n${NORM}_r14 ${model_path}V14)
m15=( ${TASK}_${ESNt}_${SeqL}${maxlen}_lapgrid_scale_L${L}_n${NORM}_r15 ${model_path}V15)
m16=( ${TASK}_${ESNt}_${SeqL}${maxlen}_lapgrid_spec_L${L}_n${NORM}_r16 ${model_path}V16)
m17=( ${TASK}_${ESNt}_${SeqL}${maxlen}_lapsw_scale_L${L}_n${NORM}_r17 ${model_path}V17)
m18=( ${TASK}_${ESNt}_${SeqL}${maxlen}_lapsw_spec_L${L}_n${NORM}_r18 ${model_path}V18)
m19=( ${TASK}_${ESNt}_${SeqL}${maxlen}_lapchain_grid_sw_scale_L${L}_n${NORM}_r19 ${model_path}V19)
m20=( ${TASK}_${ESNt}_${SeqL}${maxlen}_lapchain_grid_sw_spec_L${L}_n${NORM}_r20 ${model_path}V20)
models=(
 m01
 m02
 m03
 m04
 m05
 m06
 m07
 m08
 m09
 m10
 m11
 m12
 m13
 m14
 m15
 m16
 m17
 m18
 m19
 m20
)
function launch_exp {
  learning/brain/research/babelfish/trainer/tpu.sh \
      --cmd=reload \
      --build \
      --name="$1" \
      --model="$2" \
      --days="${FLAGS_days}" \
      --cell=vz \
      --tpu_cell="${FLAGS_train_cell}" \
      --evaler_cell="${FLAGS_evaler_cell}" \
      --topo=4x4 \
      --tpu_version="${FLAGS_tpu_type}" \
      --user=translate-train \
      --charged_user=translate-train \
      --allocator_charged_user=translate-train \
      --priority=115 \
      --cpu_priority=115 \
      --tpu_priority=115 \
      --tpu_worker_ram=80G \
      --pii=false \
      --evaler=test,dev,train \
      --decoder=
}
for model in "${models[@]}"; do
  eval arrayz=\${"$model"[@]}
  IFS=' ' read borg_name model <<< "$arrayz"
  launch_exp "${borg_name}" "${model}"
  echo "${borg_name}", "${model}"
done
popd
