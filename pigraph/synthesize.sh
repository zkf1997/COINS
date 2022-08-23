interactions=("sit on-sofa" "touch-shelving" "touch-tv_monitor" "sit down-sofa" "jump on-sofa" "touch-chair" "step down-chair" "walk on-floor" "touch-chest_of_drawers" "sit down-bed" "sit on-table" "move on-sofa" "stand on-chest_of_drawers" "turn-floor" "lie on-sofa" "stand up-bed" "lie on-bed" "step up-sofa" "side walk-floor" "sit down-cabinet" "stand up-chair" "stand up-cabinet" "touch-sofa" "sit on-cabinet" "a pose-floor" "move leg-sofa" "sit on-bed" "touch-wall" "sit on-chair" "step down-table" "stand up-sofa" "sit up-sofa" "touch-table" "step up-chair" "stand on-table" "step down-sofa" "sit down-chair" "stand on-floor" "stand on-bed" "touch-board_panel" "lie down-sofa" "step up-table" "sit on-chair+touch-table" "sit on-sofa+touch-table" "stand on-floor+touch-board_panel" "stand on-floor+touch-table" "stand on-floor+touch-tv_monitor" "stand on-floor+touch-shelving" "stand on-floor+touch-wall")
for ((i = 0; i < ${#interactions[@]}; i++))
  do
    echo ${interactions[$i]}
    source ../cluster/launch_cluster.sh python synthesize.py --use_penetration 0 --composition 0 --visualize 0 --gender neutral --num_results 32 --num_skeletons 8 --num_translations 256 --num_rotations 12 --interaction \'${interactions[$i]}\' --save_dir pigraph_normal
  done
