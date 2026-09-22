

sbatch -p gpu -A balm --time=0-08:00:00 /idiap/temp/mnafez/research/duo/slurm_scripts/idiap/eval.sh


sbatch -p gpu -A balm --time=0-08:00:00 /idiap/temp/mnafez/research/duo/slurm_scripts/idiap/eval2.sh


sbatch -p gpu -A balm --time=0-12:00:00 /idiap/temp/mnafez/research/duo/slurm_scripts/idiap/loss_term.sh




sbatch -p gpu -A balm /idiap/temp/mnafez/research/duo/slurm_scripts/idiap/step_ablation.sh


sbatch -p gpu -A balm /idiap/temp/mnafez/research/duo/slurm_scripts/idiap/step_albation_0-1.sh


sbatch -p gpu -A balm /idiap/temp/mnafez/research/duo/slurm_scripts/idiap/step_ablation_backside.sh



sbatch -p gpu -A balm /idiap/temp/mnafez/research/duo/slurm_scripts/idiap/rand_perturbation_exp.sh


sbatch -p gpu -A balm /idiap/temp/mnafez/research/duo/slurm_scripts/idiap/greedy-tail-ablation_1.sh


sbatch -p gpu -A balm /idiap/temp/mnafez/research/duo/slurm_scripts/idiap/greedy-tail-ablation_2.sh
sbatch -p gpu -A balm /idiap/temp/mnafez/research/duo/slurm_scripts/idiap/extra_radom_perturb_exp.sh

sbatch -p gpu -A balm /idiap/temp/mnafez/research/duo/slurm_scripts/idiap/5000_mauve_sample_generation.sh



sbatch -p gpu -A balm /idiap/temp/mnafez/research/duo/slurm_scripts/idiap/524B_greedy_tail.sh

sbatch -p gpu -A balm /idiap/temp/mnafez/research/duo/slurm_scripts/idiap/coef-0-1-greedy_tail.sh
sbatch -p gpu -A balm /idiap/temp/mnafez/research/duo/slurm_scripts/idiap/coef-0-1-rand-corruption-exp.sh




sbatch -p gpu -A balm /idiap/temp/mnafez/research/duo/slurm_scripts/idiap/thresh-disable-ablation_greedy_tail.sh
sbatch -p gpu -A balm /idiap/temp/mnafez/research/duo/slurm_scripts/idiap/thresh-disable-ablation_rand_corruption_exp.sh