

```
sbatch --environment=gidd --nodes=2 -A a0236 --time=0-12:00:00 slurm_scripts/cscs/training/duo_owt_nvib.sh
```

```
sbatch --environment=gidd --nodes=2 -A a0236 --time=0-12:00:00 slurm_scripts/cscs/training/duo_owt.sh
```


```
sbatch --environment=gidd --nodes=2 -A a0236 --time=0-12:00:00 slurm_scripts/cscs/training/duo_owt_sa_removal.sh
```


```
sbatch --environment=gidd --nodes=2 -A a0236 --time=0-12:00:00 slurm_scripts/cscs/training/duo_owt_clean_term_loss.sh
```

```
sbatch --environment=gidd --nodes=2 -A a0236 --time=0-12:00:00 slurm_scripts/cscs/training/duo_owt_clean_term_loss_time_dependent.sh
```



```
sbatch --environment=gidd --nodes=2 -A a0236 --time=0-12:00:00 slurm_scripts/cscs/training/duo_owt_clean_term_loss_time_dependent-v2.sh
```

```
sbatch --environment=gidd --nodes=2 -A a0236 --time=0-12:00:00 slurm_scripts/cscs/training/duo_owt_clean_term_loss_time_dependent-v2.sh
```


```
sbatch --nodes=2 -A a0236 --time=0-12:00:00 slurm_scripts/cscs/training/duo_owt_clean_term_loss_ablation_t.sh
```







sbatch --nodes=2  -A a0236 --partition=debug --time=0-00:10:00 /users/mojtaba_nafez/NLU/duo/slurm_scripts/cscs/training/duo_owt_clean_term_loss_ablation_t.sh


# Important Note:
the print is not live!!! check the wandb!!!! right now it's just look like freeze after fist backward, but it's no!! ==> I added a print every 500 step.