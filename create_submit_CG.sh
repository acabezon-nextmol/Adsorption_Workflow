#!/bin/bash

# Function to display help message
usage() {
    echo "No inputs provided!"
    echo
    echo "Usage: $0 <name> <sID> <TIME> <resources>"
    echo
    echo "Arguments:"
    echo "  name         Name for working directory "
    echo "  sID          A name to identify the run. For instance the name of the molecule"
    echo "  TIME         time for production run job"
    echo "  resources    Resources. Choose between: cpu, 1g, 2g, l40s"
    echo
    echo "Example:"
    echo "  $0 ADSORPTION_SIMULATION ADS_SIM 96 2g"
    exit 1
}

# Check if the inputs were provided
if [[ "$#" -ne 3 ]]; then
	usage
fi

echo "Name of the working directory: ${1}"
echo "System ID: ${2}"
name="${1}"
sID="${2}"
TIME="${3}"
resources="${4}"
if [[ ${resources} == "cpu" ]]; then
	cpus_task="#SBATCH --cpus-per-task=4"
    ntasks="#SBATCH --ntasks=8"
	gpu=false
elif [[ ${resources} == "1g" ]]; then
	gpu_line="#SBATCH --gres=gpu:1g.10gb:1"
	constraint="#SBATCH --constraint=zen3"
	cpus_task="#SBATCH --cpus-per-task=2"
    ntasks="#SBATCH --ntasks=2"
	gpu=true
elif [[ ${resources} == "2g" ]]; then
	gpu_line="#SBATCH --gres=gpu:2g.20gb:1"
	constraint="#SBATCH --constraint=zen3"
	cpus_task="#SBATCH --cpus-per-task=2"
    ntasks="#SBATCH --ntasks=3"
	gpu=true
elif [[ ${resources} == l40s ]]; then
	gpu_line="#SBATCH --gres=gpu:l40s:1"
	constraint="#SBATCH --constraint=zen4"
	cpus_task="#SBATCH --cpus-per-task=6"
    cpus_task="#SBATCH --cpus-per-task=2"
    ntasks="#SBATCH --ntasks=3"
	gpu=true
else
	echo "The gpu provided was not among the options: 2g or l40s"
	exit 1
fi

if $gpu ; then
	gmx_mdrun="\$GMX_BIN mdrun -v -deffnm \$tpr -ntmpi \$MPI_RANKS -ntomp \$OMP_NUM_THREADS -pin on -nb gpu -bonded gpu -dlb yes"
else
	gmx_mdrun="\$GMX_BIN mdrun -v -deffnm \$tpr -ntomp \$OMP_NUM_THREADS -ntmpi \$MPI_RANKS"
fi


cat <<EoF > submit_sim.sh
#!/bin/bash
#SBATCH --job-name=${name}_${sID}
#SBATCH --partition=main
#SBATCH --account=nextmol
#SBATCH --output=OUT/output.%x.%j.out
#SBATCH --error=ERR/output.%x.%j.err
#SBATCH --nodes=1
${ntasks}
${cpus_task}
#SBATCH --time=${TIME}:00:00
#SBATCH --mem-per-cpu=2GB
${constraint}
${gpu_line}

source /shared/environmentrc.platform_spack_0.23.0.dev0
spack load gromacs@2024.2

# Variables
GMX_BIN="gmx"
WD="/shared/projects/common/ONE_HAIR_CG/${name}"
SIMDIR="/shared/projects/common/ONE_HAIR_CG/${name}/sim"
mdps="/shared/projects/common/ONE_HAIR_CG/${name}/mdp_files"
forcefield="/home/acabezon/forcefields/martini_v300"

export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export GMXLIB="\${forcefield}"
MPI_RANKS="\${SLURM_NTASKS}"

#Execution
echo "Job \$SLURM_JOB_ID started at \$(date)"
echo "Job allocated in \$SLURM_NODELIST"

STARTTIME=\$(date +%s)
GMX_MAXH=\$(echo "scale=3; ${TIME} - 0.2" | bc) # Substract a safety buffer
# Define the runs =================================================================================
run_gmx () { #-----------------------------------------------------------------
    mkdir -p \${SIMDIR}                                                         
    cd \${SIMDIR} || exit 1                                                     
    tpr="prod"                                                                  
                                                                                
    # Check that equilibration finished                                         
    if [[ ! -f npt2.gro ]]; then                                                
        echo "Equilibration was not finished (npt2.gro missing)."               
        exit 1                                                                  
    fi                                                                          
                                                                                
    # Case 1: production already finished                                       
    if [[ -f prod.gro ]]; then                                                  
        echo "Production simulation already finished (prod.gro found)."         
        exit 0                                                                  
    fi                                                                          
                                                                                
    # Case 2: restart from checkpoint if available                              
    if [[ -f prod.cpt ]]; then                                                  
        echo "Restarting production from checkpoint prod.cpt..."                
        ${gmx_mdrun} -maxh \$GMX_MAXH -cpi prod.cpt -append                       
                                                                                
    else                                                                        
        # Case 3: no checkpoint → first production run                          
        echo "Starting new production run..."                                   
        \$GMX_BIN grompp -p \${WD}/system.top -c npt2.gro -t npt2.cpt -f \${mdps}/prod.mdp -o prod -maxwarn 10
        ${gmx_mdrun} -maxh \$GMX_MAXH                                             
    fi                                                                          
                                                                                
    # After mdrun: check if finished                                            
    if [[ -f prod.gro ]]; then                                                  
        echo "Production simulation finished."                                  
        exit 0                                                                  
    else                                                                        
        echo "Production not finished within time limit. Resubmitting job..."   
        cd \${WD} || exit 1                                                     
        sbatch submit_sim.sh                                                    
        exit 0                                                                  
    fi

cd \${WD}
} #-----------------------------------------------------------------

# Execute the runs =================================================================================
run_gmx

ENDTIME=\$(date +%s)
echo "Total runtime: \$((\$ENDTIME - \$STARTTIME)) seconds."
echo "Job completed successfully at:" \$date
exit 0
EoF


cat <<EoF > submit_equi.sh
#!/bin/bash
#SBATCH --job-name=EQ_${name}_${sID}
#SBATCH --partition=main
#SBATCH --account=nextmol
#SBATCH --output=OUT/output.%x.%j.out
#SBATCH --error=ERR/output.%x.%j.err
#SBATCH --nodes=1
${ntasks}
${cpus_task}
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2GB
${constraint}
${gpu_line}

source /shared/environmentrc.platform_spack_0.23.0.dev0
spack load gromacs@2024.2

# Variables
GMX_BIN="gmx"
WD="/shared/projects/common/ONE_HAIR_CG/${name}"
SIMDIR="/shared/projects/common/ONE_HAIR_CG/${name}/sim"
mdps="/shared/projects/common/ONE_HAIR_CG/${name}/mdp_files"
forcefield="/home/acabezon/forcefields/martini_v300"

export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export GMXLIB="\${forcefield}"
MPI_RANKS="\${SLURM_NTASKS}"

#Execution
echo "Job \$SLURM_JOB_ID started at \$(date)"
echo "Job allocated in \$SLURM_NODELIST"

STARTTIME=\$(date +%s)
# Define the runs =================================================================================
run_gmx () { #-----------------------------------------------------------------
mkdir -p sim
cd sim

if [[ ! -f minimization.gro ]]; then
	\$GMX_BIN grompp -p \${WD}/system.top -c \${WD}/final_system.gro -f \${mdps}/em.mdp -o minimization -maxwarn 10 
	\$GMX_BIN mdrun -v -deffnm minimization -ntomp \$OMP_NUM_THREADS -ntmpi \$MPI_RANKS
fi

if [[ -f minimization.gro ]]; then
        tpr="relax"                                           
        \$GMX_BIN grompp -p \${WD}/system.top -c \${SIMDIR}/minimization.gro -f \${mdps}/\${tpr}.mdp -o \${tpr} -maxwarn 10 -n ../index.ndx
        ${gmx_mdrun}
fi

if [[ -f relax.gro ]]; then
        tpr="compress-rest"                                                  
        \$GMX_BIN grompp -p \${WD}/system.top -c relax.gro -t relax.cpt -f \${mdps}/\${tpr}.mdp -o \${tpr} -maxwarn 10  -n ../index.ndx
        ${gmx_mdrun}
fi                                                                              

if [[ -f compress-rest.gro ]]; then
        tpr="compress"                                                      
        \$GMX_BIN grompp -p \${WD}/system.top -c compress-rest.gro -t compress-rest.cpt -f \${mdps}/\${tpr}.mdp -o \${tpr} -maxwarn 10  -n ../index.ndx
        ${gmx_mdrun}
fi


if [[ -f compress.gro ]]; then                                                      
        tpr="npt"                                                      
        \$GMX_BIN grompp -p \${WD}/system.top -c compress.gro -t compress.cpt -f \${mdps}/\${tpr}.mdp -o \${tpr} -maxwarn 10  -n ../index.ndx
        ${gmx_mdrun}
fi

                                                                            
if [[ -f npt.gro ]]; then                                                   
	cd \${WD} || exit 1
	#sbatch submit_sim.sh
fi

cd \${WD}
} #-----------------------------------------------------------------

# Execute the runs =================================================================================
run_gmx

ENDTIME=\$(date +%s)
echo "Total runtime: \$((\$ENDTIME - \$STARTTIME)) seconds."
echo "Job completed successfully at:" \$date
exit 0
EoF



