# in your data dir, 
# run $bash /data/home/huiyuan23/DIFUSCO/bash_scripts/data/mis/download_satlib.sh

# # DOWNLOAD SATLIB
# # Define the arrays with the different values for k, n, and m
# m_values=("m403" "m411" "m418" "m423" "m429" "m435" "m441" "m449")
# b_values=("b10" "b30" "b50" "b70" "b90")

# # Base URL
# base_url="https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS"
# current_dir=$(pwd)

# # Loop over the values of k, n, and m
# for m in "${m_values[@]}"; do
#     for b in "${b_values[@]}"; do
#         # Construct the URL
#         url="${base_url}/CBS_k3_n100_${m}_${b}.tar.gz"
        
#         # Download the file
#         wget $url
        
#         dir_name="${current_dir}/CBS_k3_n100_${m}_${b}"
        
#         mkdir -p $dir_name
        
#         # Extract the tar.gz file
#         tar -xzf "CBS_k3_n100_${m}_${b}.tar.gz" -C $dir_name
        
#         # Remove the tar.gz file
#         rm "CBS_k3_n100_${m}_${b}.tar.gz"
#     done
# done

# CREATE TEST SPLIT

# Directory to move files from
from_dir="/data/shared/huiyuan/mis/mis_satlib_train_new"

# Directory to move files to
to_dir="/data/shared/huiyuan/mis/mis_satlib_test_new"

mkdir ${to_dir}

# Text file with the list of file names
file_list="/data/home/huiyuan23/DIFUSCO/bash_scripts/data/mis/test_mis.txt"

# Read the file names from the text file
while IFS= read -r file_name
do
    # Move the file to the new director
    mv "${from_dir}/${file_name}" "${to_dir}"
    #ls ${to_dir}/${file_name}
done < "$file_list"